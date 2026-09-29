"""Knowledge-source lookup, labelled and query rows, and batch collation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import Dataset

from pie.data.evidence import EVIDENCE_KEYS, EvidenceIndex, serve_evidence
from pie.data.preprocessed import PreprocessedDir, target_gene_index
from pie.sources.contract import SOURCE_NAMES, Source

Aliases = dict[str, dict[str, str]]  # source name -> {key: canonical key}
Labels = Literal["train", "eval", "none"]
LABEL_KINDS: tuple[str, ...] = ("train", "eval", "none")
NO_SOURCE = "__no_source__"  # carrier for rows without any source token


def load_aliases(path: Path) -> Aliases:
    """Strict YAML load: top-level keys are source names, values map a key to its canonical key."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of source name to aliases")
    aliases: Aliases = {}
    for name, entries in raw.items():
        if name not in SOURCE_NAMES:
            raise ValueError(f"{path}: unknown source {name!r}")
        table = {} if entries is None else entries
        if not isinstance(table, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in table.items()
        ):
            raise ValueError(f"{path}: aliases of {name!r} must map str -> str")
        aliases[str(name)] = dict(table)
    return aliases


class SourceLookup:
    """Resolves a row's tokens per source; an alias is used only on a direct miss and only if its
    target exists in that source."""

    def __init__(self, sources: Mapping[str, Source], aliases: Aliases) -> None:
        self._sources = dict(sources)
        self._aliases = {name: dict(aliases.get(name, {})) for name in self._sources}

    @property
    def dims(self) -> dict[str, int]:
        """name -> dim, in `sources` insertion order (= data.source_dirs order)."""
        return {name: src.meta.dim for name, src in self._sources.items()}

    def _resolve(self, name: str, key: str) -> str | None:
        rows = self._sources[name].key_to_row
        if key in rows:
            return key
        target = self._aliases[name].get(key)
        if target is not None and target in rows:
            return target
        return None

    def tokens(self, name: str, *, context: str, perturbation: str) -> np.ndarray | None:
        """(m, D) tokens or None when absent. Dense rows as float32, token rows in on-disk dtype."""
        src = self._sources[name]
        key = self._resolve(name, context if src.meta.index == "context" else perturbation)
        if key is None:
            return None
        values = src.tokens(key)
        if src.meta.layout == "dense":
            return np.array(values, dtype=np.float32)
        return np.array(values)


@dataclass(frozen=True)
class RowRef:
    dir_index: int  # index into the datamodule's dirs
    row: int  # row in that dir; -1 for a query row (no labels)
    context: str
    perturbation: str


@dataclass
class GeneGroup:
    """Rows of one preprocessed dir; per-gene tensors are on that dir's local gene axis (G_d)."""

    rows: Tensor  # (b,) int64 positions in the batch
    dir_index: int
    gene_ids: Tensor  # (G_d,) int64 into the gene axis
    ctrl_means: Tensor  # (b, G_d) float32
    evidence: dict[str, Tensor]  # EVIDENCE_KEYS -> (b, G_d, width) float32
    fold_changes: Tensor | None  # (b, G_d) float32 linear, 0 where untested
    de_mask: Tensor | None  # (b, G_d) bool = (fdr < fdr_threshold) & tested
    tested: Tensor | None  # (b, G_d) bool
    delta_p: Tensor | None  # (b, G_d) float32
    lfc_true: Tensor | None  # (b, G_d) float64; eval rows only


@dataclass
class Batch:
    source_tokens: dict[str, Tensor]  # name -> (B, m_max, D), names sorted, then NO_SOURCE
    source_masks: dict[str, Tensor]  # name -> (B, m_max) bool
    dataset_ids: Tensor  # (B,) int64 dir index (= DataStats.datasets order)
    ctx_names: list[str]
    pert_names: list[str]
    row_index: Tensor  # (B,) int64 row in its dir, -1 for query rows
    target_gene: Tensor  # (B,) int64 loss target (local idx, -1 if none or untested)
    target_gene_idx: Tensor  # (B,) int64 metric target (local idx or -1)
    groups: list[GeneGroup]  # sorted by dir_index


class PieDataset(Dataset):
    """Rows of one or more preprocessed dirs, each served on its own dir's gene axis."""

    def __init__(
        self,
        dirs: Sequence[PreprocessedDir],
        rows: Sequence[RowRef],
        gene_union_ids: Sequence[np.ndarray],
        sources: SourceLookup,
        evidence: EvidenceIndex,
        fdr_threshold: float,
        labels: Labels,
    ) -> None:
        """'train' serves no lfc_true; 'eval' serves lfc_true; 'none' serves no label field."""
        if labels not in LABEL_KINDS:
            raise ValueError(f"labels must be one of {LABEL_KINDS}, got {labels!r}")
        if labels != "none" and any(ref.row < 0 for ref in rows):
            raise ValueError("a labelled dataset cannot serve query rows (row -1)")
        self.dirs = list(dirs)
        self.rows = list(rows)
        self.gene_union_ids = [np.asarray(ids, dtype=np.int64) for ids in gene_union_ids]
        self.sources = sources
        self.evidence = evidence
        self.fdr_threshold = float(fdr_threshold)
        self.labels = labels
        self._gene_index = [{gene: i for i, gene in enumerate(d.genes)} for d in self.dirs]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ref = self.rows[i]
        d = self.dirs[ref.dir_index]
        tokens: dict[str, Tensor] = {}
        for name in self.sources.dims:
            values = self.sources.tokens(name, context=ref.context, perturbation=ref.perturbation)
            if values is not None:
                tokens[name] = torch.from_numpy(values)
        gene_ids = self.gene_union_ids[ref.dir_index]
        ctrl = np.array(d.ctrl_means[d.meta.context_to_id[ref.context]], dtype=np.float32)
        blocks = serve_evidence(
            self.evidence,
            dataset=d.dataset,
            context=ref.context,
            perturbation=ref.perturbation,
            gene_union_ids=gene_ids,
        )
        target = target_gene_index(ref.perturbation, self._gene_index[ref.dir_index])
        sample: dict[str, Any] = {
            "dir_index": ref.dir_index,
            "row": ref.row,
            "context": ref.context,
            "perturbation": ref.perturbation,
            "source_tokens": tokens,
            "gene_ids": torch.from_numpy(gene_ids),
            "ctrl_mean": torch.from_numpy(ctrl),
            "evidence": {key: torch.from_numpy(value) for key, value in blocks.items()},
            "target_gene": target,
            "target_gene_idx": target,
        }
        if self.labels == "none":
            return sample
        tested = np.array(d.tested[ref.row], dtype=bool)
        fdr = np.array(d.fdr[ref.row], dtype=np.float32)
        if target >= 0 and not tested[target]:
            sample["target_gene"] = -1
        sample["fold_changes"] = torch.from_numpy(
            np.array(d.fold_changes[ref.row], dtype=np.float32)
        )
        sample["de_mask"] = torch.from_numpy((fdr < self.fdr_threshold) & tested)
        sample["tested"] = torch.from_numpy(tested)
        sample["delta_p"] = torch.from_numpy(np.array(d.delta_p[ref.row], dtype=np.float32))
        if self.labels == "eval":
            sample["lfc_true"] = torch.from_numpy(
                np.array(d.lfc_true[ref.row], dtype=np.float64)
            )
        return sample


def _collate_sources(
    tokens: Sequence[Mapping[str, Tensor]], source_dims: Mapping[str, int]
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Pad each source present in the batch to its longest row; absent rows stay masked zeros."""
    names = sorted({name for row in tokens for name in row})
    padded_tokens: dict[str, Tensor] = {}
    padded_masks: dict[str, Tensor] = {}
    for name in names:
        present = [row[name] for row in tokens if name in row]
        dtype = present[0].dtype
        dim = int(source_dims[name])
        m_max = max(int(t.shape[0]) for t in present)
        padded = torch.zeros(len(tokens), m_max, dim, dtype=dtype)
        masks = torch.zeros(len(tokens), m_max, dtype=torch.bool)
        for i, row in enumerate(tokens):
            if name in row:
                m = int(row[name].shape[0])
                padded[i, :m, :] = row[name]
                masks[i, :m] = True
        padded_tokens[name] = padded
        padded_masks[name] = masks
    has_no_source = [len(row) == 0 for row in tokens]
    if any(has_no_source):
        padded_tokens[NO_SOURCE] = torch.zeros(len(tokens), 1, 1)
        padded_masks[NO_SOURCE] = torch.tensor(has_no_source, dtype=torch.bool)[:, None]
    return padded_tokens, padded_masks


def _gene_group(
    samples: Sequence[dict[str, Any]], dir_index: int, positions: list[int]
) -> GeneGroup:
    members = [samples[p] for p in positions]
    labelled = {"fold_changes" in s for s in members}
    if len(labelled) != 1:
        raise ValueError(f"dir {dir_index}: a batch mixes labelled and query rows")
    has_labels = labelled.pop()
    has_truth = all("lfc_true" in s for s in members)

    def stack(key: str) -> Tensor:
        return torch.stack([s[key] for s in members])

    return GeneGroup(
        rows=torch.tensor(positions, dtype=torch.long),
        dir_index=dir_index,
        gene_ids=members[0]["gene_ids"],
        ctrl_means=stack("ctrl_mean"),
        evidence={key: torch.stack([s["evidence"][key] for s in members]) for key in EVIDENCE_KEYS},
        fold_changes=stack("fold_changes") if has_labels else None,
        de_mask=stack("de_mask") if has_labels else None,
        tested=stack("tested") if has_labels else None,
        delta_p=stack("delta_p") if has_labels else None,
        lfc_true=stack("lfc_true") if has_truth else None,
    )


def collate(samples: list[dict[str, Any]], source_dims: Mapping[str, int]) -> Batch:
    """Pad sources per batch and group per-gene fields by dir (groups sorted by dir index)."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    tokens, masks = _collate_sources([s["source_tokens"] for s in samples], source_dims)
    by_dir: dict[int, list[int]] = {}
    for position, sample in enumerate(samples):
        by_dir.setdefault(int(sample["dir_index"]), []).append(position)
    return Batch(
        source_tokens=tokens,
        source_masks=masks,
        dataset_ids=torch.tensor([int(s["dir_index"]) for s in samples], dtype=torch.long),
        ctx_names=[str(s["context"]) for s in samples],
        pert_names=[str(s["perturbation"]) for s in samples],
        row_index=torch.tensor([int(s["row"]) for s in samples], dtype=torch.long),
        target_gene=torch.tensor([int(s["target_gene"]) for s in samples], dtype=torch.long),
        target_gene_idx=torch.tensor(
            [int(s["target_gene_idx"]) for s in samples], dtype=torch.long
        ),
        groups=[_gene_group(samples, d, by_dir[d]) for d in sorted(by_dir)],
    )
