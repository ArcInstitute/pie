"""Lightning datamodule: preprocessed dirs, split statistics, evidence, datasets and loaders."""

from __future__ import annotations

import functools
import logging
import os
import time
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import torch
from pydantic import field_validator
from torch.utils.data import DataLoader

from pie.assets import resolve_asset
from pie.data import samplers
from pie.data.dataset import (
    LABEL_KINDS,
    Aliases,
    Batch,
    Labels,
    PieDataset,
    RowRef,
    SourceLookup,
    collate,
    load_aliases,
)
from pie.data.delta_p import DeltaPConfig, DeltaPGrid, fit_grid, train_percentile
from pie.data.evidence import (
    EvidenceConfig,
    EvidenceIndex,
    build_evidence_index,
    check_leakage,
    load_or_build_evidence,
)
from pie.data.preprocessed import PreprocessedDir
from pie.data.splits import (
    Split,
    check_disjoint,
    load_split,
    parse_split_key,
    resolve_split,
    split_pairs,
)
from pie.sources.contract import SOURCE_NAMES, read_source, read_source_aliases
from pie.utils import StrictModel, read_json, require_env, resolve_path, sha256_file, write_json

log = logging.getLogger(__name__)

DATA_STATS = "data_stats.json"
DATA_STATS_FORMAT = 1
TRAIN_JSON = "train.json"
VAL_JSON = "val.json"
_POLL_S = 5.0
LEGACY_ALIASES_PATH = "data/sources/aliases.yaml"
# Start of this launch, used when TORCHELASTIC_RUN_ID gives no launch identity: a follower then
# accepts only a handoff that rank 0 wrote after this launch started. It is kept in the
# environment so ranks that a launcher starts later as child processes share the same start.
_LAUNCH_T0 = float(os.environ.setdefault("PIE_LAUNCH_T0", repr(time.time())))

Pair = tuple[str, str, str]


def source_aliases(source_paths: Mapping[str, Path], aliases_path: str | None) -> Aliases:
    """Each source dir's aliases.yaml, then an optional extra file (old multi-source format).

    Runs saved before aliases moved into the source dirs store LEGACY_ALIASES_PATH; it maps
    to the packaged copy of that file, so their predictions do not change.
    """
    aliases: Aliases = {name: read_source_aliases(p) for name, p in source_paths.items()}
    if aliases_path is None:
        return aliases
    extra_path = (
        Path(str(files("pie.data") / "legacy_aliases.yaml"))
        if aliases_path == LEGACY_ALIASES_PATH
        else resolve_path(aliases_path)
    )
    for name, table in load_aliases(extra_path).items():
        aliases.setdefault(name, {}).update(table)
    return aliases


class DataConfig(StrictModel):
    """data.* (asset paths are local strings or explicit hf://datasets/... references)."""

    preprocessed_dirs: list[str]
    dataset_weights: dict[str, float] | None
    split_dir: str
    source_dirs: dict[str, str]
    gene_text_dir: str
    aliases_path: str | None  # an extra multi-source aliases file over each source's aliases.yaml
    delta_p: DeltaPConfig
    evidence: EvidenceConfig
    batch_size: int
    num_workers: int
    fdr_threshold: float

    @field_validator("preprocessed_dirs")
    @classmethod
    def check_preprocessed_dirs(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("data.preprocessed_dirs must list at least one dir")
        return value

    @field_validator("source_dirs")
    @classmethod
    def check_source_dirs(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = [name for name in value if name not in SOURCE_NAMES or name == "gene_text"]
        if unknown:
            raise ValueError(f"data.source_dirs names are not model sources: {unknown}")
        return value


class DataStats(StrictModel):
    """Split statistics fitted on rank 0, handed off via data_stats.json, kept in checkpoints."""

    format_version: int
    datasets: list[str]
    delta_p: DeltaPGrid
    per_dir_percentile: dict[str, float | None]
    evidence_key: str
    evidence_datasets: list[str]
    train_json_sha256: str
    source_dims: dict[str, int]
    gene_query_dim: int

    @property
    def n_donor_datasets(self) -> int:
        return len(self.evidence_datasets)


def _check_ctrl_means(d: PreprocessedDir) -> None:
    """A context row is all-NaN (no controls) or all finite; anything else is malformed."""
    ctrl = np.asarray(d.ctrl_means)
    missing = np.isnan(ctrl).all(axis=1)
    bad = ~missing & ~np.isfinite(ctrl).all(axis=1)
    if bad.any():
        names = [d.contexts[int(i)] for i in np.flatnonzero(bad)]
        raise ValueError(
            f"{d.dataset}: partially missing or non-finite control means for contexts {names}"
        )


def _dataset_weights(weights: Mapping[str, float] | None, names: list[str]) -> dict[str, float]:
    if weights is None:
        return dict.fromkeys(names, 1.0)
    if set(weights) != set(names):
        raise ValueError(
            f"data.dataset_weights keys {sorted(weights)} must equal the datasets {sorted(names)}"
        )
    if any(w < 0 for w in weights.values()):
        raise ValueError("data.dataset_weights must be non-negative")
    if sum(weights.values()) <= 0:
        raise ValueError("data.dataset_weights sum to zero")
    return {name: float(weights[name]) for name in names}


def _launch_id() -> str | None:
    """TORCHELASTIC_RUN_ID, or None when it is unset or torchrun's default 'none'."""
    run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    return None if run_id in (None, "", "none") else run_id


def _is_this_launch(payload: object, launch_id: str | None) -> bool:
    if not isinstance(payload, dict) or payload.get("launch_id") != launch_id:
        return False
    if launch_id is not None:
        return True
    written_at = payload.get("written_at")
    return isinstance(written_at, (int, float)) and written_at >= _LAUNCH_T0


def _wait_for_stats(path: Path, launch_id: str | None, rank: int, timeout_s: float) -> DataStats:
    deadline = time.monotonic() + timeout_s
    while True:
        if path.is_file():
            payload = read_json(path)
            if isinstance(payload, dict) and _is_this_launch(payload, launch_id):
                return DataStats.model_validate(payload["stats"])
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"rank {rank}: {path} for launch {launch_id!r} did not appear within "
                f"{timeout_s:g} s"
            )
        time.sleep(_POLL_S)


class PieDataModule(L.LightningDataModule):
    """Preprocessed dirs + knowledge sources -> split statistics, datasets and loaders."""

    def __init__(
        self,
        cfg: DataConfig,
        run_dir: Path | None,
        stats: DataStats | None = None,
        *,
        seed: int = 0,
        evidence_source: DataConfig | None = None,
    ) -> None:
        """`stats` given = pinned (eval/infer, resume): never refit. run_dir None only if pinned.

        `seed` seeds the train sampler. `evidence_source` is the training data config whose dirs
        and train.json define the evidence (default `cfg`).
        """
        super().__init__()
        if run_dir is None and stats is None:
            raise ValueError("run_dir is required unless data stats are pinned")
        self.cfg = cfg
        self.run_dir = run_dir
        self.seed = seed
        self._stats = stats
        self._evidence_cfg = evidence_source if evidence_source is not None else cfg
        self.dirs = [
            PreprocessedDir.open(resolve_asset(p, kind="preprocessed"))
            for p in cfg.preprocessed_dirs
        ]
        names = [d.dataset for d in self.dirs]
        if len(set(names)) != len(names):
            raise ValueError(f"data.preprocessed_dirs name a dataset twice: {names}")
        for d in self.dirs:
            _check_ctrl_means(d)
        self._weights = _dataset_weights(cfg.dataset_weights, names)
        union: dict[str, int] = {}
        for d in self.dirs:
            for gene in d.genes:
                union.setdefault(gene, len(union))
        self.gene_axis = list(union)
        self.gene_union_ids = [
            np.asarray([union[gene] for gene in d.genes], dtype=np.int64) for d in self.dirs
        ]
        source_paths = {
            name: resolve_asset(p, kind="source") for name, p in cfg.source_dirs.items()
        }
        sources = {name: read_source(path) for name, path in source_paths.items()}
        self.source_dims = {name: src.meta.dim for name, src in sources.items()}
        self._lookup = SourceLookup(sources, source_aliases(source_paths, cfg.aliases_path))
        self._gene_text = read_source(resolve_asset(cfg.gene_text_dir, kind="source"))
        if self._gene_text.meta.layout != "dense" or self._gene_text.meta.index != "gene":
            raise ValueError("data.gene_text_dir must be a dense, gene-indexed source")
        if stats is not None:
            if list(stats.source_dims.items()) != list(self.source_dims.items()):
                raise ValueError(
                    f"pinned source_dims {stats.source_dims} differ from data.source_dirs "
                    f"{self.source_dims}"
                )
            if stats.gene_query_dim != self._gene_text.meta.dim:
                raise ValueError("pinned gene_query_dim differs from data.gene_text_dir")
        self._evidence_index: EvidenceIndex | None = None
        self._train_pairs: set[Pair] = set()
        self._val_sharded = False
        self.train_dataset: PieDataset | None = None
        self.val_dataset: PieDataset | None = None

    # --- statistics -----------------------------------------------------------------------

    @property
    def stats(self) -> DataStats:
        if self._stats is None:
            raise RuntimeError("data stats are not available; call setup_stats() first")
        return self._stats

    def setup_stats(self, rank: int, timeout_s: float = 14400.0) -> DataStats:
        """Rank 0 fits the grid and builds or loads the evidence, then writes data_stats.json;
        other ranks wait for the file of their launch. Pinned stats return at once."""
        if self._stats is not None:
            return self._stats
        if self.run_dir is None:
            raise ValueError("setup_stats needs a run_dir")
        path = self.run_dir / DATA_STATS
        launch_id = _launch_id()
        if rank == 0:
            path.unlink(missing_ok=True)  # a previous launch's handoff must never be read
            stats = self._fit_stats()
            self.run_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "launch_id": launch_id,
                "written_at": time.time(),
                "stats": stats.model_dump(mode="json"),
            }
            write_json(path, payload)
            log.info("wrote %s (grid %s)", path, stats.delta_p)
        else:
            stats = _wait_for_stats(path, launch_id, rank, timeout_s)
        self._stats = stats
        return stats

    def _fit_stats(self) -> DataStats:
        controls_only = [d.dataset for d in self.dirs if d.controls_only]
        if controls_only:
            raise ValueError(f"controls-only dirs cannot be trained on: {controls_only}")
        split_dir = resolve_asset(self.cfg.split_dir, kind="splits")
        train_rows = resolve_split(load_split(split_dir / TRAIN_JSON), self.dirs)
        pct = self.cfg.delta_p.max_delta_percentile
        per_dir = {
            d.dataset: train_percentile(d.delta_p, train_rows[d.dataset], pct) for d in self.dirs
        }
        grid = fit_grid(per_dir, self.cfg.delta_p)
        index = self._ensure_evidence(split_pairs(load_split(split_dir / VAL_JSON)))
        ev = index.evidence
        return DataStats(
            format_version=DATA_STATS_FORMAT,
            datasets=[d.dataset for d in self.dirs],
            delta_p=grid,
            per_dir_percentile=per_dir,
            evidence_key=ev.key,
            evidence_datasets=list(ev.contributing_datasets),
            train_json_sha256=ev.train_json_sha256,
            source_dims=dict(self.source_dims),
            gene_query_dim=self._gene_text.meta.dim,
        )

    def _ensure_evidence(self, forbidden: set[Pair]) -> EvidenceIndex:
        """Load (or build) the evidence once; every later call re-runs the leakage check."""
        if self._evidence_index is not None:
            check_leakage(self._evidence_index.evidence, self._train_pairs, forbidden)
            return self._evidence_index
        ecfg = self._evidence_cfg
        dirs = (
            self.dirs
            if ecfg is self.cfg
            else [
                PreprocessedDir.open(resolve_asset(p, kind="preprocessed"))
                for p in ecfg.preprocessed_dirs
            ]
        )
        train_path = resolve_asset(ecfg.split_dir, kind="splits") / TRAIN_JSON
        train_sha = sha256_file(train_path)
        if self._stats is not None and train_sha != self._stats.train_json_sha256:
            raise ValueError(
                f"{train_path} sha256 {train_sha[:12]} differs from the pinned data stats"
            )
        train_split = load_split(train_path)
        cache_root = Path(require_env("PIE_CACHE_DIR")["PIE_CACHE_DIR"])
        ev = load_or_build_evidence(
            dirs, train_split, train_sha, ecfg.evidence, ecfg.fdr_threshold, cache_root, forbidden
        )
        if self._stats is not None and ev.key != self._stats.evidence_key:
            raise ValueError(
                f"evidence key {ev.key[:12]} differs from the pinned data stats "
                f"{self._stats.evidence_key[:12]}"
            )
        self._train_pairs = split_pairs(train_split)
        self._evidence_index = build_evidence_index(ev, self.gene_axis)
        return self._evidence_index

    # --- rows and datasets ----------------------------------------------------------------

    def gene_query_text(self) -> torch.Tensor:
        """(len(gene_axis), gene_query_dim) float32 gene_text rows in gene-axis order."""
        src = self._gene_text
        missing = [g for g in self.gene_axis if g not in src.key_to_row]
        if missing:
            raise KeyError(f"{len(missing)} genes missing from gene_text: {missing[:10]}")
        rows = np.asarray([src.key_to_row[g] for g in self.gene_axis], dtype=np.int64)
        return torch.from_numpy(np.array(src.embeddings[rows], dtype=np.float32))

    def _rows(self, split: Split) -> list[RowRef]:
        per_dataset = resolve_split(split, self.dirs)
        rows: list[RowRef] = []
        for di, d in enumerate(self.dirs):
            selected = per_dataset[d.dataset]
            if selected.size == 0:
                continue
            keys = d.row_keys()
            for r in selected:
                context, perturbation = keys[int(r)]
                rows.append(RowRef(di, int(r), context, perturbation))
        return rows

    def rows_from_split(self, split_path: Path) -> list[RowRef]:
        """Resolve any split file (strict keys) to RowRefs in (dir order, row order)."""
        return self._rows(load_split(split_path))

    def rows_from_query(self, query_path: Path) -> list[RowRef]:
        """Query JSON {"dataset.ctx": [perts]}: the context must exist in that dir; row = -1."""
        query = load_split(query_path)
        by_name = {d.dataset: i for i, d in enumerate(self.dirs)}
        per_dir: dict[int, list[RowRef]] = {}
        offenders: list[str] = []
        for key, perts in query.items():
            dataset, context = parse_split_key(key)
            di = by_name.get(dataset)
            if di is None or context not in self.dirs[di].meta.context_to_id:
                offenders.append(key)
                continue
            per_dir.setdefault(di, []).extend(RowRef(di, -1, context, p) for p in perts)
        if offenders:
            raise ValueError(f"query keys name unknown datasets or contexts: {offenders}")
        return [ref for di in sorted(per_dir) for ref in per_dir[di]]

    def make_dataset(self, rows: list[RowRef], labels: str) -> PieDataset:
        """Dataset over `rows`; for eval/query rows, those rows are forbidden as evidence donors."""
        if labels not in LABEL_KINDS:
            raise ValueError(f"labels must be one of {LABEL_KINDS}, got {labels!r}")
        forbidden: set[Pair] = (
            set()
            if labels == "train"
            else {(self.dirs[r.dir_index].dataset, r.context, r.perturbation) for r in rows}
        )
        index = self._ensure_evidence(forbidden)
        return PieDataset(
            self.dirs,
            rows,
            self.gene_union_ids,
            self._lookup,
            index,
            self.cfg.fdr_threshold,
            cast(Labels, labels),
        )

    def collate(self, samples: list[dict[str, Any]]) -> Batch:
        # `collate` below is the module-level function from pie.data.dataset.
        return collate(samples, self.source_dims)

    def _collate_fn(self) -> functools.partial[Batch]:
        return functools.partial(collate, source_dims=dict(self.source_dims))

    # --- Lightning hooks ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        """'fit': train (labels='train') and val (labels='eval') datasets from split_dir."""
        if stage != "fit" or self.train_dataset is not None:
            return
        if self._stats is None:
            raise RuntimeError("call setup_stats() before setup('fit')")
        split_dir = resolve_asset(self.cfg.split_dir, kind="splits")
        train = load_split(split_dir / TRAIN_JSON)
        val = load_split(split_dir / VAL_JSON)
        check_disjoint({"train": train, "val": val})
        self._ensure_evidence(split_pairs(val))
        self.train_dataset = self.make_dataset(self._rows(train), "train")
        self.val_dataset = self.make_dataset(self._rows(val), "eval")

    def _world(self) -> tuple[int, int]:
        trainer = self.trainer
        if trainer is None:
            return 1, 0
        return int(trainer.world_size), int(trainer.global_rank)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("call setup('fit') first")
        order = sorted(d.dataset for d in self.dirs)
        dir_group = [order.index(d.dataset) for d in self.dirs]
        group_ids = np.asarray(
            [dir_group[r.dir_index] for r in self.train_dataset.rows], dtype=np.int64
        )
        weights = {order.index(name): w for name, w in self._weights.items()}
        num_replicas, rank = self._world()
        sampler = samplers.build_train_sampler(group_ids, weights, num_replicas, rank, self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self._collate_fn(),
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("call setup('fit') first")
        num_replicas, rank = self._world()
        val = samplers.build_val_loader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate_fn(),
            num_replicas=num_replicas,
            rank=rank,
        )
        self._val_sharded = val.sharded
        return val.loader

    @property
    def val_sharded(self) -> bool:
        """True when val_dataloader() returned a per-rank shard (set by val_dataloader())."""
        return self._val_sharded
