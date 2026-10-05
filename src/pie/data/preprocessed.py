"""On-disk format of a preprocessed dataset dir: written by pie prep, read by the runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from pie.utils import StrictModel, atomic_dir, sha256_bytes, sha256_file, write_json

FORMAT_VERSION = 1
META = "meta.json"
FOLD_CHANGES = "fold_changes.npy"  # (N, G) float32, linear fold change, 0.0 where untested
FDR = "fdr.npy"  # (N, G) float32, 1.0 where untested
TESTED = "tested.npy"  # (N, G) bool
LFC_TRUE = "lfc_true.npy"  # (N, G) float64, log2 fold change, NaN where untested
DELTA_P = "delta_p.npy"  # (N, G) float32, pseudobulk(pert) - ctrl_mean
CTRL_MEANS = "ctrl_means.npy"  # (C, G) float32, all-NaN row for a context without controls
CTX_IDS = "ctx_ids.npy"  # (N,) int32 into context_to_id
PERT_IDS = "pert_ids.npy"  # (N,) int32 into pert_to_id
CONTEXTS_FILE = "contexts.yaml"  # optional context -> Cellosaurus map (pie sources context_text)
LABEL_ARRAYS: tuple[str, ...] = (FOLD_CHANGES, FDR, TESTED, LFC_TRUE, DELTA_P, CTX_IDS, PERT_IDS)
ALL_ARRAYS: tuple[str, ...] = (*LABEL_ARRAYS, CTRL_MEANS)
CONTROLS_ONLY_ARRAYS: tuple[str, ...] = (CTRL_MEANS,)
ARRAY_DTYPES: dict[str, np.dtype] = {
    FOLD_CHANGES: np.dtype("float32"),
    FDR: np.dtype("float32"),
    TESTED: np.dtype("bool"),
    LFC_TRUE: np.dtype("float64"),
    DELTA_P: np.dtype("float32"),
    CTRL_MEANS: np.dtype("float32"),
    CTX_IDS: np.dtype("int32"),
    PERT_IDS: np.dtype("int32"),
}


class PreprocessedMeta(StrictModel):
    """Contents of meta.json. Holds no paths and no argv."""

    format_version: int
    dataset: str
    genes: list[str]
    context_to_id: dict[str, int]
    pert_to_id: dict[str, int]
    pert_ensembl: dict[str, str] = Field(default_factory=dict)
    pert_kind: Literal["gene", "drug"]
    control_label: str
    num_rows: int
    num_genes: int
    num_contexts: int
    num_perts: int
    controls_only: bool
    tool_version: str
    array_sha256: dict[str, str]


def _check_vocab(name: str, vocab: Mapping[str, int]) -> None:
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError(f"{name} ids must be 0..{len(vocab) - 1} without gaps or repeats")


def _check_meta(meta: PreprocessedMeta) -> None:
    if meta.format_version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported preprocessed format_version {meta.format_version} "
            f"(expected {FORMAT_VERSION})"
        )
    if len(meta.genes) != meta.num_genes or len(set(meta.genes)) != meta.num_genes:
        raise ValueError("genes must hold num_genes unique symbols")
    if len(meta.context_to_id) != meta.num_contexts:
        raise ValueError("context_to_id must hold num_contexts entries")
    if len(meta.pert_to_id) != meta.num_perts:
        raise ValueError("pert_to_id must hold num_perts entries")
    _check_vocab("context_to_id", meta.context_to_id)
    _check_vocab("pert_to_id", meta.pert_to_id)
    for pert, gene_id in meta.pert_ensembl.items():
        if pert not in meta.pert_to_id:
            raise ValueError(f"pert_ensembl key {pert!r} is not a perturbation")
        if not gene_id.startswith("ENSG"):
            raise ValueError(f"pert_ensembl value {gene_id!r} for {pert!r} must start with ENSG")
    if meta.controls_only and (meta.num_rows != 0 or meta.num_perts != 0):
        raise ValueError("a controls-only dir has no rows and no perturbations")


def _expected_arrays(meta: PreprocessedMeta) -> tuple[str, ...]:
    return CONTROLS_ONLY_ARRAYS if meta.controls_only else ALL_ARRAYS


def _expected_shapes(meta: PreprocessedMeta) -> dict[str, tuple[int, ...]]:
    n, g = meta.num_rows, meta.num_genes
    return {
        FOLD_CHANGES: (n, g),
        FDR: (n, g),
        TESTED: (n, g),
        LFC_TRUE: (n, g),
        DELTA_P: (n, g),
        CTRL_MEANS: (meta.num_contexts, g),
        CTX_IDS: (n,),
        PERT_IDS: (n,),
    }


def write_preprocessed(
    out: Path,
    meta: PreprocessedMeta,
    arrays: Mapping[str, np.ndarray],
    *,
    overwrite: bool = False,
    extra_files: Mapping[str, bytes] | None = None,
) -> Path:
    """Validate names/dtypes/shapes, np.save each array, fill meta.array_sha256, write meta.json.

    `extra_files` ({file name: bytes}, e.g. CONTEXTS_FILE) are written next to meta.json.

    The dir is published atomically via atomic_dir(out). An empty existing `out` is replaced;
    a non-empty one raises FileExistsError unless `overwrite`, which replaces it once the new dir
    is complete.
    """
    _check_meta(meta)
    names = _expected_arrays(meta)
    if set(arrays) != set(names):
        raise ValueError(f"expected arrays {sorted(names)}, got {sorted(arrays)}")
    shapes = _expected_shapes(meta)
    for name in names:
        arr = arrays[name]
        if arr.dtype != ARRAY_DTYPES[name]:
            raise ValueError(f"{name}: dtype {arr.dtype}, expected {ARRAY_DTYPES[name]}")
        if arr.shape != shapes[name]:
            raise ValueError(f"{name}: shape {arr.shape}, expected {shapes[name]}")
    if not meta.controls_only:
        for name, bound in ((CTX_IDS, meta.num_contexts), (PERT_IDS, meta.num_perts)):
            ids = arrays[name]
            if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= bound):
                raise ValueError(f"{name}: ids outside [0, {bound})")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_dir() and not any(out.iterdir()):
        out.rmdir()
    with atomic_dir(out, overwrite=overwrite) as tmp:
        digests: dict[str, str] = {}
        for name in names:
            np.save(tmp / name, np.ascontiguousarray(arrays[name]))
            digests[name] = sha256_file(tmp / name)
        final = meta.model_copy(update={"array_sha256": digests})
        write_json(tmp / META, final.model_dump(mode="json"))
        for name, payload in (extra_files or {}).items():
            if "/" in name or name in names or name == META:
                raise ValueError(f"extra file {name!r} collides with the preprocessed format")
            (tmp / name).write_bytes(payload)
    return out


def _by_id(vocab: Mapping[str, int]) -> list[str]:
    return sorted(vocab, key=vocab.__getitem__)


class PreprocessedDir:
    """Read-only view of one preprocessed dir; arrays are np.load(mmap_mode='r')."""

    def __init__(self, path: Path, meta: PreprocessedMeta) -> None:
        self.path = path
        self.meta = meta
        self._arrays: dict[str, np.ndarray] = {}
        self._row_keys: list[tuple[str, str]] | None = None
        self._row_index: dict[tuple[str, str], int] | None = None

    @classmethod
    def open(cls, path: Path) -> PreprocessedDir:
        """Read meta.json (strict), check format_version and that every listed array exists."""
        path = Path(path)
        meta_path = path / META
        if not meta_path.is_file():
            raise FileNotFoundError(f"{meta_path} does not exist; not a preprocessed dir")
        meta = PreprocessedMeta.model_validate_json(meta_path.read_text())
        _check_meta(meta)
        names = _expected_arrays(meta)
        if set(meta.array_sha256) != set(names):
            raise ValueError(
                f"{meta_path}: array_sha256 lists {sorted(meta.array_sha256)}, "
                f"expected {sorted(names)}"
            )
        missing = [name for name in names if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{path}: missing array files {missing}")
        return cls(path, meta)

    def _array(self, name: str) -> np.ndarray:
        arr = self._arrays.get(name)
        if arr is None:
            arr = np.load(self.path / name, mmap_mode="r")
            if arr.dtype != ARRAY_DTYPES[name]:
                raise ValueError(
                    f"{self.path / name}: dtype {arr.dtype}, expected {ARRAY_DTYPES[name]}"
                )
            self._arrays[name] = arr
        return arr

    def _label_array(self, name: str) -> np.ndarray:
        if self.meta.controls_only:
            raise ValueError(f"{self.path} is controls-only; {name} is not available")
        return self._array(name)

    @property
    def dataset(self) -> str:
        return self.meta.dataset

    @property
    def genes(self) -> list[str]:
        return self.meta.genes

    @property
    def controls_only(self) -> bool:
        return self.meta.controls_only

    @property
    def contexts(self) -> list[str]:
        """Context names ordered by id."""
        return _by_id(self.meta.context_to_id)

    @property
    def pert_ensembl(self) -> dict[str, str]:
        """Perturbation -> Ensembl gene id, for the perturbations that have one (a copy)."""
        return dict(self.meta.pert_ensembl)

    @property
    def perts(self) -> list[str]:
        """Pert names ordered by id."""
        return _by_id(self.meta.pert_to_id)

    @property
    def fold_changes(self) -> np.ndarray:
        return self._label_array(FOLD_CHANGES)

    @property
    def fdr(self) -> np.ndarray:
        return self._label_array(FDR)

    @property
    def tested(self) -> np.ndarray:
        return self._label_array(TESTED)

    @property
    def lfc_true(self) -> np.ndarray:
        return self._label_array(LFC_TRUE)

    @property
    def delta_p(self) -> np.ndarray:
        return self._label_array(DELTA_P)

    @property
    def ctx_ids(self) -> np.ndarray:
        return self._label_array(CTX_IDS)

    @property
    def pert_ids(self) -> np.ndarray:
        return self._label_array(PERT_IDS)

    @property
    def ctrl_means(self) -> np.ndarray:
        return self._array(CTRL_MEANS)

    def row_keys(self) -> list[tuple[str, str]]:
        """(context, perturbation) of every row, in row order (cached)."""
        if self._row_keys is None:
            if self.meta.controls_only:
                self._row_keys = []
            else:
                contexts, perts = self.contexts, self.perts
                self._row_keys = [
                    (contexts[c], perts[p])
                    for c, p in zip(self.ctx_ids.tolist(), self.pert_ids.tolist(), strict=True)
                ]
        return self._row_keys

    def row_index(self) -> dict[tuple[str, str], int]:
        """(context, perturbation) -> row (cached); raises ValueError on a duplicate key."""
        if self._row_index is None:
            index: dict[tuple[str, str], int] = {}
            for row, key in enumerate(self.row_keys()):
                if key in index:
                    raise ValueError(f"{self.path}: duplicate row key {key}")
                index[key] = row
            self._row_index = index
        return self._row_index

    def meta_sha256(self) -> str:
        """sha256 of the meta.json bytes (covers array_sha256)."""
        return sha256_bytes((self.path / META).read_bytes())


def target_gene_index(perturbation: str, gene_index: Mapping[str, int]) -> int:
    """The single tolerant target rule: gene_index[perturbation] if the pert is a gene on the axis,
    else -1. Never raises."""
    idx = gene_index.get(perturbation)
    return -1 if idx is None else int(idx)
