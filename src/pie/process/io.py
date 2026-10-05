"""File helpers for pie-process: input globs, context masks and atomic, uncompressed writes."""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import anndata as ad
import numpy as np

from pie.utils import atomic_write_text, resolve_path

_GLOB_CHARS = frozenset("*?[")


class ParquetFrame(Protocol):
    """What the de stage writes: gpudge's polars DataFrame (or a test double)."""

    def __len__(self) -> int: ...

    def write_parquet(self, file: str) -> object: ...


def resolve_inputs(spec: str | Sequence[str]) -> list[Path]:
    """Expand a path or glob (or a list of them) into sorted, existing files with distinct names.

    Relative patterns resolve against the current directory (`pie.utils.resolve_path`).
    """
    raw = [spec] if isinstance(spec, str) else list(spec)
    patterns = [str(resolve_path(pattern)) for pattern in raw]
    files: set[Path] = set()
    for pattern in patterns:
        if _GLOB_CHARS.intersection(pattern):
            matches = [Path(match) for match in glob.glob(pattern) if Path(match).is_file()]
            if not matches:
                raise FileNotFoundError(f"input glob matches no files: {pattern}")
            files.update(matches)
        else:
            path = Path(pattern)
            if not path.is_file():
                raise FileNotFoundError(f"input file not found: {pattern}")
            files.add(path)
    ordered = sorted(files, key=str)
    names = [path.name for path in ordered]
    clashes = sorted({name for name in names if names.count(name) > 1})
    if clashes:
        raise ValueError(f"input files share a file name (outputs would collide): {clashes}")
    return ordered


def context_masks(
    adata: ad.AnnData, context_column: str | None, stem: str
) -> list[tuple[str, np.ndarray]]:
    """(context, cell mask) per context value, sorted; one context `stem` when column is None."""
    if context_column is None:
        return [(stem, np.ones(adata.n_obs, dtype=bool))]
    if context_column not in adata.obs.columns:
        raise KeyError(
            f"context column {context_column!r} not in obs; have {sorted(adata.obs.columns)}"
        )
    column = adata.obs[context_column]
    n_missing = int(column.isna().sum())
    if n_missing:
        raise ValueError(f"obs[{context_column!r}] has {n_missing} missing values")
    return [(str(ctx), (column == ctx).to_numpy(dtype=bool)) for ctx in sorted(column.unique())]


def check_writable(path: Path, overwrite: bool) -> None:
    """Raise FileExistsError when `path` exists and overwrite is off."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=true to replace it")


def _tmp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")


def write_h5ad_atomic(adata: ad.AnnData, path: Path, overwrite: bool) -> Path:
    """Write uncompressed to a hidden sibling tmp file, then os.replace it onto `path`."""
    check_writable(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_sibling(path)
    try:
        adata.write_h5ad(tmp, compression=None)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def write_parquet_atomic(frame: ParquetFrame, path: Path, overwrite: bool) -> Path:
    """frame.write_parquet to a hidden sibling tmp file, then os.replace it onto `path`."""
    check_writable(path, overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_sibling(path)
    try:
        frame.write_parquet(str(tmp))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def write_csv_atomic(text: str, path: Path, overwrite: bool) -> Path:
    """atomic_write_text after the overwrite check."""
    check_writable(path, overwrite)
    atomic_write_text(path, text)
    return path
