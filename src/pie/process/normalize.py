"""CP10k + natural log1p normalization of count matrices, native (no scanpy).

The arithmetic follows scanpy 1.12.2 `normalize_total(adata, target_sum=...)` then `log1p(adata)`
(`preprocessing/_normalization.py:30-45,93-124,271-276`, `_utils/__init__.py:597-661`,
`preprocessing/_simple.py:361-381`), so the float32 results are the same bits. Integer X is cast to
float32. Each cell's total is summed in float64 and stored as float32. The scale factor is
`total / target_sum` in float32, with 0 replaced by 1. Every stored value is divided by its row's
scale factor in float32, and then `np.log1p` runs in place. The work runs in blocks of rows, and
each step is elementwise within a row, so blocking changes no value.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
import scipy.sparse as sp

from pie.process.io import check_writable, write_h5ad_atomic

logger = logging.getLogger(__name__)

BLOCK_ROWS = 20_000
RTOL = 1e-4

type CountMatrix = np.ndarray | sp.csr_matrix | sp.csr_array


def _row_blocks(n_rows: int) -> Iterator[tuple[int, int]]:
    for lo in range(0, n_rows, BLOCK_ROWS):
        yield lo, min(lo + BLOCK_ROWS, n_rows)


def _block_values(X: Any, lo: int, hi: int) -> np.ndarray:
    """A writable view of the stored values of rows [lo, hi): CSR data slice or dense rows."""
    if sp.issparse(X):
        start, end = X.indptr[lo], X.indptr[hi]
        return X.data[start:end]
    return X[lo:hi]


def _row_sums64(
    X: Any, lo: int, hi: int, transform: Callable[[np.ndarray], np.ndarray] | None = None
) -> np.ndarray:
    """float64 per-row sums of (transformed) stored values for rows [lo, hi)."""
    values = _block_values(X, lo, hi).astype(np.float64)
    if transform is not None:
        values = transform(values)
    if sp.issparse(X):
        offsets = X.indptr[lo : hi + 1] - X.indptr[lo]
        csum = np.concatenate(([0.0], np.cumsum(values)))
        return csum[offsets[1:]] - csum[offsets[:-1]]
    return values.sum(axis=1)


def as_float32_counts(X: Any) -> CountMatrix:
    """X as float32 CSR or dense; integer dtypes are cast, CSC / other formats / float64 raise."""
    if sp.issparse(X):
        if X.format == "csc":
            raise ValueError("X is CSC; pie-process needs CSR or dense X (convert with X.tocsr())")
        if X.format != "csr":
            raise ValueError(f"X is sparse {X.format!r}; pie-process needs CSR or dense X")
    elif not isinstance(X, np.ndarray):
        raise TypeError(f"X is {type(X).__name__}; pie-process needs CSR or dense X")
    if np.issubdtype(X.dtype, np.integer):
        return X.astype(np.float32)
    if X.dtype != np.float32:
        raise ValueError(f"X is {X.dtype}; expected float32 or integer counts")
    return X

def check_integral(X: CountMatrix) -> None:
    """Raise unless every stored value is a whole number (raw counts, not normalized values)."""
    for lo, hi in _row_blocks(X.shape[0]):
        values = _block_values(X, lo, hi)
        if values.size and not np.array_equal(values, np.floor(values)):
            raise ValueError(
                f"X is not integral in rows [{lo}, {hi}): expected raw counts, not normalized"
            )


def check_no_empty_rows(X: CountMatrix) -> None:
    """CSR only, as in the canonical script: every cell must have a stored count."""
    if sp.issparse(X):
        empty = np.flatnonzero(np.diff(cast(sp.csr_matrix, X).indptr) == 0)
        if empty.size:
            raise ValueError(
                f"X has {empty.size} empty rows (first: row {empty[0]}); every cell needs counts"
            )


def row_totals(X: CountMatrix) -> np.ndarray:
    """Per-cell totals, summed in float64 and stored as float32 (scanpy's CSR numba path)."""
    totals = np.empty(X.shape[0], dtype=np.float32)
    for lo, hi in _row_blocks(X.shape[0]):
        totals[lo:hi] = _row_sums64(X, lo, hi)
    return totals


def normalize_log1p(X: CountMatrix, target_sum: float) -> np.ndarray:
    """In place: divide each row by (total / target_sum) in float32, then log1p. Returns totals."""
    totals = row_totals(X)
    scale = totals / np.float32(target_sum)
    scale = scale + (scale == 0)
    for lo, hi in _row_blocks(X.shape[0]):
        values = _block_values(X, lo, hi)
        if sp.issparse(X):
            divisor = np.repeat(scale[lo:hi], np.diff(cast(sp.csr_matrix, X).indptr[lo : hi + 1]))
        else:
            divisor = scale[lo:hi, None]
        np.divide(values, divisor, out=values)
        np.log1p(values, out=values)
    return totals


def stored_count(X: CountMatrix) -> int:
    """nnz for CSR, the number of non-zero entries for dense."""
    return int(cast(sp.csr_matrix, X).nnz) if sp.issparse(X) else int(np.count_nonzero(X))


def check_normalized(
    X: CountMatrix,
    *,
    shape: tuple[int, ...],
    stored: int,
    totals: np.ndarray,
    target_sum: float,
    rtol: float = RTOL,
) -> float:
    """Post-checks from the canonical script; returns the max relative expm1 row-sum deviation.

    Shape, dtype (float32) and stored-entry count are unchanged, every value is finite, and each
    cell with counts has expm1 row sum within `rtol` of `target_sum`.
    """
    if tuple(X.shape) != tuple(shape):
        raise ValueError(f"shape changed: {shape} -> {X.shape}")
    if X.dtype != np.float32:
        raise ValueError(f"dtype changed to {X.dtype}")
    if stored_count(X) != stored:
        raise ValueError(f"stored entries changed: {stored} -> {stored_count(X)}")
    worst = 0.0
    for lo, hi in _row_blocks(X.shape[0]):
        if not np.isfinite(_block_values(X, lo, hi)).all():
            raise ValueError(f"non-finite values in rows [{lo}, {hi})")
        sums = _row_sums64(X, lo, hi, np.expm1)
        live = totals[lo:hi] > 0
        if live.any():
            worst = max(worst, float(np.max(np.abs(sums[live] - target_sum))) / target_sum)
    if worst > rtol:
        raise ValueError(
            f"expm1 row sums deviate from {target_sum:g} by {worst:.2e} > rtol {rtol:.0e}"
        )
    return worst


def normalize_file(src: Path, dst: Path, target_sum: float, overwrite: bool) -> Path:
    """Count h5ad -> CP10k + log1p X (float32, same CSR/dense encoding, obs/var kept) at dst."""
    check_writable(dst, overwrite)
    adata = ad.read_h5ad(src)
    X = as_float32_counts(adata.X)
    check_no_empty_rows(X)
    check_integral(X)
    shape, stored = tuple(X.shape), stored_count(X)
    logger.info(
        "[normalize] %s: shape=%s stored=%s target_sum=%g", src, shape, f"{stored:,}", target_sum
    )
    totals = normalize_log1p(X, target_sum)
    worst = check_normalized(X, shape=shape, stored=stored, totals=totals, target_sum=target_sum)
    logger.info("[normalize] verified: max relative expm1 row-sum deviation %.2e", worst)
    adata.X = X
    return write_h5ad_atomic(adata, dst, overwrite)
