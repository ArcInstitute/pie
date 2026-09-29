"""Delta-p bin grid: fit on train rows, hard bins, two-hot targets and bin centers."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor

from pie.utils import StrictModel

MISSING_BIN = -100  # F.cross_entropy's default ignore_index
_BLOCK_ROWS = 4096


class DeltaPConfig(StrictModel):
    """data.delta_p."""

    bin_width_fold_change: float
    max_delta_percentile: float = 99.9
    max_delta: float | None = None


class DeltaPGrid(StrictModel):
    """Run-level delta-p bin grid; frozen into data_stats and checkpoints."""

    n_bins: int
    max_delta: float
    width: float


def train_percentile(delta_p: np.ndarray, rows: np.ndarray, percentile: float) -> float | None:
    """np.percentile(|finite delta_p[rows]|, percentile) as float, or None when rows is empty.

    Also None when the rows hold no finite value. Reads `rows` in blocks into one preallocated
    float32 buffer, so a memmapped (N, G) array is never materialized twice.
    """
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    if rows.size == 0:
        return None
    n_genes = int(delta_p.shape[1])
    out = np.empty(rows.size * n_genes, dtype=np.float32)
    filled = 0
    for i in range(0, rows.size, _BLOCK_ROWS):
        block = np.abs(np.asarray(delta_p[rows[i : i + _BLOCK_ROWS]], dtype=np.float32))
        finite = block[np.isfinite(block)]
        out[filled : filled + finite.size] = finite
        filled += finite.size
    if filled == 0:
        return None
    return float(np.percentile(out[:filled], percentile))


def fit_grid(per_dir_percentile: Mapping[str, float | None], cfg: DeltaPConfig) -> DeltaPGrid:
    """Grid from the per-dir train percentiles.

    max_delta = cfg.max_delta if set, else the max over non-None percentiles;
    width = ln(bin_width_fold_change); n_bins = ceil(2 * max_delta / width), +1 if even.
    """
    if cfg.bin_width_fold_change <= 1.0:
        raise ValueError(
            f"data.delta_p.bin_width_fold_change must be > 1, got {cfg.bin_width_fold_change}"
        )
    width = math.log(cfg.bin_width_fold_change)
    if cfg.max_delta is not None:
        max_delta = float(cfg.max_delta)
    else:
        contributing = [p for p in per_dir_percentile.values() if p is not None]
        if not contributing:
            raise ValueError(
                "cannot fit the delta-p grid: no dataset has finite delta-p on its train rows; "
                "set data.delta_p.max_delta"
            )
        max_delta = max(contributing)
    if max_delta <= 0.0:
        raise ValueError(f"cannot fit the delta-p grid: max_delta is {max_delta}")
    n_bins = math.ceil(2 * max_delta / width)
    if n_bins % 2 == 0:
        n_bins += 1
    return DeltaPGrid(n_bins=n_bins, max_delta=max_delta, width=width)


def quantize(delta_p: np.ndarray, grid: DeltaPGrid) -> np.ndarray:
    """Hard bins (int64): normalize over [-max_delta, max_delta], floor into n_bins, clip.

    Non-finite positions get MISSING_BIN.
    """
    valid = np.isfinite(delta_p)
    normalized = (np.nan_to_num(delta_p) / grid.max_delta + 1.0) / 2.0
    bins = np.clip(np.floor(normalized * grid.n_bins).astype(np.int64), 0, grid.n_bins - 1)
    bins[~valid] = MISSING_BIN
    return bins


def two_hot(delta_p: Tensor, grid: DeltaPGrid) -> tuple[Tensor, Tensor]:
    """(target_probs (..., n_bins), valid (...)) from continuous delta-p.

    Each value is clamped to [-max_delta, max_delta] and its unit mass is split between the two
    bracketing bin centers by linear interpolation; out-of-range mass goes to the edge bin.
    Rows are all-zero where the value is not finite.
    """
    n_bins = grid.n_bins
    max_delta = grid.max_delta
    valid = torch.isfinite(delta_p)
    d = torch.nan_to_num(delta_p, nan=0.0).clamp(-max_delta, max_delta)
    u = (d / max_delta + 1.0) * 0.5 * n_bins - 0.5
    lo = u.floor().clamp(0, n_bins - 2).long()
    frac = (u - lo.to(u.dtype)).clamp(0.0, 1.0)
    target = torch.zeros(*delta_p.shape, n_bins, device=delta_p.device, dtype=u.dtype)
    target.scatter_(-1, lo.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    target.scatter_add_(-1, (lo + 1).unsqueeze(-1), frac.unsqueeze(-1))
    return target * valid.unsqueeze(-1), valid


def dequantize(bins: Tensor, grid: DeltaPGrid) -> Tensor:
    """Bin centers ((b + 0.5) / n_bins * 2 - 1) * max_delta, float32."""
    return ((bins.float() + 0.5) / grid.n_bins * 2 - 1) * grid.max_delta
