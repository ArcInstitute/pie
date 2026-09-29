"""Tests for pie.data.delta_p."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from pie.data.delta_p import (
    MISSING_BIN,
    DeltaPConfig,
    DeltaPGrid,
    dequantize,
    fit_grid,
    quantize,
    train_percentile,
    two_hot,
)

GRID5 = DeltaPGrid(n_bins=5, max_delta=1.0, width=0.4)
WIDTH_0P1 = DeltaPConfig(bin_width_fold_change=math.exp(0.1))
NAN = float("nan")


def test_config_defaults_and_strictness() -> None:
    cfg = DeltaPConfig(bin_width_fold_change=1.1)
    assert cfg.max_delta_percentile == 99.9
    assert cfg.max_delta is None
    with pytest.raises(ValidationError):
        DeltaPConfig(bin_width_fold_change=1.1, n_bins=17)


def test_missing_bin_is_cross_entropy_ignore_index() -> None:
    assert MISSING_BIN == -100


def test_train_percentile_hand_computed() -> None:
    dp = np.array([[0.1, -0.5, np.nan], [0.3, 0.2, -0.4], [9.0, 9.0, 9.0]], dtype=np.float32)
    rows = np.array([1, 0])
    # |finite| over rows 0 and 1 = {0.1, 0.2, 0.3, 0.4, 0.5}; row 2 is excluded.
    assert train_percentile(dp, rows, 50.0) == float(np.float32(0.3))
    assert train_percentile(dp, rows, 25.0) == float(np.float32(0.2))
    assert train_percentile(dp, rows, 100.0) == float(np.float32(0.5))


def test_train_percentile_none_without_rows_or_finite_values() -> None:
    dp = np.array([[np.nan, np.nan], [1.0, 2.0]], dtype=np.float32)
    assert train_percentile(dp, np.array([], dtype=np.int64), 99.9) is None
    assert train_percentile(dp, np.array([0]), 99.9) is None


def test_train_percentile_blockwise_on_memmap_is_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    dp = rng.normal(size=(5000, 3)).astype(np.float32)
    dp[rng.random(dp.shape) < 0.1] = np.nan
    np.save(tmp_path / "delta_p.npy", dp)
    mm = np.load(tmp_path / "delta_p.npy", mmap_mode="r")
    rows = rng.permutation(5000)[:4500]
    sub = np.abs(dp[np.sort(rows)])
    expected = float(np.percentile(sub[np.isfinite(sub)], 99.9))
    assert train_percentile(mm, rows, 99.9) == expected


def test_fit_grid_wdataset_reference() -> None:
    grid = fit_grid({"replogle": 0.7726817727088928}, DeltaPConfig(bin_width_fold_change=1.1))
    assert grid == DeltaPGrid(n_bins=17, max_delta=0.7726817727088928, width=math.log(1.1))


def test_fit_grid_xdataset_reference() -> None:
    per_dir = {
        "replogle": None,
        "tahoe": 0.5923969745635986,
        "jiang": 0.23170852661132812,
        "arc_vcc_25": 0.40348678827285767,
        "orion": 0.23170852661132812,
    }
    grid = fit_grid(per_dir, DeltaPConfig(bin_width_fold_change=1.01))
    assert grid == DeltaPGrid(n_bins=121, max_delta=0.5923969745635986, width=math.log(1.01))


@pytest.mark.parametrize(("max_delta", "n_bins"), [(0.33, 7), (0.37, 9), (0.41, 9)])
def test_fit_grid_ceil_then_odd(max_delta: float, n_bins: int) -> None:
    # width = 0.1: 6.6 -> 7 (odd); 7.4 -> 8 -> 9; 8.2 -> 9 (odd).
    assert fit_grid({"a": max_delta}, WIDTH_0P1).n_bins == n_bins


def test_fit_grid_takes_max_over_contributing_dirs() -> None:
    grid = fit_grid({"a": 0.1, "b": 0.37, "c": None}, WIDTH_0P1)
    assert grid.max_delta == 0.37
    assert grid.n_bins == 9


def test_fit_grid_config_override() -> None:
    cfg = DeltaPConfig(bin_width_fold_change=math.exp(0.1), max_delta=0.33)
    assert fit_grid({"a": 0.9, "b": None}, cfg).max_delta == 0.33
    assert fit_grid({"a": None}, cfg).n_bins == 7


@pytest.mark.parametrize("per_dir", [{}, {"a": None}, {"a": 0.0}])
def test_fit_grid_rejects_no_signal(per_dir: dict[str, float | None]) -> None:
    with pytest.raises(ValueError):
        fit_grid(per_dir, WIDTH_0P1)


def test_fit_grid_rejects_non_positive_width() -> None:
    with pytest.raises(ValueError):
        fit_grid({"a": 0.5}, DeltaPConfig(bin_width_fold_change=1.0))


def test_quantize_hand_computed() -> None:
    dp = np.array([[0.0, 0.25, -0.5, -1.0, 1.0], [3.0, -7.0, np.nan, 0.75, 0.0]], dtype=np.float32)
    bins = quantize(dp, GRID5)
    assert bins.dtype == np.int64
    np.testing.assert_array_equal(bins, np.array([[2, 3, 1, 0, 4], [4, 0, MISSING_BIN, 4, 2]]))


def test_two_hot_hand_computed() -> None:
    dp = torch.tensor([[0.0, 0.25, -0.5, -1.0, 1.0], [3.0, -7.0, NAN, 0.75, 0.0]])
    target, valid = two_hot(dp, GRID5)
    expected = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.375, 0.625, 0.0],
                [0.25, 0.75, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.125, 0.875],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
        ]
    )
    assert target.dtype == torch.float32
    assert torch.equal(target, expected)
    assert torch.equal(valid, torch.isfinite(dp))


def test_dequantize_bin_centers() -> None:
    centers = dequantize(torch.arange(5), GRID5)
    assert centers.dtype == torch.float32
    torch.testing.assert_close(centers, torch.tensor([-0.8, -0.4, 0.0, 0.4, 0.8]))


def test_two_hot_mean_is_value_and_mode_is_hard_bin() -> None:
    grid = DeltaPGrid(n_bins=17, max_delta=0.7726817727088928, width=math.log(1.1))
    inner = grid.max_delta * (1 - 1 / grid.n_bins)
    values = np.random.default_rng(1).uniform(-inner, inner, size=2000)
    d = torch.from_numpy(values)
    target, valid = two_hot(d, grid)
    assert bool(valid.all())
    torch.testing.assert_close(target.sum(-1), torch.ones_like(d))
    centers = dequantize(torch.arange(grid.n_bins), grid).double()
    torch.testing.assert_close((target * centers).sum(-1), d, rtol=0.0, atol=1e-6)
    clear = target.max(-1).values > 0.5 + 1e-6
    hard = torch.from_numpy(quantize(values, grid))
    assert torch.equal(target.argmax(-1)[clear], hard[clear])
