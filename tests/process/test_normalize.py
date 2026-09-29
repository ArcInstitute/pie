from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pie.process import normalize as norm
from pie.process.normalize import (
    as_float32_counts,
    check_integral,
    check_no_empty_rows,
    check_normalized,
    normalize_file,
    normalize_log1p,
    row_totals,
    stored_count,
)
from tests.process.helpers import count_matrix, make_counts, write_counts

HAND = np.array([[1, 3, 0], [2, 6, 0], [0, 0, 4]], dtype=np.float32)  # totals 4, 8, 4


def _bits(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def _dense(x: object) -> np.ndarray:
    return x.toarray() if sp.issparse(x) else np.asarray(x)  # type: ignore[union-attr]


def _scanpy_order(counts: np.ndarray, target_sum: float) -> np.ndarray:
    """scanpy 1.12.2 normalize_total + log1p on dense counts, written out step by step."""
    x = counts.astype(np.float32)
    totals = x.astype(np.float64).sum(axis=1).astype(np.float32)
    scale = totals / np.float32(target_sum)
    scale = scale + (scale == 0)
    return np.log1p(x / scale[:, None])


def _random_counts(seed: int = 0, shape: tuple[int, int] = (50, 30)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.poisson(0.8, size=shape).astype(np.float32)
    x[:, 0] += 1  # no empty rows
    return x


def _run(counts: np.ndarray, sparse: bool, target_sum: float = 1e4) -> np.ndarray:
    x = as_float32_counts(sp.csr_matrix(counts) if sparse else counts.copy())
    normalize_log1p(x, target_sum)
    assert x.dtype == np.float32
    assert sp.issparse(x) == sparse
    return _dense(x)


@pytest.mark.parametrize("sparse", [True, False])
def test_hand_computed_bits(sparse: bool) -> None:
    # target_sum 4: scale factors 1, 2, 1, so the rows become [1, 3, 0], [1, 3, 0], [0, 0, 4].
    expected = np.log1p(np.array([[1, 3, 0], [1, 3, 0], [0, 0, 4]], dtype=np.float32))
    np.testing.assert_array_equal(_bits(_run(HAND, sparse, target_sum=4.0)), _bits(expected))


@pytest.mark.parametrize("sparse", [True, False])
def test_matches_the_scanpy_operation_order_bitwise(sparse: bool) -> None:
    counts = _random_counts()
    got = _run(counts, sparse)
    np.testing.assert_array_equal(_bits(got), _bits(_scanpy_order(counts, 1e4)))


def test_close_to_a_float64_reference() -> None:
    counts = _random_counts(seed=1).astype(np.float64)
    reference = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)
    np.testing.assert_allclose(_run(counts.astype(np.float32), True), reference, rtol=1e-6)


@pytest.mark.parametrize("sparse", [True, False])
def test_row_totals_are_exact_float32(sparse: bool) -> None:
    counts = _random_counts(seed=2)
    totals = row_totals(sp.csr_matrix(counts) if sparse else counts)
    assert totals.dtype == np.float32
    np.testing.assert_array_equal(totals, counts.astype(np.float64).sum(axis=1))


def test_blocks_do_not_change_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = _random_counts(seed=3, shape=(40, 12))
    whole = _run(counts, True)
    monkeypatch.setattr(norm, "BLOCK_ROWS", 7)
    np.testing.assert_array_equal(_bits(_run(counts, True)), _bits(whole))
    np.testing.assert_array_equal(_bits(_run(counts, False)), _bits(whole))


@pytest.mark.parametrize("dtype", [np.int16, np.int32])
def test_integer_counts_are_cast_to_float32(dtype: type) -> None:
    counts = _random_counts(seed=4)
    x = as_float32_counts(sp.csr_matrix(counts.astype(dtype)))
    assert x.dtype == np.float32
    normalize_log1p(x, 1e4)
    np.testing.assert_array_equal(_bits(_dense(x)), _bits(_run(counts, True)))


def test_csc_is_rejected() -> None:
    with pytest.raises(ValueError, match="CSC"):
        as_float32_counts(sp.csc_matrix(HAND))


def test_float64_is_rejected() -> None:
    with pytest.raises(ValueError, match="float64"):
        as_float32_counts(HAND.astype(np.float64))


def test_non_integral_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="not integral"):
        check_integral(sp.csr_matrix(HAND / 3))


def test_empty_csr_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty rows"):
        check_no_empty_rows(sp.csr_matrix(np.array([[1, 0], [0, 0]], dtype=np.float32)))


def test_dense_zero_row_stays_zero() -> None:
    x = np.array([[1, 3], [0, 0]], dtype=np.float32)
    stored = stored_count(x)
    totals = normalize_log1p(x, 4.0)
    np.testing.assert_array_equal(x[1], [0.0, 0.0])
    assert check_normalized(x, shape=(2, 2), stored=stored, totals=totals, target_sum=4.0) < 1e-6


def test_check_normalized_catches_wrong_values() -> None:
    x = sp.csr_matrix(_random_counts(seed=5))
    stored = stored_count(x)
    totals = normalize_log1p(x, 1e4)
    x.data *= 1.1
    with pytest.raises(ValueError, match="row sums"):
        check_normalized(x, shape=x.shape, stored=stored, totals=totals, target_sum=1e4)


def test_normalize_file_keeps_csr_obs_and_var(tmp_path: Path) -> None:
    src = write_counts(make_counts(), tmp_path / "counts" / "screen.h5ad")
    dst = tmp_path / "expression" / "screen.h5ad"
    assert normalize_file(src, dst, 1e4, overwrite=False) == dst
    assert sorted(p.name for p in dst.parent.iterdir()) == ["screen.h5ad"]
    before, after = ad.read_h5ad(src), ad.read_h5ad(dst)
    assert sp.issparse(after.X) and after.X.format == "csr"
    assert after.X.dtype == np.float32
    np.testing.assert_array_equal(
        _bits(after.X.toarray()), _bits(_scanpy_order(count_matrix(), 1e4))
    )
    pd.testing.assert_frame_equal(after.obs, before.obs)
    pd.testing.assert_frame_equal(after.var, before.var)
    assert after.var.index.name == "gene_name_index"


def test_normalize_file_keeps_dense_dense(tmp_path: Path) -> None:
    src = write_counts(make_counts(sparse=False), tmp_path / "counts" / "d.h5ad")
    dst = tmp_path / "expression" / "d.h5ad"
    normalize_file(src, dst, 1e4, overwrite=False)
    after = ad.read_h5ad(dst)
    assert isinstance(after.X, np.ndarray)
    np.testing.assert_array_equal(_bits(after.X), _bits(_scanpy_order(count_matrix(), 1e4)))


def test_normalize_file_casts_int16_counts(tmp_path: Path) -> None:
    src = write_counts(make_counts(dtype=np.int16), tmp_path / "counts" / "i.h5ad")
    dst = tmp_path / "expression" / "i.h5ad"
    normalize_file(src, dst, 1e4, overwrite=False)
    after = ad.read_h5ad(dst)
    assert after.X.dtype == np.float32
    np.testing.assert_array_equal(
        _bits(after.X.toarray()), _bits(_scanpy_order(count_matrix(), 1e4))
    )


def test_normalize_file_refuses_an_existing_output(tmp_path: Path) -> None:
    src = write_counts(make_counts(), tmp_path / "counts" / "s.h5ad")
    dst = tmp_path / "expression" / "s.h5ad"
    normalize_file(src, dst, 1e4, overwrite=False)
    with pytest.raises(FileExistsError):
        normalize_file(src, dst, 1e4, overwrite=False)
    normalize_file(src, dst, 1e4, overwrite=True)


def test_normalize_file_rejects_normalized_input(tmp_path: Path) -> None:
    src = write_counts(make_counts(), tmp_path / "counts" / "s.h5ad")
    once = normalize_file(src, tmp_path / "expression" / "s.h5ad", 1e4, overwrite=False)
    with pytest.raises(ValueError, match="not integral"):
        normalize_file(once, tmp_path / "twice" / "s.h5ad", 1e4, overwrite=False)
