from __future__ import annotations

import inspect
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from pie.process import de as de_mod
from pie.process.de import (
    OUTPUT_COLUMNS,
    de_kwargs,
    import_gpudge,
    preflight,
    prepare_de,
    resolve_device,
    run_de,
)
from tests.process.conftest import FakeGpudge
from tests.process.helpers import (
    CTRL,
    N_CELLS,
    count_matrix,
    de_cfg,
    make_counts,
    stack,
    write_counts,
)


def _two_contexts(tmp_path: Path, dtype: type = np.float32) -> Path:
    adata = stack(make_counts("c1", dtype=dtype), make_counts("c2", dtype=dtype))
    return write_counts(adata, tmp_path / "counts" / "screen.h5ad")


def test_de_kwargs_are_exact(tmp_path: Path) -> None:
    assert de_kwargs(de_cfg(tmp_path)) == {
        "groupby": "gene",
        "reference": CTRL,
        "normalize_target_sum": 1e4,
        "filter_gene_min_cpm_cell": 5.0,
    }
    assert de_kwargs(de_cfg(tmp_path, filter_gene_min_cpm_cell=None)) == {
        "groupby": "gene",
        "reference": CTRL,
        "normalize_target_sum": 1e4,
    }


def test_run_de_splits_per_context_on_raw_counts(tmp_path: Path, fake_gpudge: FakeGpudge) -> None:
    out = tmp_path / "de"
    cfg = de_cfg(out)
    assert run_de([_two_contexts(tmp_path)], cfg, overwrite=False) == [
        out / "c1.parquet",
        out / "c2.parquet",
    ]
    assert [kwargs for _, kwargs in fake_gpudge.calls] == [de_kwargs(cfg)] * 2
    for (sub, _), context in zip(fake_gpudge.calls, ("c1", "c2"), strict=True):
        assert sub.obs["context"].astype(str).unique().tolist() == [context]
        assert sub.n_obs == N_CELLS
        assert sub.X.dtype == np.float32
        np.testing.assert_array_equal(sub.X.toarray(), count_matrix())
    frame = pd.read_parquet(out / "c1.parquet")
    assert list(frame.columns) == list(OUTPUT_COLUMNS)
    assert sorted(set(frame["target"])) == ["G0", "G1", "G2", "G3", "NOPE"]


def test_without_a_context_column_the_file_stem_names_the_output(
    tmp_path: Path, fake_gpudge: FakeGpudge
) -> None:
    src = write_counts(make_counts("x"), tmp_path / "counts" / "lineA.h5ad")
    out = tmp_path / "de"
    assert run_de([src], de_cfg(out, context_column=None), overwrite=False) == [
        out / "lineA.parquet"
    ]
    assert [sub.n_obs for sub, _ in fake_gpudge.calls] == [N_CELLS]


def test_integer_counts_are_cast_to_float32(tmp_path: Path, fake_gpudge: FakeGpudge) -> None:
    run_de([_two_contexts(tmp_path, dtype=np.int16)], de_cfg(tmp_path / "de"), overwrite=False)
    assert {sub.X.dtype for sub, _ in fake_gpudge.calls} == {np.dtype(np.float32)}


def test_non_integral_counts_are_rejected(tmp_path: Path, fake_gpudge: FakeGpudge) -> None:
    adata = make_counts()
    adata.X = (adata.X / 3).astype(np.float32)  # scipy sparse / scalar gives float64
    src = write_counts(adata, tmp_path / "expression" / "s.h5ad")
    with pytest.raises(ValueError, match="not integral"):
        run_de([src], de_cfg(tmp_path / "de"), overwrite=False)
    assert fake_gpudge.calls == []


def test_preflight_errors() -> None:
    adata = make_counts()
    with pytest.raises(KeyError, match="target_gene"):
        preflight(adata, "target_gene", CTRL)
    with pytest.raises(ValueError, match="reference 'DMSO'"):
        preflight(adata, "gene", "DMSO")
    with pytest.raises(ValueError, match="at least 2 groups"):
        preflight(adata[(adata.obs["gene"] == CTRL).to_numpy()].copy(), "gene", CTRL)
    missing = adata.copy()
    missing.obs["gene"] = pd.Series(
        [None, *adata.obs["gene"].tolist()[1:]], index=adata.obs_names, dtype=object
    )
    with pytest.raises(ValueError, match="missing"):
        preflight(missing, "gene", CTRL)


def test_missing_gpudge_fails_with_the_install_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "gpudge", None)
    with pytest.raises(RuntimeError, match="uv sync --extra process"):
        import_gpudge()
    src = write_counts(make_counts(), tmp_path / "counts" / "s.h5ad")
    with pytest.raises(RuntimeError, match="uv sync --extra process"):
        run_de([src], de_cfg(tmp_path / "de"), overwrite=False)
    assert not (tmp_path / "de").exists()


def test_device_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(de_mod, "_cuda_available", lambda: True)
    assert [resolve_device(d) for d in ("auto", "cuda", "cpu")] == ["cuda", "cuda", "cpu"]
    monkeypatch.setattr(de_mod, "_cuda_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    with pytest.raises(RuntimeError, match="no CUDA device"):
        resolve_device("cuda")


def test_prepare_de_needs_cuda(tmp_path: Path, fake_gpudge: FakeGpudge) -> None:
    assert prepare_de(de_cfg(tmp_path)) is fake_gpudge
    with pytest.raises(RuntimeError, match="CUDA only"):
        prepare_de(de_cfg(tmp_path, device="cpu"))


def test_existing_output_is_refused(tmp_path: Path, fake_gpudge: FakeGpudge) -> None:
    src = write_counts(make_counts(), tmp_path / "counts" / "s.h5ad")
    cfg = de_cfg(tmp_path / "de")
    run_de([src], cfg, overwrite=False)
    with pytest.raises(FileExistsError):
        run_de([src], cfg, overwrite=False)
    assert len(fake_gpudge.calls) == 1
    run_de([src], cfg, overwrite=True)
    assert len(fake_gpudge.calls) == 2


def test_a_context_in_two_files_is_an_error(tmp_path: Path, fake_gpudge: FakeGpudge) -> None:
    a = write_counts(make_counts("c1"), tmp_path / "counts" / "a.h5ad")
    b = write_counts(make_counts("c1"), tmp_path / "counts" / "b.h5ad")
    with pytest.raises(ValueError, match="produced twice"):
        run_de([a, b], de_cfg(tmp_path / "de"), overwrite=False)


def test_context_names_must_be_file_names(tmp_path: Path, fake_gpudge: FakeGpudge) -> None:
    src = write_counts(make_counts("a/b"), tmp_path / "counts" / "s.h5ad")
    with pytest.raises(ValueError, match="file name"):
        run_de([src], de_cfg(tmp_path / "de"), overwrite=False)
    assert fake_gpudge.calls == []


def test_installed_gpudge_matches_the_pin() -> None:
    gpudge = pytest.importorskip("gpudge")
    from gpudge._csr_dense import HAS_NUMBA
    from gpudge._output import DEFAULT_OUTPUT_COLUMNS

    assert gpudge.__version__ == "0.9.1"
    assert HAS_NUMBA
    assert tuple(DEFAULT_OUTPUT_COLUMNS) == OUTPUT_COLUMNS
    params = inspect.signature(gpudge.de).parameters
    for name in ("groupby", "reference", "normalize_target_sum", "filter_gene_min_cpm_cell"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["mean_calc"].default == "arithmetic"
    assert params["epsilon"].default == 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gpudge needs a CUDA GPU")
def test_real_gpudge_writes_the_default_columns(tmp_path: Path) -> None:
    pytest.importorskip("gpudge")
    rng = np.random.default_rng(0)
    counts = rng.poisson(3.0, size=(60, 8)).astype(np.float32) + 1
    obs = pd.DataFrame(
        {"gene": [CTRL] * 30 + ["A"] * 15 + ["B"] * 15, "context": ["c1"] * 60},
        index=[f"cell{i}" for i in range(60)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(8)])
    src = write_counts(
        ad.AnnData(X=sp.csr_matrix(counts), obs=obs, var=var), tmp_path / "counts" / "s.h5ad"
    )
    out = tmp_path / "de"
    assert run_de([src], de_cfg(out, device="cuda"), overwrite=False) == [out / "c1.parquet"]
    frame = pd.read_parquet(out / "c1.parquet")
    assert list(frame.columns) == list(OUTPUT_COLUMNS)
    assert set(frame["target"]) <= {"A", "B"}
