from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from pie.process.knockdown import (
    DROPPED_FILE,
    STATS_FILE,
    filter_adata,
    filter_output_paths,
    knockdown_keep_mask,
    run_filter,
)
from tests.process.helpers import (
    CTRL,
    GENES,
    KEPT_ROWS,
    N_CELLS,
    filter_cfg,
    make_counts,
    stack,
    write_counts,
)

# Context c2's controls have G0 = 1, so G0 fails stage 1 there (1.5 / 1 >= 0.3).
C2_CTRL_ROW = (1.0, 10.0, 0.0, 8.0, 2.0)
C2_CONTROLS = [N_CELLS + i for i in range(4)]
STATS_HEADER = "context,cells_in,cells_out,control_cells,perts_in,perts_out,perts_dropped\n"
C1_DROPPED = ("G1", "G2", "G3", "NOPE")
C2_DROPPED = ("G0", "G1", "G2", "G3", "NOPE")


def _mask(adata: ad.AnnData, **over: object) -> np.ndarray:
    settings: dict[str, object] = {
        "perturbation_column": "gene",
        "control_label": CTRL,
        "residual_expression": 0.30,
        "cell_residual_expression": 0.50,
        "min_cells": 3,
        "layer": None,
        "var_gene_name": "gene_name",
    }
    settings.update(over)
    return knockdown_keep_mask(adata, **settings)  # type: ignore[arg-type]


def _rows(mask: np.ndarray) -> list[int]:
    return np.flatnonzero(mask).tolist()


def _two_contexts() -> ad.AnnData:
    return stack(make_counts("c1"), make_counts("c2", ctrl_row=C2_CTRL_ROW))


def test_three_stages_hand_computed() -> None:
    mask = _mask(make_counts())
    assert mask.dtype == np.bool_
    assert _rows(mask) == list(KEPT_ROWS)


def test_dense_matches_sparse() -> None:
    assert _rows(_mask(make_counts(sparse=False))) == list(KEPT_ROWS)


def test_min_cells_is_a_lower_bound() -> None:
    # G3 keeps rows 14 and 15 (0 / 8); row 16 (7 / 8) fails stage 2.
    assert _rows(_mask(make_counts(), min_cells=2)) == [*KEPT_ROWS, 14, 15]


def test_pert_threshold_is_strict() -> None:
    # G1: mean 3 / control mean 10 is exactly 0.3, which is not < 0.30 but is < 0.31.
    got = _mask(make_counts(), min_cells=2, residual_expression=0.31)
    assert _rows(got) == [*KEPT_ROWS, 8, 9, 14, 15]


def test_cell_threshold_is_strict() -> None:
    # Row 7: 5 / 10 is exactly 0.5, which is not < 0.50 but is < 0.51.
    assert _rows(_mask(make_counts(), cell_residual_expression=0.51)) == [*KEPT_ROWS, 7]


def test_zero_control_mean_and_unmatched_perts_are_dropped() -> None:
    mask = _mask(
        make_counts(), residual_expression=100.0, cell_residual_expression=100.0, min_cells=1
    )
    # Everything passes except G2 (control mean 0) and NOPE (no gene named NOPE).
    assert _rows(mask) == [row for row in range(N_CELLS) if row not in (10, 11, 12, 13)]


def test_no_matched_perturbation_keeps_controls_only() -> None:
    adata = make_counts()
    adata.var["gene_name"] = ["A", "B", "C", "D", "E"]
    assert _rows(_mask(adata)) == [0, 1, 2, 3]


def test_var_gene_name_none_matches_the_var_index() -> None:
    adata = make_counts()
    adata.var = pd.DataFrame(index=list(GENES))
    assert _rows(_mask(adata, var_gene_name=None)) == list(KEPT_ROWS)


def test_missing_var_column_raises() -> None:
    with pytest.raises(KeyError, match="symbol"):
        _mask(make_counts(), var_gene_name="symbol")


def test_layer_is_used_instead_of_x() -> None:
    adata = make_counts()
    adata.layers["counts"] = adata.X.copy()
    adata.X = adata.X * 0
    assert _rows(_mask(adata, layer="counts")) == list(KEPT_ROWS)


def test_filter_adata_runs_per_context() -> None:
    keep, stats, dropped = filter_adata(_two_contexts(), filter_cfg(None), "stem")
    assert _rows(keep) == [*KEPT_ROWS, *C2_CONTROLS]
    assert stats == [
        {"context": "c1", "cells_in": 17, "cells_out": 7, "control_cells": 4,
         "perts_in": 5, "perts_out": 1, "perts_dropped": 4},
        {"context": "c2", "cells_in": 17, "cells_out": 4, "control_cells": 4,
         "perts_in": 5, "perts_out": 0, "perts_dropped": 5},
    ]
    assert dropped == [{"context": "c1", "perturbation": p} for p in C1_DROPPED] + [
        {"context": "c2", "perturbation": p} for p in C2_DROPPED
    ]


def test_pooled_controls_would_keep_other_cells() -> None:
    # One pooled context: G0's control mean is (4 * 10 + 4 * 1) / 8 = 5.5, so c2's G0 cells
    # with 0, 0, 1 counts pass. Per-context filtering must not keep them.
    c2_g0 = {N_CELLS + 4, N_CELLS + 5, N_CELLS + 6}
    pooled, _, _ = filter_adata(_two_contexts(), filter_cfg(None, context_column=None), "all")
    assert c2_g0 <= set(_rows(pooled))
    per_context, _, _ = filter_adata(_two_contexts(), filter_cfg(None), "all")
    assert not c2_g0 & set(_rows(per_context))


def test_filter_adata_requires_unique_obs_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        filter_adata(stack(make_counts("c1"), make_counts("c1")), filter_cfg(None), "s")


def test_run_filter_writes_subsets_and_canonical_csvs(tmp_path: Path) -> None:
    src = write_counts(_two_contexts(), tmp_path / "in" / "screen.h5ad")
    out = tmp_path / "filtered"
    assert run_filter([src], filter_cfg(out), overwrite=False) == [out / "screen.h5ad"]
    original = ad.read_h5ad(src)
    got = ad.read_h5ad(out / "screen.h5ad")
    rows = [*KEPT_ROWS, *C2_CONTROLS]
    assert got.obs_names.tolist() == original.obs_names[rows].tolist()
    assert got.X.dtype == np.float32
    np.testing.assert_array_equal(got.X.toarray(), original.X.toarray()[rows])
    pd.testing.assert_frame_equal(got.var, original.var)
    pd.testing.assert_frame_equal(got.obs.astype(str), original.obs.iloc[rows].astype(str))
    assert (out / STATS_FILE).read_text() == (
        STATS_HEADER + "c1,17,7,4,5,1,4\n" + "c2,17,4,4,5,0,5\n"
    )
    assert (out / DROPPED_FILE).read_text() == (
        "context,perturbation\n"
        + "".join(f"c1,{p}\n" for p in C1_DROPPED)
        + "".join(f"c2,{p}\n" for p in C2_DROPPED)
    )


def test_run_filter_one_context_per_file_uses_the_stem(tmp_path: Path) -> None:
    a = write_counts(make_counts("x"), tmp_path / "in" / "lineA.h5ad")
    b = write_counts(make_counts("y"), tmp_path / "in" / "lineB.h5ad")
    out = tmp_path / "filtered"
    run_filter([a, b], filter_cfg(out, context_column=None), overwrite=False)
    assert (out / STATS_FILE).read_text() == (
        STATS_HEADER + "lineA,17,7,4,5,1,4\n" + "lineB,17,7,4,5,1,4\n"
    )
    assert sorted(p.name for p in out.iterdir()) == [
        DROPPED_FILE, STATS_FILE, "lineA.h5ad", "lineB.h5ad"
    ]


def test_run_filter_with_var_index_names_leaves_var_unchanged(tmp_path: Path) -> None:
    adata = make_counts()
    adata.var = pd.DataFrame(index=list(GENES))
    src = write_counts(adata, tmp_path / "in" / "vcc.h5ad")
    out = tmp_path / "filtered"
    run_filter([src], filter_cfg(out, var_gene_name=None), overwrite=False)
    got = ad.read_h5ad(out / "vcc.h5ad")
    assert got.var_names.tolist() == list(GENES)
    assert list(got.var.columns) == []
    assert got.n_obs == len(KEPT_ROWS)


def test_run_filter_writes_integer_counts_as_float32(tmp_path: Path) -> None:
    src = write_counts(make_counts(dtype=np.int32), tmp_path / "in" / "vcc.h5ad")
    out = tmp_path / "filtered"
    run_filter([src], filter_cfg(out), overwrite=False)
    original = ad.read_h5ad(src)
    got = ad.read_h5ad(out / "vcc.h5ad")
    assert original.X.dtype == np.int32
    assert got.X.dtype == np.float32
    np.testing.assert_array_equal(got.X.toarray(), original.X.toarray()[list(KEPT_ROWS)])


def test_run_filter_refuses_existing_outputs(tmp_path: Path) -> None:
    src = write_counts(make_counts(), tmp_path / "in" / "s.h5ad")
    cfg = filter_cfg(tmp_path / "filtered")
    run_filter([src], cfg, overwrite=False)
    with pytest.raises(FileExistsError):
        run_filter([src], cfg, overwrite=False)
    run_filter([src], cfg, overwrite=True)


def test_filter_output_paths(tmp_path: Path) -> None:
    out = tmp_path / "f"
    got = filter_output_paths([Path("/i/a.h5ad"), Path("/j/b.h5ad")], filter_cfg(out))
    assert got == [out / "a.h5ad", out / "b.h5ad", out / STATS_FILE, out / DROPPED_FILE]
