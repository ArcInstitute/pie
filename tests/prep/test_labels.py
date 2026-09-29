from __future__ import annotations

import math
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp
from pydantic import ValidationError

import pie.cli
from pie.cli import prep_main
from pie.data import preprocessed as pp
from pie.data.preprocessed import PreprocessedDir
from pie.prep.config import PrepConfig, compose_prep_config
from pie.prep.labels import (
    default_gene_axis,
    fold_change_and_lfc,
    h5ad_var_names,
    label_gene_set,
    normalize_obs,
    open_x_reader,
    parse_drug_dose,
    pert_ensembl_ids,
    read_gene_list,
    resolve_files,
    run_prep,
)

H5AD_GENES = ["G3", "G1", "G2", "G4"]
# (h5ad context, h5ad perturbation, expression over H5AD_GENES)
CELLS: list[tuple[str, str, list[int]]] = [
    ("A", "non-targeting", [1, 2, 3, 4]),
    ("A", "non-targeting", [3, 4, 5, 6]),
    ("A", "P1", [5, 5, 5, 5]),
    ("A", "P2", [2, 2, 2, 2]),
    ("A", "P2", [4, 4, 4, 4]),
    ("B", "non-targeting", [0, 1, 0, 1]),
    ("B", "P1", [1, 1, 1, 1]),
    ("B", "P9", [9, 9, 9, 9]),
]
# obs gene_id of the toy h5ad: P1 has an id (padded), P2 none, P9 is not a label perturbation.
TOY_GENE_IDS = {"P1": " ENSG00000000001 ", "P2": "nan", "P9": "ENSG00000000009"}


def write_h5ad(
    path: Path,
    cells: list[tuple[str, str, list[int]]] | None = None,
    genes: list[str] | None = None,
    layout: str = "dense",
    context_col: str = "cell_line",
    pert_col: str = "gene",
    gene_id: bool = True,
) -> Path:
    cells = CELLS if cells is None else cells
    genes = H5AD_GENES if genes is None else genes
    x = np.array([c[2] for c in cells], dtype=np.float32)
    matrix = {"dense": x, "csr": sp.csr_matrix(x), "csc": sp.csc_matrix(x)}[layout]
    obs = pd.DataFrame(
        {
            context_col: pd.Categorical([c[0] for c in cells]),
            pert_col: pd.Categorical([c[1] for c in cells]),
        },
        index=[f"cell{i}" for i in range(len(cells))],
    )
    if gene_id:
        obs["gene_id"] = pd.Categorical([TOY_GENE_IDS.get(c[1], c[1]) for c in cells])
    var = pd.DataFrame(index=pd.Index(genes))
    path.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(X=matrix, obs=obs, var=var).write_h5ad(path)
    return path


def test_parse_drug_dose_formats_tuple() -> None:
    assert parse_drug_dose("[('DrugA', 0.05, 'uM')]") == "DrugA_0.05uM"
    assert parse_drug_dose("[('DMSO_TF', 0.0, 'uM')]") == "DMSO_TF_0.0uM"


def test_parse_drug_dose_passes_bare_values() -> None:
    assert parse_drug_dose("control") == "control"


def test_parse_drug_dose_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        parse_drug_dose("[(")
    with pytest.raises(ValueError):
        parse_drug_dose("[('DrugA', 1.0)]")


def test_normalize_obs_maps_contexts_and_drugs() -> None:
    ctx, pert = normalize_obs(
        np.array(["A", "B", "A"], dtype=object),
        np.array(["[('D', 1.0, 'uM')]", "control", "control"], dtype=object),
        context_map={"A": "ctxa", "B": "ctxb"},
        pert_format="drug_dose",
        source="x.h5ad",
    )
    assert list(ctx) == ["ctxa", "ctxb", "ctxa"]
    assert list(pert) == ["D_1.0uM", "control", "control"]


def test_normalize_obs_rejects_unmapped_context() -> None:
    with pytest.raises(ValueError, match="'B'"):
        normalize_obs(
            np.array(["A", "B"], dtype=object),
            np.array(["P1", "P1"], dtype=object),
            context_map={"A": "ctxa"},
            pert_format="gene",
            source="x.h5ad",
        )


def test_normalize_obs_identity_without_transforms() -> None:
    ctx, pert = normalize_obs(
        np.array(["A", "B"], dtype=object),
        np.array(["[('D', 1.0, 'uM')]", "P1"], dtype=object),
        context_map=None,
        pert_format="gene",
        source="x.h5ad",
    )
    assert list(ctx) == ["A", "B"]
    assert list(pert) == ["[('D', 1.0, 'uM')]", "P1"]


def test_open_x_reader_dense_and_csr_agree(tmp_path: Path) -> None:
    x = np.array([c[2] for c in CELLS], dtype=np.float32)
    for layout in ("dense", "csr"):
        path = write_h5ad(tmp_path / f"{layout}.h5ad", layout=layout)
        with h5py.File(path, "r") as h5:
            reader = open_x_reader(h5, len(H5AD_GENES))
            scattered = reader.read_rows(np.array([6, 0, 2], dtype=np.intp))
            span = reader.read_rows(np.array([3, 1, 2], dtype=np.intp))
        scattered = scattered.toarray() if sp.issparse(scattered) else scattered
        span = span.toarray() if sp.issparse(span) else span
        np.testing.assert_array_equal(scattered, x[[0, 2, 6]])
        np.testing.assert_array_equal(span, x[1:4])


def test_open_x_reader_rejects_csc(tmp_path: Path) -> None:
    path = write_h5ad(tmp_path / "csc.h5ad", layout="csc")
    with h5py.File(path, "r") as h5, pytest.raises(ValueError, match="csc"):
        open_x_reader(h5, len(H5AD_GENES))


def test_dense_reader_rejects_duplicate_rows(tmp_path: Path) -> None:
    path = write_h5ad(tmp_path / "dense.h5ad")
    with h5py.File(path, "r") as h5:
        reader = open_x_reader(h5, len(H5AD_GENES))
        with pytest.raises(ValueError, match="Duplicate"):
            reader.read_rows(np.array([1, 1], dtype=np.intp))


def test_h5ad_var_names_first_seen_union(tmp_path: Path) -> None:
    a = write_h5ad(tmp_path / "a.h5ad")
    b = write_h5ad(tmp_path / "b.h5ad", cells=[("A", "non-targeting", [1, 2])], genes=["G5", "G1"])
    assert h5ad_var_names([str(a), str(b)]) == ["G3", "G1", "G2", "G4", "G5"]


def test_read_gene_list(tmp_path: Path) -> None:
    good = tmp_path / "genes.txt"
    good.write_text("G2\n\nG1 \n")
    assert read_gene_list(good) == ["G2", "G1"]
    dup = tmp_path / "dup.txt"
    dup.write_text("G1\nG2\nG1\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_gene_list(dup)


def test_fold_change_and_lfc_spaces() -> None:
    fc, lfc = fold_change_and_lfc(np.array([2.0, 0.5, 1.0]), "linear")
    np.testing.assert_array_equal(fc, [2.0, 0.5, 1.0])
    np.testing.assert_array_equal(lfc, [1.0, -1.0, 0.0])
    fc, lfc = fold_change_and_lfc(np.array([1.0, -1.0]), "log2")
    np.testing.assert_array_equal(fc, [2.0, 0.5])
    np.testing.assert_array_equal(lfc, [1.0, -1.0])
    fc, lfc = fold_change_and_lfc(np.array([math.log(2.0)]), "ln")
    np.testing.assert_allclose(fc, [2.0], rtol=1e-15)
    np.testing.assert_allclose(lfc, [1.0], rtol=1e-15)


NAN = np.nan
# (context, pert, gene_symbol, fdr, fold_change)
LABELS_A = [
    ("ctxa", "P2", "G1", 0.01, 2.0),
    ("ctxa", "P2", "G2", 0.5, 0.5),
    ("ctxa", "P1", "G3", 0.02, 4.0),
    ("ctxa", "P1", "GX", 0.01, 8.0),
]
LABELS_B = [("ctxb", "P1", "G1", 0.2, 1.0)]


def write_labels_csv(path: Path, rows: list[tuple[str, str, str, float, float]]) -> Path:
    """Write a labels.csv in the canonical schema (the feature column is present but unused)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "context,pert,feature,fdr,fold_change,gene_symbol",
        *(f"{c},{p},{g},{fdr!r},{fc!r},{g}" for c, p, g, fdr, fc in rows),
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


TOY_MAP = {"A": "ctxa", "B": "ctxb"}


def _hydra_map(mapping: dict[str, str] | None) -> str:
    """Render a mapping as a Hydra override value, `{k1:v1,k2:v2}`."""
    return "{" + ",".join(f"{k}:{v}" for k, v in (mapping or {}).items()) + "}"


def make_cfg(tmp_path: Path, *overrides: str, layout: str = "dense") -> PrepConfig:
    """Write the toy inputs under tmp_path; compose a config (later `key=value` replace earlier)."""
    write_labels_csv(tmp_path / "labels" / "ctxb" / "labels.csv", LABELS_B)
    write_labels_csv(tmp_path / "labels" / "ctxa" / "labels.csv", LABELS_A)
    write_h5ad(tmp_path / "expression" / "cells.h5ad", layout=layout)
    values = {
        "name": "toy",
        "labels": str(tmp_path / "labels" / "*" / "labels.csv"),
        "h5ad": str(tmp_path / "expression" / "*.h5ad"),
        "output_dir": str(tmp_path / "out"),
        "obs.context_col": "cell_line",
        "obs.pert_col": "gene",
        "obs.control_label": "non-targeting",
        "obs.context_map": _hydra_map(TOY_MAP),
    }
    for override in overrides:
        key, _, value = override.partition("=")
        values[key] = value
    return compose_prep_config([f"{key}={value}" for key, value in values.items()])


def test_default_gene_axis_h5ad_order_restricted_to_labels(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    label_files = resolve_files(str(cfg.labels), "labels")
    h5ad_files = resolve_files(cfg.h5ad, "h5ad")
    assert label_gene_set(label_files, cfg) == {"G1", "G2", "G3", "GX"}
    assert default_gene_axis(label_files, h5ad_files, cfg) == ["G3", "G1", "G2"]


def test_run_prep_hand_computed(tmp_path: Path) -> None:
    out = run_prep(make_cfg(tmp_path))
    assert out == tmp_path / "out"
    d = PreprocessedDir.open(out)
    assert d.dataset == "toy"
    assert d.genes == ["G3", "G1", "G2"]
    assert d.contexts == ["ctxa", "ctxb"]
    assert d.perts == ["P1", "P2"]
    assert d.row_keys() == [("ctxa", "P1"), ("ctxa", "P2"), ("ctxb", "P1")]
    assert d.meta.pert_kind == "gene"
    assert d.meta.control_label == "non-targeting"
    assert d.meta.num_rows == 3
    assert not d.controls_only
    np.testing.assert_array_equal(
        d.fold_changes, np.array([[4, 0, 0], [0, 2, 0.5], [0, 1, 0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        d.fdr, np.array([[0.02, 1, 1], [1, 0.01, 0.5], [1, 0.2, 1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        d.tested,
        np.array([[True, False, False], [False, True, True], [False, True, False]]),
    )
    np.testing.assert_array_equal(
        d.lfc_true, np.array([[2, NAN, NAN], [NAN, 1, -1], [NAN, 0, NAN]], dtype=np.float64)
    )
    np.testing.assert_array_equal(
        d.delta_p, np.array([[3, 2, 1], [1, 0, -1], [1, 0, 1]], dtype=np.float32)
    )
    np.testing.assert_array_equal(d.ctrl_means, np.array([[2, 3, 4], [0, 1, 0]], dtype=np.float32))
    np.testing.assert_array_equal(d.ctx_ids, np.array([0, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(d.pert_ids, np.array([0, 1, 0], dtype=np.int32))


def test_run_prep_csr_matches_dense(tmp_path: Path) -> None:
    dense = PreprocessedDir.open(run_prep(make_cfg(tmp_path / "d")))
    csr = PreprocessedDir.open(run_prep(make_cfg(tmp_path / "c", layout="csr")))
    assert dense.meta.array_sha256 == csr.meta.array_sha256


def test_run_prep_genes_file_sets_axis(tmp_path: Path) -> None:
    genes = tmp_path / "genes.txt"
    genes.write_text("G2\nGZ\nG1\n")
    d = PreprocessedDir.open(run_prep(make_cfg(tmp_path, f"genes={genes}")))
    assert d.genes == ["G2", "GZ", "G1"]
    np.testing.assert_array_equal(d.tested[:, 1], [False, False])
    np.testing.assert_array_equal(d.fold_changes[:, 1], [0, 0])
    np.testing.assert_array_equal(d.delta_p[:, 1], [0, 0])
    np.testing.assert_array_equal(d.fold_changes[:, 2], [2, 1])


def test_run_prep_label_context_without_h5ad_cells_raises(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    write_labels_csv(tmp_path / "labels" / "ctxd" / "labels.csv", [("ctxd", "P1", "G1", 0.1, 1.5)])
    with pytest.raises(ValueError, match="'ctxd'"):
        run_prep(cfg)
    assert not (tmp_path / "out").exists()


def test_run_prep_label_context_without_control_cells_raises(tmp_path: Path) -> None:
    write_h5ad(tmp_path / "extra" / "c.h5ad", cells=[("C", "P1", [1, 1, 1, 1])])
    cfg = make_cfg(
        tmp_path,
        f"h5ad={tmp_path / '*' / '*.h5ad'}",
        "obs.context_map={A:ctxa,B:ctxb,C:ctxc}",
    )
    write_labels_csv(tmp_path / "labels" / "ctxc" / "labels.csv", [("ctxc", "P1", "G1", 0.1, 1.5)])
    with pytest.raises(ValueError, match=r"'ctxc'.*control"):
        run_prep(cfg)


def test_run_prep_missing_pert_is_nan_and_logs_counts(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = make_cfg(tmp_path)
    write_labels_csv(tmp_path / "labels" / "ctxe" / "labels.csv", [("ctxa", "P7", "G1", 0.1, 1.5)])
    with caplog.at_level("WARNING", logger="pie.prep.labels"):
        d = PreprocessedDir.open(run_prep(cfg))
    assert d.row_keys()[2] == ("ctxa", "P7")
    assert np.isnan(d.delta_p[2]).all()
    assert "1 of 4 rows" in caplog.text
    assert "ctxa: 1" in caplog.text


def test_run_prep_parquet_stem_log2(tmp_path: Path) -> None:
    de = tmp_path / "de"
    de.mkdir()
    pq.write_table(
        pa.table(
            {
                "target": ["P1", "P2"],
                "feature": ["G3", "G1"],
                "log2_fold_change": [1.5, -0.25],
                "p_adj": [0.01, 0.3],
            }
        ),
        de / "ctxa.parquet",
    )
    pq.write_table(
        pa.table({"target": ["P1"], "feature": ["G2"], "log2_fold_change": [0.0], "p_adj": [0.9]}),
        de / "ctxb.parquet",
    )
    cfg = make_cfg(tmp_path, "label_format=pie_process", f"labels={de / '*.parquet'}")
    d = PreprocessedDir.open(run_prep(cfg))
    assert d.genes == ["G3", "G1", "G2"]
    assert d.row_keys() == [("ctxa", "P1"), ("ctxa", "P2"), ("ctxb", "P1")]
    expected_fc = np.zeros((3, 3), dtype=np.float32)
    expected_fc[0, 0] = np.float32(np.exp2(1.5))
    expected_fc[1, 1] = np.float32(np.exp2(-0.25))
    expected_fc[2, 2] = np.float32(1.0)
    np.testing.assert_array_equal(d.fold_changes, expected_fc)
    np.testing.assert_array_equal(
        d.lfc_true, np.array([[1.5, NAN, NAN], [NAN, -0.25, NAN], [NAN, NAN, 0.0]])
    )
    np.testing.assert_array_equal(
        d.fdr, np.array([[0.01, 1, 1], [1, 0.3, 1], [1, 1, 0.9]], dtype=np.float32)
    )


def test_run_prep_duplicate_pair_across_files_raises(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    write_labels_csv(tmp_path / "labels" / "ctxc" / "labels.csv", [("ctxa", "P1", "G1", 0.1, 1.0)])
    with pytest.raises(ValueError, match="appears in both"):
        run_prep(cfg)


def test_run_prep_drug_dose(tmp_path: Path) -> None:
    cells = [
        ("A", "[('DMSO_TF', 0.0, 'uM')]", [1, 1, 1, 1]),
        ("A", "[('DrugA', 0.05, 'uM')]", [3, 3, 3, 3]),
    ]
    write_h5ad(
        tmp_path / "expression" / "cells.h5ad",
        cells=cells,
        context_col="context",
        pert_col="perturbation",
    )
    write_labels_csv(
        tmp_path / "labels" / "A" / "labels.csv", [("A", "DrugA_0.05uM", "G1", 0.01, 2.0)]
    )
    cfg = compose_prep_config(
        [
            "name=drugs",
            f"labels={tmp_path / 'labels' / '*' / 'labels.csv'}",
            f"h5ad={tmp_path / 'expression' / '*.h5ad'}",
            f"output_dir={tmp_path / 'out'}",
            "obs.context_col=context",
            "obs.pert_col=perturbation",
            "obs.control_label=DMSO_TF_0.0uM",
            "obs.pert_format=drug_dose",
        ]
    )
    d = PreprocessedDir.open(run_prep(cfg))
    assert d.meta.pert_kind == "drug"
    assert d.perts == ["DrugA_0.05uM"]
    assert d.genes == ["G1"]
    np.testing.assert_array_equal(d.delta_p, np.array([[2.0]], dtype=np.float32))
    np.testing.assert_array_equal(d.ctrl_means, np.array([[1.0]], dtype=np.float32))


def _relabel_contexts(tmp_path: Path, names: dict[str, str]) -> str:
    """Write LABELS_A/B again with renamed label contexts; returns their glob."""
    for rows in (LABELS_A, LABELS_B):
        ctx = names[rows[0][0]]
        write_labels_csv(
            tmp_path / "relabelled" / ctx / "labels.csv", [(ctx, *r[1:]) for r in rows]
        )
    return str(tmp_path / "relabelled" / "*" / "labels.csv")


def test_run_prep_label_context_case_lower(tmp_path: Path) -> None:
    ref = PreprocessedDir.open(run_prep(make_cfg(tmp_path / "ref")))
    labels = _relabel_contexts(tmp_path / "alt", {"ctxa": "CTXA", "ctxb": "CtxB"})
    cfg = make_cfg(tmp_path / "alt", "label.context_case=lower", f"labels={labels}")
    d = PreprocessedDir.open(run_prep(cfg))
    assert d.contexts == ["ctxa", "ctxb"]
    assert d.meta.array_sha256 == ref.meta.array_sha256


def test_run_prep_label_context_map(tmp_path: Path) -> None:
    ref = PreprocessedDir.open(run_prep(make_cfg(tmp_path / "ref")))
    labels = _relabel_contexts(tmp_path / "alt", {"ctxa": "K1", "ctxb": "K2"})
    cfg = make_cfg(tmp_path / "alt", "label.context_map={K1:ctxa,K2:ctxb}", f"labels={labels}")
    d = PreprocessedDir.open(run_prep(cfg))
    assert d.contexts == ["ctxa", "ctxb"]
    assert d.meta.model_dump() == ref.meta.model_dump()


def test_run_prep_label_context_map_rejects_unmapped(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"'ctxb'.*label context map"):
        run_prep(make_cfg(tmp_path, "label.context_map={ctxa:ctxa}"))


def test_run_prep_label_pert_format_drug_dose(tmp_path: Path) -> None:
    cells = [
        ("A", "[('DMSO_TF', 0.0, 'uM')]", [1, 1]),
        ("A", "[('DrugA', 0.05, 'uM')]", [3, 3]),
        ("A", "[('DrugB', 1.0, 'uM')]", [5, 5]),
    ]
    write_h5ad(tmp_path / "expression" / "cells.h5ad", cells=cells, genes=["G1", "G2"])
    (tmp_path / "de").mkdir()
    pq.write_table(
        pa.table(
            {
                "target": ["[('DrugA', 0.05, 'uM')]", "DrugB_1.0uM"],
                "feature": ["G1", "G2"],
                "fold_change": [2.0, 4.0],
                "fdr": [0.01, 0.02],
            }
        ),
        tmp_path / "de" / "A.parquet",
    )
    cfg = compose_prep_config(
        [
            "name=drugs",
            f"labels={tmp_path / 'de' / '*.parquet'}",
            f"h5ad={tmp_path / 'expression' / '*.h5ad'}",
            f"output_dir={tmp_path / 'out'}",
            "obs.context_col=cell_line",
            "obs.pert_col=gene",
            "obs.control_label=DMSO_TF_0.0uM",
            "obs.pert_format=drug_dose",
            "label.pert_format=drug_dose",
            "label.pert_col=target",
            "label.gene_col=feature",
            "label.context_from=stem",
            "label.context_col=null",
        ]
    )
    d = PreprocessedDir.open(run_prep(cfg))
    assert d.perts == ["DrugA_0.05uM", "DrugB_1.0uM"]
    np.testing.assert_array_equal(d.delta_p, np.array([[2.0, 2.0], [4.0, 4.0]], dtype=np.float32))


def test_run_prep_controls_only(tmp_path: Path) -> None:
    d = PreprocessedDir.open(run_prep(make_cfg(tmp_path, "controls_only=true", "labels=null")))
    assert d.controls_only
    assert d.genes == ["G3", "G1", "G2", "G4"]
    assert d.contexts == ["ctxa", "ctxb"]
    assert d.perts == []
    assert d.meta.num_rows == 0
    assert d.pert_ensembl == {}
    assert set(d.meta.array_sha256) == {pp.CTRL_MEANS}
    np.testing.assert_array_equal(
        d.ctrl_means, np.array([[2, 3, 4, 5], [0, 1, 0, 1]], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="controls-only"):
        _ = d.delta_p


def test_run_prep_requires_labels(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="labels"):
        make_cfg(tmp_path, "labels=null")


def test_run_prep_refuses_non_empty_out(tmp_path: Path) -> None:
    out = tmp_path / "taken"
    out.mkdir()
    (out / "x.txt").write_text("x")
    with pytest.raises(FileExistsError, match="overwrite=true"):
        run_prep(make_cfg(tmp_path, f"output_dir={out}"))


def test_run_prep_rows_sorted_by_key_not_file_path(tmp_path: Path) -> None:
    ref = PreprocessedDir.open(run_prep(make_cfg(tmp_path / "ref")))
    write_labels_csv(tmp_path / "swapped" / "a_file" / "labels.csv", LABELS_B)
    write_labels_csv(tmp_path / "swapped" / "b_file" / "labels.csv", LABELS_A)
    cfg = make_cfg(tmp_path / "alt", f"labels={tmp_path / 'swapped' / '*' / 'labels.csv'}")
    d = PreprocessedDir.open(run_prep(cfg))
    assert d.row_keys() == [("ctxa", "P1"), ("ctxa", "P2"), ("ctxb", "P1")]
    assert d.meta.array_sha256 == ref.meta.array_sha256


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pie.cli, "load_common_env", lambda *_a, **_k: {})
    for name in ("PIE_DATA_ROOT", "PIE_RUNS_ROOT", "PIE_CACHE_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))


@pytest.mark.usefixtures("cli_env")
def test_prep_main_matches_run_prep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_cfg(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    argv = ["name=toy", f"labels={cfg.labels}", f"h5ad={cfg.h5ad}",
            f"output_dir={tmp_path / 'cli'}",
            "obs.context_col=cell_line", "obs.pert_col=gene", "obs.control_label=non-targeting",
            f"obs.context_map={_hydra_map(cfg.obs.context_map)}"]  # fmt: skip
    assert prep_main(argv) == 0
    assert list(work.iterdir()) == []
    ref = PreprocessedDir.open(run_prep(cfg))
    got = PreprocessedDir.open(tmp_path / "cli")
    assert got.meta.model_dump() == ref.meta.model_dump()


@pytest.mark.usefixtures("cli_env")
def test_prep_main_help_and_bad_values(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert prep_main(["--help"]) == 0
    assert capsys.readouterr().out.startswith("usage: pie-prep")
    with pytest.raises(ValidationError):
        prep_main(
            ["dataset=jiang", "labels=/l", "h5ad=/h", "output_dir=/o", "label.fc_space=log10"]
        )


def test_output_dir_overwrite(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    run_prep(cfg)
    with pytest.raises(FileExistsError):
        run_prep(cfg)
    run_prep(cfg.model_copy(update={"overwrite": True}))
    assert PreprocessedDir.open(tmp_path / "out").meta.num_rows > 0


def test_label_rewrites_are_no_ops_on_pie_label_tables(tmp_path: Path) -> None:
    # Lower-case contexts and already-parsed perturbations: the replogle/tahoe rewrites are no-ops.
    plain = PreprocessedDir.open(run_prep(make_cfg(tmp_path / "a")))
    rewritten = PreprocessedDir.open(
        run_prep(
            make_cfg(tmp_path / "b", "label.context_case=lower", "label.pert_format=drug_dose")
        )
    )
    assert rewritten.meta.array_sha256 == plain.meta.array_sha256


def _write_obs_h5ad(path: Path, gene: list[str], gene_id: list[str] | None) -> Path:
    """A tiny AnnData whose obs has `gene` and, unless gene_id is None, a categorical gene_id."""
    columns: dict[str, object] = {"gene": gene}
    if gene_id is not None:
        columns["gene_id"] = pd.Categorical(gene_id)
    obs = pd.DataFrame(columns, index=[f"cell{i}" for i in range(len(gene))])
    x = np.ones((len(gene), 2), dtype=np.float32)
    ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=pd.Index(["G1", "G2"]))).write_h5ad(path)
    return path


def test_pert_ensembl_from_the_h5ad(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, "obs.pert_id_col=gene_id")  # toy h5ad: P1 -> " ENSG00000000001 ",
    d = PreprocessedDir.open(run_prep(cfg))  # P2 -> "nan", P9 (no label rows) -> ENSG00000000009
    assert d.meta.pert_ensembl == {"P1": "ENSG00000000001"}
    assert d.pert_ensembl == {"P1": "ENSG00000000001"}


def test_pert_ensembl_is_empty_without_the_column(tmp_path: Path) -> None:
    assert PreprocessedDir.open(run_prep(make_cfg(tmp_path))).pert_ensembl == {}


def test_pert_ensembl_ids_rules(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, "obs.pert_id_col=gene_id")
    one = _write_obs_h5ad(
        tmp_path / "one.h5ad", gene=["GA", "GB", "GC"], gene_id=["ENSG1", "7157", "-"]
    )
    two = _write_obs_h5ad(tmp_path / "two.h5ad", gene=["GA", "GB"], gene_id=["nan", "ENSG2"])
    assert pert_ensembl_ids([str(one), str(two)], cfg, {"GA", "GB", "GC"}) == {
        "GA": "ENSG1",
        "GB": "ENSG2",
    }
    assert pert_ensembl_ids([str(one)], cfg, {"GB"}) == {}  # not ENSG; GA not asked for
    clash = _write_obs_h5ad(tmp_path / "clash.h5ad", gene=["GA", "GA"], gene_id=["ENSG1", "ENSG9"])
    with pytest.raises(ValueError, match="conflicting Ensembl ids"):
        pert_ensembl_ids([str(clash)], cfg, {"GA"})
    other = _write_obs_h5ad(tmp_path / "other.h5ad", gene=["GA"], gene_id=["ENSG9"])
    with pytest.raises(ValueError, match="conflicting Ensembl ids"):
        pert_ensembl_ids([str(one), str(other)], cfg, {"GA"})
    missing = _write_obs_h5ad(tmp_path / "missing.h5ad", gene=["GA"], gene_id=None)
    with pytest.raises(ValueError, match="gene_id"):
        pert_ensembl_ids([str(missing)], cfg, {"GA"})
