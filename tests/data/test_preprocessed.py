from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from pie.data import preprocessed as pp
from pie.data.preprocessed import (
    PreprocessedDir,
    PreprocessedMeta,
    target_gene_index,
    write_preprocessed,
)
from pie.utils import sha256_bytes, sha256_file

GENES = ["G3", "G1", "G2"]


def _meta(**overrides: object) -> PreprocessedMeta:
    fields: dict[str, object] = {
        "format_version": pp.FORMAT_VERSION,
        "dataset": "toy",
        "genes": GENES,
        "context_to_id": {"ctxB": 0, "ctxA": 1},
        "pert_to_id": {"P2": 0, "P1": 1},
        "pert_kind": "gene",
        "control_label": "non-targeting",
        "num_rows": 3,
        "num_genes": 3,
        "num_contexts": 2,
        "num_perts": 2,
        "controls_only": False,
        "tool_version": "0.0.0",
        "array_sha256": {},
    }
    fields.update(overrides)
    return PreprocessedMeta.model_validate(fields)


def _arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        pp.FOLD_CHANGES: rng.random((3, 3), dtype=np.float32),
        pp.FDR: rng.random((3, 3), dtype=np.float32),
        pp.TESTED: np.array([[True, False, True]] * 3),
        pp.LFC_TRUE: rng.random((3, 3)),
        pp.DELTA_P: rng.random((3, 3), dtype=np.float32),
        pp.CTRL_MEANS: rng.random((2, 3), dtype=np.float32),
        pp.CTX_IDS: np.array([1, 1, 0], dtype=np.int32),
        pp.PERT_IDS: np.array([1, 0, 1], dtype=np.int32),
    }


def test_write_and_open_round_trip(tmp_path: Path) -> None:
    arrays = _arrays()
    out = write_preprocessed(tmp_path / "toy", _meta(), arrays)
    assert out == tmp_path / "toy"
    d = PreprocessedDir.open(out)
    assert d.dataset == "toy"
    assert d.genes == GENES
    assert not d.controls_only
    assert d.contexts == ["ctxB", "ctxA"]
    assert d.perts == ["P2", "P1"]
    for name, arr in arrays.items():
        loaded = np.load(out / name, mmap_mode="r")
        assert loaded.dtype == pp.ARRAY_DTYPES[name]
        np.testing.assert_array_equal(loaded, arr)
    assert isinstance(d.fold_changes, np.memmap)
    np.testing.assert_array_equal(d.fdr, arrays[pp.FDR])
    np.testing.assert_array_equal(d.tested, arrays[pp.TESTED])
    np.testing.assert_array_equal(d.lfc_true, arrays[pp.LFC_TRUE])
    np.testing.assert_array_equal(d.delta_p, arrays[pp.DELTA_P])
    np.testing.assert_array_equal(d.ctrl_means, arrays[pp.CTRL_MEANS])
    np.testing.assert_array_equal(d.ctx_ids, arrays[pp.CTX_IDS])
    np.testing.assert_array_equal(d.pert_ids, arrays[pp.PERT_IDS])
    assert set(d.meta.array_sha256) == set(pp.ALL_ARRAYS)
    for name, digest in d.meta.array_sha256.items():
        assert digest == sha256_file(out / name)
    assert d.meta_sha256() == sha256_bytes((out / pp.META).read_bytes())


def test_row_keys_and_row_index(tmp_path: Path) -> None:
    d = PreprocessedDir.open(write_preprocessed(tmp_path / "toy", _meta(), _arrays()))
    assert d.row_keys() == [("ctxA", "P1"), ("ctxA", "P2"), ("ctxB", "P1")]
    assert d.row_index() == {("ctxA", "P1"): 0, ("ctxA", "P2"): 1, ("ctxB", "P1"): 2}


def test_row_index_rejects_duplicate_keys(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays[pp.CTX_IDS] = np.array([0, 0, 0], dtype=np.int32)
    arrays[pp.PERT_IDS] = np.array([0, 0, 1], dtype=np.int32)
    d = PreprocessedDir.open(write_preprocessed(tmp_path / "toy", _meta(), arrays))
    with pytest.raises(ValueError, match="duplicate"):
        d.row_index()


def test_meta_json_holds_no_paths(tmp_path: Path) -> None:
    out = write_preprocessed(tmp_path / "toy", _meta(), _arrays())
    text = (out / pp.META).read_text()
    assert str(tmp_path) not in text
    assert set(json.loads(text)) == set(PreprocessedMeta.model_fields)


def test_write_rejects_wrong_dtype(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays[pp.FDR] = arrays[pp.FDR].astype(np.float64)
    with pytest.raises(ValueError, match=r"fdr\.npy"):
        write_preprocessed(tmp_path / "toy", _meta(), arrays)
    assert not (tmp_path / "toy").exists()


def test_write_rejects_wrong_shape(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays[pp.CTRL_MEANS] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=r"ctrl_means\.npy"):
        write_preprocessed(tmp_path / "toy", _meta(), arrays)


def test_write_rejects_missing_or_extra_arrays(tmp_path: Path) -> None:
    missing = _arrays()
    del missing[pp.DELTA_P]
    with pytest.raises(ValueError, match="expected arrays"):
        write_preprocessed(tmp_path / "a", _meta(), missing)
    extra = _arrays()
    extra["bin_labels.npy"] = np.zeros((3, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="expected arrays"):
        write_preprocessed(tmp_path / "b", _meta(), extra)


def test_write_rejects_out_of_range_ids(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays[pp.CTX_IDS] = np.array([0, 2, 0], dtype=np.int32)
    with pytest.raises(ValueError, match=r"ctx_ids\.npy"):
        write_preprocessed(tmp_path / "toy", _meta(), arrays)


def test_write_refuses_non_empty_out(tmp_path: Path) -> None:
    out = tmp_path / "toy"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        write_preprocessed(out, _meta(), _arrays())
    assert (out / "keep.txt").read_text() == "x"


def test_write_accepts_empty_existing_out(tmp_path: Path) -> None:
    out = tmp_path / "toy"
    out.mkdir()
    write_preprocessed(out, _meta(), _arrays())
    assert PreprocessedDir.open(out).meta.num_rows == 3


def test_controls_only_dir(tmp_path: Path) -> None:
    meta = _meta(pert_to_id={}, num_rows=0, num_perts=0, controls_only=True)
    ctrl = np.arange(6, dtype=np.float32).reshape(2, 3)
    d = PreprocessedDir.open(write_preprocessed(tmp_path / "q", meta, {pp.CTRL_MEANS: ctrl}))
    assert d.controls_only
    assert set(d.meta.array_sha256) == {pp.CTRL_MEANS}
    np.testing.assert_array_equal(d.ctrl_means, ctrl)
    assert d.row_keys() == []
    assert d.perts == []
    with pytest.raises(ValueError, match="controls-only"):
        _ = d.fold_changes
    with pytest.raises(ValueError, match="controls-only"):
        _ = d.ctx_ids


def test_open_rejects_unknown_meta_key(tmp_path: Path) -> None:
    out = write_preprocessed(tmp_path / "toy", _meta(), _arrays())
    raw = json.loads((out / pp.META).read_text())
    raw["source_files"] = []
    (out / pp.META).write_text(json.dumps(raw))
    with pytest.raises(ValidationError):
        PreprocessedDir.open(out)


def test_open_rejects_wrong_format_version(tmp_path: Path) -> None:
    out = write_preprocessed(tmp_path / "toy", _meta(), _arrays())
    raw = json.loads((out / pp.META).read_text())
    raw["format_version"] = 99
    (out / pp.META).write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="format_version"):
        PreprocessedDir.open(out)


def test_open_rejects_missing_array_file(tmp_path: Path) -> None:
    out = write_preprocessed(tmp_path / "toy", _meta(), _arrays())
    (out / pp.TESTED).unlink()
    with pytest.raises(FileNotFoundError, match=r"tested\.npy"):
        PreprocessedDir.open(out)


def test_target_gene_index() -> None:
    gene_index = {"G3": 0, "G1": 1}
    assert target_gene_index("G1", gene_index) == 1
    assert target_gene_index("G3", gene_index) == 0
    assert target_gene_index("DrugA_0.05uM", gene_index) == -1
    assert target_gene_index("", gene_index) == -1


def test_pert_ensembl_round_trips_and_defaults_to_empty(tmp_path: Path) -> None:
    meta = _meta(pert_ensembl={"P1": "ENSG00000000001"})
    d = PreprocessedDir.open(write_preprocessed(tmp_path / "a", meta, _arrays()))
    assert d.pert_ensembl == {"P1": "ENSG00000000001"}
    raw = json.loads((tmp_path / "a" / pp.META).read_text())
    raw.pop("pert_ensembl")
    (tmp_path / "a" / pp.META).write_text(json.dumps(raw))
    assert PreprocessedDir.open(tmp_path / "a").pert_ensembl == {}


@pytest.mark.parametrize(
    ("mapping", "message"),
    [({"NOPE": "ENSG00000000001"}, "not a perturbation"), ({"P1": "7157"}, "ENSG")],
)
def test_pert_ensembl_is_checked(tmp_path: Path, mapping: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        write_preprocessed(tmp_path / "a", _meta(pert_ensembl=mapping), _arrays())


def test_write_preprocessed_overwrite(tmp_path: Path) -> None:
    write_preprocessed(tmp_path / "a", _meta(), _arrays())
    with pytest.raises(FileExistsError):
        write_preprocessed(tmp_path / "a", _meta(), _arrays())
    newer = _meta(pert_ensembl={"P1": "ENSG00000000002"})
    write_preprocessed(tmp_path / "a", newer, _arrays(), overwrite=True)
    assert PreprocessedDir.open(tmp_path / "a").pert_ensembl == {"P1": "ENSG00000000002"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a"]
