from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import pie.config as config_mod
from pie.data.dataset import RowRef
from pie.data.preprocessed import PreprocessedDir
from pie.predict import (
    PREDICTION_COLUMNS,
    PredictionBlock,
    Predictions,
    RowSource,
    batch_rows,
    load_checkpoint,
    predict,
    write_predictions_parquet,
)
from tests.fixtures import ALPHA_GENES, BETA_GENES
from tests.pipeline import TrainedRun, controls_only_dir, write_query


def _refs(*keys: tuple[int, str]) -> list[RowRef]:
    return [
        RowRef(dir_index=di, row=i, context=context, perturbation=f"p{i}")
        for i, (di, context) in enumerate(keys)
    ]


def test_batch_rows_cuts_each_context_run_into_chunks() -> None:
    rows = _refs((0, "a1"), (0, "a1"), (0, "a1"), (0, "a2"), (1, "a2"), (1, "b1"))
    assert batch_rows(rows, 2) == [[0, 1], [2], [3], [4], [5]]
    assert batch_rows(rows, 16) == [[0, 1, 2], [3], [4], [5]]
    assert batch_rows([], 4) == []
    with pytest.raises(ValueError, match="batch_size"):
        batch_rows(rows, 0)


def test_load_checkpoint_resolves_portable_paths(pipeline: TrainedRun) -> None:
    raw = torch.load(pipeline.ckpt, map_location="cpu", weights_only=False)
    assert raw["pie"]["config"]["run_dir"] == "${PIE_RUNS_ROOT}/tiny"
    first_dir = raw["pie"]["config"]["data"]["preprocessed_dirs"][0]
    assert first_dir == "${PIE_DATA_ROOT}/preprocessed/alpha"
    loaded = load_checkpoint(pipeline.ckpt)
    assert loaded.config.run_dir == str(pipeline.run_dir)
    assert loaded.config.data.preprocessed_dirs == [
        str(pipeline.tiny.preprocessed["alpha"]),
        str(pipeline.tiny.preprocessed["beta"]),
    ]
    assert loaded.stats.datasets == ["alpha", "beta"]
    assert "latents" in loaded.state_dict
    assert not any(key.startswith("model.") for key in loaded.state_dict)
    assert "gene_query_text" not in loaded.state_dict


def test_load_checkpoint_rejects_a_foreign_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "foreign.ckpt"
    torch.save({"state_dict": {"w": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="not a pie checkpoint"):
        load_checkpoint(path)


def test_predict_split_rows_follow_dir_row_order(pipeline: TrainedRun) -> None:
    preds = predict(pipeline.ckpt, RowSource("split", pipeline.test_split), None, "cpu")
    alpha, beta = preds.blocks
    assert (alpha.dataset, beta.dataset) == ("alpha", "beta")
    assert alpha.genes == ALPHA_GENES
    assert beta.genes == BETA_GENES
    assert list(zip(alpha.contexts, alpha.perts, strict=True)) == [("a2", "GC"), ("a2", "OLDX")]
    assert list(zip(beta.contexts, beta.perts, strict=True)) == [
        ("b2", "drugA"),
        ("b2", "drugB"),
        ("b2", "drugC"),
    ]
    index = PreprocessedDir.open(pipeline.tiny.preprocessed["alpha"]).row_index()
    assert alpha.row_index.tolist() == [index[("a2", "GC")], index[("a2", "OLDX")]]
    grid = load_checkpoint(pipeline.ckpt).stats.delta_p
    centres = ((np.arange(grid.n_bins) + 0.5) / grid.n_bins * 2 - 1) * grid.max_delta
    for block, n_rows in ((alpha, 2), (beta, 3)):
        for name in PREDICTION_COLUMNS:
            values = getattr(block, name)
            assert values.shape == (n_rows, 6)
            assert values.dtype == np.float32
        assert np.all((block.p_de >= 0.0) & (block.p_de <= 1.0))
        assert np.isfinite(block.lfc_pred).all()
        on_grid = np.isclose(block.delta_p_pred[..., None], centres.astype(np.float32), atol=1e-6)
        assert on_grid.any(axis=-1).all()


def test_predict_does_not_depend_on_the_batch_size(pipeline: TrainedRun) -> None:
    rows = RowSource("split", pipeline.test_split)
    one = predict(pipeline.ckpt, rows, None, "cpu", batch_size=1)
    many = predict(pipeline.ckpt, rows, None, "cpu", batch_size=16)
    for a, b in zip(one.blocks, many.blocks, strict=True):
        assert (a.contexts, a.perts) == (b.contexts, b.perts)
        np.testing.assert_allclose(a.p_de, b.p_de, atol=1e-5)
        np.testing.assert_allclose(a.lfc_pred, b.lfc_pred, atol=1e-5)


def test_predict_restores_the_deterministic_setting(pipeline: TrainedRun) -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        predict(pipeline.ckpt, RowSource("split", pipeline.test_split), None, "cpu")
        assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        torch.use_deterministic_algorithms(previous)


def test_predict_query_rows_on_a_controls_only_dir(pipeline: TrainedRun, tmp_path: Path) -> None:
    gamma = controls_only_dir(tmp_path / "gamma", pipeline.tiny.preprocessed["alpha"])
    query = write_query(tmp_path / "query.json")
    preds = predict(pipeline.ckpt, RowSource("query", query), [gamma], "cpu")
    (block,) = preds.blocks
    assert block.dataset == "gamma"
    assert block.genes == ALPHA_GENES
    assert list(zip(block.contexts, block.perts, strict=True)) == [
        ("a1", "GA"),
        ("a1", "NEWPERT"),
        ("a2", "GB"),
    ]
    assert block.row_index.tolist() == [-1, -1, -1]
    assert block.p_de.shape == (3, 6)
    assert np.isfinite(block.lfc_pred).all()


def test_predict_query_rejects_an_unknown_context(pipeline: TrainedRun, tmp_path: Path) -> None:
    gamma = controls_only_dir(tmp_path / "gamma", pipeline.tiny.preprocessed["alpha"])
    query = write_query(tmp_path / "query.json", {"gamma.zz": ["GA"]})
    with pytest.raises(ValueError, match="unknown datasets or contexts"):
        predict(pipeline.ckpt, RowSource("query", query), [gamma], "cpu")


def _block(
    dataset: str, genes: list[str], keys: list[tuple[str, str]], offset: float
) -> PredictionBlock:
    n, g = len(keys), len(genes)
    base = np.arange(n * g, dtype=np.float32).reshape(n, g) / 100 + np.float32(offset)
    return PredictionBlock(
        dataset=dataset,
        genes=genes,
        contexts=[c for c, _ in keys],
        perts=[p for _, p in keys],
        row_index=np.arange(n, dtype=np.int64),
        p_de=base,
        lfc_pred=-base,
        delta_p_pred=base * 2,
    )


def test_predictions_parquet_is_slim_with_the_gene_axis_in_metadata(tmp_path: Path) -> None:
    preds = Predictions(
        blocks=[
            _block("alpha", ["GA", "GB", "GC"], [("a1", "GA"), ("a2", "GB")], 0.0),
            _block("beta", ["GC", "GD"], [("b1", "drugA")], 1.0),
        ]
    )
    path = write_predictions_parquet(preds, tmp_path / "out" / "predictions.parquet")
    table = pq.read_table(path)
    assert table.column_names == [
        "dataset",
        "context",
        "perturbation",
        "p_de",
        "lfc_pred",
        "delta_p_pred",
    ]
    for name in PREDICTION_COLUMNS:
        assert table.schema.field(name).type == pa.list_(pa.float32())
    meta = json.loads(table.schema.metadata[b"pie"])
    genes = {"alpha": ["GA", "GB", "GC"], "beta": ["GC", "GD"]}
    assert meta == {"format_version": 1, "genes": genes}
    assert table.column("dataset").to_pylist() == ["alpha", "alpha", "beta"]
    assert table.column("context").to_pylist() == ["a1", "a2", "b1"]
    assert table.column("perturbation").to_pylist() == ["GA", "GB", "drugA"]
    rows = [np.array(r, dtype=np.float32) for r in table.column("lfc_pred").to_pylist()]
    np.testing.assert_array_equal(rows[1], preds.blocks[0].lfc_pred[1])
    np.testing.assert_array_equal(rows[2], preds.blocks[1].lfc_pred[0])
    assert list((tmp_path / "out").iterdir()) == [path]


def test_predictions_parquet_rejects_empty_predictions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no prediction blocks"):
        write_predictions_parquet(Predictions(blocks=[]), tmp_path / "p.parquet")


def test_checkpoint_with_legacy_split_dir_loads(pipeline: TrainedRun) -> None:
    ckpt = torch.load(pipeline.ckpt, map_location="cpu", weights_only=False)
    ckpt["pie"]["config"]["data"]["split_dir"] = "data/splits/replogle_xdataset"
    legacy = pipeline.run_dir / "legacy.ckpt"
    torch.save(ckpt, legacy)
    loaded = load_checkpoint(legacy)
    assert loaded.config.data.split_dir == f"{config_mod.PIE_SPLITS_URI}/replogle_xdataset"
