from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

from pie.config import InferConfig
from pie.infer import run_infer
from pie.predict import PREDICTION_COLUMNS, RowSource, predict
from tests.fixtures import ALPHA_GENES
from tests.pipeline import TrainedRun, controls_only_dir, write_query


def _infer_config(run: TrainedRun, out: Path, **overrides: Any) -> InferConfig:
    values: dict[str, Any] = {
        "experiment_name": "tiny",
        "run_dir": str(run.run_dir),
        "ckpt": "best_auprc",
        "rows_kind": "split",
        "rows_path": str(run.test_split),
        "preprocessed_dirs": None,
        "output_path": str(out),
        "overwrite": False,
        "device": "cpu",
        "batch_size": 16,
    }
    values.update(overrides)
    return InferConfig(**values)


def test_run_infer_writes_query_predictions(pipeline: TrainedRun, tmp_path: Path) -> None:
    gamma = controls_only_dir(tmp_path / "gamma", pipeline.tiny.preprocessed["alpha"])
    query = write_query(tmp_path / "query.json")
    cfg = _infer_config(
        pipeline,
        tmp_path / "infer" / "predictions.parquet",
        rows_kind="query",
        rows_path=str(query),
        preprocessed_dirs=[str(gamma)],
    )
    out = run_infer(cfg)
    assert out == tmp_path / "infer" / "predictions.parquet"
    table = pq.read_table(out)
    assert table.num_rows == 3
    assert json.loads(table.schema.metadata[b"pie"])["genes"] == {"gamma": ALPHA_GENES}
    assert table.column("perturbation").to_pylist() == ["GA", "NEWPERT", "GB"]
    expected = predict(pipeline.ckpt, RowSource("query", query), [gamma], "cpu").blocks[0]
    for name in PREDICTION_COLUMNS:
        got = np.array(table.column(name).to_pylist(), dtype=np.float32)
        np.testing.assert_array_equal(got, getattr(expected, name))


def test_run_infer_on_split_rows(pipeline: TrainedRun, tmp_path: Path) -> None:
    out = run_infer(_infer_config(pipeline, tmp_path / "split.parquet"))
    table = pq.read_table(out)
    assert table.column("dataset").to_pylist() == ["alpha", "alpha", "beta", "beta", "beta"]
    assert [len(v) for v in table.column("p_de").to_pylist()] == [6, 6, 6, 6, 6]


def test_run_infer_needs_the_checkpoint(pipeline: TrainedRun, tmp_path: Path) -> None:
    cfg = _infer_config(pipeline, tmp_path / "x.parquet", run_dir=str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match=r"best_auprc\.ckpt"):
        run_infer(cfg)


def test_run_infer_refuses_an_existing_file_unless_overwrite(
    pipeline: TrainedRun, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "predictions.parquet"
    cfg = _infer_config(pipeline, out)
    run_infer(cfg)

    def _fail(*_a: object, **_k: object) -> None:
        raise AssertionError("predict ran before the overwrite check")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("pie.infer.predict", _fail)
        with pytest.raises(FileExistsError, match="overwrite=true"):
            run_infer(cfg)
    assert run_infer(_infer_config(pipeline, out, overwrite=True)) == out
