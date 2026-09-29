from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from pie.config import EvalConfig
from pie.data.preprocessed import PreprocessedDir
from pie.evaluate import (
    GRANULAR_COLUMNS,
    METRICS_COLUMNS,
    granular_table,
    metrics_table,
    run_eval,
    scoring_inputs,
)
from pie.metrics import METRIC_KEYS, ScoringInputs, score
from pie.predict import PredictionBlock, RowSource, predict
from tests.fixtures import TinyData
from tests.pipeline import TrainedRun


def _eval_config(run: TrainedRun, row_set: str, **overrides: Any) -> EvalConfig:
    values: dict[str, Any] = {
        "experiment_name": "tiny",
        "run_dir": str(run.run_dir),
        "ckpt": "best_auprc",
        "split_path": str(run.test_split),
        "row_set": row_set,
        "preprocessed_dirs": None,
        "save_predictions": False,
        "overwrite": False,
        "device": "cpu",
        "batch_size": 16,
    }
    values.update(overrides)
    return EvalConfig(**values)


def _alpha_block(
    d: PreprocessedDir, keys: list[tuple[str, str]], rows: list[int]
) -> PredictionBlock:
    zeros = np.zeros((len(keys), len(d.genes)), dtype=np.float32)
    return PredictionBlock(
        dataset="alpha",
        genes=list(d.genes),
        contexts=[c for c, _ in keys],
        perts=[p for _, p in keys],
        row_index=np.asarray(rows, dtype=np.int64),
        p_de=zeros,
        lfc_pred=zeros,
        delta_p_pred=zeros,
    )


def test_scoring_inputs_reads_the_labels_of_the_block_rows(tiny_data: TinyData) -> None:
    d = PreprocessedDir.open(tiny_data.preprocessed["alpha"])
    keys = [("a2", "GC"), ("a2", "OLDX")]
    rows = np.asarray([d.row_index()[k] for k in keys], dtype=np.int64)
    got = scoring_inputs(_alpha_block(d, keys, rows.tolist()), d, 0.05)
    assert got.dataset == "alpha"
    assert got.genes == list(d.genes)
    assert (got.contexts, got.perts) == (["a2", "a2"], ["GC", "OLDX"])
    np.testing.assert_array_equal(got.tested, d.tested[rows])
    np.testing.assert_array_equal(got.de_true, (d.fdr[rows] < 0.05) & d.tested[rows])
    assert got.lfc_true.dtype == np.float64
    np.testing.assert_array_equal(got.lfc_true, d.lfc_true[rows])
    np.testing.assert_array_equal(got.delta_p_true, d.delta_p[rows])
    ctrl = np.asarray(d.ctrl_means[d.meta.context_to_id["a2"]], dtype=np.float32)
    np.testing.assert_array_equal(got.ctrl_means, np.stack([ctrl, ctrl]))


def test_scoring_inputs_rejects_query_rows_and_mismatched_dirs(tiny_data: TinyData) -> None:
    d = PreprocessedDir.open(tiny_data.preprocessed["alpha"])
    with pytest.raises(ValueError, match="query rows"):
        scoring_inputs(_alpha_block(d, [("a1", "GA")], [-1]), d, 0.05)
    beta = PreprocessedDir.open(tiny_data.preprocessed["beta"])
    with pytest.raises(ValueError, match="does not match"):
        scoring_inputs(_alpha_block(d, [("a1", "GA")], [0]), beta, 0.05)
    with pytest.raises(ValueError, match="do not match the dir rows"):
        scoring_inputs(_alpha_block(d, [("a2", "GA")], [0]), d, 0.05)


def test_run_eval_writes_the_metric_tables(pipeline: TrainedRun) -> None:
    out = run_eval(_eval_config(pipeline, "unit_eval", save_predictions=True))
    assert out == pipeline.run_dir / "eval" / "unit_eval"
    metrics = pd.read_csv(out / "metrics_best_auprc.csv")
    granular = pd.read_csv(out / "granular_best_auprc.csv")
    assert tuple(metrics.columns) == METRICS_COLUMNS
    assert tuple(granular.columns) == GRANULAR_COLUMNS
    assert metrics["context"].tolist()[-1] == "all"
    assert sorted(metrics["context"].tolist()[:-1]) == ["a2", "b2"]
    assert metrics["n_rows"].tolist()[-1] == 5
    assert int(metrics["n_rows"].iloc[:-1].sum()) == 5
    assert len(granular) == 5
    preds = predict(pipeline.ckpt, RowSource("split", pipeline.test_split), None, "cpu")
    dirs = {name: PreprocessedDir.open(path) for name, path in pipeline.tiny.preprocessed.items()}
    result = score([scoring_inputs(b, dirs[b.dataset], 0.05) for b in preds.blocks])
    pd.testing.assert_frame_equal(metrics, metrics_table(result), check_dtype=False)
    pd.testing.assert_frame_equal(granular, granular_table(result), check_dtype=False)
    assert pq.read_table(out / "predictions_best_auprc.parquet").num_rows == 5


def test_run_eval_removes_stale_predictions(pipeline: TrainedRun) -> None:
    run_eval(_eval_config(pipeline, "unit_stale", save_predictions=True))
    out = run_eval(_eval_config(pipeline, "unit_stale", save_predictions=False, overwrite=True))
    assert (out / "metrics_best_auprc.csv").is_file()
    assert not (out / "predictions_best_auprc.parquet").exists()


def test_run_eval_refuses_existing_tables_unless_overwrite(
    pipeline: TrainedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_eval(_eval_config(pipeline, "unit_refuse"))

    def _fail(*_a: object, **_k: object) -> None:
        raise AssertionError("work ran before the overwrite check")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("pie.evaluate.load_checkpoint", _fail)
        mp.setattr("pie.evaluate.PreprocessedDir.open", _fail)
        mp.setattr("pie.evaluate.predict_loaded", _fail)
        with pytest.raises(FileExistsError, match="overwrite=true"):
            run_eval(_eval_config(pipeline, "unit_refuse"))
    out = run_eval(_eval_config(pipeline, "unit_refuse", overwrite=True))
    assert out == pipeline.run_dir / "eval" / "unit_refuse"


def test_run_eval_needs_the_checkpoint(pipeline: TrainedRun, tmp_path: Path) -> None:
    cfg = _eval_config(pipeline, "missing", run_dir=str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match=r"best_auprc\.ckpt"):
        run_eval(cfg)


def test_run_eval_with_explicit_dirs_matches_the_training_dirs(pipeline: TrainedRun) -> None:
    explicit = [str(pipeline.tiny.preprocessed[name]) for name in ("alpha", "beta")]
    default = run_eval(_eval_config(pipeline, "unit_default_dirs"))
    override = run_eval(_eval_config(pipeline, "unit_explicit_dirs", preprocessed_dirs=explicit))
    pd.testing.assert_frame_equal(
        pd.read_csv(default / "metrics_best_auprc.csv"),
        pd.read_csv(override / "metrics_best_auprc.csv"),
    )


def _random_inputs(dataset: str, contexts: list[str], seed: int) -> ScoringInputs:
    rng = np.random.default_rng(seed)
    shape = (len(contexts), 6)
    ctrl = {c: rng.random(shape[1], dtype=np.float32) for c in sorted(set(contexts))}
    return ScoringInputs(
        dataset=dataset,
        genes=[f"G{i}" for i in range(shape[1])],
        contexts=contexts,
        perts=[f"P{i}" for i in range(shape[0])],
        p_de=rng.random(shape, dtype=np.float32),
        lfc_pred=rng.normal(size=shape).astype(np.float32),
        delta_p_pred=rng.normal(size=shape).astype(np.float32),
        de_true=np.tile([True, False, True, False, False, True], (shape[0], 1)),
        tested=np.ones(shape, dtype=bool),
        lfc_true=rng.normal(size=shape),
        delta_p_true=rng.normal(size=shape).astype(np.float32),
        ctrl_means=np.stack([ctrl[c] for c in contexts]),
    )


def test_metrics_csv_all_row_is_the_context_mean_not_the_aggregate(tmp_path: Path) -> None:
    """R4: the 'all' row averages the context rows, not the datasets (validation's aggregate)."""
    inputs = [
        _random_inputs("x", ["c1", "c1", "c2", "c2", "c3", "c3"], seed=0),
        _random_inputs("y", ["d1", "d1"], seed=1),
    ]
    result = score(inputs)
    path = tmp_path / "metrics.csv"
    path.write_text(metrics_table(result).to_csv(index=False))
    table = pd.read_csv(path)
    assert table["context"].tolist() == ["c1", "c2", "c3", "d1", "all"]
    overall = table.iloc[-1]
    per_context = table.iloc[:-1]
    differs = []
    for key in METRIC_KEYS:
        expected = float(per_context[key].astype(float).mean(skipna=True))
        assert overall[key] == pytest.approx(expected, rel=1e-12, nan_ok=True)
        assert overall[key] == pytest.approx(result.context_mean[key], rel=1e-12, nan_ok=True)
        differs.append(abs(result.aggregate[key] - result.context_mean[key]) > 1e-6)
    assert sum(differs) >= 3  # a regression to `aggregate` breaks the rows above
