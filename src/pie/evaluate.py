"""pie eval: predict a split file with a checkpoint, score the rows, write the metric tables."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from pie.assets import resolve_asset, resolve_split_file
from pie.config import EvalConfig
from pie.data.preprocessed import PreprocessedDir
from pie.metrics import METRIC_KEYS, PAIR_METRIC_KEYS, ScoreResult, ScoringInputs, score
from pie.predict import (
    PredictionBlock,
    RowSource,
    load_checkpoint,
    predict_loaded,
    write_predictions_parquet,
)
from pie.utils import atomic_write_text, resolve_path

log = logging.getLogger(__name__)

ALL_CONTEXT = "all"
METRICS_COLUMNS: tuple[str, ...] = ("context", "n_rows", *METRIC_KEYS)
GRANULAR_COLUMNS: tuple[str, ...] = ("context", "perturbation", "n_rows", *PAIR_METRIC_KEYS)


def scoring_inputs(
    block: PredictionBlock, d: PreprocessedDir, fdr_threshold: float
) -> ScoringInputs:
    """Predictions of one dir joined with that dir's labels for the same rows."""
    if block.dataset != d.dataset or list(block.genes) != list(d.genes):
        raise ValueError(
            f"prediction block {block.dataset!r} does not match preprocessed dir {d.dataset!r}"
        )
    rows = np.asarray(block.row_index, dtype=np.int64)
    if rows.size == 0 or bool((rows < 0).any()):
        raise ValueError(f"{block.dataset}: query rows (row -1) have no labels to score")
    keys = d.row_keys()
    if [keys[int(r)] for r in rows] != list(zip(block.contexts, block.perts, strict=True)):
        raise ValueError(f"{block.dataset}: block contexts/perturbations do not match the dir rows")
    tested = np.asarray(d.tested[rows], dtype=bool)
    fdr = np.asarray(d.fdr[rows], dtype=np.float32)
    ctx_ids = np.asarray(d.ctx_ids[rows], dtype=np.int64)
    return ScoringInputs(
        dataset=d.dataset,
        genes=list(d.genes),
        contexts=list(block.contexts),
        perts=list(block.perts),
        p_de=block.p_de,
        lfc_pred=block.lfc_pred,
        delta_p_pred=block.delta_p_pred,
        de_true=(fdr < fdr_threshold) & tested,
        tested=tested,
        lfc_true=np.asarray(d.lfc_true[rows], dtype=np.float64),
        delta_p_true=np.asarray(d.delta_p[rows], dtype=np.float32),
        ctrl_means=np.asarray(d.ctrl_means, dtype=np.float32)[ctx_ids],
    )


def metrics_table(result: ScoreResult) -> pd.DataFrame:
    """Per-context rows, then 'all' (n_rows summed; metrics = ScoreResult.context_mean)."""
    per_context = result.per_context.loc[:, list(METRICS_COLUMNS)].reset_index(drop=True)
    overall: dict[str, object] = {
        "context": ALL_CONTEXT,
        "n_rows": int(per_context["n_rows"].sum()),
        **{key: float(result.context_mean[key]) for key in METRIC_KEYS},
    }
    return pd.concat([per_context, pd.DataFrame([overall])], ignore_index=True)


def granular_table(result: ScoreResult) -> pd.DataFrame:
    """One row per (context, perturbation) with the pair-level metrics."""
    return result.per_pair.loc[:, list(GRANULAR_COLUMNS)].reset_index(drop=True)


def run_eval(cfg: EvalConfig) -> Path:
    """Writes <run_dir>/eval/<row_set>/metrics_<ckpt>.csv, granular_<ckpt>.csv and, with
    save_predictions, predictions_<ckpt>.parquet; returns that dir."""
    run_dir = resolve_path(cfg.run_dir)
    ckpt_path = run_dir / f"{cfg.ckpt}.ckpt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    out_dir = run_dir / "eval" / cfg.row_set
    existing = [out_dir / f"metrics_{cfg.ckpt}.csv", out_dir / f"granular_{cfg.ckpt}.csv"]
    if cfg.save_predictions:
        existing.append(out_dir / f"predictions_{cfg.ckpt}.parquet")
    if not cfg.overwrite:
        for path in existing:
            if path.exists():
                raise FileExistsError(f"{path} exists; set overwrite=true to replace it")

    loaded = load_checkpoint(ckpt_path)
    override: list[Path] | None = None
    if cfg.preprocessed_dirs is not None:
        override = [resolve_asset(p, kind="preprocessed") for p in cfg.preprocessed_dirs]
    paths = override or [
        resolve_asset(p, kind="preprocessed") for p in loaded.config.data.preprocessed_dirs
    ]
    dirs = {d.dataset: d for d in (PreprocessedDir.open(p) for p in paths)}

    rows = RowSource("split", resolve_split_file(cfg.split_path))
    preds = predict_loaded(loaded, rows, override, cfg.device, cfg.batch_size)

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / f"predictions_{cfg.ckpt}.parquet"
    if cfg.save_predictions:
        write_predictions_parquet(preds, parquet)
    else:
        parquet.unlink(missing_ok=True)
    fdr = loaded.config.data.fdr_threshold
    result = score([scoring_inputs(block, dirs[block.dataset], fdr) for block in preds.blocks])
    atomic_write_text(
        out_dir / f"metrics_{cfg.ckpt}.csv", metrics_table(result).to_csv(index=False)
    )
    atomic_write_text(
        out_dir / f"granular_{cfg.ckpt}.csv", granular_table(result).to_csv(index=False)
    )
    for key in METRIC_KEYS:
        log.info("%s/%s %s = %.6f", cfg.row_set, cfg.ckpt, key, result.context_mean[key])
    return out_dir
