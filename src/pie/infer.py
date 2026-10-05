"""pie-infer: predictions for query rows (controls-only dirs) or a split file, as parquet."""

from __future__ import annotations

import logging
from pathlib import Path

from pie.assets import resolve_asset, resolve_split_file
from pie.config import InferConfig
from pie.predict import RowSource, predict, write_predictions_parquet
from pie.utils import resolve_path

log = logging.getLogger(__name__)


def run_infer(cfg: InferConfig) -> Path:
    """predict + write_predictions_parquet(output_path); returns output_path."""
    ckpt_path = resolve_path(cfg.run_dir) / f"{cfg.ckpt}.ckpt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    output_path = resolve_path(cfg.output_path)
    if not cfg.overwrite and output_path.exists():
        raise FileExistsError(f"{output_path} exists; set overwrite=true to replace it")
    dirs: list[Path] | None = None
    if cfg.preprocessed_dirs is not None:
        dirs = [resolve_asset(p, kind="preprocessed") for p in cfg.preprocessed_dirs]
    split = cfg.rows_kind == "split"
    rows_path = resolve_split_file(cfg.rows_path) if split else resolve_path(cfg.rows_path)
    rows = RowSource(cfg.rows_kind, rows_path)
    preds = predict(ckpt_path, rows, dirs, cfg.device, cfg.batch_size)
    out = write_predictions_parquet(preds, output_path)
    log.info("wrote %d rows to %s", sum(len(b.contexts) for b in preds.blocks), out)
    return out
