"""Per-context differential expression with gpudge (optional extra `process`).

Per context this calls `gpudge.de(adata, ...)` with `groupby`, `reference`, `normalize_target_sum`
on raw counts, adding `filter_gene_min_cpm_cell=...` when it is set. Every other gpudge parameter
keeps its default. gpudge runs on CUDA only (its `de()` raises without a GPU), so the stage
checks for one before any work starts.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import anndata as ad

from pie.process.config import DEConfig
from pie.process.io import check_writable, context_masks, write_parquet_atomic
from pie.process.normalize import as_float32_counts, check_integral
from pie.utils import resolve_path

logger = logging.getLogger(__name__)

GPUDGE_MISSING = (
    "the de stage needs gpudge, which comes with the optional 'process' extra; "
    'install it with `pip install "arc-pie[process]"` (or `uv sync --extra process` in a checkout)'
)
OUTPUT_COLUMNS: tuple[str, ...] = (
    "target",
    "feature",
    "target_mean",
    "ref_mean",
    "target_ncells",
    "ref_ncells",
    "log2_fold_change",
    "p_value",
    "Ueffect",
    "p_adj",
)


def import_gpudge() -> ModuleType:
    """Import gpudge lazily; raise RuntimeError(GPUDGE_MISSING) when the extra is not installed."""
    try:
        return importlib.import_module("gpudge")
    except ImportError as err:
        raise RuntimeError(GPUDGE_MISSING) from err


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def resolve_device(device: str) -> str:
    """'auto' -> 'cuda' when torch sees a GPU, else 'cpu'; 'cuda' without a GPU raises."""
    if device == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if device == "cuda" and not _cuda_available():
        raise RuntimeError("de.device=cuda but torch sees no CUDA device")
    return device


def prepare_de(cfg: DEConfig) -> ModuleType:
    """Import gpudge and check the device; called before any stage does work."""
    gpudge = import_gpudge()
    device = resolve_device(cfg.device)
    if device != "cuda":
        raise RuntimeError(
            f"de.device={cfg.device} resolved to {device}, but gpudge 0.7.0 runs on CUDA only; "
            "run the de stage on a machine with a CUDA GPU"
        )
    return gpudge


def de_kwargs(cfg: DEConfig) -> dict[str, Any]:
    """The gpudge.de keyword arguments (canonical run_de.py:236-252)."""
    kwargs: dict[str, Any] = {
        "groupby": cfg.groupby,
        "reference": cfg.reference,
        "normalize_target_sum": cfg.normalize_target_sum,
    }
    if cfg.filter_gene_min_cpm_cell is not None:
        kwargs["filter_gene_min_cpm_cell"] = cfg.filter_gene_min_cpm_cell
    return kwargs


def preflight(adata: ad.AnnData, groupby: str, reference: str) -> None:
    """Canonical run_de.py:153-171: groupby present, no NA, at least 2 groups, reference present."""
    if groupby not in adata.obs.columns:
        raise KeyError(f"groupby column {groupby!r} not in obs; have {sorted(adata.obs.columns)}")
    groups = adata.obs[groupby]
    n_missing = int(groups.isna().sum())
    if n_missing:
        raise ValueError(f"obs[{groupby!r}] has {n_missing} missing values")
    labels = groups.astype(str)
    n_groups = int(labels.nunique())
    if n_groups < 2:
        raise ValueError(f"need at least 2 groups in obs[{groupby!r}], found {n_groups}")
    if not bool((labels == reference).any()):
        raise ValueError(f"reference {reference!r} matches no cells in obs[{groupby!r}]")


def _file_name(context: str) -> str:
    if not context or "/" in context or context.startswith("."):
        raise ValueError(f"context {context!r} cannot be used as a file name")
    return context


def run_de(
    inputs: Sequence[Path], cfg: DEConfig, overwrite: bool, gpudge: ModuleType | None = None
) -> list[Path]:
    """DE for every context of every input count h5ad; writes `<output_dir>/<context>.parquet`."""
    if gpudge is None:
        gpudge = prepare_de(cfg)
    if cfg.output_dir is None:
        raise ValueError("de.output_dir is not set")
    out_dir = resolve_path(cfg.output_dir)
    kwargs = de_kwargs(cfg)
    claimed: set[Path] = set()
    written: list[Path] = []
    for src in inputs:
        adata = ad.read_h5ad(src)
        contexts = context_masks(adata, cfg.context_column, src.stem)
        targets = [out_dir / f"{_file_name(context)}.parquet" for context, _ in contexts]
        for path in targets:
            if path in claimed:
                raise ValueError(
                    f"context output {path.name} is produced twice; "
                    "a context may appear in one input file only"
                )
            check_writable(path, overwrite)
        claimed.update(targets)
        for (context, mask), dst in zip(contexts, targets, strict=True):
            sub = adata if mask.all() else adata[mask].copy()
            sub.X = as_float32_counts(sub.X)
            check_integral(sub.X)
            preflight(sub, str(cfg.groupby), str(cfg.reference))
            logger.info("[de] %s: %s cells; gpudge.de(**%s)", context, f"{sub.n_obs:,}", kwargs)
            frame = gpudge.de(sub, **kwargs)
            write_parquet_atomic(frame, dst, overwrite)
            logger.info("[de] %s: %s rows -> %s", context, f"{len(frame):,}", dst)
            written.append(dst)
    return written
