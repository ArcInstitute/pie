"""On-target knockdown filter for genetic perturbation screens.

`knockdown_keep_mask` ports `filter_on_target_knockdown` from cell-load 0.10.4
(`cell_load/utils/data_utils.py:244-411`). The numpy operations are unchanged, so the kept cells are
the same: a dense slice of the matched target columns, control means with `.mean(axis=0)` in X's
dtype, per-perturbation means from float64 `bincount`, `isclose(control mean, 0)` drops a
perturbation, and both thresholds are strict `<`. It returns the keep mask instead of a subset copy.
`filter_adata` applies it once per context, and `run_filter` writes the original object subset by
the kept cells, so obs and var are unchanged apart from the row selection. X keeps its values;
integer X is written as float32, as the canonical arc_vcc_25 wrapper does.

The ported code is used under the MIT License:

    Copyright (c) 2025 Arc Research Institute

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from pie.process.config import FilterConfig
from pie.process.io import check_writable, context_masks, write_csv_atomic, write_h5ad_atomic
from pie.utils import resolve_path

logger = logging.getLogger(__name__)

STATS_FILE = "filter_stats.csv"
DROPPED_FILE = "filter_dropped_perturbations.csv"
STATS_COLUMNS = (
    "context",
    "cells_in",
    "cells_out",
    "control_cells",
    "perts_in",
    "perts_out",
    "perts_dropped",
)
DROPPED_COLUMNS = ("context", "perturbation")


def knockdown_keep_mask(
    adata: ad.AnnData,
    *,
    perturbation_column: str,
    control_label: str,
    residual_expression: float,
    cell_residual_expression: float,
    min_cells: int,
    layer: str | None,
    var_gene_name: str | None,
) -> np.ndarray:
    """Boolean keep mask over adata's cells.

    1. Keep perturbations whose mean target expression / control mean < residual_expression.
    2. Within those, keep cells whose target expression / control mean < cell_residual_expression.
    3. Drop perturbations with fewer than `min_cells` cells left. Control cells are always kept.
    Genes are matched on `var[var_gene_name]`, or on the var index when it is None.
    """
    if var_gene_name is None:
        gene_index = pd.Index(adata.var.index.astype(str))
    else:
        if var_gene_name not in adata.var.columns:
            raise KeyError(f"Column {var_gene_name!r} not found in adata.var.")
        gene_index = pd.Index(adata.var[var_gene_name])

    X: Any = adata.layers[layer] if layer is not None else adata.X
    perts = adata.obs[perturbation_column]
    n_cells = adata.n_obs

    control_mask = (perts == control_label).to_numpy(dtype=bool, na_value=False)

    unique_perts = [p for p in perts.unique() if p != control_label]
    matched_perts = [p for p in unique_perts if p in gene_index]
    n_matched = len(matched_perts)

    logger.info(
        "[input] %s cells | %d perturbations (excl. control)", f"{n_cells:,}", len(unique_perts)
    )

    if n_matched == 0:
        logger.info("[no matched perturbations] returning only control cells")
        return control_mask.copy()

    # Dense (n_cells, n_matched) slice of the matched target columns.
    pert_positions = gene_index.get_indexer(matched_perts)
    X_sub = X[:, pert_positions].toarray() if sp.issparse(X) else np.asarray(X)[:, pert_positions]

    ctrl_means = X_sub[control_mask].mean(axis=0)

    # Each cell's column in X_sub (-1 for control / unmatched).
    pert_to_col = {p: i for i, p in enumerate(matched_perts)}
    cell_col_mapped = perts.map(pert_to_col)
    cell_col = np.full(n_cells, -1, dtype=np.int32)
    valid = ~cell_col_mapped.isna()
    cell_col[valid.values] = cell_col_mapped[valid].values.astype(np.int32)

    matched_cells = cell_col >= 0
    valid_row = np.where(matched_cells)[0]
    valid_col = cell_col[valid_row]

    # Stage 1: perturbation-level filter.
    diag_expr = X_sub[valid_row, valid_col]
    pert_sums = np.bincount(valid_col, weights=diag_expr, minlength=n_matched)
    pert_counts = np.bincount(valid_col, minlength=n_matched).astype(np.float64)
    pert_means = np.where(pert_counts > 0, pert_sums / np.maximum(pert_counts, 1.0), 0.0)

    valid_ctrl = ~np.isclose(ctrl_means, 0.0)
    kd_ratio = np.where(valid_ctrl, pert_means / np.where(valid_ctrl, ctrl_means, 1.0), np.inf)
    stage1_pass = valid_ctrl & (kd_ratio < residual_expression)

    cells_s1 = int(control_mask.sum()) + int(stage1_pass[valid_col].sum())
    perts_s1 = len(unique_perts) - int(stage1_pass.sum())
    logger.info(
        "[stage 1 (pert avg filter)] removed %s cells | %d perturbations",
        f"{n_cells - cells_s1:,}",
        perts_s1,
    )
    prev_cells = cells_s1

    # Stage 2: cell-level filter, for cells whose perturbation passed stage 1.
    passed_sel = stage1_pass[valid_col]
    passed_row = valid_row[passed_sel]
    passed_col = valid_col[passed_sel]

    expr_vals = X_sub[passed_row, passed_col]
    ctrl_means_per_cell = ctrl_means[passed_col]

    nonzero_ctrl = ~np.isclose(ctrl_means_per_cell, 0.0)
    cell_keep = np.zeros(len(passed_row), dtype=bool)
    cell_keep[nonzero_ctrl] = (
        expr_vals[nonzero_ctrl] / ctrl_means_per_cell[nonzero_ctrl] < cell_residual_expression
    )

    keep_mask = control_mask.copy()
    keep_mask[passed_row] = cell_keep

    cells_s2 = int(keep_mask.sum())
    perts_after_s1 = set(np.array(matched_perts)[stage1_pass])
    perts_after_s2 = set(perts.values[keep_mask & matched_cells])
    logger.info(
        "[stage 2 (cell filter)    ] removed %s cells | %d perturbations",
        f"{prev_cells - cells_s2:,}",
        len(perts_after_s1 - perts_after_s2),
    )
    prev_cells = cells_s2

    # Stage 3: minimum cells per perturbation.
    kept_mask = keep_mask & matched_cells
    kept_row = np.where(kept_mask)[0]
    kept_col = cell_col[kept_row]

    perts_removed = 0
    if len(kept_col) > 0:
        pert_kept_counts = np.bincount(kept_col, minlength=n_matched)
        drop_pert = pert_kept_counts < min_cells
        cell_drop = drop_pert[kept_col]
        keep_mask[kept_row[cell_drop]] = False
        perts_removed = int((drop_pert & (pert_kept_counts > 0)).sum())

    cells_s3 = int(keep_mask.sum())
    out_perts = len(set(perts.values[keep_mask]) - {control_label})
    logger.info(
        "[stage 3 (min cells filter)] removed %s cells | %d perturbations",
        f"{prev_cells - cells_s3:,}",
        perts_removed,
    )
    logger.info(
        "[output] %s cells | %d perturbations (excl. control)", f"{cells_s3:,}", out_perts
    )
    return keep_mask


def filter_adata(
    adata: ad.AnnData, cfg: FilterConfig, stem: str
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    """Filter each context on its own controls (canonical wrapper, jiang filter_knockdown.py:47-79).

    Returns the keep mask over all of adata's cells, one stats row per context and the
    (context, perturbation) pairs that were dropped. Contexts come from `cfg.context_column`
    (sorted values), or are the single context `stem` when it is None.
    """
    if not adata.obs_names.is_unique:
        raise ValueError("obs names must be unique for the knockdown filter")
    column = cfg.perturbation_column
    if column not in adata.obs.columns:
        raise KeyError(
            f"perturbation column {column!r} not in obs; have {sorted(adata.obs.columns)}"
        )
    keep = np.zeros(adata.n_obs, dtype=bool)
    stats: list[dict[str, object]] = []
    dropped: list[dict[str, object]] = []
    for context, mask in context_masks(adata, cfg.context_column, stem):
        sub = adata if mask.all() else adata[mask].copy()
        perts = sub.obs[column]
        perts_in = set(perts.tolist()) - {cfg.control_label}
        n_ctrl = int((perts == cfg.control_label).sum())
        logger.info("=== %s ===", context)
        kept = knockdown_keep_mask(
            sub,
            perturbation_column=column,
            control_label=cfg.control_label,
            residual_expression=cfg.residual_expression,
            cell_residual_expression=cfg.cell_residual_expression,
            min_cells=cfg.min_cells,
            layer=cfg.layer,
            var_gene_name=cfg.var_gene_name,
        )
        perts_out = set(perts.to_numpy()[kept].tolist()) - {cfg.control_label}
        keep[np.flatnonzero(mask)[kept]] = True
        stats.append(
            {
                "context": context,
                "cells_in": int(mask.sum()),
                "cells_out": int(kept.sum()),
                "control_cells": n_ctrl,
                "perts_in": len(perts_in),
                "perts_out": len(perts_out),
                "perts_dropped": len(perts_in - perts_out),
            }
        )
        dropped.extend(
            {"context": context, "perturbation": p} for p in sorted(perts_in - perts_out)
        )
    return keep, stats, dropped


def filter_output_paths(inputs: Sequence[Path], cfg: FilterConfig) -> list[Path]:
    """The files run_filter writes: one h5ad per input (same name), then the two CSVs."""
    if cfg.output_dir is None:
        raise ValueError("filter.output_dir is not set")
    out_dir = resolve_path(cfg.output_dir)
    return [*(out_dir / path.name for path in inputs), out_dir / STATS_FILE, out_dir / DROPPED_FILE]


def run_filter(inputs: Sequence[Path], cfg: FilterConfig, overwrite: bool) -> list[Path]:
    """Filter every input count h5ad; write the subsets and filter_stats / dropped CSVs."""
    paths = filter_output_paths(inputs, cfg)
    for path in paths:
        check_writable(path, overwrite)
    targets = paths[: len(inputs)]
    stats_rows: list[dict[str, object]] = []
    dropped_rows: list[dict[str, object]] = []
    for src, dst in zip(inputs, targets, strict=True):
        adata = ad.read_h5ad(src)
        logger.info("[load] %s: %s cells x %s genes", src, f"{adata.n_obs:,}", f"{adata.n_vars:,}")
        keep, stats, dropped = filter_adata(adata, cfg, src.stem)
        logger.info("[keep] %s / %s cells", f"{int(keep.sum()):,}", f"{adata.n_obs:,}")
        out = adata[keep].copy()
        matrix: Any = out.X
        if np.issubdtype(matrix.dtype, np.integer):
            # Canonical arc_vcc_25 wrapper: integer counts are written as float32 (lossless).
            logger.info("[cast] X %s -> float32", matrix.dtype)
            if sp.issparse(matrix):
                matrix.data = matrix.data.astype(np.float32)
            else:
                out.X = matrix.astype(np.float32)
        write_h5ad_atomic(out, dst, overwrite)
        stats_rows.extend(stats)
        dropped_rows.extend(dropped)
    stats_csv = pd.DataFrame(stats_rows, columns=list(STATS_COLUMNS)).to_csv(index=False)
    dropped_csv = pd.DataFrame(dropped_rows, columns=list(DROPPED_COLUMNS)).to_csv(index=False)
    write_csv_atomic(stats_csv, paths[-2], overwrite)
    write_csv_atomic(dropped_csv, paths[-1], overwrite)
    return targets
