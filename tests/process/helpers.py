"""Synthetic count h5ads shared by the pie-process tests."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import DTypeLike

from pie.process.config import DEConfig, FilterConfig

GENES = ("G0", "G1", "G2", "G3", "H0")
CTRL = "non-targeting"
CTRL_ROW = (10.0, 10.0, 0.0, 8.0, 2.0)
# (perturbation, count of its own target gene); every other entry of a perturbed row is 2.
CELLS: tuple[tuple[str, float | None], ...] = (
    (CTRL, None), (CTRL, None), (CTRL, None), (CTRL, None),
    ("G0", 0.0), ("G0", 0.0), ("G0", 1.0), ("G0", 5.0),
    ("G1", 3.0), ("G1", 3.0),
    ("G2", 1.0), ("G2", 1.0),
    ("NOPE", None), ("NOPE", None),
    ("G3", 0.0), ("G3", 0.0), ("G3", 7.0),
)
N_CELLS = len(CELLS)
KEPT_ROWS = (0, 1, 2, 3, 4, 5, 6)


def count_matrix(ctrl_row: tuple[float, ...] = CTRL_ROW) -> np.ndarray:
    """Dense float64 counts for CELLS x GENES."""
    x = np.full((N_CELLS, len(GENES)), 2.0)
    for i, (pert, own) in enumerate(CELLS):
        if pert == CTRL:
            x[i] = ctrl_row
        elif own is not None:
            x[i, GENES.index(pert)] = own
    return x


def make_counts(
    context: str = "c1",
    *,
    ctrl_row: tuple[float, ...] = CTRL_ROW,
    sparse: bool = True,
    dtype: DTypeLike = np.float32,
) -> ad.AnnData:
    """One context of the synthetic screen; obs names are '<context>_<row>'."""
    x = count_matrix(ctrl_row).astype(dtype)
    obs = pd.DataFrame(
        {"gene": [pert for pert, _ in CELLS], "context": [context] * N_CELLS},
        index=[f"{context}_{i}" for i in range(N_CELLS)],
    )
    var = pd.DataFrame({"gene_name": list(GENES)}, index=[f"g{i}" for i in range(len(GENES))])
    var.index.name = "gene_name_index"
    return ad.AnnData(X=sp.csr_matrix(x) if sparse else x, obs=obs, var=var)


def stack(*parts: ad.AnnData) -> ad.AnnData:
    """Concatenate the cells of several make_counts() objects (same genes, same var)."""
    out = ad.concat(list(parts))
    out.var = parts[0].var.copy()
    return out


def write_counts(adata: ad.AnnData, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path


def filter_cfg(out: Path | None, **over: object) -> FilterConfig:
    settings: dict[str, object] = {
        "enabled": True,
        "perturbation_column": "gene",
        "control_label": CTRL,
        "residual_expression": 0.30,
        "cell_residual_expression": 0.50,
        "min_cells": 3,
        "layer": None,
        "var_gene_name": "gene_name",
        "context_column": "context",
        "output_dir": None if out is None else str(out),
    }
    settings.update(over)
    return FilterConfig.model_validate(settings)


def de_cfg(out: Path, **over: object) -> DEConfig:
    settings: dict[str, object] = {
        "enabled": True,
        "groupby": "gene",
        "reference": CTRL,
        "context_column": "context",
        "normalize_target_sum": 1e4,
        "filter_gene_min_cpm_cell": 5.0,
        "device": "auto",
        "output_dir": str(out),
    }
    settings.update(over)
    return DEConfig.model_validate(settings)
