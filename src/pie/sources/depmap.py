"""DepMap CRISPR gene effect: one float32 row per gene over the models, in ModelID order."""

from __future__ import annotations

import re
from collections.abc import Iterable
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pie.sources.common import base_provenance, input_record
from pie.sources.contract import FORMAT_VERSION, SourceMeta, write_source

if TYPE_CHECKING:
    from pie.sources.registry import RunContext

GENE_COLUMN = re.compile(r"^(.+)\s+\((\d+)\)$")
NAN_FILL = 0.0
DOWNLOAD_PAGE = "https://depmap.org/portal/data_page/?tab=allData"


def parse_gene_columns(columns: Iterable[object]) -> list[tuple[str, str]]:
    """(symbol, column) for every 'SYMBOL (ENTREZ)' header; a repeated symbol keeps the first."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for column in columns:
        text = str(column)
        match = GENE_COLUMN.match(text)
        if match is None:
            continue
        symbol = match.group(1).strip()
        if symbol in seen:
            continue
        seen.add(symbol)
        out.append((symbol, text))
    return out


def read_gene_effect(csv: Path) -> pd.DataFrame:
    """CRISPRGeneEffect.csv with ModelID as the index, rows sorted by ModelID."""
    return pd.read_csv(csv, index_col=0).sort_index()


def gene_effect_matrix(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """(symbols sorted, float32 (genes, models)) with missing effects set to NAN_FILL."""
    columns = sorted(parse_gene_columns(frame.columns), key=itemgetter(0))
    matrix = np.stack(
        [frame[column].fillna(NAN_FILL).to_numpy(dtype=np.float64) for _, column in columns]
    ).astype(np.float32)
    return [symbol for symbol, _ in columns], matrix


def build_depmap(csv: Path) -> tuple[list[str], np.ndarray]:
    """Gene keys and the (genes, models) float32 gene-effect matrix of a CRISPRGeneEffect CSV."""
    return gene_effect_matrix(read_gene_effect(csv))


def run_depmap(ctx: RunContext) -> Path:
    """depmap_gene_effect: a user-supplied CRISPRGeneEffect.csv -> gene rows."""
    csv = ctx.options.depmap_csv
    if csv is None:
        raise ValueError(
            "depmap_gene_effect needs options.depmap_csv: download CRISPRGeneEffect.csv from "
            f"{DOWNLOAD_PAGE}"
        )
    frame = read_gene_effect(csv)
    keys, matrix = gene_effect_matrix(frame)
    provenance = base_provenance(
        {"crispr_gene_effect": input_record(csv, None)},
        {
            "nan_fill_value": NAN_FILL,
            "column_order": "ModelID ascending",
            "model_ids": [str(model) for model in frame.index],
        },
    )
    meta = SourceMeta(
        format_version=FORMAT_VERSION,
        name="depmap_gene_effect",
        layout="dense",
        index="pert",
        keys=keys,
        dim=int(matrix.shape[1]),
        dtype="float32",
        provenance=provenance,
    )
    return write_source(
        ctx.out_root / "depmap_gene_effect", meta, matrix, overwrite=ctx.overwrite
    )
