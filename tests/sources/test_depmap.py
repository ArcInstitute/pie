"""Tests for the DepMap gene-effect builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tests.sources.inputs import DEPMAP_KEYS, DEPMAP_VALUES, write_depmap_csv

from pie.sources import depmap


def test_parse_gene_columns_keeps_the_first_symbol() -> None:
    columns = ["TP53 (7157)", "BAD", "AAA  (1)", "TP53 (7158)"]
    assert depmap.parse_gene_columns(columns) == [("TP53", "TP53 (7157)"), ("AAA", "AAA  (1)")]


def test_read_gene_effect_sorts_models(tmp_path: Path) -> None:
    frame = depmap.read_gene_effect(write_depmap_csv(tmp_path / "effect.csv"))
    assert frame.index.tolist() == ["ACH-1", "ACH-2"]


def test_build_depmap_sorts_genes_and_fills_missing(tmp_path: Path) -> None:
    keys, matrix = depmap.build_depmap(write_depmap_csv(tmp_path / "effect.csv"))
    assert keys == DEPMAP_KEYS
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix, DEPMAP_VALUES)
