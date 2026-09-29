"""Tests for the NCBI Gene FTP parser and the gene-text renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.sources.helpers import make_options
from tests.sources.inputs import NCBI_SYMBOLS, FakeSession, write_ncbi_ftp

from pie.sources.text.ncbi import (
    FTP_URLS,
    NCBI_FTP_FILES,
    TABLE_COLUMNS,
    build_genes_table,
    describe_ncbi_genes,
    fetch_ncbi_ftp,
    protein_coding_symbols,
    read_human_rows,
    render_gene_text,
)

GO_BP = "Gene Ontology biological process annotations, sorted by evidence count."
GO_MF = "Gene Ontology molecular function annotations, sorted by evidence count."
GO_CC = "Gene Ontology cellular component annotations, sorted by evidence count."
DISEASE = (
    "Known disease associations with confidence scores and OMIM "
    "(Online Mendelian Inheritance in Man) identifiers."
)
GROUPS = "Gene family and group memberships by relationship type."
AAA_TEXT = "\n".join(
    [
        "Gene: AAA",
        "",
        "=== GO BIOLOGICAL PROCESS ===",
        GO_BP,
        "- GO:0000001: term one (evidence: 2)",
        "- GO:0000003: term three (evidence: 1)",
        "",
        "=== GO MOLECULAR FUNCTION ===",
        GO_MF,
        "- GO:0000002: term two (evidence: 1)",
        "",
        "=== GO CELLULAR COMPONENT ===",
        GO_CC,
        "None",
        "",
        "=== DISEASE ASSOCIATIONS ===",
        DISEASE,
        "- Disease one (confidence: 4, MIM: 100100)",
        "- Disease two (confidence: 1, MIM: 100200)",
        "",
        "=== GENE GROUPS ===",
        GROUPS,
        "- Related functional gene: CCC",
        "- Potential readthrough sibling: DDD",
    ]
)
EEE_TEXT = "\n".join(
    [
        "Gene: EEE",
        "",
        "=== GO BIOLOGICAL PROCESS ===",
        GO_BP,
        "None",
        "",
        "=== GO MOLECULAR FUNCTION ===",
        GO_MF,
        "None",
        "",
        "=== GO CELLULAR COMPONENT ===",
        GO_CC,
        "None",
        "",
        "=== DISEASE ASSOCIATIONS ===",
        DISEASE,
        "None",
        "",
        "=== GENE GROUPS ===",
        GROUPS,
        "None",
    ]
)
NO_ANNOTATIONS = (
    "go_biological_process",
    "go_molecular_function",
    "go_cellular_component",
    "disease_associations",
    "gene_groups",
)


def test_fetch_ncbi_ftp_downloads_under_upstream_names(tmp_path: Path) -> None:
    bodies = {url: name.encode() for name, url in FTP_URLS.items()}
    files = fetch_ncbi_ftp(tmp_path, session=FakeSession(bodies))
    assert list(files) == list(NCBI_FTP_FILES)
    assert files["medgen_names"] == tmp_path / "NAMES.csv.gz"
    assert files["gene_info"].read_bytes() == b"gene_info"
    session = FakeSession(bodies)
    only = fetch_ncbi_ftp(tmp_path / "other", session=session, names=("gene_info",))
    assert list(only) == ["gene_info"]
    assert session.calls == [(FTP_URLS["gene_info"], None)]


def test_fetch_ncbi_ftp_rejects_unknown_files(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="gene_summary"):
        fetch_ncbi_ftp(tmp_path, session=FakeSession({}), names=("gene_summary",))


def test_read_human_rows_filters_the_tax_id_in_chunks(tmp_path: Path) -> None:
    files = write_ncbi_ftp(tmp_path)
    rows = read_human_rows(files["gene_info"], chunk_rows=2)
    assert rows["GeneID"].tolist() == ["1", "2", "3", "4", "5", "6"]
    assert set(rows["tax_id"]) == {"9606"}
    assert "tax_id" in rows.columns


def test_protein_coding_symbols_keep_gene_info_order(tmp_path: Path) -> None:
    files = write_ncbi_ftp(tmp_path)
    assert protein_coding_symbols(files["gene_info"]) == list(NCBI_SYMBOLS)


def test_build_genes_table_parses_every_annotation(tmp_path: Path) -> None:
    table = build_genes_table(write_ncbi_ftp(tmp_path))
    assert list(table.columns) == list(TABLE_COLUMNS)
    assert table["symbol"].tolist() == list(NCBI_SYMBOLS)
    assert table["gene_id"].tolist() == [1, 2, 3, 4, 5]
    aaa = table.iloc[0]
    assert aaa["name"] == "alpha gene"
    assert aaa["synonyms"] == ["A1", "A2"]
    assert aaa["db_xrefs"] == ["MIM:1", "HGNC:1"]
    assert aaa["go_biological_process"] == [
        {"go_id": "GO:0000001", "term": "term one", "evidence_count": 2},
        {"go_id": "GO:0000003", "term": "term three", "evidence_count": 1},
    ]
    assert aaa["go_molecular_function"] == [
        {"go_id": "GO:0000002", "term": "term two", "evidence_count": 1}
    ]
    assert aaa["go_cellular_component"] == []
    assert aaa["disease_associations"] == [
        {"disease_name": "Disease one", "mim_number": "100100", "comment_confidence": 4,
         "n_sources": 2},
        {"disease_name": "Disease two", "mim_number": "100200", "comment_confidence": 1,
         "n_sources": 1},
    ]
    assert aaa["gene_groups"] == {
        "Related functional gene": ["CCC"],
        "Potential readthrough sibling": ["DDD"],
    }
    assert table.iloc[1]["disease_associations"] == [
        {"disease_name": "Disease three", "mim_number": "100400", "comment_confidence": 3,
         "n_sources": 0}
    ]
    assert table.iloc[1]["go_cellular_component"][0]["go_id"] == "GO:0000004"
    assert table.iloc[2]["gene_groups"] == {"Related functional gene": ["AAA"]}
    assert table.iloc[4]["gene_groups"] == {}


def test_describe_renders_the_golden_texts(tmp_path: Path) -> None:
    texts = describe_ncbi_genes(build_genes_table(write_ncbi_ftp(tmp_path)))
    assert list(texts) == list(NCBI_SYMBOLS)
    assert texts["AAA"] == AAA_TEXT
    assert texts["EEE"] == EEE_TEXT


def test_render_gene_info_and_summary_sections() -> None:
    row = {
        "symbol": "BBB",
        "name": "beta gene",
        "gene_type": "protein-coding",
        "chromosome": "2",
        "map_location": "2q2",
        "synonyms": [],
        "db_xrefs": ["HGNC:2"],
    }
    assert render_gene_text(row, exclude=NO_ANNOTATIONS) == "\n".join(
        [
            "=== GENE INFO ===",
            "Basic gene identifiers, genomic location, synonyms, and database cross-references.",
            "Symbol: BBB",
            "Full Name: beta gene",
            "Gene Type: protein-coding",
            "Chromosome: 2",
            "Map Location: 2q2",
            "Synonyms: None",
            "Database Cross-References: HGNC:2",
            "",
            "=== SUMMARY ===",
            "Free-text summary of gene function from NCBI.",
            "No summary available.",
        ]
    )
    no_symbol = render_gene_text(
        row, exclude=("gene_info", *NO_ANNOTATIONS), keep_symbol_line=False
    )
    assert no_symbol == (
        "=== SUMMARY ===\nFree-text summary of gene function from NCBI.\nNo summary available."
    )


def test_render_accepts_json_encoded_columns(tmp_path: Path) -> None:
    table = build_genes_table(write_ncbi_ftp(tmp_path))
    encoded = {
        column: json.dumps(value) if isinstance(value, list | dict) else value
        for column, value in table.iloc[0].items()
    }
    assert render_gene_text(pd.Series(encoded)) == AAA_TEXT


def test_render_rejects_unknown_sections() -> None:
    with pytest.raises(ValueError, match="unknown section"):
        render_gene_text({"symbol": "X"}, exclude=("gene_info", "interactions"))


def test_describe_rejects_duplicate_symbols(tmp_path: Path) -> None:
    table = build_genes_table(write_ncbi_ftp(tmp_path))
    with pytest.raises(ValueError, match="duplicate gene symbol"):
        describe_ncbi_genes(pd.concat([table, table.iloc[[0]]], ignore_index=True))


def test_run_ncbi_text_uses_the_pinned_gene_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pie.sources.registry import RunContext
    from pie.sources.text import ncbi

    pinned = tmp_path / "pinned_gene_info.gz"
    pinned.write_bytes(b"pinned")
    fetched: list[tuple[str, ...]] = []
    seen: dict[str, dict[str, Path]] = {}

    def fake_fetch(cache_dir: Path, names: tuple[str, ...] = (), **_k: object) -> dict:
        fetched.append(tuple(names))
        return {name: Path(cache_dir) / name for name in names}

    class _Stop(Exception):
        pass

    def fake_table(files: dict[str, Path]) -> None:
        seen["files"] = dict(files)
        raise _Stop

    monkeypatch.setattr(ncbi, "fetch_ncbi_ftp", fake_fetch)
    monkeypatch.setattr(ncbi, "build_genes_table", fake_table)
    ctx = RunContext(
        datasets=[],
        prior_root=None,
        out_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        options=make_options(tmp_path, device="cpu", gene_info=pinned),
    )
    with pytest.raises(_Stop):
        ncbi.run_ncbi_text(ctx)
    assert all("gene_info" not in names for names in fetched)
    assert seen["files"]["gene_info"] == pinned
    assert list(seen["files"]) == list(ncbi.NCBI_FTP_FILES)
