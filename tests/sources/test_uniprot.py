"""Tests for the UniProt stream matcher and the proteins table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tests.sources.inputs import FakeSession, write_ncbi_ftp, write_uniprot_tsv

from pie.sources import uniprot


def _stream(tmp_path: Path) -> pd.DataFrame:
    path = write_uniprot_tsv(tmp_path / "stream.tsv")
    return pd.read_csv(path, sep="\t", dtype=str).rename(columns=uniprot.TSV_COLUMNS)


def test_match_symbols_is_two_pass_first_claim(tmp_path: Path) -> None:
    matched, unmatched = uniprot.match_symbols(
        _stream(tmp_path), ["AAA", "BBB", "CCC", "DDD", "EEE", "ZZZ"]
    )
    # BBB's primary row (P1) is already claimed by AAA: BBB is dropped, not unmatched.
    assert matched["symbol"].tolist() == ["AAA", "CCC", "DDD", "EEE"]
    assert matched["accession"].tolist() == ["P1", "P2", "P4", "P3"]
    assert unmatched == ["ZZZ"]


def test_match_symbols_is_case_insensitive(tmp_path: Path) -> None:
    matched, _ = uniprot.match_symbols(_stream(tmp_path), ["ccc", "xdd"])
    assert matched["symbol"].tolist() == ["ccc"]
    assert matched["accession"].tolist() == ["P2"]


def test_build_proteins_table_drops_invalid_sequences(tmp_path: Path) -> None:
    gene_info = write_ncbi_ftp(tmp_path / "ncbi")["gene_info"]
    table = uniprot.build_proteins_table(write_uniprot_tsv(tmp_path / "s.tsv"), gene_info)
    assert list(table.columns) == ["symbol", "uniprot_accession", "sequence"]
    assert table.values.tolist() == [
        ["AAA", "P1", "MAAA"],
        ["CCC", "P2", "MCCC"],
        ["DDD", "P4", "MDDD"],
    ]


def test_valid_sequence() -> None:
    assert uniprot.valid_sequence("mkvU")
    assert not uniprot.valid_sequence("")
    assert not uniprot.valid_sequence("MK1")


def test_fetch_reviewed_human_streams_the_query(tmp_path: Path) -> None:
    session = FakeSession({uniprot.STREAM_URL: b"Entry\n"})
    path = uniprot.fetch_reviewed_human(tmp_path, session=session)
    assert path == tmp_path / "uniprot_reviewed_9606.tsv"
    assert session.calls == [
        (
            uniprot.STREAM_URL,
            {
                "query": "(organism_id:9606) AND (reviewed:true)",
                "format": "tsv",
                "fields": "accession,gene_primary,gene_synonym,protein_name,organism_id,"
                "sequence,length",
            },
        )
    ]


def test_proteins_sha256_tracks_content() -> None:
    table = pd.DataFrame({"symbol": ["A"], "uniprot_accession": ["P1"], "sequence": ["MK"]})
    other = table.assign(sequence=["MKV"])
    assert uniprot.proteins_sha256(table) == uniprot.proteins_sha256(table.copy())
    assert uniprot.proteins_sha256(table) != uniprot.proteins_sha256(other)
