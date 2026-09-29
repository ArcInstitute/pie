"""Reviewed human UniProt proteome matched to the NCBI protein-coding gene symbols."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from pie.sources.common import download_file
from pie.sources.text.ncbi import protein_coding_symbols
from pie.utils import sha256_bytes

if TYPE_CHECKING:
    import requests

STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
TAX_ID = 9606
QUERY = f"(organism_id:{TAX_ID}) AND (reviewed:true)"
FIELDS: tuple[str, ...] = (
    "accession",
    "gene_primary",
    "gene_synonym",
    "protein_name",
    "organism_id",
    "sequence",
    "length",
)
TSV_COLUMNS: dict[str, str] = {
    "Entry": "accession",
    "Gene Names (primary)": "gene_primary",
    "Gene Names (synonym)": "gene_synonyms",
    "Protein names": "protein_name",
    "Organism (ID)": "organism_id",
    "Sequence": "sequence",
    "Length": "length",
}
STREAM_FILE = "uniprot_reviewed_9606.tsv"
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")


def fetch_reviewed_human(
    cache_dir: Path, session: requests.Session | None = None, offline: bool = False
) -> Path:
    """Stream the reviewed human proteome as TSV into `cache_dir` (a cached file is reused)."""
    params = {"query": QUERY, "format": "tsv", "fields": ",".join(FIELDS)}
    dest = Path(cache_dir) / STREAM_FILE
    return download_file(STREAM_URL, dest, session=session, params=params, offline=offline)


def valid_sequence(sequence: str) -> bool:
    """True for a non-empty string of amino-acid letters (case-insensitive)."""
    return bool(sequence) and all(c in VALID_AMINO_ACIDS for c in sequence.upper())


def _synonym_tokens(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(token) for token in value]
    if isinstance(value, str) and value:
        return [t.strip() for part in value.split(";") for t in part.split() if t.strip()]
    return []


def match_symbols(
    stream: pd.DataFrame, symbols: Sequence[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Match NCBI symbols to stream rows: primary gene names first, then synonyms.

    Names compare upper-case, and a name maps to the first row that lists it. Each row is claimed
    once. A symbol whose primary row is already claimed is dropped (neither matched nor
    unmatched); a symbol without a primary row tries its synonym row. Returns the claimed rows in
    claim order with a ``symbol`` column, and the unmatched symbols.
    """
    stream = stream.reset_index(drop=True)
    primary_row: dict[str, int] = {}
    synonym_row: dict[str, int] = {}
    for idx, (_, row) in enumerate(stream.iterrows()):
        for name in str(row.get("gene_primary", "")).strip().split(";"):
            key = name.strip().upper()
            if key and key != "NAN":
                primary_row.setdefault(key, idx)
        for token in _synonym_tokens(row.get("gene_synonyms", "")):
            key = token.upper()
            if key and key != "NAN":
                synonym_row.setdefault(key, idx)
    claimed: set[int] = set()
    rows: list[int] = []
    matched: list[str] = []
    remaining: list[str] = []
    for symbol in symbols:
        position: int | None = primary_row.get(symbol.upper())
        if position is None:
            remaining.append(symbol)
        elif position not in claimed:
            rows.append(position)
            matched.append(symbol)
            claimed.add(position)
    unmatched: list[str] = []
    for symbol in remaining:
        position = synonym_row.get(symbol.upper())
        if position is not None and position not in claimed:
            rows.append(position)
            matched.append(symbol)
            claimed.add(position)
        else:
            unmatched.append(symbol)
    out = stream.iloc[rows].copy().reset_index(drop=True)
    out["symbol"] = matched
    return out, unmatched


def build_proteins_table(uniprot_tsv: Path, gene_info_gz: Path) -> pd.DataFrame:
    """(symbol, uniprot_accession, sequence) for every matched NCBI protein-coding gene.

    Row order is the claim order of match_symbols. Rows with an invalid sequence are dropped,
    then repeated symbols (the first is kept).
    """
    stream = pd.read_csv(uniprot_tsv, sep="\t", dtype=str).rename(columns=TSV_COLUMNS)
    matched, _ = match_symbols(stream, protein_coding_symbols(gene_info_gz))
    keep = [isinstance(s, str) and valid_sequence(s) for s in matched["sequence"]]
    matched = matched.loc[keep].drop_duplicates(subset=["symbol"], keep="first")
    return pd.DataFrame(
        {
            "symbol": matched["symbol"].tolist(),
            "uniprot_accession": matched["accession"].tolist(),
            "sequence": matched["sequence"].tolist(),
        }
    )


def proteins_sha256(table: pd.DataFrame) -> str:
    """sha256 over the (symbol, accession, sequence) rows, in order (tab/newline separated)."""
    lines = [
        f"{symbol}\t{accession}\t{sequence}\n"
        for symbol, accession, sequence in zip(
            table["symbol"], table["uniprot_accession"], table["sequence"], strict=True
        )
    ]
    return sha256_bytes("".join(lines).encode("utf-8"))
