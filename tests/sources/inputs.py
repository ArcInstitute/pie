"""Tiny synthetic upstream files and fake encoders shared by the source-tool tests."""

from __future__ import annotations

import gzip
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from pie.utils import sha256_file

# --- fake HTTP -------------------------------------------------------------------------------


class FakeResponse:
    """A streamed response; an unknown URL answers 404 on raise_for_status."""

    def __init__(self, url: str, body: bytes | None, headers: Mapping[str, str]) -> None:
        self.url = url
        self._body = body
        self.headers = dict(headers)
        self.status_code = 200 if body is not None else 404

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self._body is None:
            raise RuntimeError(f"404 Not Found: {self.url}")

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        assert self._body is not None
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


class FakeSession:
    """Serves fixed bytes per URL (lower-case header names) and records (url, params) per GET."""

    def __init__(
        self, bodies: Mapping[str, bytes], headers: Mapping[str, str] | None = None
    ) -> None:
        self.bodies = dict(bodies)
        self.headers = dict(headers or {})
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        stream: bool = False,
        timeout: object = None,
    ) -> FakeResponse:
        self.calls.append((url, None if params is None else dict(params)))
        return FakeResponse(url, self.bodies.get(url), self.headers)


# --- file writers ----------------------------------------------------------------------------


def write_tsv(
    path: Path, header: Sequence[str], rows: Sequence[Sequence[str]], *, sep: str = "\t"
) -> Path:
    """Write a delimited table; gzip (mtime 0) when the name ends in .gz."""
    text = "\n".join(sep.join(row) for row in [header, *rows]) + "\n"
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
    return path


GENE_INFO_HEADER = (
    "#tax_id",
    "GeneID",
    "Symbol",
    "LocusTag",
    "Synonyms",
    "dbXrefs",
    "chromosome",
    "map_location",
    "description",
    "type_of_gene",
)
GENE_INFO_ROWS = (
    ("9606", "1", "AAA", "-", "A1|A2", "MIM:1|HGNC:1", "1", "1p1", "alpha gene", "protein-coding"),
    ("9606", "2", "BBB", "-", "-", "-", "2", "2q2", "beta gene", "protein-coding"),
    ("9606", "3", "CCC", "-", "-", "HGNC:3", "3", "3p3", "gamma gene", "protein-coding"),
    ("9606", "4", "DDD", "-", "-", "-", "4", "4p4", "delta gene", "protein-coding"),
    ("9606", "5", "EEE", "-", "-", "-", "5", "5p5", "epsilon gene", "protein-coding"),
    ("9606", "6", "PSG", "-", "-", "-", "6", "6p6", "a pseudogene", "pseudo"),
    ("10090", "7", "Mmm", "-", "-", "-", "7", "7", "mouse gene", "protein-coding"),
)
NCBI_SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE")
GENE2GO_HEADER = (
    "#tax_id", "GeneID", "GO_ID", "Evidence", "Qualifier", "GO_term", "PubMed", "Category"
)
GENE2GO_ROWS = (
    ("9606", "1", "GO:0000002", "IDA", "enables", "term two", "1", "Function"),
    ("9606", "1", "GO:0000001", "IEA", "involved_in", "term one", "-", "Process"),
    ("9606", "1", "GO:0000001", "IDA", "involved_in", "term one", "2", "Process"),
    ("9606", "1", "GO:0000003", "TAS", "involved_in", "term three", "-", "Process"),
    ("9606", "2", "GO:0000004", "IDA", "located_in", "term four", "-", "Component"),
    ("10090", "7", "GO:0000001", "IDA", "involved_in", "term one", "-", "Process"),
)
MIM_HEADER = ("#MIM number", "GeneID", "type", "Source", "MedGenCUI", "Comment")
MIM_ROWS = (
    ("100100", "1", "phenotype", "GeneReviews;OMIM", "C0001", "-"),
    ("100200", "1", "phenotype", "OMIM", "C0002", "question of association"),
    ("100300", "1", "gene", "-", "-", "-"),
    ("100400", "2", "phenotype", "-", "C0003", "somatic"),
)
NAMES_HEADER = ("CUI", "name", "source", "SUPPRESS")
NAMES_ROWS = (
    ("C0001", "Disease one", "MSH", "N"),
    ("C0002", "Disease two", "OMIM", "N"),
    ("C0003", "Disease three", "OMIM", "N"),
)
GROUP_HEADER = ("#tax_id", "GeneID", "relationship", "Other_tax_id", "Other_GeneID")
GROUP_ROWS = (
    ("9606", "1", "Related functional gene", "9606", "3"),
    ("9606", "4", "Potential readthrough sibling", "9606", "1"),
)


def write_ncbi_ftp(cache_dir: Path) -> dict[str, Path]:
    """The five NCBI FTP inputs of ncbi_text, under their upstream file names."""
    return {
        "gene_info": write_tsv(cache_dir / "gene_info.gz", GENE_INFO_HEADER, GENE_INFO_ROWS),
        "gene2go": write_tsv(cache_dir / "gene2go.gz", GENE2GO_HEADER, GENE2GO_ROWS),
        "mim2gene_medgen": write_tsv(cache_dir / "mim2gene_medgen", MIM_HEADER, MIM_ROWS),
        "medgen_names": write_tsv(cache_dir / "NAMES.csv.gz", NAMES_HEADER, NAMES_ROWS, sep=","),
        "gene_group": write_tsv(cache_dir / "gene_group.gz", GROUP_HEADER, GROUP_ROWS),
    }


UNIPROT_HEADER = (
    "Entry",
    "Gene Names (primary)",
    "Gene Names (synonym)",
    "Protein names",
    "Organism (ID)",
    "Sequence",
    "Length",
)
UNIPROT_ROWS = (
    ("P1", "AAA; BBB", "", "prot one", "9606", "MAAA", "4"),
    ("P2", "CCC", "DDD XDD", "prot two", "9606", "MCCC", "4"),
    ("P3", "", "EEE", "prot three", "9606", "MEE1", "4"),
    ("P4", "DDD", "", "prot four", "9606", "MDDD", "4"),
)


def write_uniprot_tsv(path: Path) -> Path:
    """Stream TSV: a shared primary name, a synonym-only hit and an invalid sequence."""
    return write_tsv(path, UNIPROT_HEADER, UNIPROT_ROWS)


STRING_PROTEINS = (b"9606.P3", b"9606.P1", b"9606.P2", b"9606.P4", b"9606.PX")
STRING_EMBEDDINGS = np.arange(10, dtype=np.float16).reshape(5, 2)


def write_string_files(cache_dir: Path, release: str) -> dict[str, Path]:
    """SPACE h5 (P3 and P1 share a name; PX has no name) and protein.info, upstream names."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    h5 = cache_dir / f"9606.protein.network.embeddings.{release}.h5"
    with h5py.File(h5, "w") as handle:
        handle.create_dataset("proteins", data=np.array(STRING_PROTEINS, dtype="S16"))
        handle.create_dataset("embeddings", data=STRING_EMBEDDINGS)
    info = write_tsv(
        cache_dir / f"9606.protein.info.{release}.txt.gz",
        ("#string_protein_id", "preferred_name", "protein_size", "annotation"),
        [
            ("9606.P1", "BBB", "10", "x"),
            ("9606.P2", "AAA", "10", "x"),
            ("9606.P3", "BBB", "10", "x"),
            ("9606.P4", "CCC", "10", "x"),
        ],
    )
    return {"space_h5": h5, "protein_info": info}


DEPMAP_KEYS = ["AAA", "TP53"]
DEPMAP_VALUES = np.array([[0.0, 0.5], [-2.0, -1.0]], dtype=np.float32)


def write_depmap_csv(path: Path) -> Path:
    """Unsorted ModelIDs, a NaN, an unparseable header and a repeated symbol."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ModelID,AAA (1),BAD,TP53 (7157),TP53 (7158)\n"
        "ACH-2,0.5,,-1.0,9\n"
        "ACH-1,,1,-2.0,9\n"
    )
    return path


CHEM_DRUG_KEYS = ("drugA_1.0uM", "drugA_10.0uM", "drugB_1.0uM", "drugC_1.0uM")
FAKE_INCHI: dict[str, tuple[str, str]] = {
    "CCO": ("KEYA", "PARENTA"),
    "CCN": ("KEYB", "KEYB"),
}


def chem_metadata() -> pd.DataFrame:
    """Drug table: drugB carries stray whitespace, drugC has no SMILES."""
    return pd.DataFrame(
        {
            "drug": ["drugA", "drugB ", "drugC"],
            "canonical_smiles": ["CCO", "CCN", ""],
            "pubchem_cid": [1, 2, 3],
        }
    )


def fake_inchi_keys(smiles: str) -> tuple[str | None, str | None]:
    """Stand-in for the RDKit InChIKey pair (full, charge parent)."""
    return FAKE_INCHI.get(str(smiles), (None, None))


def release_table(files: Mapping[str, Path], source: str) -> dict[str, dict[str, str]]:
    """A RELEASES-style table pinning the given local files."""
    return {
        name: {
            "url": f"https://example.invalid/{name}",
            "sha256": sha256_file(path),
            "source": source,
        }
        for name, path in files.items()
    }


def write_l1000_files(cache_dir: Path) -> dict[str, Path]:
    """Two series: drugA by InChIKey, drugC by name (its only TAS is -666), drugB in micro-molar."""
    from pie.sources.chem_profiles import l1000_file

    info = ("pert_id", "pert_iname", "pert_type", "inchi_key", "canonical_smiles")
    sig = ("sig_id", "pert_id", "pert_type", "cell_id", "pert_idose", "pert_itime")
    metrics = ("sig_id", "tas")
    tables = {
        l1000_file("GSE92742", "pert_info"): (
            info,
            [
                ("BRD-A", "alias of a", "trt_cp", "KEYA", ""),
                ("BRD-C", "drugc", "trt_cp", "", ""),
                ("CTL", "DMSO", "ctl_vehicle", "", ""),
            ],
        ),
        l1000_file("GSE92742", "sig_info"): (
            sig,
            [
                ("s1", "BRD-A", "trt_cp", "A375", "10 uM", "24 h"),
                ("s2", "BRD-A", "trt_cp", "A375", "1 uM", "24 h"),
                ("s3", "BRD-A", "trt_cp", "PC3", "1000 nM", "6 h"),
                ("s4", "BRD-C", "trt_cp", "PC3", "1 uM", "24 h"),
                ("s5", "CTL", "ctl_vehicle", "MCF7", "0.1 %", "24 h"),
            ],
        ),
        l1000_file("GSE92742", "sig_metrics"): (
            metrics,
            [("s1", "0.5"), ("s2", "0.2"), ("s3", "0.3"), ("s4", "-666"), ("s5", "0.9")],
        ),
        l1000_file("GSE70138", "pert_info"): (info, [("BRD-B", "drug b", "trt_cp", "KEYB", "")]),
        l1000_file("GSE70138", "sig_info"): (
            sig,
            [("s6", "BRD-B", "trt_cp", "A375", "1 µM", "24 h")],
        ),
        l1000_file("GSE70138", "sig_metrics"): (metrics, [("s6", "0.7")]),
    }
    return {
        name: write_tsv(cache_dir / name, header, rows)
        for name, (header, rows) in tables.items()
    }


def write_prism_files(cache_dir: Path) -> dict[str, Path]:
    """drugA has a redo (MTS010) column; ACH-1 misses drugA's redo; R3 fails QC."""
    treatment = write_tsv(
        cache_dir / "secondary-screen-replicate-collapsed-treatment-info.csv",
        ("column_name", "broad_id", "name", "smiles", "dose", "screen_id"),
        [
            ("c1", "BRD-A", "drugA", "CCO", "1.0", "HTS002"),
            ("c2", "BRD-A", "drugA", "CCO", "10.0", "HTS002"),
            ("c3", "BRD-A", "drugA", "CCO", "1.0", "MTS010"),
            ("c4", "BRD-B", "drug_b", "CCN", "2.0", "HTS002"),
        ],
        sep=",",
    )
    lfc = write_tsv(
        cache_dir / "secondary-screen-replicate-collapsed-logfold-change.csv",
        ("row_name", "c1", "c2", "c3", "c4"),
        [("R1", "-1.0", "-2.0", "-1.5", "0.5"), ("R2", "", "-3.0", "", ""), ("R3", "", "", "", "")],
        sep=",",
    )
    cells = write_tsv(
        cache_dir / "secondary-screen-cell-line-info.csv",
        ("row_name", "depmap_id", "passed_str_profiling"),
        [("R1", "ACH-2", "TRUE"), ("R2", "ACH-1", "True"), ("R3", "ACH-3", "FALSE")],
        sep=",",
    )
    return {
        "secondary-screen-replicate-collapsed-treatment-info.csv": treatment,
        "secondary-screen-replicate-collapsed-logfold-change.csv": lfc,
        "secondary-screen-cell-line-info.csv": cells,
    }


def write_jump_files(cache_dir: Path) -> dict[str, Path]:
    """JCP1 matches drugA by full key, JCP2 by parent key, JCP3 matches nothing."""
    compound = write_tsv(
        cache_dir / "jump_compound.csv.gz",
        ("Metadata_JCP2022", "Metadata_InChIKey"),
        [("JCP1", "KEYA"), ("JCP2", "PARENTA"), ("JCP3", "KEYX")],
        sep=",",
    )
    profiles = cache_dir / "jump_profiles.parquet"
    pd.DataFrame(
        {
            "Metadata_JCP2022": ["JCP1", "JCP2", "JCP3", "JCP1"],
            "Metadata_Plate": ["p1", "p1", "p2", "p2"],
            "f1": [1.0, 3.0, 5.0, 2.0],
            "f2": [10.0, 30.0, 50.0, 20.0],
        }
    ).to_parquet(profiles, index=False)
    return {"jump_compound.csv.gz": compound, "jump_profiles.parquet": profiles}


# --- fake encoders ---------------------------------------------------------------------------


class FakeQwenTokenizer:
    """One token per character (its code point); records the call's keyword arguments."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def __call__(self, texts: Sequence[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        self.kwargs = dict(kwargs)
        limit = int(kwargs["max_length"])
        return {"input_ids": [[ord(c) for c in text][:limit] for text in texts]}


class FakeQwenModel:
    """hidden[t] = (id, 2 * id, t) for token t of the single input sequence."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.config = SimpleNamespace(hidden_size=3)
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise RuntimeError("simulated failure")
        assert input_ids.shape[0] == 1
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        ids = input_ids[0].to(torch.float32)
        pos = torch.arange(ids.shape[0], dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=torch.stack([ids, 2 * ids, pos], dim=-1)[None])


class FakeEsmTokenizer:
    """<cls> = 0, residue X = ord(X) - 64, <eos> = 99; records every max_length."""

    def __init__(self) -> None:
        self.max_lengths: list[int] = []

    def __call__(self, sequence: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.max_lengths.append(int(kwargs["max_length"]))
        ids = [0, *(ord(c) - 64 for c in sequence), 99]
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}


class FakeEsmModel:
    """hidden[t] = (id, t); t = 0 is <cls>, so residue r of a window sits at t = r + 1."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(hidden_size=2)

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        ids = input_ids[0].to(torch.float32)
        pos = torch.arange(ids.shape[0], dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=torch.stack([ids, pos], dim=-1)[None])


class FakeChemTokenizer:
    """ids = [1, ord(c)..., 2]; with return_tensors='pt' the batch is right-padded with 0."""

    def __call__(
        self,
        smiles: Sequence[str],
        *,
        return_tensors: str | None = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        ids = [[1, *(ord(c) for c in s), 2] for s in smiles]
        if truncation and max_length is not None:
            ids = [row[:max_length] for row in ids]
        if return_tensors is None:
            return {"input_ids": ids}
        width = max(len(row) for row in ids)
        input_ids = torch.zeros((len(ids), width), dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, row in enumerate(ids):
            input_ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


class FakeChemModel:
    """hidden[b, t] = (id, 1)."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(hidden_size=2)

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        ids = input_ids.to(torch.float32)
        return SimpleNamespace(last_hidden_state=torch.stack([ids, torch.ones_like(ids)], dim=-1))
