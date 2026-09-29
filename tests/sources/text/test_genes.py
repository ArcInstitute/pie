from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from tests.sources.text.fakes import FakeResponse, FakeSession, NoNetwork

from pie.sources.text import _http, genes

HEADER = [
    "#tax_id", "GeneID", "Symbol", "LocusTag", "Synonyms", "dbXrefs", "chromosome",
    "map_location", "description", "type_of_gene", "Symbol_from_nomenclature_authority",
    "Full_name_from_nomenclature_authority", "Nomenclature_status", "Other_designations",
    "Modification_date", "Feature_type",
]
# Three rows of NCBI Homo_sapiens.gene_info (public domain), snapshot sha256 9422fe8d.
REAL_ROWS = [
    ["9606", "54707", "GPN2", "-", "ATPBD1B",
     "MIM:621544|HGNC:HGNC:25513|Ensembl:ENSG00000142751|AllianceGenome:HGNC:25513", "1",
     "1p36.11", "GPN-loop GTPase 2", "protein-coding", "GPN2", "GPN-loop GTPase 2", "O",
     "GPN-loop GTPase 2|ATP-binding domain 1 family member B", "20260705", "-"],
    ["9606", "57456", "KIAA1143", "-", "-",
     "HGNC:HGNC:29198|Ensembl:ENSG00000163807|AllianceGenome:HGNC:29198", "3", "3p21.31",
     "KIAA1143", "protein-coding", "KIAA1143", "KIAA1143", "O",
     "uncharacterized protein KIAA1143", "20260705", "-"],
    ["9606", "148223", "C19orf25", "-", "-",
     "HGNC:HGNC:26711|Ensembl:ENSG00000119559|AllianceGenome:HGNC:26711", "19", "19p13.3",
     "chromosome 19 open reading frame 25", "protein-coding", "C19orf25",
     "chromosome 19 open reading frame 25", "O", "UPF0449 protein C19orf25", "20260519", "-"],
]
# Synthetic rows (GeneIDs 9000000xx do not exist) for the resolution edge cases.
SYNTH_ROWS = [
    ["9606", "900000001", "SYNA", "-", "SHARED|ALIASA", "Ensembl:ENSG09000000001", "1", "1p1",
     "synthetic a", "pseudo", "SYNA", "synthetic gene a", "O", "-", "20260101", "-"],
    ["9606", "900000002", "SYNB", "-", "SHARED", "Ensembl:ENSG09000000001", "2", "2p1",
     "synthetic b", "protein-coding", "SYNB", "synthetic gene b", "O", "-", "20260101", "-"],
]
LINEAGE = (
    "Eukaryota; Metazoa; Chordata; Craniata; Vertebrata; Euteleostomi; Mammalia; Eutheria; "
    "Euarchontoglires; Primates; Haplorrhini; Catarrhini; Hominidae; Homo"
)
GPN2_SUMMARY = (
    "Predicted to enable GTPase activity. [provided by Alliance of Genome Resources, Jul 2025]"
)
# Canonical texts, quoted from the published replogle perturbation descriptions.
GPN2_TEXT = (
    "Gene Name: GPN2 ;\nHGNC Gene Symbol: GPN2 ;\nFull Name: GPN-loop GTPase 2 ;\n"
    "Synonyms: ATPBD1B ;\nGene Type: protein-coding ;\nOrganism: Homo sapiens ;\n"
    f"Lineage: {LINEAGE} ;\nMap Location: 1p36.11 ;\n"
    "Alt Designations: ATP-binding domain 1 family member B, GPN-loop GTPase 2 ;\n"
    f"Gene Summary: {GPN2_SUMMARY}"
)
KIAA1143_TEXT = (
    "Gene Name: KIAA1143 ;\nHGNC Gene Symbol: KIAA1143 ;\nFull Name: KIAA1143 ;\n"
    "Gene Type: protein-coding ;\nOrganism: Homo sapiens ;\n"
    f"Lineage: {LINEAGE} ;\nMap Location: 3p21.31 ;\n"
    "Alt Designations: uncharacterized protein KIAA1143"
)
C19ORF25_TEXT = (
    "Gene Name: C19orf25 ;\nHGNC Gene Symbol: C19orf25 ;\n"
    "Full Name: chromosome 19 open reading frame 25 ;\nGene Type: protein-coding ;\n"
    f"Organism: Homo sapiens ;\nLineage: {LINEAGE} ;\nMap Location: 19p13.3 ;\n"
    "Alt Designations: UPF0449 protein C19orf25"
)


def write_gene_info(path: Path, rows: list[list[str]] | None = None) -> Path:
    rows = REAL_ROWS + SYNTH_ROWS if rows is None else rows
    lines = ["\t".join(HEADER)] + ["\t".join(row) for row in rows]
    path.write_bytes(gzip.compress(("\n".join(lines) + "\n").encode("utf-8"), mtime=0))
    return path


@pytest.fixture
def index(tmp_path: Path) -> genes.GeneInfoIndex:
    return genes.GeneInfoIndex.load(write_gene_info(tmp_path / "Homo_sapiens.gene_info.gz"))


def _record(index: genes.GeneInfoIndex, gene_id: str, summary: str = "") -> dict:
    fields = index.fields(gene_id)
    fields.update(organism="Homo sapiens", lineage=LINEAGE, gene_summary=summary)
    return fields


@pytest.mark.parametrize(
    ("key", "gene_id", "summary", "text"),
    [
        ("GPN2", "54707", GPN2_SUMMARY, GPN2_TEXT),
        ("KIAA1143", "57456", "", KIAA1143_TEXT),
        ("C19orf25", "148223", "", C19ORF25_TEXT),
    ],
)
def test_render_gene_matches_canonical(index, key, gene_id, summary, text) -> None:
    assert genes.render_gene(key, _record(index, gene_id, summary)) == text


def test_render_gene_minimal_fallback() -> None:
    assert genes.render_gene("ABALON", None) == "Gene Name: ABALON"


def test_fields_extracts_gene_info_columns(index) -> None:
    assert index.fields("900000001") == {
        "hgnc_gene_symbol": "SYNA",
        "full_name": "synthetic gene a",
        "synonyms": ["ALIASA", "SHARED"],
        "gene_type": "pseudogene",
        "map_location": "1p1",
        "alt_designations": [],
        "ncbi_gene_id": "900000001",
        "ensembl_gene_id": "ENSG09000000001",
    }
    assert len(index.sha256) == 64


@pytest.mark.parametrize(
    ("key", "ensembl", "expected"),
    [
        ("GPN2", "ENSG00000142751", ("54707", "ensembl_id")),
        ("gpn2", "", ("54707", "official_symbol")),
        ("ATPBD1B", "", ("54707", "synonym")),
        ("SYNA", "ENSG09000000001", ("900000001", "ensembl_id")),  # shared id, symbol breaks tie
        ("SHARED", "", (None, "ambiguous")),
        ("NOPE", "", (None, "unresolved")),
    ],
)
def test_resolve_with_method(index, key, ensembl, expected) -> None:
    assert index.resolve_with_method(key, ensembl) == expected


def test_resolve_returns_row_or_none(index) -> None:
    assert index.resolve("KIAA1143")["geneid"] == "57456"
    assert index.resolve("SHARED") is None


_HEADER = (
    "#tax_id\tGeneID\tSymbol\tLocusTag\tSynonyms\tdbXrefs\tchromosome\tmap_location\t"
    "description\ttype_of_gene\tSymbol_from_nomenclature_authority\t"
    "Full_name_from_nomenclature_authority\tNomenclature_status\tOther_designations\t"
    "Modification_date\tFeature_type\n"
)
_HUMAN_TP53 = ["9606", "7157", "TP53", "-", "LFS1|P53", "HGNC:HGNC:11998|Ensembl:ENSG00000141510",
               "17", "17p13.1", "tumor protein p53", "protein-coding", "TP53", "tumor protein p53",
               "O", "cellular tumor antigen p53", "20240101", "-"]  # fmt: skip
_CHIMP_TP53 = ["9598", "449628", "TP53", "-", "-", "Ensembl:ENSPTRG00000008842", "17", "-",
               "tumor protein p53", "protein-coding", "-", "-", "-", "-",
               "20240101", "-"]  # fmt: skip


def _gene_info(path: Path, *rows: list[str]) -> Path:
    with gzip.open(path, "wt") as handle:
        handle.write(_HEADER)
        for row in rows:
            handle.write("\t".join(row) + "\n")
    return path


def test_gene_info_index_keeps_only_human_rows(tmp_path: Path) -> None:
    every = genes.GeneInfoIndex.load(_gene_info(tmp_path / "all.gz", _HUMAN_TP53, _CHIMP_TP53))
    human = genes.GeneInfoIndex.load(_gene_info(tmp_path / "hs.gz", _HUMAN_TP53))
    assert every.rows == human.rows
    assert every.resolve_with_method("TP53") == ("7157", "official_symbol")  # not "ambiguous"
    assert every.resolve_with_method("TP53", "ENSG00000141510") == ("7157", "ensembl_id")


def test_the_homo_sapiens_file_gives_the_same_index(tmp_path: Path) -> None:
    # Rows of a Homo_sapiens file are all 9606, so the filter keeps every one of them.
    path = _gene_info(tmp_path / "hs.gz", _HUMAN_TP53)
    assert set(genes._load_gene_info(path)) == {"7157"}


ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
GENE_INFO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
)
SYN_LINEAGE = "Synthetic; Lineage"
GZ = gzip.compress(b"#tax_id\tGeneID\n", mtime=0)
SYNTH_EXTRA = [
    ["9606", "900000003", "SYNC", "-", "OLDC", "Ensembl:ENSG09000000003", "3", "3p1",
     "synthetic c", "protein-coding", "SYNC", "synthetic gene c", "O", "synthetic c protein",
     "20260101", "-"],
    ["9606", "900000004", "SYND", "-", "-", "Ensembl:ENSG09000000004", "4", "4p1",
     "synthetic d", "protein-coding", "SYND", "synthetic gene d", "O", "-", "20260101", "-"],
]


def _efetch_xml(organism: str = "Homo sapiens", lineage: str = SYN_LINEAGE) -> bytes:
    return (
        "<Entrezgene-Set><Entrezgene><Entrezgene_source><BioSource><BioSource_org><Org-ref>"
        f"<Org-ref_taxname>{organism}</Org-ref_taxname><Org-ref_orgname><OrgName>"
        f"<OrgName_lineage>{lineage}</OrgName_lineage></OrgName></Org-ref_orgname></Org-ref>"
        "</BioSource_org></BioSource></Entrezgene_source></Entrezgene></Entrezgene-Set>"
    ).encode()


def _esummary(summaries: dict[str, str]) -> FakeResponse:
    result: dict[str, object] = {"uids": list(summaries)}
    result.update({gid: {"uid": gid, "summary": text} for gid, text in summaries.items()})
    return FakeResponse(200, {"result": result})


def _seed_cache(
    cache: Path, summaries: dict[str, str], lineage_ids: Sequence[str] | None = None
) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    ids = list(summaries) if lineage_ids is None else list(lineage_ids)
    lineage = {gid: {"organism": "Homo sapiens", "lineage": SYN_LINEAGE} for gid in ids}
    (cache / "esummary_cache.json").write_text(json.dumps(summaries), encoding="utf-8")
    (cache / "lineage_cache.json").write_text(json.dumps(lineage), encoding="utf-8")
    return cache


def _offline(cache: Path) -> genes.EsummaryClient:
    return genes.EsummaryClient(cache, None, NoNetwork(), offline=True)


def _summary_record(summary: str) -> dict[str, object]:
    return {"organism": "Homo sapiens", "lineage": SYN_LINEAGE, "gene_summary": summary}


def _fresh(index: genes.GeneInfoIndex, key: str, gene_id: str, summary: str) -> str:
    fields = index.fields(gene_id)
    fields.update(_summary_record(summary))
    return genes.render_gene(key, fields)


@pytest.fixture
def synth_index(tmp_path: Path) -> genes.GeneInfoIndex:
    path = write_gene_info(tmp_path / "synth.gene_info.gz", SYNTH_ROWS + SYNTH_EXTRA)
    return genes.GeneInfoIndex.load(path)


def test_fetch_gene_info_downloads_once_under_its_digest(tmp_path: Path) -> None:
    assert genes.GENE_INFO_URL == GENE_INFO_URL
    session = FakeSession({GENE_INFO_URL: [FakeResponse(200, content=GZ)]})
    path = genes.fetch_gene_info(tmp_path, session)
    digest = hashlib.sha256(GZ).hexdigest()
    assert path == tmp_path / "gene_info" / digest[:12] / "Homo_sapiens.gene_info.gz"
    assert path.read_bytes() == GZ
    assert [p.name for p in tmp_path.iterdir()] == ["gene_info"]  # no temp file left
    assert genes.fetch_gene_info(tmp_path, NoNetwork()) == path
    assert genes.fetch_gene_info(tmp_path, NoNetwork(), offline=True, digest=digest) == path
    assert len(session.calls) == 1


def test_fetch_gene_info_offline_miss_mismatch_and_ambiguity(tmp_path: Path) -> None:
    with pytest.raises(_http.CacheMissError, match="NCBI gene_info"):
        genes.fetch_gene_info(tmp_path / "empty", NoNetwork(), offline=True)
    session = FakeSession({GENE_INFO_URL: [FakeResponse(200, content=GZ)]})
    with pytest.raises(ValueError, match="does not match the expected digest"):
        genes.fetch_gene_info(tmp_path / "m", session, digest="0" * 64)
    assert not [p for p in (tmp_path / "m").rglob("*") if p.is_file()]
    for digest in ("a" * 64, "b" * 64):
        target = tmp_path / "two" / "gene_info" / digest[:12] / genes.GENE_INFO_NAME
        target.parent.mkdir(parents=True)
        target.write_bytes(GZ)
    with pytest.raises(ValueError, match="2 cached gene_info snapshots"):
        genes.fetch_gene_info(tmp_path / "two", NoNetwork(), offline=True)
    chosen = genes.fetch_gene_info(tmp_path / "two", NoNetwork(), offline=True, digest="b" * 64)
    assert chosen.parent.name == "b" * 12


def test_esummary_fetches_only_missing_ids_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    cache = _seed_cache(tmp_path / "c", {"900000001": "cached a"}, lineage_ids=())
    session = FakeSession({
        ESUMMARY_URL: [_esummary({"900000002": "fresh b"})],
        EFETCH_URL: [FakeResponse(200, content=_efetch_xml())],
    })
    sleeps: list[float] = []
    client = genes.EsummaryClient(cache, None, session, sleep=sleeps.append)
    expected = {"900000001": _summary_record("cached a"), "900000002": _summary_record("fresh b")}
    assert client.summaries(["900000001", "900000002"]) == expected
    assert session.calls == [
        (ESUMMARY_URL, {"db": "gene", "id": "900000002", "retmode": "json", "tool": "pie-sources"}),
        (EFETCH_URL, {"db": "gene", "id": "900000001", "retmode": "xml", "tool": "pie-sources"}),
    ]
    assert sleeps == [genes.NO_KEY_PACING_SECONDS] * 2
    summaries = json.loads(client.summary_path.read_text(encoding="utf-8"))
    assert summaries == {"900000001": "cached a", "900000002": "fresh b"}
    lineage = json.loads(client.lineage_path.read_text(encoding="utf-8"))
    assert lineage == {"900000001": {"lineage": SYN_LINEAGE, "organism": "Homo sapiens"}}
    assert _offline(cache).summaries(["900000001", "900000002"]) == expected
    prov = client.provenance()
    assert list(prov) == ["ncbi_esummary", "ncbi_lineage"]
    assert prov["ncbi_esummary"]["url"] == genes.EUTILS
    assert len(prov["ncbi_esummary"]["sha256"]) == 64


def test_esummary_batches_and_sends_key_and_email(tmp_path: Path) -> None:
    session = FakeSession({
        ESUMMARY_URL: [_esummary({"900000002": ""}), _esummary({"900000001": "a"})],
        EFETCH_URL: [FakeResponse(200, content=_efetch_xml())],
    })
    sleeps: list[float] = []
    client = genes.EsummaryClient(
        tmp_path, "k", session, email="lab@example.org", batch_size=1, sleep=sleeps.append
    )
    got = client.summaries(["900000002", "900000001"])
    assert list(got) == ["900000002", "900000001"]
    assert [got[g]["gene_summary"] for g in got] == ["", "a"]
    assert [params["id"] for _, params in session.calls] == ["900000002", "900000001", "900000001"]
    for _, params in session.calls:
        assert (params["api_key"], params["email"], params["tool"]) == (
            "k", "lab@example.org", "pie-sources",
        )
    assert sleeps == []


def test_esummary_offline_misses_and_non_human_lineage(tmp_path: Path) -> None:
    cache = _seed_cache(tmp_path / "c", {"900000001": "a"})
    with pytest.raises(_http.CacheMissError, match="gene ids 900000002"):
        _offline(cache).summaries(["900000001", "900000002"])
    other = _seed_cache(tmp_path / "d", {"900000002": "b"}, lineage_ids=["900000001"])
    with pytest.raises(_http.CacheMissError, match="lineage cache miss for gene 900000002"):
        _offline(other).summaries(["900000002"])
    mouse = _seed_cache(tmp_path / "m", {"900000001": "a"}, lineage_ids=())
    session = FakeSession({EFETCH_URL: [FakeResponse(200, content=_efetch_xml("Mus musculus"))]})
    with pytest.raises(ValueError, match="non-human organism 'Mus musculus'"):
        genes.EsummaryClient(mouse, "k", session).summaries(["900000001"])


def test_describe_genetic_perts_resolves_every_target(
    synth_index: genes.GeneInfoIndex, tmp_path: Path
) -> None:
    cache = _seed_cache(tmp_path / "c", {"900000001": "sum a", "900000003": "sum c"})
    texts = genes.describe_genetic_perts(
        ["SYNC", "ctrl", "SYNA", "SYNC"],
        synth_index,
        _offline(cache),
        {"SYNA": "ENSG09000000001", "ctrl": ""},
        "ctrl",
    )
    assert texts == {
        "ctrl": genes.CONTROL_TEXT,
        "SYNA": _fresh(synth_index, "SYNA", "900000001", "sum a"),
        "SYNC": _fresh(synth_index, "SYNC", "900000003", "sum c"),
    }
    assert list(texts) == ["ctrl", "SYNA", "SYNC"]
    alone = genes.describe_genetic_perts(["SYNC"], synth_index, _offline(cache), {}, "ctrl")
    assert alone == {"SYNC": texts["SYNC"]}


def test_describe_genetic_perts_rejects_unresolved(
    synth_index: genes.GeneInfoIndex, tmp_path: Path
) -> None:
    expected = r"unresolved production records: NOPE \(unresolved\), SHARED \(ambiguous\)"
    with pytest.raises(ValueError, match=expected):
        genes.describe_genetic_perts(
            ["SHARED", "NOPE", "ctrl"], synth_index, _offline(tmp_path), {}, "ctrl"
        )


def test_describe_gene_queries_tiers(synth_index: genes.GeneInfoIndex, tmp_path: Path) -> None:
    cache = _seed_cache(tmp_path / "c", {"900000003": "sum c", "900000004": "sum d"})
    pert_output = {"SYNA": "pert a", "SYNB": "pert b", "SYNC": "pert c", "ctrl": "control"}
    query = ["SYND", "SYNA", "SYNB", "SYNC", "ENSG09000000004", "OLDC_ENSG09000000003"]
    texts = genes.describe_gene_queries(
        [*query, "SHARED", "NOPE", "SYNA"],
        {"SYNA": "prior a"},
        pert_output,
        ["SYNB", "ctrl"],
        synth_index,
        _offline(cache),
    )
    assert texts == {
        "ENSG09000000004": _fresh(synth_index, "ENSG09000000004", "900000004", "sum d"),
        "NOPE": "Gene Name: NOPE",
        "OLDC_ENSG09000000003": _fresh(synth_index, "OLDC_ENSG09000000003", "900000003", "sum c"),
        "SHARED": "Gene Name: SHARED",
        "SYNA": "prior a",
        "SYNB": "pert b",
        "SYNC": _fresh(synth_index, "SYNC", "900000003", "sum c"),
        "SYND": _fresh(synth_index, "SYND", "900000004", "sum d"),
    }
    assert list(texts) == sorted(texts, key=genes.sort_key)


def test_describe_gene_queries_copies_without_network(
    synth_index: genes.GeneInfoIndex, tmp_path: Path
) -> None:
    client = genes.EsummaryClient(tmp_path / "empty", None, NoNetwork(), offline=True)
    texts = genes.describe_gene_queries(
        ["SYNB"], None, {"SYNB": "pert b"}, ["SYNB"], synth_index, client
    )
    assert texts == {"SYNB": "pert b"}
    assert not (tmp_path / "empty").exists()
