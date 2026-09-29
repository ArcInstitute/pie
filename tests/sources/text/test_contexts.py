from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.sources.text.fakes import FIXTURES, FakeResponse, FakeSession, NoNetwork

from pie.sources.text import _http, contexts

CELLO: Path = FIXTURES / "cellosaurus"

# Canonical embedding texts for synthetic (non-real) Cellosaurus-format fixture records.
SC_ALPHA1_TEXT = (
    "Context Name: stemline_a ;\nRecommended Name: SC-Alpha1 ;\n"
    "Synonyms: Alpha-1, Alpha1 ES, SCA1, Stem Alpha 1 ;\n"
    "Cell Line Category: Embryonic stem cell ;\nOrganism: Homo sapiens ;\nSex: Male ;\n"
    "Age: Blastocyst stage ;\nDerived From Site: Blastocyst ;\nCell Type: Embryonic stem cell"
)
TX_KIDNEY9_TEXT = (
    "Context Name: translineB ;\nRecommended Name: TX-Kidney9 ;\n"
    "Synonyms: Kidney-9, Kidney9, Transformed Kidney 9, TXK9 ;\n"
    "Cell Line Category: Transformed cell line ;\nOrganism: Homo sapiens ;\nSex: Female ;\n"
    "Age: Fetus ;\nDerived From Site: Fetal kidney ;\nParent Cell Line: TX-Kidney (parent) ;\n"
    "Genetic Integration: Synthetic viral early element V1 via Transfection, "
    "Transposon element Tn-Fictional neo via Transfection"
)
PANC7_BASE_TEXT = (
    "Recommended Name: PA-Panc7 ;\n"
    "Synonyms: Fictional Panc Line 7, PA7, Panc-7, PANC7 ;\n"
    "Cell Line Category: Cancer cell line ;\nOrganism: Homo sapiens ;\nSex: Female ;\n"
    "Age: 61 years ;\nPopulation: Fictional Population X ;\n"
    "Disease: Fictional pancreatic neoplasm ;\nDerived From Site: Pancreas ;\n"
    "Sequence Variation: GENEA: Gene deletion (Homozygous), "
    "GENEB: Gene deletion (Homozygous), GENEC: Mutation, "
    "GENED: Mutation, p.Xxx123Tyr (c.367A>T) (Heterozygous), "
    "GENEE: Mutation, p.Yyy45Cys (c.135T>G) (Homozygous)"
)


def _record(accession: str) -> dict:
    return json.loads((CELLO / "56.0" / f"{accession}.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("key", "accession", "text"),
    [
        ("stemline_a", "CVCL_Y001", SC_ALPHA1_TEXT),
        ("translineB", "CVCL_Y002", TX_KIDNEY9_TEXT),
    ],
)
def test_build_context_entry_renders_canonical_text(key: str, accession: str, text: str) -> None:
    entry = contexts.build_context_entry(key, _record(accession))
    assert tuple(entry) == contexts.ENTRY_FIELDS
    assert entry["embedding_text"] == text
    assert entry["cellosaurus_id"] == accession
    assert entry["organism"] == "Homo sapiens"
    assert entry["ncbi_taxonomy_id"] == "9606"


def test_build_context_entry_structured_fields_match_canonical() -> None:
    entry = contexts.build_context_entry("translineB", _record("CVCL_Y002"))
    assert entry["parent_cell_lines"] == [
        {"name": "TX-Kidney (parent)", "cellosaurus_id": "CVCL_Y005"}
    ]
    assert entry["genetic_integrations"] == [
        {
            "method": "Transfection",
            "label": "Synthetic viral early element V1",
            "database": "UniProtKB",
            "accession": "Q00001",
        },
        {
            "method": "Transfection",
            "label": "Transposon element Tn-Fictional neo",
            "database": "UniProtKB",
            "accession": "Q00002",
        },
    ]
    assert entry["cross_references"] == {
        "atcc": ["CRL-9999"], "depmap": [], "cell_model_passport": []
    }
    stem = contexts.build_context_entry("stemline_a", _record("CVCL_Y001"))
    assert stem["cell_type"] == {
        "name": "Embryonic stem cell", "database": "CL", "accession": "CL_0002322"
    }


def test_sequence_variations_are_sorted_and_blank_fields_dropped() -> None:
    entry = contexts.build_context_entry("cancerline_c", _record("CVCL_Y003"))
    assert entry["embedding_text"] == "Context Name: cancerline_c ;\n" + PANC7_BASE_TEXT
    assert entry["sequence_variations"][2] == {
        "genes": ["GENEC"],
        "variation_type": "Mutation",
        "mutation_description": "",
        "zygosity": "",
        "note": "",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("61Y", "61 years"),
        ("1Y2M", "1 year, 2 months"),
        ("3D", "3 days"),
        ("Age unspecified", ""),
        ("Adult", "Adult"),
        (None, ""),
    ],
)
def test_humanize_age(raw: object, expected: str) -> None:
    assert contexts._humanize_age(raw) == expected


def test_humanize_age_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="unrecognized Cellosaurus age"):
        contexts._humanize_age("about 40")


def test_non_human_record_is_rejected() -> None:
    payload = _record("CVCL_Y001")
    payload["Cellosaurus"]["cell-line-list"][0]["species-list"] = [
        {"accession": "10090", "database": "NCBI_TaxID", "label": "Mus musculus (Mouse)"}
    ]
    with pytest.raises(ValueError, match="exactly one human species"):
        contexts.build_context_entry("x", payload)


def test_accession_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match primary accession 'CVCL_Y001'"):
        contexts._verify_requested_accession(_record("CVCL_Y001"), "CVCL_0004")


def test_sort_key_is_casefold_then_raw() -> None:
    assert sorted(["b", "B", "a", "ARC_H1"], key=contexts.sort_key) == ["a", "ARC_H1", "B", "b"]


REL_URL = contexts.RELEASE_URL


def _rec_url(accession: str) -> str:
    return contexts.CELL_LINE_URL.format(accession=accession)


def _seeded_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cello"
    shutil.copytree(CELLO, cache)
    return cache


def _release_payload(version: str) -> dict:
    payload = json.loads((CELLO / "release-info.json").read_text(encoding="utf-8"))
    payload["Cellosaurus"]["header"]["release"]["version"] = version
    return payload


def _client(cache: Path, release: str = "56.0") -> contexts.CellosaurusClient:
    return contexts.CellosaurusClient(cache, release, offline=True, session=NoNetwork())


def _online(cache: Path, record_accession: str, record: dict) -> tuple:
    session = FakeSession({
        REL_URL: [FakeResponse(200, _release_payload("56.0"))],
        _rec_url(record_accession): [FakeResponse(200, record)],
    })
    return contexts.CellosaurusClient(cache, "56.0", offline=False, session=session), session


def test_offline_client_reads_cached_records_without_network(tmp_path: Path) -> None:
    record = _client(_seeded_cache(tmp_path)).get("CVCL_Y001")
    assert contexts.build_context_entry("stemline_a", record)["embedding_text"] == SC_ALPHA1_TEXT


def test_offline_cache_miss_names_the_record(tmp_path: Path) -> None:
    with pytest.raises(_http.CacheMissError, match="CVCL_0004"):
        _client(_seeded_cache(tmp_path)).get("CVCL_0004")


def test_offline_requires_cached_release_info(tmp_path: Path) -> None:
    cache = _seeded_cache(tmp_path)
    (cache / "release-info.json").unlink()
    with pytest.raises(_http.CacheMissError, match=r"release-info\.json"):
        _client(cache).get("CVCL_Y001")


def test_offline_cached_release_must_equal_pin(tmp_path: Path) -> None:
    with pytest.raises(contexts.CellosaurusReleaseError, match=r"56\.0"):
        _client(_seeded_cache(tmp_path), "55.0").get("CVCL_Y001")


def test_online_fetch_populates_cache_once(tmp_path: Path) -> None:
    record = json.loads((CELLO / "56.0" / "CVCL_Y003.json").read_text(encoding="utf-8"))
    cache = tmp_path / "fresh"
    client, session = _online(cache, "CVCL_Y003", record)
    assert client.get("CVCL_Y003") == record
    assert client.get("CVCL_Y003") == record
    assert [url for url, _ in session.calls] == [REL_URL, _rec_url("CVCL_Y003")]
    assert json.loads(client.record_path("CVCL_Y003").read_text(encoding="utf-8")) == record
    assert _client(cache).get("CVCL_Y003") == record


def test_online_release_drift_is_refused_before_any_record_fetch(tmp_path: Path) -> None:
    session = FakeSession({REL_URL: [FakeResponse(200, _release_payload("57.0"))]})
    client = contexts.CellosaurusClient(tmp_path / "c", "56.0", offline=False, session=session)
    match = r"57\.0 differs from the pinned 56\.0"
    with pytest.raises(contexts.CellosaurusReleaseError, match=match):
        client.get("CVCL_Y003")
    assert [url for url, _ in session.calls] == [REL_URL]


def test_fetched_record_must_carry_the_requested_accession(tmp_path: Path) -> None:
    wrong = json.loads((CELLO / "56.0" / "CVCL_Y001.json").read_text(encoding="utf-8"))
    client, _ = _online(tmp_path / "c", "CVCL_Y003", wrong)
    with pytest.raises(ValueError, match="does not match primary accession"):
        client.get("CVCL_Y003")
    assert not client.record_path("CVCL_Y003").exists()


def test_invalid_accession_is_rejected(tmp_path: Path) -> None:
    client = contexts.CellosaurusClient(tmp_path, "56.0", offline=True, session=NoNetwork())
    with pytest.raises(ValueError, match="invalid Cellosaurus accession"):
        client.get("k562")


def test_provenance_hashes_every_used_record(tmp_path: Path) -> None:
    cache = _seeded_cache(tmp_path)
    client = contexts.CellosaurusClient(cache, "56.0", offline=True, session=NoNetwork())
    client.get("CVCL_Y001")
    prov = client.provenance(["CVCL_Y001", "CVCL_Y001"])
    assert prov["release"] == "56.0"
    assert prov["url"] == "https://api.cellosaurus.org"
    assert list(prov["records"]) == ["CVCL_Y001"]
    assert len(prov["records"]["CVCL_Y001"]) == 64


IFNG = {
    "name": "Interferon gamma",
    "abbreviation": "IFN-gamma",
    "family": "type II interferon cytokine",
    "receptors": ["IFNGR1", "IFNGR2"],
    "signaling": (
        "JAK1/JAK2 signaling through STAT1 at gamma-activated sequence (GAS) promoter elements"
    ),
    "description": (
        "Pro-inflammatory type II interferon produced mainly by T and NK cells; activates "
        "macrophages and up-regulates MHC class I and II antigen presentation."
    ),
}
# Synthetic (non-real) base render for the fictional CVCL_0186 fixture record, sans context name.
BXPC3_BASE_TEXT = (
    "Recommended Name: PA-Pancreas-X9 ;\n"
    "Synonyms: Fictional Panc Stim Line, PPX9 ;\n"
    "Cell Line Category: Cancer cell line ;\nOrganism: Homo sapiens ;\nSex: Female ;\n"
    "Age: 50 years ;\nDisease: Fictional ductal neoplasm ;\nDerived From Site: Pancreas"
)
BXPC3_IFNG_TEXT = (
    "Context Name: bxpc3_ifng ;\n" + BXPC3_BASE_TEXT + " ;\n"
    "Stimulation: Interferon gamma (IFN-gamma) ;\n"
    "Stimulation Family: type II interferon cytokine ;\n"
    "Stimulation Receptors: IFNGR1, IFNGR2 ;\n"
    "Stimulation Signaling: JAK1/JAK2 signaling through STAT1 at gamma-activated sequence (GAS) "
    "promoter elements ;\n"
    "Stimulation Description: Pro-inflammatory type II interferon produced mainly by T and NK "
    "cells; activates macrophages and up-regulates MHC class I and II antigen presentation."
)
# Synthetic (non-real) render for the fictional CVCL_9771 fixture record (context ARC_H1).
ARC_H1_TEXT = (
    "Context Name: ARC_H1 ;\nRecommended Name: ARC-Kidney-H1 ;\nSynonyms: ARC H1 Line ;\n"
    "Cell Line Category: Immortalized cell line ;\nOrganism: Homo sapiens ;\nSex: Female ;\n"
    "Age: Fetus ;\nDerived From Site: Kidney ;\nCell Type: Epithelial cell"
)


def _write_yaml(path: Path, body: dict) -> Path:
    path.write_text(json.dumps(body), encoding="utf-8")  # JSON is valid YAML
    return path


def _context_dir(tmp_path: Path) -> Path:
    root = tmp_path / "contexts"
    root.mkdir()
    _write_yaml(root / "arc_vcc_25.yaml", {"contexts": {
        "ARC_H1": {"cellosaurus": "CVCL_9771"}, "ARC_H1_VAL": {"cellosaurus": "CVCL_9771"},
    }})
    _write_yaml(root / "jiang.yaml", {
        "contexts": {"bxpc3_ifng": {"cellosaurus": "CVCL_0186", "stimulation": "ifng"}},
        "stimulations": {"ifng": IFNG},
    })
    return root


def test_load_context_file_parses_entries_and_stimulations(tmp_path: Path) -> None:
    parsed = contexts.load_context_file(_context_dir(tmp_path) / "jiang.yaml")
    assert parsed.contexts["bxpc3_ifng"].cellosaurus == "CVCL_0186"
    assert parsed.contexts["bxpc3_ifng"].stimulation == "ifng"
    assert parsed.stimulations["ifng"].model_dump() == IFNG
    arc = contexts.load_context_file(tmp_path / "contexts" / "arc_vcc_25.yaml")
    assert list(arc.contexts) == ["ARC_H1", "ARC_H1_VAL"]
    assert arc.stimulations == {}


@pytest.mark.parametrize(
    "body",
    [
        {"contexts": {"a": {"cellosaurus": "CVCL_0004", "stimulation": "ifng"}}},
        {"contexts": {"a": {"cellosaurus": "k562"}}},
        {"contexts": {"a": {"cellosaurus": "CVCL_0004", "dose": 1}}},
        {"contexts": {}, "extra": 1},
    ],
)
def test_load_context_file_rejects_bad_files(tmp_path: Path, body: dict) -> None:
    with pytest.raises(ValueError):
        contexts.load_context_file(_write_yaml(tmp_path / "bad.yaml", body))


def test_render_context_appends_the_stimulation_segments() -> None:
    stimulation = contexts.StimulationEntry(**IFNG)
    assert contexts.render_context("bxpc3_ifng", _record("CVCL_0186"), stimulation) == (
        BXPC3_IFNG_TEXT
    )
    assert contexts.render_context("ARC_H1", _record("CVCL_9771"), None) == ARC_H1_TEXT


def test_render_context_omits_parentheses_for_empty_abbreviation() -> None:
    stimulation = contexts.StimulationEntry(**{**IFNG, "abbreviation": ""})
    text = contexts.render_context("bxpc3_ifng", _record("CVCL_0186"), stimulation)
    assert " ;\nStimulation: Interferon gamma ;\n" in text
    assert "()" not in text


def test_describe_contexts_orders_by_dataset_then_sort_key(tmp_path: Path) -> None:
    datasets = [
        SimpleNamespace(dataset="arc_vcc_25", contexts=["ARC_H1_VAL", "ARC_H1"]),
        SimpleNamespace(dataset="jiang", contexts=["bxpc3_ifng"]),
    ]
    client = _client(_seeded_cache(tmp_path))
    texts = contexts.describe_contexts(datasets, _context_dir(tmp_path), client)
    assert list(texts) == ["ARC_H1", "ARC_H1_VAL", "bxpc3_ifng"]
    assert texts["ARC_H1"] == ARC_H1_TEXT
    assert texts["ARC_H1_VAL"] == ARC_H1_TEXT.replace("ARC_H1 ;", "ARC_H1_VAL ;", 1)
    assert texts["bxpc3_ifng"] == BXPC3_IFNG_TEXT
    assert list(client.provenance()["records"]) == ["CVCL_0186", "CVCL_9771"]


def test_describe_contexts_requires_every_dataset_context(tmp_path: Path) -> None:
    datasets = [SimpleNamespace(dataset="arc_vcc_25", contexts=["ARC_H1", "ARC_H2"])]
    with pytest.raises(KeyError, match="ARC_H2"):
        client = _client(_seeded_cache(tmp_path))
        contexts.describe_contexts(datasets, _context_dir(tmp_path), client)


def test_describe_contexts_requires_a_context_file(tmp_path: Path) -> None:
    datasets = [SimpleNamespace(dataset="orion", contexts=["hek293t"])]
    with pytest.raises(FileNotFoundError, match=r"orion\.yaml"):
        client = _client(_seeded_cache(tmp_path))
        contexts.describe_contexts(datasets, _context_dir(tmp_path), client)
