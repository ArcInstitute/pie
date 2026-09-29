"""Drug-dose perturbation text: key parsing, the PubChem client, normalization and rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pie.sources.text import _http, drugs


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """PUG-REST payloads from {cid: {endpoint: payload}}; `fail_first`: one 429 per URL."""

    def __init__(self, records: dict[int, dict[str, object]], fail_first: bool = False) -> None:
        self.records = records
        self.fail_first = fail_first
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        params: object = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> FakeResponse:
        assert timeout == drugs.REQUEST_TIMEOUT_SECONDS
        assert params is None
        assert stream is False
        self.calls.append(url)
        if self.fail_first and self.calls.count(url) == 1:
            return FakeResponse(429, {})
        cid = int(url.split("/compound/cid/")[1].split("/")[0])
        endpoint = next(name for name, part in drugs.ENDPOINT_MARKERS.items() if part in url)
        return FakeResponse(200, self.records[cid][endpoint])


def pubchem_record(
    cid: int, properties: dict[str, Any], synonyms: list[str] | None, descriptions: list[dict]
) -> dict[str, object]:
    synonym_row: dict[str, Any] = {"CID": cid}
    if synonyms is not None:
        synonym_row["Synonym"] = synonyms
    return {
        "properties": {"PropertyTable": {"Properties": [{"CID": cid, **properties}]}},
        "synonyms": {"InformationList": {"Information": [synonym_row]}},
        "descriptions": {
            "InformationList": {"Information": [{"CID": cid, **row} for row in descriptions]}
        },
    }


TESTAMIDE = pubchem_record(
    1001,
    {
        "Title": "Testamide",
        "MolecularFormula": "C10H12N2O",
        "MolecularWeight": "176.21",
        "SMILES": "CC(=O)NC1=CC=CC=C1N",
        "ConnectivitySMILES": "CC(=O)NC1=CC=CC=C1N",
        "InChI": "InChI=1S/C10H12N2O/test",
        "InChIKey": "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
        "XLogP": 1.2,
        "TPSA": 55.1,
        "Complexity": 180,
        "Charge": 0,
        "HBondDonorCount": 2,
        "HBondAcceptorCount": 2,
        "RotatableBondCount": 1,
    },
    [
        "Testamide",
        "1001-00-1",
        "TSTM-7",
        "CHEMBL123",
        "testamide",
        "UNII-ABC123",
        "http://example.org/x",
        "InChI=1S/abc",
        "AAAAAAAAAAAAAA-BBBBBBBBBB-C",
        "12345",
        "Testamid A",
        "tstm-7",
    ],
    [
        {"Title": "Testamide"},
        {
            "Description": "Synthetic test description one.",
            "DescriptionSourceName": "Example Source",
            "DescriptionURL": "https://example.org/1",
        },
        {
            "Description": "Synthetic ChEBI-style description.",
            "DescriptionSourceName": "ChEBI",
            "DescriptionURL": "https://example.org/2",
        },
    ],
)


def test_parse_drug_dose_keeps_the_raw_name() -> None:
    assert drugs.parse_drug_dose("Erdafitinib _0.05uM") == ("Erdafitinib ", 0.05)
    assert drugs.parse_drug_dose("DMSO_TF_0.0uM") == ("DMSO_TF", 0.0)
    assert drugs.parse_drug_dose("Testamide_5.0uM") == ("Testamide", 5.0)
    for bad in ("Testamide", "Testamide_5nM", "_5.0uM", "Testamide_xuM", "Testamide_nanuM"):
        with pytest.raises(ValueError):
            drugs.parse_drug_dose(bad)


def test_is_control_matches_the_label_or_the_parsed_name() -> None:
    assert drugs.is_control("DMSO_TF_0.0uM", "DMSO_TF")
    assert drugs.is_control("DMSO_TF_0.0uM", "DMSO_TF_0.0uM")
    assert not drugs.is_control("Testamide_5.0uM", "DMSO_TF")
    assert not drugs.is_control("not a key", "DMSO_TF")


def test_record_fetches_once_then_serves_the_cache(tmp_path: Path) -> None:
    session = FakeSession({1001: TESTAMIDE})
    client = drugs.PubChemClient(tmp_path, session, sleep=lambda _s: None)
    record = client.record(1001)
    assert set(record) == {"properties", "synonyms", "descriptions"}
    assert len(session.calls) == 3
    cached = json.loads((tmp_path / "pubchem" / "1001" / "synonyms.json").read_text())
    assert cached == TESTAMIDE["synonyms"]
    again = drugs.PubChemClient(tmp_path, FakeSession({}), offline=True).record(1001)
    assert again == record


def test_record_retries_429_and_spaces_requests(tmp_path: Path) -> None:
    waits: list[float] = []
    session = FakeSession({1001: TESTAMIDE}, fail_first=True)
    drugs.PubChemClient(tmp_path, session, sleep=waits.append).record(1001)
    assert len(session.calls) == 6
    assert waits.count(1) == 3


def test_record_offline_miss_and_bad_payloads(tmp_path: Path) -> None:
    with pytest.raises(_http.CacheMissError, match="offline"):
        drugs.PubChemClient(tmp_path, FakeSession({}), offline=True).record(1001)
    wrong = pubchem_record(9, {"Title": "X"}, [], [])
    with pytest.raises(ValueError, match="CID 1001"):
        client = drugs.PubChemClient(tmp_path, FakeSession({1001: wrong}), sleep=lambda _s: None)
        client.record(1001)
    assert not (tmp_path / "pubchem" / "1001" / "properties.json").exists()
    for bad in (0, -3, True, "1001"):
        with pytest.raises(ValueError):
            drugs.PubChemClient(tmp_path).record(bad)  # type: ignore[arg-type]


TESTAMIDE_TEXT = (
    "Compound Name: Testamide ;\n"
    "Synonyms: TSTM-7, Testamid A ;\n"
    "Molecular Formula: C10H12N2O ;\n"
    "Molecular Weight: 176.21 g/mol ;\n"
    "SMILES: CC(=O)NC1=CC=CC=C1N ;\n"
    "XLogP: 1.2 ;\n"
    "Topological Polar Surface Area: 55.1 square angstroms ;\n"
    "Complexity: 180 ;\n"
    "Formal Charge: 0 ;\n"
    "Hydrogen Bond Donors: 2 ;\n"
    "Hydrogen Bond Acceptors: 2 ;\n"
    "Rotatable Bonds: 1 ;\n"
    "Compound Description: Synthetic ChEBI-style description. ;\n"
    "Dose: 0.5 micromolar"
)
MIX_A = pubchem_record(
    2001,
    {"Title": "Mixamide A", "MolecularFormula": "C1H1"},
    None,
    [{"Description": "Alpha form.", "DescriptionSourceName": "DrugBank", "DescriptionURL": "u"}],
)
MIX_B = pubchem_record(
    2002,
    {"Title": "CID 2002", "MolecularFormula": "C1H1"},
    [],
    [{"Description": "Beta form.", "DescriptionSourceName": "Other", "DescriptionURL": "v"}],
)
MIXAMIDE_TEXT = (
    "Compound Name: Mixamide ;\n"
    "PubChem Compound: Mixamide A (isomer) ;\n"
    "Molecular Formula: C1H1 ;\n"
    "Molecular Formula: C1H1 ;\n"
    "Compound Description: Mixamide A: Alpha form. | Beta form. ;\n"
    "Dose: 5 micromolar"
)


def _synonyms(record: dict[str, Any]) -> list[str]:
    return record["synonyms"]["InformationList"]["Information"][0]["Synonym"]


def test_normalize_synonyms_filters_identifiers_and_caps_at_20() -> None:
    assert drugs.normalize_synonyms("Testamide", "Testamide", _synonyms(TESTAMIDE)) == [
        "TSTM-7",
        "Testamid A",
    ]
    many = [f"Name {i}" for i in range(30)]
    assert drugs.normalize_synonyms("X", "Y", many) == many[:20]


def test_render_keeps_only_the_primary_description_and_no_identifiers() -> None:
    component = drugs.build_component(1001, "primary", "Testamide", TESTAMIDE)
    assert component["pubchem_cid"] == "1001"
    assert len(component["descriptions"]) == 2
    text = drugs.render_drug("Testamide_0.5uM", {"components": [component]})
    assert text == TESTAMIDE_TEXT
    for absent in ("Synthetic test description one.", "example.org", "Example Source", "ChEBI:",
                   "1001", "InChI", "AAAAAAAAAAAAAA"):
        assert absent not in text
    component["synonyms"] = [f"S{i}" for i in range(20)]
    capped = drugs.render_drug("Testamide_0.5uM", {"components": [component]})
    assert "Synonyms: " + ", ".join(f"S{i}" for i in range(10)) + " ;\n" in capped
    assert drugs.render_drug("DMSO_TF_0.0uM", None) == drugs.CONTROL_TEXT


def test_describe_drug_perts_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = ((2001, "isomer"), (2002, "isomer"))
    monkeypatch.setitem(drugs.IDENTITY_OVERRIDES, "Mixamide", overrides)
    session = FakeSession({1001: TESTAMIDE, 2001: MIX_A, 2002: MIX_B})
    client = drugs.PubChemClient(tmp_path, session, sleep=lambda _s: None)
    metadata = pd.DataFrame(
        {"drug": ["Testamide", "Mixamide", "DMSO_TF"], "pubchem_cid": [1001.0, None, None]}
    )
    keys = ["Testamide_0.5uM", "Testamide _5.0uM", "Mixamide_5.0uM", "DMSO_TF_0.0uM"]
    texts = drugs.describe_drug_perts(keys, metadata, client, "DMSO_TF")
    assert list(texts) == keys
    assert texts["Testamide_0.5uM"] == TESTAMIDE_TEXT
    assert texts["Testamide _5.0uM"] == TESTAMIDE_TEXT.replace("0.5 micromolar", "5 micromolar")
    assert texts["Mixamide_5.0uM"] == MIXAMIDE_TEXT
    assert texts["DMSO_TF_0.0uM"] == drugs.CONTROL_TEXT
    assert len(session.calls) == 9


def test_describe_drug_perts_rejects_unresolvable_drugs(tmp_path: Path) -> None:
    client = drugs.PubChemClient(tmp_path, FakeSession({}), sleep=lambda _s: None)
    no_cid = pd.DataFrame({"drug": ["Testamide"], "pubchem_cid": [None]})
    with pytest.raises(ValueError, match="no PubChem CID"):
        drugs.describe_drug_perts(["Testamide_5.0uM"], no_cid, client, "DMSO_TF")
    with pytest.raises(ValueError, match="missing from the drug metadata"):
        drugs.describe_drug_perts(["Other_5.0uM"], no_cid, client, "DMSO_TF")
    duplicated = pd.DataFrame({"drug": ["Testamide", "Testamide "], "pubchem_cid": [1001, 1001]})
    with pytest.raises(ValueError, match="duplicate"):
        drugs.describe_drug_perts(["Testamide_5.0uM"], duplicated, client, "DMSO_TF")
    wrong = pd.DataFrame({"drug": ["Verteporfin"], "pubchem_cid": [42]})
    with pytest.raises(ValueError, match="override"):
        drugs.describe_drug_perts(["Verteporfin_5.0uM"], wrong, client, "DMSO_TF")
    with pytest.raises(ValueError, match="columns"):
        drugs.describe_drug_perts(["Testamide_5.0uM"], pd.DataFrame({"drug": ["x"]}), client, "C")


def test_build_component_has_a_docstring() -> None:
    assert drugs.build_component.__doc__
