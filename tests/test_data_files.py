"""Tracked data files: the context maps."""

from __future__ import annotations

from typing import Any

import pytest

from pie.sources.text.contexts import load_context_file
from tests.conftest import REPO_ROOT

SOURCES = REPO_ROOT / "data" / "sources"
DATASETS = ("replogle", "tahoe", "jiang", "arc_vcc_25", "orion")

REPLOGLE_CONTEXTS = {
    "hepg2": "CVCL_0027", "jurkat": "CVCL_0367", "k562": "CVCL_0004", "rpe1": "CVCL_4388",
}
ORION_CONTEXTS = {"hct116": "CVCL_0291", "hek293t": "CVCL_0063"}
ARC_VCC_25_CONTEXTS = {"ARC_H1": "CVCL_9771", "ARC_H1_VAL": "CVCL_9771"}
JIANG_LINES = {
    "a549": "CVCL_0023",
    "bxpc3": "CVCL_0186",
    "hap1": "CVCL_Y019",
    "ht29": "CVCL_0320",
    "k562": "CVCL_0004",
    "mcf7": "CVCL_0031",
}
JIANG_STIMULATION_KEYS = ("ifnb", "ifng", "ins", "tgfb", "tnfa")
TAHOE_CONTEXTS = (
    "CVCL_0023", "CVCL_0028", "CVCL_0069", "CVCL_0099", "CVCL_0131", "CVCL_0152", "CVCL_0179",
    "CVCL_0218", "CVCL_0292", "CVCL_0293", "CVCL_0320", "CVCL_0332", "CVCL_0334", "CVCL_0359",
    "CVCL_0366", "CVCL_0371", "CVCL_0397", "CVCL_0399", "CVCL_0428", "CVCL_0459", "CVCL_0480",
    "CVCL_0504", "CVCL_0546", "CVCL_1055", "CVCL_1056", "CVCL_1094", "CVCL_1097", "CVCL_1098",
    "CVCL_1119", "CVCL_1125", "CVCL_1239", "CVCL_1285", "CVCL_1381", "CVCL_1478", "CVCL_1495",
    "CVCL_1517", "CVCL_1531", "CVCL_1547", "CVCL_1550", "CVCL_1571", "CVCL_1577", "CVCL_1635",
    "CVCL_1666", "CVCL_1693", "CVCL_1715", "CVCL_1716", "CVCL_1717", "CVCL_1724", "CVCL_1731",
    "CVCL_C466",
)
JIANG_STIMULATIONS = {
    "ifnb": {
        "name": "Interferon beta",
        "abbreviation": "IFN-beta",
        "family": "type I interferon cytokine",
        "receptors": ["IFNAR1", "IFNAR2"],
        "signaling": (
            "JAK1/TYK2 signaling through STAT1/STAT2-IRF9 (ISGF3), inducing "
            "interferon-stimulated genes"
        ),
        "description": (
            "Antiviral type I interferon secreted in response to pathogen sensing; "
            "induces a broad interferon-stimulated gene program that restricts viral "
            "replication and modulates innate immunity."
        ),
    },
    "ifng": {
        "name": "Interferon gamma",
        "abbreviation": "IFN-gamma",
        "family": "type II interferon cytokine",
        "receptors": ["IFNGR1", "IFNGR2"],
        "signaling": (
            "JAK1/JAK2 signaling through STAT1 at gamma-activated sequence (GAS) promoter elements"
        ),
        "description": (
            "Pro-inflammatory type II interferon produced mainly by T and NK cells; "
            "activates macrophages and up-regulates MHC class I and II antigen "
            "presentation."
        ),
    },
    "ins": {
        "name": "Insulin",
        "abbreviation": "INS",
        "family": "peptide hormone",
        "receptors": ["INSR"],
        "signaling": (
            "insulin receptor tyrosine kinase signaling through PI3K-AKT-mTOR and RAS-MAPK cascades"
        ),
        "description": (
            "Anabolic peptide hormone secreted by pancreatic beta cells; stimulates "
            "glucose uptake, glycogen and lipid synthesis, and cell growth."
        ),
    },
    "tgfb": {
        "name": "Transforming growth factor beta",
        "abbreviation": "TGF-beta",
        "family": "TGF-beta superfamily cytokine",
        "receptors": ["TGFBR1", "TGFBR2"],
        "signaling": (
            "type I/type II receptor serine/threonine kinase signaling through SMAD2/SMAD3-SMAD4"
        ),
        "description": (
            "Pleiotropic cytokine that arrests epithelial proliferation, promotes "
            "epithelial-mesenchymal transition and extracellular-matrix production, "
            "and regulates immune tolerance."
        ),
    },
    "tnfa": {
        "name": "Tumor necrosis factor alpha",
        "abbreviation": "TNF-alpha",
        "family": "TNF superfamily pro-inflammatory cytokine",
        "receptors": ["TNFRSF1A", "TNFRSF1B"],
        "signaling": (
            "TNFR1/TNFR2 signaling through NF-kappa-B and MAPK/JNK, with "
            "context-dependent caspase-8-mediated cell death"
        ),
        "description": (
            "Central pro-inflammatory cytokine; activates NF-kappa-B-dependent "
            "inflammatory gene expression and can trigger apoptosis or necroptosis "
            "depending on cellular context."
        ),
    },
}


def _expected_contexts(dataset: str) -> dict[str, tuple[str, str | None]]:
    if dataset == "replogle":
        return {key: (acc, None) for key, acc in REPLOGLE_CONTEXTS.items()}
    if dataset == "orion":
        return {key: (acc, None) for key, acc in ORION_CONTEXTS.items()}
    if dataset == "arc_vcc_25":
        return {key: (acc, None) for key, acc in ARC_VCC_25_CONTEXTS.items()}
    if dataset == "jiang":
        return {
            f"{line}_{stim}": (acc, stim)
            for line, acc in JIANG_LINES.items()
            for stim in JIANG_STIMULATION_KEYS
        }
    return {key: (key, None) for key in TAHOE_CONTEXTS}


def _plain(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else dict(value)


@pytest.mark.parametrize("dataset", DATASETS)
def test_contexts_map_every_context_to_its_accession(dataset: str) -> None:
    parsed = load_context_file(SOURCES / "contexts" / f"{dataset}.yaml")
    got = {key: (entry.cellosaurus, entry.stimulation) for key, entry in parsed.contexts.items()}
    expected = _expected_contexts(dataset)
    assert list(got) == list(expected)
    assert got == expected
    if dataset != "jiang":
        assert dict(parsed.stimulations) == {}


def test_contexts_jiang_stimulations_are_the_reviewed_registry() -> None:
    parsed = load_context_file(SOURCES / "contexts" / "jiang.yaml")
    assert list(parsed.stimulations) == list(JIANG_STIMULATION_KEYS)
    assert {key: _plain(value) for key, value in parsed.stimulations.items()} == JIANG_STIMULATIONS

