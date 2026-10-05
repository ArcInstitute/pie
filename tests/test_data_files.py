"""Tracked data files: canonical splits, curated source aliases and context maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pie.data.dataset import load_aliases
from pie.data.splits import Split, check_disjoint, load_split, parse_split_key
from pie.sources.text.contexts import load_context_file
from pie.utils import sha256_file
from tests.conftest import REPO_ROOT

SPLITS = REPO_ROOT / "data" / "splits"
WDATASET = SPLITS / "replogle_wdataset"
XDATASET = SPLITS / "replogle_xdataset"
SOURCES = REPO_ROOT / "data" / "sources"
FOLDS = ("hepg2", "jurkat", "k562", "rpe1")
SETTINGS = ("unseen_ctx", "unseen_pert", "unseen_ctx_pert")
DATASETS = ("replogle", "tahoe", "jiang", "arc_vcc_25", "orion")
XDATASET_FILES = ("train.json", "val.json", "test_seen.json", "test_unseen.json")

WDATASET_TRAIN_SHA256 = {
    "hepg2": "828efbc978904d3637133fb69a2862bc0fb256ac0a4bdf4a231f0a7a71dcec0c",
    "jurkat": "65d072b95eb9e79edbf2cc6fe5c44bc92d2d8909ed8489c587a304f487b9a123",
    "k562": "6143b4d1bdc30f09a323a5af944fe9049c77c724909fb20fdde0039b7faeddfb",
    "rpe1": "9862e35811aaf7b1d104c090612af0f41d4d3d4db9580a6ebea13cb393811ab6",
}
WDATASET_VAL_SHA256 = {
    "hepg2": "5ffc07afa54dce5c9139cf203d6bafb703a237c2b3f95b712dfd4dbc7fde85f0",
    "jurkat": "31ecf26911b194c36d968ed5b3afdc560ab6ae17fc9f26f3089a40a2d6231221",
    "k562": "bde657c9750f0d4ac1fa4826510147834fbc2161f6cfbd11f59c97479d90621a",
    "rpe1": "d2bbb95839cf2618465ff0fbd0ed25dde67b0c64cd208d12f30399667d6729ba",
}
WDATASET_TEST_SHA256 = {
    ("unseen_ctx", "hepg2"): "c5dda0a2afbd71e0c81f72072ac93ff3be7a22c5df5bdfd9c2b50e01d9623a2e",
    ("unseen_ctx", "jurkat"): "6286889c032caeca9902a9da6ad3956ebb358372d7e0d787d3e040da202d73d0",
    ("unseen_ctx", "k562"): "e5aedaeb92860b7d3082c82470218f5b8787f524eebfd2987e301e4da96ed850",
    ("unseen_ctx", "rpe1"): "b1597e981ada3b510d29f5fce96a04bfbff066a89036a025c2773d31275fc949",
    ("unseen_pert", "hepg2"): "683d28c9da80dff9a303f92d8115b45578a3a3b4dc82dac68902ae034f8d3cca",
    ("unseen_pert", "jurkat"): "da2121562fdcf3ed95b34381380660051068a0b344dd98a802c5be6e9322394e",
    ("unseen_pert", "k562"): "5805fbee43c3c60239f80dfdfc9ebd5dbe0011b672f18575d927060027c3525e",
    ("unseen_pert", "rpe1"): "e3498b2405a30ab48561616559b3239b6c4ec5af3ba2aaf5c8fbf973b516f9f8",
    ("unseen_ctx_pert", "hepg2"): (
        "9fb754ec39a9698c1684aee95aef4bc1e9cb09283315d28aa9a99de8caa16963"
    ),
    ("unseen_ctx_pert", "jurkat"): (
        "3c4a4550d7347e62338f3f606ba7d1a6bd5124d3b9361af1b38c8a31a84d83cf"
    ),
    ("unseen_ctx_pert", "k562"): "2d155bdec7d7577ce85e0241b15275752e5504f2f52cc6e074f27b80e4801d8c",
    ("unseen_ctx_pert", "rpe1"): (
        "f752dc487a1c3c28338a1332f7ba4dd3eb360daa6225edbf7e716f74af86a54f"
    ),
}
XDATASET_SHA256 = {
    "train.json": "8d050f28aebf482135d42f758f6b440e0333b3dec8efc0dddab9dcc3099d4be4",
    "val.json": "36b2ea97734f9a470fd9906741182e6a64b4833f0b2713f83e07b5df110f1c9c",
}
XDATASET_LINES = ("replogle.hepg2", "replogle.jurkat", "replogle.k562", "replogle.rpe1")
XDATASET_SEEN_COUNTS = {
    "replogle.hepg2": 490, "replogle.jurkat": 626, "replogle.k562": 573, "replogle.rpe1": 597,
}
XDATASET_UNSEEN_COUNTS = {
    "replogle.hepg2": 763, "replogle.jurkat": 813, "replogle.k562": 741, "replogle.rpe1": 856,
}
XDATASET_PANEL_COUNTS = {
    "replogle.hepg2": 1253, "replogle.jurkat": 1439, "replogle.k562": 1314, "replogle.rpe1": 1453,
}

EXPECTED_ALIASES = {
    "esm2": {
        "TAZ": "TAFAZZIN",
        "ADAL": "MAPDA",
        "C15orf48": "COXFA4L3",
        "C17orf49": "BACC1",
        "C4orf3": "ARLN",
        "ILVBL": "HACL2",
        "MFSD10": "SLC75A1",
        "SLC22A18": "SLC67A1",
        "SMIM6": "ERLN",
        "STK19": "WHR1",
        "TMEM104": "SLC38A12",
        "TMEM30B": "CDC50B",
    },
    "ncbi_text": {
        "TAZ": "TAFAZZIN",
        "ADAL": "MAPDA",
        "C15orf48": "COXFA4L3",
        "C16orf74": "CLMB",
        "C17orf49": "BACC1",
        "C4orf3": "ARLN",
        "ILVBL": "HACL2",
        "MFSD10": "SLC75A1",
        "SLC22A18": "SLC67A1",
        "SMIM6": "ERLN",
        "STK19": "WHR1",
        "TMEM104": "SLC38A12",
        "TMEM30B": "CDC50B",
    },
    "string_space": {},
    "depmap_gene_effect": {
        "TAZ": "TAFAZZIN",
        "ADAL": "MAPDA",
        "C17orf49": "BACC1",
        "C4orf3": "ARLN",
        "SLC22A18": "SLC67A1",
        "SMIM6": "ERLN",
        "STK19": "WHR1",
    },
}

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


def _wdataset_cases() -> list[Any]:
    cases = []
    for setting in SETTINGS:
        for fold in FOLDS:
            expected = {
                "train.json": WDATASET_TRAIN_SHA256[fold],
                "val.json": WDATASET_VAL_SHA256[fold],
                "test.json": WDATASET_TEST_SHA256[(setting, fold)],
            }
            for name, sha256 in expected.items():
                cases.append(
                    pytest.param(setting, fold, name, sha256, id=f"{setting}-{fold}-{name}")
                )
    return cases


def _split_dir_cases() -> list[Any]:
    cases = [
        pytest.param(WDATASET / setting / fold, ("train.json", "val.json", "test.json"),
                     id=f"wdataset-{setting}-{fold}")
        for setting in SETTINGS
        for fold in FOLDS
    ]
    cases.append(pytest.param(XDATASET, XDATASET_FILES, id="xdataset"))
    return cases


def _perts(split: Split) -> set[str]:
    return {pert for perts in split.values() for pert in perts}


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


@pytest.mark.parametrize(("setting", "fold", "name", "sha256"), _wdataset_cases())
def test_split_wdataset_file_is_canonical(setting: str, fold: str, name: str, sha256: str) -> None:
    assert sha256_file(WDATASET / setting / fold / name) == sha256


def test_split_wdataset_tree_holds_only_the_canonical_files() -> None:
    found = sorted(p.relative_to(WDATASET).as_posix() for p in WDATASET.rglob("*") if p.is_file())
    expected = sorted(
        f"{setting}/{fold}/{name}"
        for setting in SETTINGS
        for fold in FOLDS
        for name in ("train.json", "val.json", "test.json")
    )
    assert found == expected


@pytest.mark.parametrize("name", sorted(XDATASET_SHA256))
def test_split_xdataset_train_and_val_are_canonical(name: str) -> None:
    assert sha256_file(XDATASET / name) == XDATASET_SHA256[name]


def test_split_xdataset_tree_holds_only_four_files() -> None:
    found = sorted(p.name for p in XDATASET.glob("*") if p.is_file())
    assert found == sorted(XDATASET_FILES)


def test_split_xdataset_test_rows_follow_the_train_perturbations() -> None:
    train_perts = _perts(load_split(XDATASET / "train.json"))
    seen = load_split(XDATASET / "test_seen.json")
    unseen = load_split(XDATASET / "test_unseen.json")
    assert {key: len(perts) for key, perts in seen.items()} == XDATASET_SEEN_COUNTS
    assert {key: len(perts) for key, perts in unseen.items()} == XDATASET_UNSEEN_COUNTS
    assert _perts(seen) <= train_perts
    assert not _perts(unseen) & train_perts


def test_split_xdataset_test_sets_partition_the_full_panel() -> None:
    seen = load_split(XDATASET / "test_seen.json")
    unseen = load_split(XDATASET / "test_unseen.json")
    assert list(seen) == list(XDATASET_LINES)
    assert list(unseen) == list(XDATASET_LINES)
    for key in XDATASET_LINES:
        assert not set(seen[key]) & set(unseen[key])
    assert {key: len(seen[key]) + len(unseen[key]) for key in XDATASET_LINES} == (
        XDATASET_PANEL_COUNTS
    )


@pytest.mark.parametrize(("split_dir", "names"), _split_dir_cases())
def test_split_files_are_pairwise_disjoint(split_dir: Path, names: tuple[str, ...]) -> None:
    splits = {name: load_split(split_dir / name) for name in names}
    assert all(splits.values())
    check_disjoint(splits)


def test_aliases_are_the_curated_entries() -> None:
    aliases = load_aliases(SOURCES / "aliases.yaml")
    assert aliases == EXPECTED_ALIASES
    assert list(aliases) == list(EXPECTED_ALIASES)
    assert {name: len(entries) for name, entries in aliases.items()} == {
        "esm2": 12, "ncbi_text": 13, "string_space": 0, "depmap_gene_effect": 7,
    }


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


def test_contexts_cover_every_key_of_every_split_file() -> None:
    known = {
        dataset: set(load_context_file(SOURCES / "contexts" / f"{dataset}.yaml").contexts)
        for dataset in DATASETS
    }
    files = sorted(SPLITS.rglob("*.json"))
    assert len(files) == 40
    missing = []
    for path in files:
        for key in load_split(path):
            dataset, context = parse_split_key(key)
            if context not in known.get(dataset, set()):
                missing.append(f"{path.relative_to(SPLITS).as_posix()}:{key}")
    assert missing == []
