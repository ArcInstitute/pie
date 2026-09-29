"""Tests for the measured chemical-profile builders (RDKit and downloads are faked)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.sources.inputs import (
    CHEM_DRUG_KEYS,
    chem_metadata,
    fake_inchi_keys,
    release_table,
    write_jump_files,
    write_l1000_files,
    write_prism_files,
)

from pie.sources import chem_profiles


@pytest.fixture(autouse=True)
def _fake_rdkit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chem_profiles, "_inchi_keys", fake_inchi_keys)


def _pin(monkeypatch: pytest.MonkeyPatch, files: Mapping[str, Path], source: str) -> None:
    monkeypatch.setattr(chem_profiles, "RELEASES", release_table(files, source))


def test_parse_perturbation() -> None:
    assert chem_profiles.parse_perturbation("drug_1.5uM") == {
        "pert": "drug_1.5uM",
        "drug": "drug",
        "dose_uM": 1.5,
    }
    assert chem_profiles.parse_perturbation("a_b_10uM")["drug"] == "a_b"
    assert chem_profiles.parse_perturbation("x_2\u00b5M")["dose_uM"] == 2.0
    for bad in ("bad", "x_0uM", "x_1.0nM"):
        with pytest.raises(ValueError):
            chem_profiles.parse_perturbation(bad)


def test_releases_pin_every_profile_input() -> None:
    sources = [entry["source"] for entry in chem_profiles.RELEASES.values()]
    assert sorted(set(sources)) == sorted(chem_profiles.PROFILES)
    assert len(chem_profiles.RELEASES) == 11
    assert all(len(entry["sha256"]) == 64 for entry in chem_profiles.RELEASES.values())
    name = chem_profiles.l1000_file("GSE70138", "sig_info")
    assert name == f"GSE70138_Broad_LINCS_sig_info_{chem_profiles.GSE70138_RELEASE}.txt.gz"
    assert chem_profiles.RELEASES[name]["url"].startswith(
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE70nnn/GSE70138/suppl/"
    )


def test_matcher_prefers_keys_then_smiles_then_name() -> None:
    identities = chem_profiles.drug_identities(["drugA", "drugB", "drugC"], chem_metadata())
    assert identities[0] == {
        "drug": "drugA",
        "canonical_smiles": "CCO",
        "inchi_key": "KEYA",
        "parent_inchi_key": "PARENTA",
    }
    assert identities[2]["inchi_key"] is None
    matcher = chem_profiles.CompoundMatcher(identities)
    assert matcher.match(key="PARENTA") == ["drugA"]
    assert matcher.match(smiles="CCN") == ["drugB"]
    assert matcher.match(name="Drug-C") == ["drugC"]
    assert matcher.match(name="nothing", key="KEYX", smiles="C") == []


def test_matcher_narrows_ambiguous_hits_by_exact_name() -> None:
    matcher = chem_profiles.CompoundMatcher(
        [{"drug": "X", "inchi_key": "K"}, {"drug": "Y", "inchi_key": "K"}]
    )
    assert matcher.match(name="y", key="K") == ["Y"]
    assert matcher.match(key="K") == []


def test_drug_identities_rejects_missing_and_duplicate_drugs() -> None:
    with pytest.raises(ValueError, match="missing drug metadata"):
        chem_profiles.drug_identities(["drugZ"], chem_metadata())
    with pytest.raises(ValueError, match="duplicate drug names"):
        chem_profiles.drug_identities(["drugA"], pd.concat([chem_metadata(), chem_metadata()]))


def test_l1000_takes_the_nearest_dose_and_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin(monkeypatch, write_l1000_files(tmp_path), "l1000_tas")
    monkeypatch.setattr(chem_profiles, "_L1000_CELLS", 2)
    keys, values = chem_profiles.build_profile(
        "l1000_tas", CHEM_DRUG_KEYS, chem_metadata(), tmp_path
    )
    # drugC's only signature has TAS -666 (missing), so drugC is not covered
    assert keys == ["drugA_1.0uM", "drugA_10.0uM", "drugB_1.0uM"]
    assert values.dtype == np.float32
    np.testing.assert_allclose(values, [[0.2, 0.3], [0.5, 0.3], [0.7, 0.0]], rtol=1e-6)


def test_prism_prefers_redo_screens_then_the_nearest_dose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin(monkeypatch, write_prism_files(tmp_path), "prism_secondary")
    monkeypatch.setattr(chem_profiles, "_PRISM_CELLS", 2)
    keys, values = chem_profiles.build_profile(
        "prism_secondary", CHEM_DRUG_KEYS, chem_metadata(), tmp_path
    )
    # cells sorted by model id: ACH-1 (row R2), ACH-2 (row R1); R3 fails QC
    assert keys == ["drugA_1.0uM", "drugA_10.0uM", "drugB_1.0uM"]
    np.testing.assert_allclose(values, [[-3.0, -1.5], [-3.0, -1.5], [0.0, 0.5]])


def test_jump_takes_the_median_well_profile_per_drug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _pin(monkeypatch, write_jump_files(tmp_path), "jump_morphology")
    monkeypatch.setattr(chem_profiles, "_JUMP_COORDS", 2)
    keys, values = chem_profiles.build_profile(
        "jump_morphology", CHEM_DRUG_KEYS, chem_metadata(), tmp_path
    )
    assert keys == ["drugA_1.0uM", "drugA_10.0uM"]
    np.testing.assert_allclose(values, [[2.0, 20.0], [2.0, 20.0]])


def test_row_count_guard_and_pinned_checksums(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    files = write_jump_files(tmp_path)
    _pin(monkeypatch, files, "jump_morphology")
    with pytest.raises(ValueError, match="737"):
        chem_profiles.build_profile("jump_morphology", CHEM_DRUG_KEYS, chem_metadata(), tmp_path)
    table = release_table(files, "jump_morphology")
    table["jump_compound.csv.gz"]["sha256"] = "0" * 64
    monkeypatch.setattr(chem_profiles, "RELEASES", table)
    with pytest.raises(ValueError, match="does not match"):
        chem_profiles.build_profile("jump_morphology", CHEM_DRUG_KEYS, chem_metadata(), tmp_path)
    with pytest.raises(KeyError, match="dose_onehot"):
        chem_profiles.build_profile("dose_onehot", CHEM_DRUG_KEYS, chem_metadata(), tmp_path)
