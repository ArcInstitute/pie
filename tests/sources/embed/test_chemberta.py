"""Tests for the ChemBERTa SMILES encoder and its drug-dose mapping (the model is faked)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from tests.sources.inputs import FakeChemModel, FakeChemTokenizer

from pie.sources.embed import chemberta


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(model_id: str, revision: str, device: str) -> tuple[object, object]:
        assert (model_id, revision, device) == (chemberta.MODEL, chemberta.REVISION, "cpu")
        return FakeChemTokenizer(), FakeChemModel()

    monkeypatch.setattr(chemberta, "_load_model", load)


def test_masked_mean_ignores_padding() -> None:
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    np.testing.assert_allclose(chemberta.masked_mean(hidden, mask).numpy(), [[2.0, 3.0]])


@pytest.mark.parametrize("batch", [1, 64])
def test_embed_smiles_pools_each_string(monkeypatch: pytest.MonkeyPatch, batch: int) -> None:
    _patch(monkeypatch)
    monkeypatch.setattr(chemberta, "BATCH", batch)
    out = chemberta.embed_smiles(["CC", "N"], device="cpu")
    assert out.dtype == np.float32
    # CC -> ids 1, 67, 67, 2 (mean 34.25); N -> 1, 78, 2 (mean 27); padding is masked out
    np.testing.assert_allclose(out, [[34.25, 1.0], [27.0, 1.0]])


def test_smiles_inputs_maps_every_dose_to_its_drug_row() -> None:
    metadata = pd.DataFrame(
        {
            "drug": ["drugA", "drugB ", "drugC", "drugD"],
            "canonical_smiles": ["CC", "N", None, "CC"],
        }
    )
    keys = ["drugD_1.0uM", "drugA_2.0uM", "drugA_1.0uM", "drugB_1.0uM", "drugC_1.0uM"]
    assert chemberta.smiles_inputs(keys, metadata) == chemberta.SmilesInputs(
        keys=["drugA_1.0uM", "drugA_2.0uM", "drugB_1.0uM", "drugD_1.0uM"],
        smiles=["CC", "N"],
        rows=[0, 0, 1, 0],
        omitted=["drugC"],
    )


def test_smiles_inputs_rejects_bad_metadata() -> None:
    good = pd.DataFrame({"drug": ["drugA"], "canonical_smiles": ["CC"]})
    with pytest.raises(ValueError, match="missing drug metadata"):
        chemberta.smiles_inputs(["drugZ_1.0uM"], good)
    with pytest.raises(ValueError, match="duplicate drug names"):
        chemberta.smiles_inputs(["drugA_1.0uM"], pd.concat([good, good]))
    with pytest.raises(ValueError, match="canonical_smiles"):
        chemberta.smiles_inputs(["drugA_1.0uM"], good.rename(columns={"canonical_smiles": "s"}))
    with pytest.raises(ValueError, match="no SMILES"):
        chemberta.smiles_inputs(["drugA_1.0uM"], good.assign(canonical_smiles=[""]))
