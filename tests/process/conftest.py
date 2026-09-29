"""A fake `gpudge` module for the CPU pie-process tests (the real gpudge needs CUDA)."""

from __future__ import annotations

import sys
import types

import anndata as ad
import pandas as pd
import pytest

from pie.process import de as de_mod
from pie.process.de import OUTPUT_COLUMNS


class FakeFrame:
    """Stands in for gpudge's polars DataFrame: len() and write_parquet()."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def __len__(self) -> int:
        return len(self.frame)

    def write_parquet(self, file: str) -> None:
        self.frame.to_parquet(file, index=False)


class FakeGpudge(types.ModuleType):
    """A `gpudge` module that records every de() call and returns one row per (target, gene)."""

    def __init__(self) -> None:
        super().__init__("gpudge")
        self.calls: list[tuple[ad.AnnData, dict[str, object]]] = []

    def de(self, adata: ad.AnnData, **kwargs: object) -> FakeFrame:
        self.calls.append((adata.copy(), dict(kwargs)))
        groups = adata.obs[str(kwargs["groupby"])].astype(str)
        reference = str(kwargs["reference"])
        rows = [
            {
                "target": target,
                "feature": gene,
                "target_mean": 1.0,
                "ref_mean": 1.0,
                "target_ncells": int((groups == target).sum()),
                "ref_ncells": int((groups == reference).sum()),
                "log2_fold_change": 0.0,
                "p_value": 1.0,
                "Ueffect": 0.5,
                "p_adj": 1.0,
            }
            for target in sorted(set(groups) - {reference})
            for gene in adata.var_names
        ]
        return FakeFrame(pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS)))


@pytest.fixture
def fake_gpudge(monkeypatch: pytest.MonkeyPatch) -> FakeGpudge:
    fake = FakeGpudge()
    monkeypatch.setitem(sys.modules, "gpudge", fake)
    monkeypatch.setattr(de_mod, "_cuda_available", lambda: True)
    return fake
