"""Tests for the ESM2 sliding-window mean encoder (tokenizer and model are faked)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.sources.helpers import make_options
from tests.sources.inputs import FakeEsmModel, FakeEsmTokenizer

from pie.sources import uniprot
from pie.sources.embed import esm2
from pie.sources.registry import RunContext


def _patch(monkeypatch: pytest.MonkeyPatch) -> FakeEsmTokenizer:
    tokenizer = FakeEsmTokenizer()

    def load(model_id: str, revision: str, device: str) -> tuple[object, object]:
        assert (model_id, revision, device) == (esm2.MODEL, esm2.REVISION, "cpu")
        return tokenizer, FakeEsmModel()

    monkeypatch.setattr(esm2, "_load_model", load)
    return tokenizer


def test_short_sequence_is_the_mean_residue_state(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _patch(monkeypatch)
    out = esm2.embed_proteins(["ABC"], device="cpu")
    assert out.dtype == np.float32
    assert out.shape == (1, 2)
    # residues A, B, C have ids 1, 2, 3 at positions 1, 2, 3 (position 0 is <cls>)
    np.testing.assert_allclose(out[0], [2.0, 2.0])
    assert tokenizer.max_lengths == [esm2.WINDOW + 2]


def test_long_sequence_averages_overlapping_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _patch(monkeypatch)
    monkeypatch.setattr(esm2, "WINDOW", 4)
    monkeypatch.setattr(esm2, "STRIDE", 2)
    out = esm2.embed_proteins(["ABC", "ABCDEF"], device="cpu")
    # windows ABCD, CDEF, EF: per-residue window positions A1 B2 C(3,1) D(4,2) E(3,1) F(4,2)
    # -> averages 1, 2, 2, 3, 2, 3 (mean 13/6); residue ids 1..6 (mean 3.5)
    np.testing.assert_allclose(out, [[2.0, 2.0], [3.5, 13 / 6]], rtol=1e-6)
    assert tokenizer.max_lengths == [6, 6, 6, 6]


def test_empty_sequence_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch)
    with pytest.raises(ValueError, match="empty"):
        esm2.embed_proteins(["ABC", ""], device="cpu")


def test_run_esm2_fetches_under_its_own_cache_subdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both raw-input fetches must cache under cache_dir/esm2, per contract §16 (R10)."""
    seen: dict[str, Path] = {}

    def fake_fetch_reviewed_human(cache_dir: Path, offline: bool = False) -> Path:
        seen["uniprot"] = Path(cache_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        path = Path(cache_dir) / "stream.tsv"
        path.write_text("stream")
        return path

    def fake_fetch_ncbi_ftp(
        cache_dir: Path, names: tuple[str, ...] = (), offline: bool = False
    ) -> dict[str, Path]:
        seen["ncbi"] = Path(cache_dir)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        path = Path(cache_dir) / "gene_info.gz"
        path.write_bytes(b"gene_info")
        return {"gene_info": path}

    monkeypatch.setattr(uniprot, "fetch_reviewed_human", fake_fetch_reviewed_human)
    monkeypatch.setattr(esm2, "fetch_ncbi_ftp", fake_fetch_ncbi_ftp)
    table = pd.DataFrame({"sequence": ["AC"], "symbol": ["G1"]})
    monkeypatch.setattr(uniprot, "build_proteins_table", lambda *_a, **_k: table)
    monkeypatch.setattr(uniprot, "proteins_sha256", lambda _table: "deadbeef")
    monkeypatch.setattr(
        esm2, "embed_proteins", lambda seqs, **_k: np.zeros((len(seqs), 2), dtype=np.float32)
    )

    ctx = RunContext(
        datasets=[],
        prior_root=None,
        out_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        options=make_options(device="cpu"),
    )
    esm2.run_esm2(ctx)

    expected = tmp_path / "cache" / "esm2"
    assert seen["uniprot"] == expected
    assert seen["ncbi"] == expected


def test_run_esm2_uses_the_pinned_gene_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinned = tmp_path / "pinned_gene_info.gz"
    pinned.write_bytes(b"pinned")
    stream = tmp_path / "stream.tsv"
    stream.write_text("stream")
    seen: dict[str, Path] = {}

    def no_fetch(*_a: object, **_k: object) -> dict[str, Path]:
        raise AssertionError("gene_info was fetched although --gene-info is set")

    def fake_table(_stream: Path, gene_info: Path) -> pd.DataFrame:
        seen["gene_info"] = Path(gene_info)
        return pd.DataFrame({"sequence": ["AC"], "symbol": ["G1"]})

    monkeypatch.setattr(uniprot, "fetch_reviewed_human", lambda *_a, **_k: stream)
    monkeypatch.setattr(esm2, "fetch_ncbi_ftp", no_fetch)
    monkeypatch.setattr(uniprot, "build_proteins_table", fake_table)
    monkeypatch.setattr(uniprot, "proteins_sha256", lambda _table: "deadbeef")
    monkeypatch.setattr(
        esm2, "embed_proteins", lambda seqs, **_k: np.zeros((len(seqs), 2), dtype=np.float32)
    )
    ctx = RunContext(
        datasets=[],
        prior_root=None,
        out_root=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        options=make_options(device="cpu", gene_info=pinned),
    )
    esm2.run_esm2(ctx)
    assert seen["gene_info"] == pinned
