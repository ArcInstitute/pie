"""Tests for pie-sources verify: key coverage after aliases, and diffs against a reference."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from tests.fixtures import TinyData

from pie.cli import sources_main
from pie.data.preprocessed import PreprocessedDir
from pie.sources import registry
from pie.sources.contract import read_source, write_source
from pie.sources.verify import verify_sources


def _dirs(tiny: TinyData, *names: str) -> list[PreprocessedDir]:
    return [PreprocessedDir.open(tiny.preprocessed[name]) for name in names]


def _sources(tiny: TinyData) -> dict[str, Path]:
    return {
        "esm2": tiny.sources["esm2"],
        "ncbi_text": tiny.sources["ncbi_text"],
        "context_text": tiny.sources["context_text"],
        "gene_text": tiny.gene_text,
    }


def test_coverage_counts_direct_alias_and_missing_keys(tiny_data: TinyData) -> None:
    report = verify_sources(
        _sources(tiny_data), _dirs(tiny_data, "alpha", "beta"), tiny_data.aliases, None
    )
    coverage = report["coverage"]
    assert coverage["esm2"]["alpha"] == {
        "n": 4, "direct": 3, "alias": 1, "missing": 0, "missing_keys": []
    }
    assert coverage["esm2"]["beta"] == {
        "n": 3, "direct": 0, "alias": 0, "missing": 3, "missing_keys": ["drugA", "drugB", "drugC"]
    }
    # ncbi_text aliases OLDX -> GZ, but GZ is not a key of that source
    assert coverage["ncbi_text"]["alpha"] == {
        "n": 4, "direct": 2, "alias": 0, "missing": 2, "missing_keys": ["GC", "OLDX"]
    }
    assert coverage["context_text"]["beta"] == {
        "n": 2, "direct": 2, "alias": 0, "missing": 0, "missing_keys": []
    }
    assert coverage["gene_text"]["alpha"]["n"] == 6
    assert coverage["gene_text"]["alpha"]["missing"] == 0
    assert report["reference"] == {}


def test_without_aliases_the_alias_hit_is_missing(tiny_data: TinyData) -> None:
    report = verify_sources(
        {"esm2": tiny_data.sources["esm2"]}, _dirs(tiny_data, "alpha"), None, None
    )
    assert report["coverage"]["esm2"]["alpha"]["missing_keys"] == ["OLDX"]


def test_reference_diff_reports_cosines_and_bitwise_rows(
    tiny_data: TinyData, tmp_path: Path
) -> None:
    reference = tmp_path / "reference"
    esm2 = read_source(tiny_data.sources["esm2"])
    flipped = np.array(esm2.embeddings)
    flipped[1] *= -1.0
    write_source(reference / "esm2", esm2.meta, flipped)
    shutil.copytree(tiny_data.sources["ncbi_text"], reference / "ncbi_text")
    report = verify_sources(
        _sources(tiny_data), _dirs(tiny_data, "alpha"), tiny_data.aliases, reference
    )
    assert sorted(report["reference"]) == ["esm2", "ncbi_text"]
    dense = report["reference"]["esm2"]
    assert dense["keys_equal"]
    assert dense["n_common"] == 4
    assert dense["n_bitwise_equal"] == 3
    assert dense["cosine_min"] == pytest.approx(-1.0)
    assert dense["cosine_mean"] == pytest.approx(0.5)
    assert dense["descriptions_equal"] is None
    token = report["reference"]["ncbi_text"]
    assert (token["n_length_equal"], token["n_sampled"], token["n_sampled_bitwise_equal"]) == (
        3,
        3,
        3,
    )
    assert token["cosine_min"] == pytest.approx(1.0, abs=1e-6)


def test_registry_delegates_to_verify(tiny_data: TinyData) -> None:
    sources = {"esm2": tiny_data.sources["esm2"]}
    dirs = _dirs(tiny_data, "alpha")
    assert registry.verify_sources(sources, dirs, tiny_data.aliases, None) == verify_sources(
        sources, dirs, tiny_data.aliases, None
    )


def test_cli_verify_prints_the_report(
    tiny_data: TinyData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = tmp_path / "reference"
    esm2 = read_source(tiny_data.sources["esm2"])
    flipped = np.array(esm2.embeddings)
    flipped[1] *= -1.0
    write_source(reference / "esm2", esm2.meta, flipped)
    argv = [
        "mode=verify",
        "tools=[esm2]",
        f"preprocessed_dirs=[{tiny_data.preprocessed['alpha']}]",
        f"output_root={tiny_data.root / 'sources'}",
        f"verify.aliases={tiny_data.aliases}",
        f"verify.reference={reference}",
    ]
    assert sources_main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    expected = verify_sources(
        {"esm2": tiny_data.sources["esm2"]}, _dirs(tiny_data, "alpha"), tiny_data.aliases, reference
    )
    assert report == json.loads(json.dumps(expected))
    assert report["coverage"]["esm2"]["alpha"]["alias"] == 1
    assert report["reference"]["esm2"]["n_bitwise_equal"] == 3


def test_cli_verify_needs_no_openai_key_or_cache(
    tiny_data: TinyData, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PIE_CACHE_DIR", raising=False)
    argv = [
        "mode=verify",
        "tools=[context_text]",
        f"preprocessed_dirs=[{tiny_data.preprocessed['beta']}]",
        f"output_root={tiny_data.root / 'sources'}",
        "verify.aliases=null",
    ]
    assert sources_main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["coverage"]["context_text"]["beta"]["missing"] == 0
    assert report["reference"] == {}
