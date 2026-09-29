"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures import TinyData, build_tiny_data


@pytest.fixture
def tiny_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TinyData:
    """Two synthetic datasets, five sources, gene_text, splits and aliases under tmp_path."""
    data = build_tiny_data(tmp_path / "tiny")
    monkeypatch.setenv("PIE_CACHE_DIR", str(data.cache_dir))
    return data
