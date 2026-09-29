"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from pie.train import run_train
from tests.fixtures import TinyData, build_tiny_data
from tests.pipeline import TrainedRun, tiny_train_config


@pytest.fixture
def tiny_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TinyData:
    """Two synthetic datasets, five sources, gene_text, splits and aliases under tmp_path."""
    data = build_tiny_data(tmp_path / "tiny")
    monkeypatch.setenv("PIE_CACHE_DIR", str(data.cache_dir))
    return data


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> TrainedRun:
    """Train the tiny model for 2 CPU steps, once per test session."""
    root = tmp_path_factory.mktemp("pipeline")
    tiny = build_tiny_data(root / "tiny")
    run = TrainedRun(root=root, tiny=tiny, run_dir=root / "runs" / "tiny")
    with pytest.MonkeyPatch.context() as mp:
        run.apply_env(mp)
        run_train(tiny_train_config(tiny, run.run_dir))
    return run


@pytest.fixture
def pipeline(trained_run: TrainedRun, monkeypatch: pytest.MonkeyPatch) -> TrainedRun:
    """The session's trained run, with its env roots set for this test."""
    trained_run.apply_env(monkeypatch)
    return trained_run
