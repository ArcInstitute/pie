from __future__ import annotations

import tomllib
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from pydantic import ValidationError

import pie.cli
from pie.cli import _launch, process_main
from pie.process.config import DATASETS
from pie.utils import ENV_ROOTS, REPO_ROOT, MissingEnvError
from tests.process.helpers import make_counts, write_counts


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pie.cli, "load_common_env", lambda *_a, **_k: {})
    for name in ENV_ROOTS:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))


def test_help_lists_the_datasets(capsys: pytest.CaptureFixture[str]) -> None:
    assert process_main(["--help"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("usage: pie-process")
    for name in DATASETS:
        assert name in text


def test_process_main_runs_the_enabled_stages(tmp_path: Path) -> None:
    src = write_counts(make_counts(), tmp_path / "counts" / "s.h5ad")
    out = tmp_path / "expression"
    argv = [f"input={src}", "normalize.enabled=true", f"normalize.output_dir={out}"]
    assert process_main(argv) == 0
    assert ad.read_h5ad(out / "s.h5ad").X.dtype == np.float32


def test_process_main_rejects_an_invalid_config(tmp_path: Path) -> None:
    src = write_counts(make_counts(), tmp_path / "counts" / "s.h5ad")
    with pytest.raises(ValidationError, match="no stage enabled"):
        process_main([f"input={src}"])


def test_process_main_needs_no_env_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_ROOTS:
        monkeypatch.delenv(name)
    src = write_counts(make_counts(), tmp_path / "counts" / "s.h5ad")
    argv = [f"input={src}", "normalize.enabled=true", f"normalize.output_dir={tmp_path / 'e'}"]
    assert process_main(argv) == 0


def test_launch_checks_the_required_env_after_help(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PIE_CACHE_DIR")
    calls: list[list[str]] = []
    assert _launch(["--help"], "usage: x\n", calls.append, require=ENV_ROOTS) == 0
    assert capsys.readouterr().out == "usage: x\n"
    with pytest.raises(MissingEnvError, match="PIE_CACHE_DIR"):
        _launch(["a=1"], "usage: x\n", calls.append, require=ENV_ROOTS)
    assert calls == []
    assert _launch(["a=1"], "usage: x\n", calls.append) == 0
    assert calls == [["a=1"]]


def test_process_main_writes_nothing_into_the_working_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = write_counts(make_counts(), tmp_path / "counts" / "s.h5ad")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    argv = [f"input={src}", "normalize.enabled=true", f"normalize.output_dir={tmp_path / 'e'}"]
    assert process_main(argv) == 0
    assert list(work.iterdir()) == []


def test_console_script_is_registered() -> None:
    meta = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert meta["project"]["scripts"]["pie-process"] == "pie.cli:process_main"
