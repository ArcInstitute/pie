"""pie-sources: config mapping, env checks, dependency order and overwrite."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.conftest import REPO_ROOT

import pie.cli
from pie.cli import sources_main
from pie.data.preprocessed import PreprocessedDir
from pie.sources import registry
from pie.utils import MissingEnvError


@pytest.fixture
def seen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    """Two fake tools (fake_b needs fake_a and OpenAI); returns (tool, ctx) in run order."""
    calls: list[tuple[str, Any]] = []

    def tool(name: str) -> Any:
        def run(ctx: Any) -> Path:
            calls.append((name, ctx))
            return ctx.out_root / name

        return run

    monkeypatch.setitem(
        registry.TOOLS, "fake_a", registry.SourceTool("fake_a", (), False, tool("fake_a"))
    )
    monkeypatch.setitem(
        registry.TOOLS, "fake_b", registry.SourceTool("fake_b", ("fake_a",), True, tool("fake_b"))
    )
    monkeypatch.setattr(pie.cli, "load_common_env", lambda *_a, **_k: {})
    monkeypatch.setattr(
        PreprocessedDir,
        "open",
        classmethod(lambda cls, path: SimpleNamespace(dataset=path.name, path=path)),
    )
    for name in ("PIE_DATA_ROOT", "PIE_RUNS_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("PIE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return calls


def test_build_orders_dependencies_and_maps_keys(
    tmp_path: Path, seen: list[tuple[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "sources"
    a, b = tmp_path / "pre" / "alpha", tmp_path / "pre" / "beta"
    argv = ["tools=[fake_b]", f"preprocessed_dirs=[{a},{b}]", f"output_root={out}",
            "options.on_conflict=keep-prior", "options.offline=true"]  # fmt: skip
    assert sources_main(argv) == 0
    assert [name for name, _ in seen] == ["fake_a", "fake_b"]
    ctx = seen[0][1]
    assert [d.path for d in ctx.datasets] == [a, b]
    assert ctx.out_root == out and ctx.prior_root is None and ctx.overwrite is False
    assert ctx.cache_dir == tmp_path / "cache" / "http"
    assert ctx.options.on_conflict == "keep-prior" and ctx.options.offline
    assert ctx.options.contexts_dir == REPO_ROOT / "data" / "sources" / "contexts"
    assert json.loads(capsys.readouterr().out) == {n: str(out / n) for n in ("fake_a", "fake_b")}


def test_with_deps_false_runs_only_the_named_tools(
    tmp_path: Path, seen: list[tuple[str, Any]]
) -> None:
    argv = ["tools=[fake_b]", "with_deps=false", f"preprocessed_dirs=[{tmp_path / 'alpha'}]",
            f"prior_root={tmp_path / 'p'}", f"output_root={tmp_path / 'o'}",
            f"options.pert_output={tmp_path / 'pt'}"]  # fmt: skip
    assert sources_main(argv) == 0
    assert [name for name, _ in seen] == ["fake_b"]
    ctx = seen[0][1]
    assert ctx.prior_root == tmp_path / "p" and ctx.out_root == tmp_path / "o"
    assert ctx.options.pert_output == tmp_path / "pt"


def test_existing_targets_fail_before_any_tool_runs(
    tmp_path: Path, seen: list[tuple[str, Any]]
) -> None:
    out = tmp_path / "o"
    (out / "fake_a").mkdir(parents=True)
    (out / "fake_a" / "meta.json").write_text("{}")
    argv = ["tools=[fake_b]", f"preprocessed_dirs=[{tmp_path / 'alpha'}]", f"output_root={out}"]
    with pytest.raises(FileExistsError, match="fake_a"):
        sources_main(argv)
    assert seen == []
    assert sources_main([*argv, "overwrite=true"]) == 0
    assert all(ctx.overwrite for _, ctx in seen)


def test_env_and_key_are_checked_before_any_tool_runs(
    seen: list[tuple[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv = ["tools=[fake_b]", f"preprocessed_dirs=[{tmp_path / 'a'}]",
            f"output_root={tmp_path / 's'}"]  # fmt: skip
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(MissingEnvError):
        sources_main(argv)
    monkeypatch.delenv("PIE_CACHE_DIR")
    with pytest.raises(MissingEnvError):
        sources_main(["tools=[fake_a]", *argv[1:]])
    assert seen == []


def test_help_and_script_entry(capsys: pytest.CaptureFixture[str]) -> None:
    assert sources_main(["--help"]) == 0
    assert capsys.readouterr().out.startswith("usage: pie sources")
    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["pie-sources"] == "pie.cli:sources_alias"
