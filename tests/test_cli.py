"""Tests for pie.cli entry points."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from pie import cli
from pie.utils import REPO_ROOT, MissingEnvError
from tests.pipeline import TrainedRun


def _no_common_env(path: Path | None = None) -> dict[str, str]:
    return {}


@pytest.mark.parametrize("main", ["eval_main", "infer_main"])
def test_help(main: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert getattr(cli, main)(["--help"]) == 0
    assert capsys.readouterr().out.startswith(f"usage: pie {main.removesuffix('_main')}")


def test_pie_eval_writes_the_metric_tables(
    pipeline: TrainedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_common_env", _no_common_env)
    argv = [
        "experiment_name=tiny",
        f"run_dir={pipeline.run_dir}",
        f"split_path={pipeline.test_split}",
        "row_set=cli_eval",
        "device=cpu",
    ]
    cli.eval_main(argv)
    out = pipeline.run_dir / "eval" / "cli_eval"
    assert (out / "metrics_best_auprc.csv").is_file()
    assert (out / "granular_best_auprc.csv").is_file()
    assert not (out / "predictions_best_auprc.parquet").exists()


def test_pie_eval_refuses_existing_tables_unless_overwrite(
    pipeline: TrainedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_common_env", _no_common_env)
    argv = ["experiment_name=tiny", f"run_dir={pipeline.run_dir}",
            f"split_path={pipeline.test_split}", "row_set=cli_twice", "device=cpu"]  # fmt: skip
    cli.eval_main(argv)
    with pytest.raises(FileExistsError, match="overwrite=true"):
        cli.eval_main(argv)
    assert cli.eval_main([*argv, "overwrite=true"]) == 0


def test_pie_infer_writes_the_parquet(
    pipeline: TrainedRun, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "cli" / "predictions.parquet"
    monkeypatch.setattr(cli, "load_common_env", _no_common_env)
    argv = [
        "experiment_name=tiny",
        f"run_dir={pipeline.run_dir}",
        "rows_kind=split",
        f"rows_path={pipeline.test_split}",
        f"output_path={out}",
        "device=cpu",
    ]
    cli.infer_main(argv)
    assert out.is_file()


def test_pie_eval_fails_fast_without_the_env_roots(
    pipeline: TrainedRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_common_env", _no_common_env)
    monkeypatch.delenv("PIE_CACHE_DIR")
    argv = [
        "experiment_name=tiny",
        f"run_dir={pipeline.run_dir}",
        f"split_path={pipeline.test_split}",
        "row_set=cli_noenv",
        "device=cpu",
    ]
    with pytest.raises(MissingEnvError, match="PIE_CACHE_DIR"):
        cli.eval_main(argv)


def test_pie_without_arguments_prints_the_command_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: pie <command>")
    for name in ("process", "prep", "sources", "train", "eval", "infer"):
        assert f"  {name}" in out


def test_pie_help_matches_no_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main([])
    bare = capsys.readouterr().out
    assert cli.main(["--help"]) == 0
    assert capsys.readouterr().out == bare


def test_pie_version(capsys: pytest.CaptureFixture[str]) -> None:
    from pie import __version__

    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == f"pie {__version__}\n"


def test_pie_unknown_command_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["trian"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'trian'" in err
    assert "usage: pie <command>" in err


@pytest.mark.parametrize("name", ["process", "prep", "sources", "train", "eval", "infer"])
def test_pie_command_help(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([name, "--help"]) == 0
    assert capsys.readouterr().out.startswith(f"usage: pie {name}")


def test_deprecated_alias_warns_and_delegates(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.train_alias(["--help"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("usage: pie train")
    assert "'pie-train' is deprecated; use 'pie train'" in captured.err


def test_console_scripts_are_registered() -> None:
    scripts = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["pie"] == "pie.cli:main"
    for name in ("process", "prep", "sources", "train", "eval", "infer"):
        assert scripts[f"pie-{name}"] == f"pie.cli:{name}_alias"
