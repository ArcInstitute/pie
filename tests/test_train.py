"""Tests for pie.train."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from pie import cli, train
from pie.config import (
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
    compose_train_config,
)
from pie.data.datamodule import DATA_STATS, DataStats
from pie.utils import MissingEnvError, read_json
from tests.fixtures import TinyData, build_tiny_data, set_run_env, train_overrides


@pytest.fixture(autouse=True)
def _wandb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_train checks the wandb env up front; tests that need it unset delete it."""
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")


SPEC_KEYS = {
    "train/loss", "train/de_loss", "train/lfc_loss", "train/lfc_dir_loss", "train/dp_loss",
    "train/grad_norm", "val/loss", "val/de_loss", "val/lfc_loss", "val/lfc_dir_loss",
    "val/dp_loss", "val/binary_auprc", "val/sig_jaccard", "val/direction_match",
    "val/spearman_nsig", "val/spearman_lfc", "val/discrimination_score_l1", "lr-AdamW",
}


def _trainer_cfg(val_every_n_steps: int, accumulate_grad_batches: int) -> TrainerConfig:
    return TrainerConfig(
        accelerator="gpu",
        devices=2,
        num_nodes=1,
        precision="bf16-mixed",
        max_steps=5000,
        accumulate_grad_batches=accumulate_grad_batches,
        val_every_n_steps=val_every_n_steps,
        gradient_clip_val=10.0,
        log_every_n_steps=50,
    )


@pytest.mark.parametrize(("every", "accumulate", "expected"), [(100, 4, 400), (100, 1, 100)])
def test_val_check_interval_counts_micro_batches(
    every: int, accumulate: int, expected: int
) -> None:
    assert train.val_check_interval(_trainer_cfg(every, accumulate)) == expected


def _lrs(sched: SchedulerConfig, max_steps: int, lr: float = 1e-4) -> list[float]:
    param = torch.nn.Parameter(torch.zeros(1))
    opt = OptimizerConfig(lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
    optimizer, scheduler = train.build_optimizer([param], opt, sched, max_steps)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)
    lrs: list[float] = []
    for _ in range(max_steps):
        lrs.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    return lrs


def test_schedule_warmup_constant_cosine() -> None:
    sched = SchedulerConfig(warmup_fraction=0.25, constant_steps=5, eta_min=1e-6)
    assert sched.resolve_steps(20) == (5, 5, 10)
    lrs = _lrs(sched, 20)
    base, eta = 1e-4, 1e-6
    start = 1e-8 / base
    for i in range(5):
        assert lrs[i] == pytest.approx(base * (start + (1 - start) * i / 5), rel=1e-6)
    assert lrs[5:10] == pytest.approx([base] * 5, rel=1e-12)
    for i in range(10, 20):
        cosine = eta + (base - eta) * (1 + math.cos(math.pi * (i - 10) / 10)) / 2
        assert lrs[i] == pytest.approx(cosine, rel=1e-6)


def test_schedule_without_constant_phase() -> None:
    sched = SchedulerConfig(warmup_fraction=0.25, constant_steps=0, eta_min=1e-6)
    assert sched.resolve_steps(20) == (5, 0, 15)
    lrs = _lrs(sched, 20)
    assert lrs[0] == pytest.approx(1e-8, rel=1e-6)
    assert lrs[5] == pytest.approx(1e-4, rel=1e-12)
    last = 1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * 14 / 15)) / 2
    assert lrs[19] == pytest.approx(last, rel=1e-6)


def test_grad_total_norm_reads_without_writing() -> None:
    a = torch.nn.Parameter(torch.zeros(2))
    b = torch.nn.Parameter(torch.zeros(3))
    c = torch.nn.Parameter(torch.zeros(1))
    a.grad = torch.tensor([3.0, 0.0])
    b.grad = torch.tensor([0.0, 4.0, 12.0])
    before = (a.grad.clone(), b.grad.clone())
    norm = train.grad_total_norm([a, b, c])
    assert norm.item() == pytest.approx(13.0)
    assert torch.equal(a.grad, before[0])
    assert torch.equal(b.grad, before[1])
    assert c.grad is None


def test_wandb_keys_are_the_spec_list() -> None:
    assert set(train.WANDB_KEYS) == SPEC_KEYS
    noisy = dict.fromkeys([*SPEC_KEYS, "epoch", "val/raw_loss", "diagnostics/tested_frac"], 0.0)
    assert set(train.filter_wandb_metrics(noisy)) == SPEC_KEYS | {"epoch"}
    assert train.filter_wandb_metrics({"epoch": 0.0}) == {}


def test_pie_wandb_logger_filters_metrics_and_skips_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[dict[str, float], int | None]] = []

    def _record(self: WandbLogger, metrics: dict[str, float], step: int | None = None) -> None:
        seen.append((dict(metrics), step))

    monkeypatch.setattr(WandbLogger, "log_metrics", _record)
    logger = train.PieWandbLogger(project="p", offline=True, save_dir=str(tmp_path))
    logger.log_metrics({"train/loss": 1.0, "epoch": 0.0, "diagnostics/x": 2.0}, step=3)
    logger.log_metrics({"epoch": 1.0}, step=4)
    assert seen == [({"train/loss": 1.0, "epoch": 0.0}, 3)]
    logger.log_hyperparams({"lr": 1.0})
    assert logger._experiment is None


def test_fresh_and_empty_run_dirs_are_accepted(tmp_path: Path) -> None:
    run = tmp_path / "a" / "run"
    train.prepare_run_dir(run, resume=False, overwrite=False)
    assert run.is_dir()
    train.prepare_run_dir(run, resume=False, overwrite=False)
    assert list(run.iterdir()) == []


def test_non_empty_run_dir_needs_a_flag(tmp_path: Path) -> None:
    (tmp_path / "old.txt").write_text("x")
    with pytest.raises(FileExistsError, match="overwrite=true"):
        train.prepare_run_dir(tmp_path, resume=False, overwrite=False)
    assert (tmp_path / "old.txt").read_text() == "x"


def test_overwrite_clears_the_run_dir(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "sub").mkdir(parents=True)
    (run / "last.ckpt").write_text("x")
    train.prepare_run_dir(run, resume=False, overwrite=True)
    assert run.is_dir()
    assert list(run.iterdir()) == []


def test_resume_needs_last_ckpt_and_data_stats(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"last\.ckpt"):
        train.prepare_run_dir(tmp_path, resume=True, overwrite=False)
    (tmp_path / "last.ckpt").write_text("x")
    with pytest.raises(FileNotFoundError, match=r"data_stats\.json"):
        train.prepare_run_dir(tmp_path, resume=True, overwrite=False)
    (tmp_path / "data_stats.json").write_text("{}")
    train.prepare_run_dir(tmp_path, resume=True, overwrite=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data_stats.json", "last.ckpt"]


def test_resume_and_overwrite_are_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        train.prepare_run_dir(tmp_path, resume=True, overwrite=True)


def _part(rows: list[int], contexts: list[str]) -> train._ValRows:
    n, g = len(rows), 2
    base = np.asarray(rows, dtype=np.float32)[:, None] * np.ones((1, g), dtype=np.float32)
    return train._ValRows(
        dir_index=0,
        row_index=np.asarray(rows, dtype=np.int64),
        contexts=contexts,
        perts=[f"p{r}" for r in rows],
        p_de=base,
        lfc_pred=base + 0.5,
        delta_p_pred=base - 0.5,
        de_true=np.ones((n, g), dtype=bool),
        tested=np.ones((n, g), dtype=bool),
        lfc_true=base.astype(np.float64),
        delta_p_true=base,
        ctrl_means=np.ones((n, g), dtype=np.float32),
    )


def test_scoring_inputs_drop_repeated_rows_and_follow_row_order() -> None:
    parts = [_part([2, 0], ["c2", "c0"]), _part([0, 1], ["c0", "c1"])]
    got = train._scoring_inputs(parts, "alpha", ["GA", "GB"])
    assert got.dataset == "alpha"
    assert got.genes == ["GA", "GB"]
    assert got.contexts == ["c0", "c1", "c2"]
    assert got.perts == ["p0", "p1", "p2"]
    np.testing.assert_array_equal(got.p_de[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(got.lfc_pred[:, 1], [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(got.lfc_true[:, 1], [0.0, 1.0, 2.0])
    assert got.lfc_true.dtype == np.float64
    assert got.de_true.dtype == np.bool_
    assert got.ctrl_means.shape == (3, 2)


class _CapturingLogger(Logger):
    """Records every metric key Lightning (or LearningRateMonitor) sends to a logger."""

    def __init__(self) -> None:
        super().__init__()
        self.keys: set[str] = set()

    @property
    def name(self) -> str:
        return "capture"

    @property
    def version(self) -> str:
        return "0"

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self.keys.update(metrics)

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        return None


class _Capture:
    """Stand-in for pie.train._make_logger that records the run ids it is given."""

    def __init__(self) -> None:
        self.run_ids: list[str | None] = []
        self.loggers: list[_CapturingLogger] = []

    def make(self, cfg: TrainConfig, run_dir: Path, run_id: str | None) -> Logger:
        self.run_ids.append(run_id)
        logger = _CapturingLogger()
        self.loggers.append(logger)
        return logger


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def test_build_trainer_ddp_cadence_and_callbacks(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    cfg = compose_train_config(
        train_overrides(
            tiny_data, devices=2, accumulate_grad_batches=4, val_every_n_steps=100, max_steps=5000
        )
    )
    trainer = train.build_trainer(cfg, tmp_path / "run", None)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert trainer.strategy._ddp_kwargs == {"find_unused_parameters": True}
    assert trainer.val_check_interval == 400
    assert trainer.check_val_every_n_epoch is None
    assert trainer.accumulate_grad_batches == 4
    assert trainer.max_steps == 5000
    assert trainer.gradient_clip_val == 10.0
    assert torch.are_deterministic_algorithms_enabled()
    checkpoints = {cb.filename: cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)}
    assert sorted(checkpoints) == ["best_auprc", "last"]
    assert (checkpoints["best_auprc"].monitor, checkpoints["best_auprc"].mode) == (
        train.MONITOR,
        "max",
    )
    assert checkpoints["last"].monitor is None
    assert not any(isinstance(cb, LearningRateMonitor) for cb in trainer.callbacks)


def test_build_trainer_single_device_with_logger(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    cfg = compose_train_config(train_overrides(tiny_data))
    logger = _CapturingLogger()
    trainer = train.build_trainer(cfg, tmp_path / "run", logger)
    assert not isinstance(trainer.strategy, DDPStrategy)
    assert trainer.val_check_interval == 1
    assert trainer.logger is logger
    assert any(isinstance(cb, LearningRateMonitor) for cb in trainer.callbacks)


def test_run_train_checks_wandb_env_before_touching_the_run_dir(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    cfg = compose_train_config(train_overrides(tiny_data, logger=True, extra=("overwrite=true",)))
    run_dir = train.resolve_path(cfg.run_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "keep.txt").write_text("old run")

    def no_stats(*_a: object, **_k: object) -> None:
        raise AssertionError("setup_stats ran before the wandb env check")

    monkeypatch.setattr(train.PieDataModule, "setup_stats", no_stats)
    with pytest.raises(MissingEnvError, match="WANDB_PROJECT"):
        train.run_train(cfg)
    assert (run_dir / "keep.txt").read_text() == "old run"


def test_make_logger_reads_wandb_env_and_run_name(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    cfg = compose_train_config(train_overrides(tiny_data, logger=True))
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    with pytest.raises(MissingEnvError, match="WANDB_ENTITY"):
        train._make_logger(cfg, tmp_path, "abcd1234")
    monkeypatch.setenv("WANDB_ENTITY", "ent")
    monkeypatch.setenv("WANDB_PROJECT", "proj")
    logger = train._make_logger(cfg, tmp_path, "abcd1234")
    assert isinstance(logger, train.PieWandbLogger)
    init = logger._wandb_init
    assert (init["name"], init["project"], init["entity"], init["id"]) == (
        "tiny_run",
        "proj",
        "ent",
        "abcd1234",
    )
    assert logger._experiment is None


@dataclass
class _Fitted:
    tiny: TinyData
    runs: Path
    run_dir: Path
    capture: _Capture


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> _Fitted:
    """One 2-step CPU fit on the tiny data (validation after each step, capturing logger)."""
    root = tmp_path_factory.mktemp("fit")
    tiny = build_tiny_data(root / "tiny")
    capture = _Capture()
    with pytest.MonkeyPatch.context() as mp:
        set_run_env(mp, tiny, root / "runs")
        mp.setenv("WANDB_ENTITY", "test-entity")
        mp.setenv("WANDB_PROJECT", "test-project")
        mp.setattr(train, "_make_logger", capture.make)
        run_dir = train.run_train(compose_train_config(train_overrides(tiny, logger=True)))
    return _Fitted(tiny=tiny, runs=root / "runs", run_dir=run_dir, capture=capture)


def test_fit_writes_checkpoints_and_run_files(fitted: _Fitted) -> None:
    assert fitted.run_dir == fitted.runs / "tiny_run"
    for name in (train.CKPT_BEST, train.CKPT_LAST, DATA_STATS, train.CONFIG_FILE):
        assert (fitted.run_dir / name).is_file(), name
    assert fitted.capture.run_ids == [(fitted.run_dir / train.WANDB_ID_FILE).read_text().strip()]
    assert _load(fitted.run_dir / train.CKPT_LAST)["global_step"] == 2
    assert _load(fitted.run_dir / train.CKPT_BEST)["global_step"] in (1, 2)


def test_checkpoints_embed_portable_config_and_data_stats(fitted: _Fitted) -> None:
    handoff = read_json(fitted.run_dir / DATA_STATS)
    assert isinstance(handoff, dict)
    stats = DataStats.model_validate(handoff["stats"])
    for name in (train.CKPT_BEST, train.CKPT_LAST):
        ckpt = _load(fitted.run_dir / name)
        assert "hyper_parameters" not in ckpt
        block = ckpt["pie"]
        assert block["format_version"] == train.CHECKPOINT_FORMAT
        assert DataStats.model_validate(block["data_stats"]) == stats
        cfg = TrainConfig.model_validate(block["config"])
        assert cfg.run_dir == "${PIE_RUNS_ROOT}/tiny_run"
        assert cfg.data.preprocessed_dirs == [
            "${PIE_DATA_ROOT}/preprocessed/alpha",
            "${PIE_DATA_ROOT}/preprocessed/beta",
        ]
        assert cfg.data.gene_text_dir == "${PIE_DATA_ROOT}/sources/gene_text"
        keys = list(ckpt["state_dict"])
        assert keys and all(k.startswith("model.") for k in keys)
        assert not any("gene_query_text" in k for k in keys)


def test_config_yaml_holds_portable_paths(fitted: _Fitted) -> None:
    text = (fitted.run_dir / train.CONFIG_FILE).read_text()
    assert "${PIE_RUNS_ROOT}/tiny_run" in text
    assert "${PIE_DATA_ROOT}/preprocessed/alpha" in text
    assert str(fitted.runs) not in text
    assert str(fitted.tiny.root) not in text


def test_logged_keys_equal_the_spec_list(fitted: _Fitted) -> None:
    keys = fitted.capture.loggers[0].keys
    assert keys - {"epoch"} == set(train.WANDB_KEYS)
    kept = set(train.filter_wandb_metrics(dict.fromkeys(keys, 0.0)))
    assert kept == set(train.WANDB_KEYS) | {"epoch"}


def test_rerun_into_a_non_empty_run_dir_is_refused(
    fitted: _Fitted, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_run_env(monkeypatch, fitted.tiny, fitted.runs)
    with pytest.raises(FileExistsError, match="overwrite=true"):
        train.run_train(compose_train_config(train_overrides(fitted.tiny)))


def test_resume_continues_from_last_and_reuses_the_run_id(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    capture = _Capture()
    monkeypatch.setattr(train, "_make_logger", capture.make)
    run_dir = train.run_train(compose_train_config(train_overrides(tiny_data, logger=True)))
    first_id = (run_dir / train.WANDB_ID_FILE).read_text().strip()
    stats_bytes = (run_dir / DATA_STATS).read_bytes()
    resumed = train_overrides(
        tiny_data, logger=True, max_steps=3, val_every_n_steps=2, extra=("resume=true",)
    )
    assert train.run_train(compose_train_config(resumed)) == run_dir
    assert _load(run_dir / train.CKPT_LAST)["global_step"] == 3
    assert capture.run_ids == [first_id, first_id]
    assert (run_dir / DATA_STATS).read_bytes() == stats_bytes


def _no_common_env(path: Path | None = None) -> dict[str, str]:
    return {}


def test_train_main_composes_and_runs(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    seen: list[TrainConfig] = []

    def _record(cfg: TrainConfig) -> Path:
        seen.append(cfg)
        return Path(cfg.run_dir)

    monkeypatch.setattr(cli, "load_common_env", _no_common_env)
    monkeypatch.setattr(train, "run_train", _record)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert cli.train_main(train_overrides(tiny_data, experiment_name="cli_run")) == 0
    assert [cfg.experiment_name for cfg in seen] == ["cli_run"]
    assert seen[0].run_dir == str(tmp_path / "runs" / "cli_run")
    assert not (tmp_path / "runs" / "cli_run").exists()
    assert list(work.iterdir()) == []


def test_train_main_reads_sys_argv(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Lightning's DDP subprocess launcher re-executes `pie-train <same argv>` for rank 1.
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    seen: list[str] = []
    monkeypatch.setattr(cli, "load_common_env", _no_common_env)
    monkeypatch.setattr(train, "run_train", lambda cfg: seen.append(cfg.experiment_name))
    argv = ["pie-train", *train_overrides(tiny_data, experiment_name="argv")]
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.train_main() == 0
    assert seen == ["argv"]


def test_train_main_help_needs_no_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("PIE_DATA_ROOT", "PIE_RUNS_ROOT", "PIE_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    assert cli.train_main(["--help"]) == 0
    assert capsys.readouterr().out.startswith("usage: pie-train")


def test_train_main_fails_fast_on_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PIE_DATA_ROOT", "PIE_RUNS_ROOT", "PIE_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "load_common_env", _no_common_env)
    with pytest.raises(MissingEnvError):
        cli.train_main([])
