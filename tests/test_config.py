"""Tests for pie.config: the strict schema and Hydra composition of configs/train.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import BaseModel, ValidationError

from pie.config import (
    CONFIG_DIR,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
    compose_train_config,
    portable_train_config,
    resolved_train_config,
)
from pie.data.datamodule import DataConfig
from pie.data.evidence import EvidenceConfig
from pie.model.pie import EvidenceModelConfig, ModelConfig, TemperatureConfig
from pie.utils import REPO_ROOT
from tests.fixtures import TinyData, required_train_overrides, set_run_env


def _required(tiny: TinyData) -> list[str]:
    return required_train_overrides(
        tiny,
        experiment_name="cfg_test",
        devices=2,
        num_nodes=1,
        max_steps=5000,
        accumulate_grad_batches=4,
    )


def test_train_yaml_lives_in_the_config_dir() -> None:
    assert CONFIG_DIR == REPO_ROOT / "configs"
    assert (CONFIG_DIR / "train.yaml").is_file()


def test_train_yaml_has_no_hydra_node() -> None:
    assert "hydra" not in OmegaConf.load(CONFIG_DIR / "train.yaml")


def test_base_alone_is_incomplete(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    with pytest.raises(MissingMandatoryValue):
        compose_train_config([])


def test_required_keys_compose_to_the_shared_recipe(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    cfg = compose_train_config(_required(tiny_data))
    assert cfg.vars == {}
    assert cfg.experiment_name == "cfg_test"
    assert cfg.seed == 42
    assert cfg.run_dir == f"{tmp_path / 'runs'}/cfg_test"
    assert (cfg.overwrite, cfg.resume) == (False, False)
    assert cfg.data.preprocessed_dirs == [
        str(tiny_data.preprocessed["alpha"]),
        str(tiny_data.preprocessed["beta"]),
    ]
    assert cfg.data.dataset_weights is None
    assert list(cfg.data.source_dirs) == [
        "esm2", "ncbi_text", "context_text", "perturbation_text", "smiles",
    ]
    assert cfg.data.gene_text_dir == f"{tiny_data.root}/sources/gene_text"
    assert cfg.data.aliases_path == "data/sources/aliases.yaml"
    assert cfg.data.delta_p.bin_width_fold_change == 1.5
    assert cfg.data.delta_p.max_delta_percentile == 99.9
    assert cfg.data.delta_p.max_delta is None
    assert cfg.data.evidence == EvidenceConfig(seed=0, chunk=2, lfc_clip_percentile=95.0)
    assert (cfg.data.batch_size, cfg.data.num_workers, cfg.data.fdr_threshold) == (16, 4, 0.05)
    assert cfg.model == ModelConfig.model_validate(
        {
            "d_model": 768,
            "n_latents": 512,
            "n_encoder_layers": 4,
            "n_processor_layers": 6,
            "n_decoder_layers": 4,
            "num_heads": 8,
            "ff_mult": 2,
            "dropout": 0.1,
            "drop_src": 0.05,
            "inference_chunk_size": 2048,
            "class_weight_cap": 10.0,
            "class_weight_cap_delta_p": 5.0,
            "lfc_huber_delta": 1.0,
            "lfc_target_gene_alpha": 0.001,
            "lfc_direction_temperature": 0.25,
            "evidence": {"encoder_dim": 64, "dim": 128, "dropout": 0.1, "response_dropout": 0.25},
            "temperature": {"de": 4.0, "delta_p": 4.5},
        }
    )
    assert cfg.optimizer == OptimizerConfig(lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
    assert cfg.scheduler == SchedulerConfig(warmup_fraction=0.05, constant_steps=4250, eta_min=1e-6)
    assert cfg.trainer == TrainerConfig(
        accelerator="gpu",
        devices=2,
        num_nodes=1,
        precision="bf16-mixed",
        max_steps=5000,
        accumulate_grad_batches=4,
        val_every_n_steps=100,
        gradient_clip_val=10.0,
        log_every_n_steps=50,
    )
    assert cfg.logger == LoggerConfig(enabled=True, group=None, tags=[])


@pytest.mark.parametrize("precision", ["bf16-mixed", "32-true"])
def test_trainer_precision_accepts_what_prediction_supports(precision: str) -> None:
    fields = _trainer_fields()
    assert TrainerConfig(**{**fields, "precision": precision}).precision == precision


@pytest.mark.parametrize("precision", ["16-mixed", "bf16-true", "16-true", "64-true", "32"])
def test_trainer_precision_rejects_what_prediction_cannot_replay(precision: str) -> None:
    with pytest.raises(ValidationError, match="precision"):
        TrainerConfig(**{**_trainer_fields(), "precision": precision})


def _trainer_fields() -> dict[str, object]:
    return {
        "accelerator": "cpu",
        "devices": 1,
        "num_nodes": 1,
        "precision": "32-true",
        "max_steps": 1,
        "accumulate_grad_batches": 1,
        "val_every_n_steps": 1,
        "gradient_clip_val": 1.0,
        "log_every_n_steps": 1,
    }


@pytest.mark.parametrize(
    "override",
    [
        "+bogus=1",
        "+data.bogus=1",
        "+data.delta_p.bogus=1",
        "+model.bogus=1",
        "+model.evidence.bogus=1",
        "+trainer.bogus=1",
        "+optimizer.bogus=1",
        "+scheduler.bogus=1",
        "+logger.bogus=1",
    ],
)
def test_unknown_keys_are_rejected(
    override: str, tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    with pytest.raises(ValidationError, match="bogus"):
        compose_train_config([*_required(tiny_data), override])


@pytest.mark.parametrize(
    "model",
    [
        TrainConfig,
        TrainerConfig,
        OptimizerConfig,
        SchedulerConfig,
        LoggerConfig,
        DataConfig,
        EvidenceConfig,
        ModelConfig,
        EvidenceModelConfig,
        TemperatureConfig,
    ],
)
def test_schema_holds_no_defaults(model: type[BaseModel]) -> None:
    """Every value lives once in YAML; only data.delta_p (DeltaPConfig) has pydantic defaults."""
    assert [name for name, field in model.model_fields.items() if not field.is_required()] == []


def test_portable_paths_round_trip(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    cfg = compose_train_config(_required(tiny_data))
    portable = portable_train_config(cfg)
    assert portable.run_dir == "${PIE_RUNS_ROOT}/cfg_test"
    assert portable.data.preprocessed_dirs == [
        "${PIE_DATA_ROOT}/preprocessed/alpha",
        "${PIE_DATA_ROOT}/preprocessed/beta",
    ]
    assert portable.data.split_dir == "${PIE_DATA_ROOT}/splits"
    assert portable.data.source_dirs["esm2"] == "${PIE_DATA_ROOT}/sources/esm2"
    assert portable.data.gene_text_dir == "${PIE_DATA_ROOT}/sources/gene_text"
    assert portable.data.aliases_path == "data/sources/aliases.yaml"
    back = resolved_train_config(portable).model_dump()
    assert back["data"]["aliases_path"] == str(REPO_ROOT / "data/sources/aliases.yaml")
    back["data"]["aliases_path"] = cfg.data.aliases_path
    assert back == cfg.model_dump()


def test_resolve_steps_matches_both_recipes() -> None:
    sched = SchedulerConfig(warmup_fraction=0.05, constant_steps=4250, eta_min=1e-6)
    assert sched.resolve_steps(5000) == (250, 4250, 500)
    assert sched.resolve_steps(10000) == (500, 4250, 5250)
    assert sched.resolve_steps(10) == (1, 4250, 1)
