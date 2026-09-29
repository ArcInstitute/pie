"""Strict run configuration (training) and Hydra composition of configs/.

Every value lives once in the YAML files. These models only validate it; unknown keys are
errors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from pie.data.datamodule import DataConfig
from pie.model.pie import ModelConfig
from pie.utils import REPO_ROOT, StrictModel, compose_config, resolve_path, to_portable

CONFIG_DIR: Path = REPO_ROOT / "configs"


class TrainerConfig(StrictModel):
    """trainer.*: Lightning Trainer settings (determinism and DDP are fixed in code)."""

    accelerator: str
    devices: int
    num_nodes: int
    # Only the precisions pie.predict replays exactly: bf16 autocast or full fp32.
    precision: Literal["bf16-mixed", "32-true"]
    max_steps: int
    accumulate_grad_batches: int
    val_every_n_steps: int  # optimizer steps; pie.train converts it to micro-batches
    gradient_clip_val: float
    log_every_n_steps: int


class OptimizerConfig(StrictModel):
    """optimizer.*: AdamW with a single parameter group."""

    lr: float
    weight_decay: float
    betas: tuple[float, float]


class SchedulerConfig(StrictModel):
    """scheduler.*: linear warmup, constant hold, then one cosine cycle down to eta_min."""

    warmup_fraction: float
    constant_steps: int
    eta_min: float

    def resolve_steps(self, max_steps: int) -> tuple[int, int, int]:
        """(warmup_steps, constant_steps, T_0) for a run of `max_steps` optimizer steps."""
        warmup_steps = max(1, int(max_steps * self.warmup_fraction))
        constant_steps = max(0, self.constant_steps)
        t_0 = max(1, max_steps - warmup_steps - constant_steps)
        return warmup_steps, constant_steps, t_0


class LoggerConfig(StrictModel):
    """logger.*: wandb on or off, plus run grouping. Entity and project come from the env."""

    enabled: bool
    group: str | None
    tags: list[str]


class TrainConfig(StrictModel):
    """The resolved configs/train.yaml plus an experiment overlay."""

    vars: dict[str, str]
    experiment_name: str
    seed: int
    run_dir: str
    overwrite: bool
    resume: bool
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainerConfig
    logger: LoggerConfig


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Resolve interpolations (a `???` left anywhere is an error) and drop the `hydra` node."""
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(container, dict):
        raise TypeError("a run config must be a mapping")
    container.pop("hydra", None)
    return {str(key): value for key, value in container.items()}


def _compose(config_name: str, overrides: Sequence[str]) -> dict[str, Any]:
    """Compose configs/<config_name>.yaml with Hydra overrides into plain resolved data."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=list(overrides))
        return config_to_dict(cfg)


def compose_train_config(overrides: Sequence[str]) -> TrainConfig:
    """Compose configs/train.yaml (an experiment overlay comes in via `experiment=<name>`)."""
    return TrainConfig.model_validate(compose_config("train", overrides))


def _map_paths(cfg: TrainConfig, convert: Callable[[str], str]) -> TrainConfig:
    data = cfg.data
    new_data = data.model_copy(
        update={
            "preprocessed_dirs": [convert(p) for p in data.preprocessed_dirs],
            "split_dir": convert(data.split_dir),
            "source_dirs": {name: convert(p) for name, p in data.source_dirs.items()},
            "gene_text_dir": convert(data.gene_text_dir),
            "aliases_path": convert(data.aliases_path),
        }
    )
    return cfg.model_copy(update={"run_dir": convert(cfg.run_dir), "data": new_data})


def _resolved(path: str) -> str:
    return str(resolve_path(path))


def portable_train_config(cfg: TrainConfig) -> TrainConfig:
    """Copy with every path rewritten by pie.utils.to_portable ('${PIE_DATA_ROOT}/...')."""
    return _map_paths(cfg, to_portable)


def resolved_train_config(cfg: TrainConfig) -> TrainConfig:
    """Inverse of portable_train_config: every path through pie.utils.resolve_path."""
    return _map_paths(cfg, _resolved)
