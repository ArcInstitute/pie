"""Strict run configuration (training) and Hydra composition of configs/.

Every value lives once in the YAML files. These models only validate it; unknown keys are
errors.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from pydantic import field_validator

from pie.assets import parse_hf_reference, pin_asset_reference
from pie.data.datamodule import LEGACY_ALIASES_PATH, DataConfig
from pie.model.pie import ModelConfig
from pie.utils import StrictModel, compose_config, resolve_path, to_portable


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


def compose_train_config(overrides: Sequence[str]) -> TrainConfig:
    """Compose configs/train.yaml (an experiment overlay comes in via `experiment=<name>`)."""
    return TrainConfig.model_validate(compose_config("train", overrides))


def _map_paths(cfg: TrainConfig, convert: Callable[[str], str]) -> TrainConfig:
    data = cfg.data
    aliases = data.aliases_path
    new_data = data.model_copy(
        update={
            "preprocessed_dirs": [convert(p) for p in data.preprocessed_dirs],
            "split_dir": convert(data.split_dir),
            "source_dirs": {name: convert(p) for name, p in data.source_dirs.items()},
            "gene_text_dir": convert(data.gene_text_dir),
            # The legacy literal stays as written, so resume keeps it portable.
            "aliases_path": (
                aliases if aliases is None or aliases == LEGACY_ALIASES_PATH else convert(aliases)
            ),
        }
    )
    return cfg.model_copy(update={"run_dir": convert(cfg.run_dir), "data": new_data})


def _resolved(path: str) -> str:
    if path.startswith("hf://"):
        return path
    return str(resolve_path(path))


PIE_SPLITS_URI = "hf://datasets/arcinstitute/PIE_splits@396ab9563175ee887750c9eed7ccaea6f5fdbf50"
_LEGACY_SPLITS = "data/splits/"


def load_saved_train_config(raw: Mapping[str, Any]) -> TrainConfig:
    """A saved run config (config.yaml or a checkpoint) with repo-era paths moved to HF.

    Runs saved before splits moved to PIE_splits store `data/splits/<rest>`; the same files live
    at PIE_SPLITS_URI/<rest>, and the train.json sha256 in the data stats still guards them.
    """
    data = dict(raw["data"])
    split_dir = str(data["split_dir"])
    if split_dir.startswith(_LEGACY_SPLITS):
        data["split_dir"] = f"{PIE_SPLITS_URI}/{split_dir.removeprefix(_LEGACY_SPLITS)}"
    return TrainConfig.model_validate({**raw, "data": data})


def pinned_train_config(cfg: TrainConfig, *, previous: TrainConfig | None = None) -> TrainConfig:
    """Pin remote assets for saved configs; resume replays the previous run's commits."""
    prior = previous.data if previous is not None else None

    def pin(value: str, old: str | None = None) -> str:
        if value.startswith("hf://") and old is not None and old.startswith("hf://"):
            requested, saved = parse_hf_reference(value), parse_hf_reference(old)
            if (requested.repo_id, requested.subdir) != (saved.repo_id, saved.subdir):
                raise ValueError(f"resume HF asset {value} differs from the saved asset {old}")
            return pin_asset_reference(old)
        return pin_asset_reference(value)

    data = cfg.data
    dirs = [
        pin(p, prior.preprocessed_dirs[i] if prior and i < len(prior.preprocessed_dirs) else None)
        for i, p in enumerate(data.preprocessed_dirs)
    ]
    sources = {
        name: pin(p, prior.source_dirs.get(name) if prior else None)
        for name, p in data.source_dirs.items()
    }
    return cfg.model_copy(update={"data": data.model_copy(update={
        "preprocessed_dirs": dirs,
        "source_dirs": sources,
        "gene_text_dir": pin(data.gene_text_dir, prior.gene_text_dir if prior else None),
        "split_dir": pin(data.split_dir, prior.split_dir if prior else None),
    })})


def portable_train_config(cfg: TrainConfig) -> TrainConfig:
    """Copy with every path rewritten by pie.utils.to_portable ('${PIE_DATA_ROOT}/...')."""
    return _map_paths(cfg, to_portable)


def resolved_train_config(cfg: TrainConfig) -> TrainConfig:
    """Inverse of portable_train_config: every path through pie.utils.resolve_path."""
    return _map_paths(cfg, _resolved)


_ROW_SET = re.compile(r"[A-Za-z0-9_\-][A-Za-z0-9_.\-]*")


class EvalConfig(StrictModel):
    """pie-eval (configs/eval.yaml)."""

    experiment_name: str
    run_dir: str
    ckpt: Literal["best_auprc", "last"]
    split_path: str  # a split file, e.g. data/splits/replogle_xdataset/test_seen.json
    row_set: str  # output dir name under <run_dir>/eval/
    preprocessed_dirs: list[str] | None  # None = the checkpoint's training dirs
    save_predictions: bool
    overwrite: bool
    device: str
    batch_size: int

    @field_validator("row_set")
    @classmethod
    def check_row_set(cls, value: str) -> str:
        if not _ROW_SET.fullmatch(value):
            raise ValueError(f"row_set must be a single directory name, got {value!r}")
        return value


class InferConfig(StrictModel):
    """pie-infer (configs/infer.yaml)."""

    experiment_name: str
    run_dir: str
    ckpt: Literal["best_auprc", "last"]
    rows_kind: Literal["split", "query"]
    rows_path: str
    preprocessed_dirs: list[str] | None
    output_path: str  # predictions.parquet path
    overwrite: bool
    device: str
    batch_size: int


def compose_eval_config(overrides: Sequence[str]) -> EvalConfig:
    """Compose configs/eval.yaml (+overrides), resolve, validate."""
    return EvalConfig.model_validate(compose_config("eval", overrides))


def compose_infer_config(overrides: Sequence[str]) -> InferConfig:
    """Compose configs/infer.yaml (+overrides), resolve, validate."""
    return InferConfig.model_validate(compose_config("infer", overrides))
