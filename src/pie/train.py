"""Training: wandb key policy, schedule, run-dir policy, the Lightning module and run_train."""

from __future__ import annotations

import logging
import shutil
import uuid
from argparse import Namespace
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR

from pie.config import (
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
    portable_train_config,
)
from pie.data.datamodule import DATA_STATS, DataStats, PieDataModule
from pie.data.dataset import Batch
from pie.metrics import METRIC_KEYS, ScoringInputs, score
from pie.model import LOSS_KEYS, GroupOutput, PieModel, compute_losses, readout
from pie.utils import (
    atomic_write_text,
    configure_determinism,
    env_global_rank,
    read_json,
    require_env,
    resolve_path,
)

log = logging.getLogger(__name__)

CKPT_BEST = "best_auprc.ckpt"
CKPT_LAST = "last.ckpt"
CONFIG_FILE = "config.yaml"
WANDB_ID_FILE = "wandb_run_id"
MONITOR = "val/binary_auprc"
CHECKPOINT_FORMAT = 1
WANDB_KEYS: frozenset[str] = frozenset(
    {
        "train/loss",
        "train/de_loss",
        "train/lfc_loss",
        "train/lfc_dir_loss",
        "train/dp_loss",
        "train/grad_norm",
        "val/loss",
        "val/de_loss",
        "val/lfc_loss",
        "val/lfc_dir_loss",
        "val/dp_loss",
        "val/binary_auprc",
        "val/sig_jaccard",
        "val/direction_match",
        "val/spearman_nsig",
        "val/spearman_lfc",
        "val/discrimination_score_l1",
        "lr-AdamW",
    }
)


def filter_wandb_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only WANDB_KEYS; pass Lightning's `epoch` through alongside a kept key.

    A dict holding only `epoch` (no WANDB_KEYS key) still filters to empty.
    """
    kept = {key: value for key, value in metrics.items() if key in WANDB_KEYS}
    if kept and "epoch" in metrics:
        kept["epoch"] = metrics["epoch"]
    return kept


class PieWandbLogger(WandbLogger):
    """WandbLogger that logs only WANDB_KEYS and never uploads a config."""

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        kept = filter_wandb_metrics(metrics)
        if kept:
            super().log_metrics(kept, step)

    def log_hyperparams(
        self, params: dict[str, Any] | Namespace, *args: Any, **kwargs: Any
    ) -> None:
        """No config upload."""


def val_check_interval(cfg: TrainerConfig) -> int:
    """trainer.val_every_n_steps counts optimizer steps; Lightning's int interval counts batches."""
    return cfg.val_every_n_steps * cfg.accumulate_grad_batches


def build_optimizer(
    params: Iterable[nn.Parameter],
    opt: OptimizerConfig,
    sched: SchedulerConfig,
    max_steps: int,
) -> tuple[AdamW, SequentialLR]:
    """AdamW (one group) + SequentialLR[LinearLR, ConstantLR, CosineAnnealingWarmRestarts].

    Scheduler creation order (warmup, cosine, constant) is the original's and must not change.
    """
    optimizer = AdamW(
        [{"params": list(params), "weight_decay": opt.weight_decay}], lr=opt.lr, betas=opt.betas
    )
    warmup_steps, constant_steps, t_0 = sched.resolve_steps(max_steps)
    warmup = LinearLR(
        optimizer,
        start_factor=1e-8 / max(opt.lr, 1e-10),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=1, eta_min=sched.eta_min)
    if constant_steps > 0:
        constant = ConstantLR(optimizer, factor=1.0, total_iters=constant_steps)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, constant, cosine],
            milestones=[warmup_steps, warmup_steps + constant_steps],
        )
    else:
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    return optimizer, scheduler


def grad_total_norm(params: Iterable[nn.Parameter]) -> Tensor:
    """2-norm over every present gradient; reads gradients only (never rescales them)."""
    grads = [g for p in params if (g := p.grad) is not None]
    if not grads:
        return torch.zeros(())
    return torch.nn.utils.get_total_norm(grads, norm_type=2.0)


def prepare_run_dir(run_dir: Path, *, resume: bool, overwrite: bool) -> None:
    """Rank-0 run-dir policy.

    Fresh or empty dir: used as is. Non-empty dir: an error unless overwrite=true (clears it).
    resume=true: needs last.ckpt and data_stats.json and leaves the dir untouched.
    """
    if resume and overwrite:
        raise ValueError("resume=true and overwrite=true are mutually exclusive")
    if resume:
        for name in (CKPT_LAST, DATA_STATS):
            if not (run_dir / name).is_file():
                raise FileNotFoundError(f"resume=true but {run_dir / name} does not exist")
        return
    if run_dir.is_dir() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{run_dir} is not empty; set overwrite=true to clear it or resume=true to "
                f"continue from {CKPT_LAST}"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


_VAL_ARRAYS: tuple[str, ...] = (
    "p_de",
    "lfc_pred",
    "delta_p_pred",
    "de_true",
    "tested",
    "lfc_true",
    "delta_p_true",
    "ctrl_means",
)


@dataclass
class _ValRows:
    """Readout and truth of one gene group of one validation batch (the dir's local gene axis)."""

    dir_index: int
    row_index: np.ndarray  # (b,) int64 row in the dir
    contexts: list[str]
    perts: list[str]
    p_de: np.ndarray  # (b, G_d) float32
    lfc_pred: np.ndarray  # (b, G_d) float32
    delta_p_pred: np.ndarray  # (b, G_d) float32
    de_true: np.ndarray  # (b, G_d) bool
    tested: np.ndarray  # (b, G_d) bool
    lfc_true: np.ndarray  # (b, G_d) float64
    delta_p_true: np.ndarray  # (b, G_d) float32
    ctrl_means: np.ndarray  # (b, G_d) float32


def _scoring_inputs(parts: list[_ValRows], dataset: str, genes: Sequence[str]) -> ScoringInputs:
    """One dir's chunks -> ScoringInputs: repeated rows (sampler padding) dropped, row order."""
    row_index = np.concatenate([p.row_index for p in parts])
    order = np.argsort(row_index, kind="stable")
    _, first = np.unique(row_index[order], return_index=True)
    keep = order[first]
    picked = keep.tolist()
    arrays = {
        name: np.concatenate([getattr(p, name) for p in parts])[keep] for name in _VAL_ARRAYS
    }
    contexts = [c for p in parts for c in p.contexts]
    perts = [q for p in parts for q in p.perts]
    return ScoringInputs(
        dataset=dataset,
        genes=list(genes),
        contexts=[contexts[i] for i in picked],
        perts=[perts[i] for i in picked],
        p_de=arrays["p_de"],
        lfc_pred=arrays["lfc_pred"],
        delta_p_pred=arrays["delta_p_pred"],
        de_true=arrays["de_true"],
        tested=arrays["tested"],
        lfc_true=arrays["lfc_true"],
        delta_p_true=arrays["delta_p_true"],
        ctrl_means=arrays["ctrl_means"],
    )


def _numpy(t: Tensor, dtype: type[np.generic]) -> np.ndarray:
    x = t.detach().cpu()
    if x.dtype == torch.bfloat16:
        x = x.float()
    return x.numpy().astype(dtype, copy=False)


def _require(t: Tensor | None, name: str) -> Tensor:
    if t is None:
        raise ValueError(f"validation batch has no {name}")
    return t


def _score_rows(rows: list[_ValRows], dm: PieDataModule) -> list[float]:
    """The 6 validation metrics (METRIC_KEYS order) from the collected validation rows."""
    by_dir: dict[int, list[_ValRows]] = {}
    for part in rows:
        by_dir.setdefault(part.dir_index, []).append(part)
    inputs = [
        _scoring_inputs(by_dir[i], dm.dirs[i].dataset, dm.dirs[i].genes) for i in sorted(by_dir)
    ]
    result = score(inputs)
    return [float(result.aggregate[key]) for key in METRIC_KEYS]


class PieLightningModule(L.LightningModule):
    """PieModel plus losses, rank-0 validation scoring, the optimizer and checkpoint contents."""

    def __init__(
        self,
        cfg: TrainConfig,
        stats: DataStats,
        source_dims: dict[str, int],
        gene_query_text: Tensor,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stats = stats
        self.model = PieModel(cfg.model, stats, source_dims, gene_query_text)
        self._val_rows: list[_ValRows] = []

    def _datamodule(self) -> PieDataModule:
        dm = getattr(self.trainer, "datamodule", None)
        if not isinstance(dm, PieDataModule):
            raise TypeError("PieLightningModule must be fitted with a PieDataModule")
        return dm

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        # Token sources arrive in their on-disk half dtype. Mixed precision autocasts them; a
        # full-precision run casts them to the parameter dtype (bf16-mixed runs are untouched).
        if "mixed" in str(self.trainer.precision):
            return batch
        dtype = self.model.latents.dtype
        batch.source_tokens = {
            name: t.to(dtype) if t.is_floating_point() and t.dtype != dtype else t
            for name, t in batch.source_tokens.items()
        }
        return batch

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        outputs = self.model(batch)
        losses = compute_losses(
            outputs, batch, self.cfg.model, self.stats.delta_p, temperatures=False
        )
        size = len(batch.ctx_names)
        for key in LOSS_KEYS:
            self.log(f"train/{key}", losses[key], prog_bar=key == "loss", batch_size=size)
        return losses["loss"]

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        # Full accumulated gradient, before Lightning clips it (clipping runs after this hook).
        self.log("train/grad_norm", grad_total_norm(self.parameters()))

    def on_validation_epoch_start(self) -> None:
        self._val_rows = []

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        outputs = self.model(batch)
        losses = compute_losses(
            outputs, batch, self.cfg.model, self.stats.delta_p, temperatures=True
        )
        size = len(batch.ctx_names)
        for key in LOSS_KEYS:
            self.log(
                f"val/{key}", losses[key], prog_bar=key == "loss", sync_dist=True, batch_size=size
            )
        if self._datamodule().val_sharded or self.trainer.is_global_zero:
            self._collect(batch, outputs)

    def _collect(self, batch: Batch, outputs: list[GroupOutput]) -> None:
        grid = self.stats.delta_p
        row_index = batch.row_index.detach().cpu().numpy()
        for out, group in zip(outputs, batch.groups, strict=True):
            pred = readout(out, self.cfg.model, grid)
            pos = group.rows.detach().cpu().numpy()
            self._val_rows.append(
                _ValRows(
                    dir_index=group.dir_index,
                    row_index=row_index[pos],
                    contexts=[batch.ctx_names[i] for i in pos.tolist()],
                    perts=[batch.pert_names[i] for i in pos.tolist()],
                    p_de=_numpy(pred.p_de, np.float32),
                    lfc_pred=_numpy(pred.lfc, np.float32),
                    delta_p_pred=_numpy(pred.delta_p, np.float32),
                    de_true=_numpy(_require(group.de_mask, "de_mask"), np.bool_),
                    tested=_numpy(_require(group.tested, "tested"), np.bool_),
                    lfc_true=_numpy(_require(group.lfc_true, "lfc_true"), np.float64),
                    delta_p_true=_numpy(_require(group.delta_p, "delta_p"), np.float32),
                    ctrl_means=_numpy(group.ctrl_means, np.float32),
                )
            )

    def on_validation_epoch_end(self) -> None:
        dm = self._datamodule()
        rows, self._val_rows = self._val_rows, []
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if dm.val_sharded and distributed:
            gathered: list[list[_ValRows] | None] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = [part for shard in gathered if shard is not None for part in shard]
        values: list[float] | None = _score_rows(rows, dm) if self.trainer.is_global_zero else None
        values = self.trainer.strategy.broadcast(values, src=0)
        assert values is not None
        for key, value in zip(METRIC_KEYS, values, strict=True):
            self.log(f"val/{key}", value, sync_dist=False)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        optimizer, scheduler = build_optimizer(
            self.parameters(), self.cfg.optimizer, self.cfg.scheduler, self.trainer.max_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["pie"] = {
            "format_version": CHECKPOINT_FORMAT,
            "config": portable_train_config(self.cfg).model_dump(mode="json"),
            "data_stats": self.stats.model_dump(mode="json"),
        }


class _LastCheckpoint(ModelCheckpoint):
    """last.ckpt after every validation, and once more at the end of training if not yet saved."""

    def __init__(self, dirpath: Path) -> None:
        super().__init__(
            dirpath=str(dirpath),
            filename=Path(CKPT_LAST).stem,
            monitor=None,
            save_top_k=1,
            enable_version_counter=False,
        )

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if (
            getattr(trainer, "fast_dev_run", False)
            or self._last_global_step_saved == trainer.global_step
        ):
            return
        self._save_none_monitor_checkpoint(trainer, self._monitor_candidates(trainer))


def build_trainer(cfg: TrainConfig, run_dir: Path, logger: Logger | None) -> L.Trainer:
    """Lightning Trainer for `cfg` (deterministic, DDP with unused parameters when world > 1)."""
    t = cfg.trainer
    strategy: DDPStrategy | str = (
        DDPStrategy(find_unused_parameters=True) if t.devices * t.num_nodes > 1 else "auto"
    )
    callbacks: list[L.Callback] = [
        ModelCheckpoint(
            dirpath=str(run_dir),
            filename=Path(CKPT_BEST).stem,
            monitor=MONITOR,
            mode="max",
            save_top_k=1,
            enable_version_counter=False,
        ),
        _LastCheckpoint(run_dir),
    ]
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    return L.Trainer(
        accelerator=t.accelerator,
        devices=t.devices,
        num_nodes=t.num_nodes,
        strategy=strategy,
        precision=cast(Any, t.precision),
        max_steps=t.max_steps,
        max_epochs=-1,
        deterministic=True,
        gradient_clip_val=t.gradient_clip_val,
        accumulate_grad_batches=t.accumulate_grad_batches,
        val_check_interval=val_check_interval(t),
        check_val_every_n_epoch=None,
        log_every_n_steps=t.log_every_n_steps,
        use_distributed_sampler=False,
        enable_checkpointing=True,
        default_root_dir=str(run_dir),
        logger=logger if logger is not None else False,
        callbacks=callbacks,
    )


def _make_logger(cfg: TrainConfig, run_dir: Path, run_id: str | None) -> Logger:
    """wandb logger: entity/project from the env, run name = experiment_name (test seam)."""
    env = require_env("WANDB_ENTITY", "WANDB_PROJECT")
    return PieWandbLogger(
        name=cfg.experiment_name,
        project=env["WANDB_PROJECT"],
        entity=env["WANDB_ENTITY"],
        group=cfg.logger.group,
        tags=list(cfg.logger.tags),
        id=run_id,
        save_dir=str(run_dir),
    )


def _wandb_run_id(run_dir: Path, resume: bool) -> str:
    path = run_dir / WANDB_ID_FILE
    if resume and path.is_file():
        return path.read_text().strip()
    run_id = uuid.uuid4().hex[:8]
    atomic_write_text(path, run_id + "\n")
    return run_id


def _read_data_stats(run_dir: Path) -> DataStats:
    payload = read_json(run_dir / DATA_STATS)
    if not isinstance(payload, dict) or "stats" not in payload:
        raise ValueError(f"{run_dir / DATA_STATS} is not a data-stats handoff file")
    return DataStats.model_validate(payload["stats"])


def _write_config(cfg: TrainConfig, run_dir: Path) -> None:
    dump = portable_train_config(cfg).model_dump(mode="json")
    atomic_write_text(run_dir / CONFIG_FILE, OmegaConf.to_yaml(OmegaConf.create(dump)))


def run_train(cfg: TrainConfig) -> Path:
    """Seed, run-dir policy (rank 0), split statistics (fit or pinned), model, then fit."""
    if cfg.logger.enabled:
        # Fail before the run-dir policy and the stats fit, not after them in _make_logger.
        require_env("WANDB_ENTITY", "WANDB_PROJECT")
    configure_determinism(cfg.seed)
    run_dir = resolve_path(cfg.run_dir)
    rank = env_global_rank(cfg.trainer.devices)
    if rank == 0:
        prepare_run_dir(run_dir, resume=cfg.resume, overwrite=cfg.overwrite)
    pinned = _read_data_stats(run_dir) if cfg.resume else None
    datamodule = PieDataModule(cfg.data, run_dir, pinned, seed=cfg.seed)
    stats = datamodule.setup_stats(rank)
    # No global RNG draw may happen between configure_determinism and this constructor.
    module = PieLightningModule(cfg, stats, datamodule.source_dims, datamodule.gene_query_text())
    logger: Logger | None = None
    if cfg.logger.enabled:
        run_id = _wandb_run_id(run_dir, cfg.resume) if rank == 0 else None
        logger = _make_logger(cfg, run_dir, run_id)
    if rank == 0:
        _write_config(cfg, run_dir)
    trainer = build_trainer(cfg, run_dir, logger)
    log.info("training %s in %s (resume=%s)", cfg.experiment_name, run_dir, cfg.resume)
    trainer.fit(
        module, datamodule=datamodule, ckpt_path=str(run_dir / CKPT_LAST) if cfg.resume else None
    )
    if isinstance(logger, WandbLogger) and trainer.is_global_zero:
        wandb.finish()
    return run_dir
