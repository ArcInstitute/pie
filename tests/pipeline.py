"""A tiny trained run shared by the prediction, evaluation, inference and CLI tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import pie
from pie.config import (
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
)
from pie.data.preprocessed import (
    CTRL_MEANS,
    FORMAT_VERSION,
    PreprocessedDir,
    PreprocessedMeta,
    write_preprocessed,
)
from tests.fixtures import ALPHA_GENES, TinyData, tiny_data_config
from tests.model.helpers import tiny_config

QUERY: dict[str, list[str]] = {"gamma.a1": ["GA", "NEWPERT"], "gamma.a2": ["GB"]}


@dataclass(frozen=True)
class TrainedRun:
    root: Path
    tiny: TinyData
    run_dir: Path

    @property
    def ckpt(self) -> Path:
        return self.run_dir / "best_auprc.ckpt"

    @property
    def test_split(self) -> Path:
        return self.tiny.split_dir / "test.json"

    def apply_env(self, mp: pytest.MonkeyPatch) -> None:
        """The env roots the run was trained under (checkpoint paths are portable against them)."""
        mp.setenv("PIE_DATA_ROOT", str(self.tiny.root))
        mp.setenv("PIE_RUNS_ROOT", str(self.root / "runs"))
        mp.setenv("PIE_CACHE_DIR", str(self.tiny.cache_dir))


def tiny_train_config(tiny: TinyData, run_dir: Path) -> TrainConfig:
    """A 2-step CPU fit over the tiny data (no logger)."""
    return TrainConfig(
        vars={},
        experiment_name="tiny",
        seed=0,
        run_dir=str(run_dir),
        overwrite=False,
        resume=False,
        data=tiny_data_config(tiny),
        model=tiny_config(),
        optimizer=OptimizerConfig(lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999)),
        scheduler=SchedulerConfig(warmup_fraction=0.5, constant_steps=0, eta_min=1e-6),
        trainer=TrainerConfig(
            accelerator="cpu",
            devices=1,
            num_nodes=1,
            precision="32-true",
            max_steps=2,
            accumulate_grad_batches=1,
            val_every_n_steps=1,
            gradient_clip_val=10.0,
            log_every_n_steps=1,
        ),
        logger=LoggerConfig(enabled=False, group=None, tags=[]),
    )


def controls_only_dir(out: Path, source: Path) -> Path:
    """A controls-only dir 'gamma' with the gene axis, contexts and control means of `source`."""
    src = PreprocessedDir.open(source)
    meta = PreprocessedMeta(
        format_version=FORMAT_VERSION,
        dataset="gamma",
        genes=list(ALPHA_GENES),
        context_to_id=dict(src.meta.context_to_id),
        pert_to_id={},
        pert_kind="gene",
        control_label="non-targeting",
        num_rows=0,
        num_genes=len(ALPHA_GENES),
        num_contexts=len(src.contexts),
        num_perts=0,
        controls_only=True,
        tool_version=pie.__version__,
        array_sha256={},
    )
    return write_preprocessed(out, meta, {CTRL_MEANS: np.array(src.ctrl_means, dtype=np.float32)})


def write_query(path: Path, query: dict[str, list[str]] | None = None) -> Path:
    path.write_text(json.dumps(QUERY if query is None else query, indent=1) + "\n")
    return path
