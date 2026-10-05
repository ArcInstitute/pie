"""Tests for pie.config: the strict schema and Hydra composition of configs/train.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import BaseModel, ValidationError

import pie.config as config_mod
from pie.config import (
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
    compose_eval_config,
    compose_infer_config,
    compose_train_config,
    portable_train_config,
    resolved_train_config,
)
from pie.data.datamodule import DataConfig
from pie.data.evidence import EvidenceConfig
from pie.model.pie import EvidenceModelConfig, ModelConfig, TemperatureConfig
from pie.utils import CONFIG_DIR
from tests.conftest import REPO_ROOT
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


def test_config_dir_is_inside_the_package() -> None:
    import pie

    assert Path(pie.__file__).resolve().parent / "configs" == CONFIG_DIR
    assert (CONFIG_DIR / "train.yaml").is_file()
    assert (CONFIG_DIR / "experiment" / "replogle_wdataset.yaml").is_file()


def test_compose_from_another_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _experiment_roots(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    cfg = compose_train_config(["experiment=replogle_wdataset", "vars.fold=k562"])
    assert cfg.experiment_name == "replogle_wdataset/k562"


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
    monkeypatch.chdir(REPO_ROOT)
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
    assert portable.data.aliases_path == str(REPO_ROOT / "data/sources/aliases.yaml")
    back = resolved_train_config(portable).model_dump()
    assert back["data"]["aliases_path"] == str(REPO_ROOT / "data/sources/aliases.yaml")
    back["data"]["aliases_path"] = cfg.data.aliases_path
    assert back == cfg.model_dump()


def test_resolve_steps_matches_both_recipes() -> None:
    sched = SchedulerConfig(warmup_fraction=0.05, constant_steps=4250, eta_min=1e-6)
    assert sched.resolve_steps(5000) == (250, 4250, 500)
    assert sched.resolve_steps(10000) == (500, 4250, 5250)
    assert sched.resolve_steps(10) == (1, 4250, 1)


def test_eval_config_composes_with_its_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PIE_RUNS_ROOT", str(tmp_path))
    cfg = compose_eval_config(
        [
            "experiment_name=replogle_wdataset/k562",
            "split_path=data/splits/replogle_wdataset/unseen_ctx/k562/test.json",
            "row_set=unseen_ctx",
        ]
    )
    assert cfg.run_dir == f"{tmp_path}/replogle_wdataset/k562"
    assert cfg.ckpt == "best_auprc"
    assert cfg.preprocessed_dirs is None
    assert cfg.save_predictions is False
    assert cfg.overwrite is False
    assert (cfg.device, cfg.batch_size) == ("cuda", 16)


def test_eval_config_requires_split_and_row_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PIE_RUNS_ROOT", str(tmp_path))
    with pytest.raises(MissingMandatoryValue):
        compose_eval_config(["experiment_name=x", "split_path=s.json"])


def test_eval_config_rejects_unknown_keys_and_nested_row_sets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PIE_RUNS_ROOT", str(tmp_path))
    base = ["experiment_name=x", "split_path=s.json"]
    with pytest.raises(ValidationError):
        compose_eval_config([*base, "row_set=r", "+bogus=1"])
    with pytest.raises(ValidationError, match="row_set"):
        compose_eval_config([*base, "row_set=a/b"])


def test_infer_config_composes_with_its_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PIE_RUNS_ROOT", str(tmp_path))
    cfg = compose_infer_config(["experiment_name=replogle_xdataset", "rows_path=query.json"])
    assert cfg.rows_kind == "query"
    assert cfg.run_dir == f"{tmp_path}/replogle_xdataset"
    assert cfg.output_path == f"{tmp_path}/replogle_xdataset/infer/predictions.parquet"
    assert cfg.overwrite is False
    assert (cfg.ckpt, cfg.device, cfg.batch_size) == ("best_auprc", "cuda", 16)


@pytest.mark.parametrize("name", ["eval.yaml", "infer.yaml"])
def test_eval_and_infer_yaml_have_no_hydra_node(name: str) -> None:
    assert "hydra" not in OmegaConf.load(CONFIG_DIR / name)


def test_infer_config_rejects_an_unknown_rows_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PIE_RUNS_ROOT", str(tmp_path))
    with pytest.raises(ValidationError):
        compose_infer_config(["experiment_name=x", "rows_path=q.json", "rows_kind=table"])


# --- canonical experiment overlays (configs/experiment/) ---------------------------------------
# Golden values are the resolved reference recipes, restated in the new schema.

_EXPERIMENT_FOLDS = ("hepg2", "jurkat", "k562", "rpe1")
_EXPERIMENT_SETTINGS = ("unseen_ctx", "unseen_pert", "unseen_ctx_pert")
_WDATASET_SOURCES = (
    "esm2", "ncbi_text", "string_space", "depmap_gene_effect", "context_text", "perturbation_text",
)
_XDATASET_SOURCES = (
    *_WDATASET_SOURCES, "smiles", "l1000_tas", "prism_secondary", "jump_morphology",
)
_XDATASET_DATASETS = ("replogle", "tahoe", "jiang", "arc_vcc_25", "orion")
_HF_DATASETS = {
    "replogle": "PIE_replogle_nadig_essential@f243f5473b68c7b62422e22645277835f9b04d0d",
    "tahoe": "PIE_tahoe100m@8f003b086289aecfa6f2f07e3ba49738f69fa69b",
    "jiang": "PIE_jiang@bcf4ceedd2e1232e8a3a362c1a10b1c49c3585ac",
    "arc_vcc_25": "PIE_arc_vcc_25@fe629a02e7d584c4b6fd0f9ea0865aa584cb854f",
    "orion": "PIE_x_atlas_orion@516a10f46461f168cc5dde5c3b449e9c45c70b8f",
}
_HF_SOURCES = "hf://datasets/arcinstitute/PIE_sources@7a27f6e647d8d3d640a37b3c7f0c93559a628b37"
_XDATASET_WEIGHTS = {
    "replogle": 0.0, "tahoe": 0.68, "jiang": 0.01, "arc_vcc_25": 0.01, "orion": 0.3,
}
_EXPERIMENT_MODEL = {
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


def _experiment_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    data_root = tmp_path / "data_root"
    runs_root = tmp_path / "runs_root"
    monkeypatch.setenv("PIE_DATA_ROOT", str(data_root))
    monkeypatch.setenv("PIE_RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("PIE_CACHE_DIR", str(tmp_path / "cache_root"))
    return str(data_root), str(runs_root)


def _experiment_golden(
    data_root: str,
    runs_root: str,
    *,
    experiment_name: str,
    vars_: dict[str, str],
    datasets: tuple[str, ...],
    weights: dict[str, float] | None,
    split_dir: str,
    sources: tuple[str, ...],
    bin_width: float,
    evidence: tuple[int, int],
    trainer: tuple[int, int, int, int],
    group: str,
    tags: list[str],
) -> dict[str, object]:
    devices, num_nodes, max_steps, accumulate = trainer
    return {
        "vars": vars_,
        "experiment_name": experiment_name,
        "seed": 42,
        "run_dir": f"{runs_root}/{experiment_name}",
        "overwrite": False,
        "resume": False,
        "data": {
            "preprocessed_dirs": [
                f"hf://datasets/arcinstitute/{_HF_DATASETS[name]}/preprocessed"
                for name in datasets
            ],
            "dataset_weights": weights,
            "split_dir": split_dir,
            "source_dirs": {name: f"{_HF_SOURCES}/{name}" for name in sources},
            "gene_text_dir": f"{_HF_SOURCES}/gene_text",
            "aliases_path": "data/sources/aliases.yaml",
            "delta_p": {
                "bin_width_fold_change": bin_width,
                "max_delta_percentile": 99.9,
                "max_delta": None,
            },
            "evidence": {"seed": evidence[0], "chunk": evidence[1], "lfc_clip_percentile": 95.0},
            "batch_size": 16,
            "num_workers": 4,
            "fdr_threshold": 0.05,
        },
        "model": _EXPERIMENT_MODEL,
        "optimizer": {"lr": 1e-4, "weight_decay": 0.01, "betas": [0.9, 0.999]},
        "scheduler": {"warmup_fraction": 0.05, "constant_steps": 4250, "eta_min": 1e-6},
        "trainer": {
            "accelerator": "gpu",
            "devices": devices,
            "num_nodes": num_nodes,
            "precision": "bf16-mixed",
            "max_steps": max_steps,
            "accumulate_grad_batches": accumulate,
            "val_every_n_steps": 100,
            "gradient_clip_val": 10.0,
            "log_every_n_steps": 50,
        },
        "logger": {"enabled": True, "group": group, "tags": tags},
    }


@pytest.mark.parametrize(
    ("fold", "overrides"),
    [
        ("k562", []),
        ("hepg2", ["vars.fold=hepg2"]),
        ("jurkat", ["vars.fold=jurkat"]),
        ("rpe1", ["vars.fold=rpe1"]),
    ],
)
def test_experiment_wdataset_matches_the_reference_recipe(
    fold: str, overrides: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, runs_root = _experiment_roots(tmp_path, monkeypatch)
    cfg = compose_train_config(["experiment=replogle_wdataset", *overrides])
    assert cfg.model_dump(mode="json") == _experiment_golden(
        data_root,
        runs_root,
        experiment_name=f"replogle_wdataset/{fold}",
        vars_={"fold": fold},
        datasets=("replogle",),
        weights=None,
        split_dir=f"{config_mod.PIE_SPLITS_URI}/replogle_wdataset/unseen_ctx/{fold}",
        sources=_WDATASET_SOURCES,
        bin_width=1.1,
        evidence=(42, 256),
        trainer=(2, 1, 5000, 4),
        group="replogle_wdataset",
        tags=["replogle_wdataset", fold],
    )
    assert list(cfg.data.source_dirs) == list(_WDATASET_SOURCES)


def test_experiment_xdataset_matches_the_reference_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, runs_root = _experiment_roots(tmp_path, monkeypatch)
    cfg = compose_train_config(["experiment=replogle_xdataset"])
    assert cfg.model_dump(mode="json") == _experiment_golden(
        data_root,
        runs_root,
        experiment_name="replogle_xdataset",
        vars_={},
        datasets=_XDATASET_DATASETS,
        weights=_XDATASET_WEIGHTS,
        split_dir=f"{config_mod.PIE_SPLITS_URI}/replogle_xdataset",
        sources=_XDATASET_SOURCES,
        bin_width=1.01,
        evidence=(0, 2048),
        trainer=(4, 2, 10000, 1),
        group="replogle_xdataset",
        tags=["replogle_xdataset"],
    )
    assert list(cfg.data.source_dirs) == list(_XDATASET_SOURCES)
    assert cfg.data.dataset_weights is not None
    assert list(cfg.data.dataset_weights) == list(_XDATASET_DATASETS)


def test_xdataset_dataset_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _experiment_roots(tmp_path, monkeypatch)
    cfg = compose_train_config(["experiment=replogle_xdataset"])
    assert cfg.data.preprocessed_dirs == [
        f"hf://datasets/arcinstitute/{_HF_DATASETS[name]}/preprocessed"
        for name in _XDATASET_DATASETS
    ]
    names = list(_XDATASET_DATASETS)
    # The sorted names seed the sampler and order the evidence donors.
    assert sorted(names) == ["arc_vcc_25", "jiang", "orion", "replogle", "tahoe"]
    assert sorted(cfg.data.dataset_weights or {}) == sorted(names)


@pytest.mark.parametrize("experiment", ["replogle_wdataset", "replogle_xdataset"])
def test_experiment_overlays_reject_unknown_keys(
    experiment: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _experiment_roots(tmp_path, monkeypatch)
    with pytest.raises(ValidationError, match="bogus"):
        compose_train_config([f"experiment={experiment}", "+data.bogus=1"])


def test_pinned_train_config_pins_the_split_dir(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pie.config as config_mod

    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    monkeypatch.setattr(
        config_mod, "pin_asset_reference", lambda v: v.replace("@main", "@" + "b" * 40)
    )
    uri = "hf://datasets/arcinstitute/PIE_splits@main/exp/fold"
    cfg = compose_train_config([*_required(tiny_data), f"data.split_dir={uri}"])
    pinned = config_mod.pinned_train_config(cfg)
    assert pinned.data.split_dir == f"hf://datasets/arcinstitute/PIE_splits@{'b' * 40}/exp/fold"


def _saved(tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    set_run_env(monkeypatch, tiny_data, tmp_path / "runs")
    return compose_train_config(_required(tiny_data)).model_dump(mode="json")


def test_legacy_split_dir_maps_to_pie_splits(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _saved(tiny_data, tmp_path, monkeypatch)
    raw["data"]["split_dir"] = "data/splits/replogle_wdataset/unseen_ctx/k562"
    cfg = config_mod.load_saved_train_config(raw)
    assert cfg.data.split_dir == f"{config_mod.PIE_SPLITS_URI}/replogle_wdataset/unseen_ctx/k562"


def test_saved_config_without_legacy_paths_is_unchanged(
    tiny_data: TinyData, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _saved(tiny_data, tmp_path, monkeypatch)
    assert config_mod.load_saved_train_config(raw).model_dump(mode="json") == raw


def test_experiment_split_dirs_are_pinned_pie_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _experiment_roots(tmp_path, monkeypatch)
    uri = config_mod.PIE_SPLITS_URI
    for fold in _EXPERIMENT_FOLDS:
        cfg = compose_train_config(["experiment=replogle_wdataset", f"vars.fold={fold}"])
        assert cfg.data.split_dir == f"{uri}/replogle_wdataset/unseen_ctx/{fold}"
    cfg = compose_train_config(["experiment=replogle_xdataset"])
    assert cfg.data.split_dir == f"{uri}/replogle_xdataset"
