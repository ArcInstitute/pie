from __future__ import annotations

import tomllib

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pydantic import ValidationError

from pie.process.config import DATASETS, ProcessConfig, compose_process_config
from pie.utils import CONFIG_DIR, REPO_ROOT

GPUDGE_PIN = (
    "gpudge[fast] @ git+https://github.com/ArcInstitute/gpudge.git"
    "@bfbde26903b5580a4cf7692985f7b3c20f3bfebd"
)
OUTS = [
    "filter.output_dir=/o/filtered",
    "normalize.output_dir=/o/expression",
    "de.output_dir=/o/de",
]
NORM_ONLY = ["input=/i/a.h5ad", "normalize.enabled=true", "normalize.output_dir=/o/expression"]
JIANG_FILTER = {
    "enabled": True,
    "perturbation_column": "gene",
    "control_label": "non-targeting",
    "residual_expression": 0.30,
    "cell_residual_expression": 0.50,
    "min_cells": 30,
    "layer": None,
    "var_gene_name": "gene_name",
    "context_column": "context",
}
OFF = {"enabled": False}
# dataset -> (filter keys to check, de.groupby, de.reference, de.context_column)
EXPECTED: dict[str, tuple[dict[str, object], str, str, str | None]] = {
    "replogle": (OFF, "target_gene", "non-targeting", "cell_line"),
    "tahoe": (OFF, "perturbation", "[('DMSO_TF', 0.0, 'uM')]", None),
    "jiang": (JIANG_FILTER, "gene", "non-targeting", "context"),
    "orion": (JIANG_FILTER, "gene", "non-targeting", None),
    "arc_vcc_25": (
        {**JIANG_FILTER, "perturbation_column": "target_gene", "var_gene_name": None},
        "target_gene",
        "non-targeting",
        None,
    ),
}


def test_config_files_exist() -> None:
    assert CONFIG_DIR == REPO_ROOT / "configs"
    assert (CONFIG_DIR / "process.yaml").is_file()
    overlays = sorted(p.stem for p in (CONFIG_DIR / "process" / "dataset").glob("*.yaml"))
    assert overlays == sorted(DATASETS)


def test_base_config_defaults() -> None:
    cfg = compose_process_config(NORM_ONLY)
    assert cfg.input == "/i/a.h5ad"
    assert cfg.overwrite is False
    assert cfg.normalize.model_dump() == {
        "enabled": True,
        "target_sum": 1e4,
        "output_dir": "/o/expression",
    }
    assert cfg.filter.model_dump() == {
        **JIANG_FILTER,
        "enabled": False,
        "context_column": None,
        "output_dir": None,
    }
    assert cfg.de.model_dump() == {
        "enabled": False,
        "groupby": None,
        "reference": None,
        "context_column": None,
        "normalize_target_sum": 1e4,
        "filter_gene_min_cpm_cell": 5.0,
        "device": "auto",
        "output_dir": None,
    }


def test_input_is_mandatory() -> None:
    with pytest.raises(MissingMandatoryValue):
        compose_process_config(["normalize.enabled=true", "normalize.output_dir=/o/n"])


def test_input_accepts_a_list() -> None:
    cfg = compose_process_config(["input=[/i/a.h5ad,/i/b.h5ad]", *NORM_ONLY[1:]])
    assert cfg.input == ["/i/a.h5ad", "/i/b.h5ad"]


@pytest.mark.parametrize("name", DATASETS)
def test_dataset_overlay_composes_to_the_canonical_settings(name: str) -> None:
    cfg = compose_process_config([f"dataset={name}", "input=/i/*.h5ad", *OUTS])
    filt, groupby, reference, de_context = EXPECTED[name]
    got_filter = cfg.filter.model_dump()
    assert {key: got_filter[key] for key in filt} == filt
    assert cfg.filter.output_dir == "/o/filtered"
    assert cfg.normalize.model_dump() == {
        "enabled": True,
        "target_sum": 1e4,
        "output_dir": "/o/expression",
    }
    assert cfg.de.model_dump() == {
        "enabled": True,
        "groupby": groupby,
        "reference": reference,
        "context_column": de_context,
        "normalize_target_sum": 1e4,
        "filter_gene_min_cpm_cell": 5.0,
        "device": "auto",
        "output_dir": "/o/de",
    }


def _leaves(node: object, prefix: str = "") -> dict[str, object]:
    if not isinstance(node, dict):
        return {prefix: node}
    out: dict[str, object] = {}
    for key, value in node.items():
        out.update(_leaves(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


@pytest.mark.parametrize("name", DATASETS)
def test_overlays_set_only_values_that_differ_from_the_base(name: str) -> None:
    base = _leaves(OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "process.yaml")))
    overlay = _leaves(
        OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / "process" / "dataset" / f"{name}.yaml"))
    )
    same = sorted(key for key, value in overlay.items() if base.get(key) == value)
    assert same == []


def test_relative_input_resolves_against_the_repo_root() -> None:
    from pie.process.io import resolve_inputs

    assert resolve_inputs("pyproject.toml") == [REPO_ROOT / "pyproject.toml"]


def test_short_and_long_dataset_forms_agree() -> None:
    short = compose_process_config(["dataset=jiang", "input=/i/a.h5ad", *OUTS])
    long = compose_process_config(["process/dataset=jiang", "input=/i/a.h5ad", *OUTS])
    assert short == long


def test_unknown_dataset_fails() -> None:
    with pytest.raises(ConfigCompositionException, match="nope"):
        compose_process_config(["dataset=nope", *NORM_ONLY])


def test_unknown_key_fails() -> None:
    with pytest.raises(ConfigCompositionException):
        compose_process_config([*NORM_ONLY, "filter.bogus=1"])
    with pytest.raises(ValidationError, match="bogus"):
        compose_process_config([*NORM_ONLY, "+filter.bogus=1"])


def test_enabled_stage_needs_output_dir() -> None:
    with pytest.raises(ValidationError, match=r"normalize\.output_dir"):
        compose_process_config(["input=/i/a.h5ad", "normalize.enabled=true"])


def test_some_stage_must_be_enabled() -> None:
    with pytest.raises(ValidationError, match="no stage enabled"):
        compose_process_config(["input=/i/a.h5ad"])


def test_de_needs_groupby_and_reference() -> None:
    with pytest.raises(ValidationError, match=r"de\.groupby"):
        compose_process_config(["input=/i/a.h5ad", "de.enabled=true", "de.output_dir=/o/de"])


def test_stage_output_dirs_must_differ() -> None:
    overrides = [
        "dataset=jiang",
        "input=/i/a.h5ad",
        "filter.output_dir=/o/x",
        "normalize.output_dir=/o/x",
        "de.output_dir=/o/de",
    ]
    with pytest.raises(ValidationError, match="must differ"):
        compose_process_config(overrides)


def test_device_values_are_checked() -> None:
    with pytest.raises(ValidationError, match="device"):
        compose_process_config(["dataset=jiang", "input=/i/a.h5ad", *OUTS, "de.device=tpu"])


def test_strict_model_rejects_extra_keys() -> None:
    data = compose_process_config(NORM_ONLY).model_dump()
    data["extra"] = 1
    with pytest.raises(ValidationError):
        ProcessConfig.model_validate(data)


def test_pyproject_declares_the_process_extra() -> None:
    meta = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert meta["project"]["optional-dependencies"]["process"] == [GPUDGE_PIN]
    assert meta["tool"]["uv"]["override-dependencies"] == ["torch==2.10.0"]
    assert meta["tool"]["hatch"]["metadata"]["allow-direct-references"] is True
