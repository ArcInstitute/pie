"""pie process configuration: configs/process.yaml, a dataset overlay and key=value overrides."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import model_validator

from pie.utils import StrictModel, compose_config, resolve_path

CONFIG_NAME = "process"
DATASET_GROUP = "process/dataset"
DATASETS: tuple[str, ...] = ("arc_vcc_25", "jiang", "orion", "replogle", "tahoe")


class FilterConfig(StrictModel):
    """On-target knockdown filter (pie.process.knockdown), applied per context."""

    enabled: bool
    perturbation_column: str
    control_label: str
    residual_expression: float
    cell_residual_expression: float
    min_cells: int
    layer: str | None
    var_gene_name: str | None
    context_column: str | None
    output_dir: str | None


class NormalizeConfig(StrictModel):
    """CP10k + natural log1p into X (pie.process.normalize)."""

    enabled: bool
    target_sum: float
    output_dir: str | None


class DEConfig(StrictModel):
    """Per-context differential expression with gpudge (pie.process.de)."""

    enabled: bool
    groupby: str | None
    reference: str | None
    context_column: str | None
    normalize_target_sum: float
    filter_gene_min_cpm_cell: float | None
    device: Literal["auto", "cuda", "cpu"]
    output_dir: str | None


class ProcessConfig(StrictModel):
    """The composed pie process config."""

    input: str | list[str]
    overwrite: bool
    filter: FilterConfig
    normalize: NormalizeConfig
    de: DEConfig

    @model_validator(mode="after")
    def _check_stages(self) -> ProcessConfig:
        stages: dict[str, FilterConfig | NormalizeConfig | DEConfig] = {
            "filter": self.filter,
            "normalize": self.normalize,
            "de": self.de,
        }
        enabled = [name for name, stage in stages.items() if stage.enabled]
        if not enabled:
            raise ValueError("no stage enabled: enable one of filter, normalize, de")
        for name in enabled:
            if not stages[name].output_dir:
                raise ValueError(f"{name}.enabled=true needs {name}.output_dir")
        dirs = [resolve_path(str(stages[name].output_dir)).resolve() for name in enabled]
        if len(set(dirs)) != len(dirs):
            raise ValueError("the output_dir of each enabled stage must differ")
        if self.de.enabled and not (self.de.groupby and self.de.reference):
            raise ValueError("de.enabled=true needs de.groupby and de.reference")
        if not self.input:
            raise ValueError("input is empty")
        return self


def compose_process_config(overrides: Sequence[str]) -> ProcessConfig:
    """Compose configs/process.yaml, the `dataset=<name>` overlay and overrides; validate."""
    data = compose_config(CONFIG_NAME, overrides, groups={"dataset": DATASET_GROUP})
    return ProcessConfig.model_validate(data)
