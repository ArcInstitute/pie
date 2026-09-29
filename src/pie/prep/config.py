"""pie-prep configuration: configs/prep.yaml, a dataset and a label-format overlay, overrides."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import model_validator

from pie.utils import StrictModel, compose_config

CONFIG_NAME = "prep"
DATASET_GROUP = "prep/dataset"
LABEL_FORMAT_GROUP = "prep/label_format"
DATASETS: tuple[str, ...] = ("arc_vcc_25", "jiang", "orion", "replogle", "tahoe")
LABEL_FORMATS: tuple[str, ...] = ("pie_process", "table")


class LabelConfig(StrictModel):
    """Label-table columns and the rewrites applied to label contexts and perturbations."""

    context_from: Literal["column", "stem", "parent"]
    context_col: str | None
    pert_col: str
    gene_col: str
    fc_col: str
    fc_space: Literal["linear", "log2", "ln"]
    fdr_col: str
    context_case: Literal["asis", "lower"]
    context_map: dict[str, str] | None
    pert_format: Literal["plain", "drug_dose"]

    @model_validator(mode="after")
    def _check_context_col(self) -> LabelConfig:
        if (self.context_col is not None) != (self.context_from == "column"):
            raise ValueError("label.context_col is set exactly when label.context_from=column")
        return self


class ObsConfig(StrictModel):
    """The h5ad obs columns, the control label and the obs-side rewrites."""

    context_col: str
    pert_col: str
    pert_id_col: str | None
    control_label: str
    context_map: dict[str, str] | None
    pert_format: Literal["gene", "drug_dose"]


class PrepConfig(StrictModel):
    """The composed pie-prep config."""

    name: str
    labels: str | None
    h5ad: str
    output_dir: str
    genes: str | None
    controls_only: bool
    overwrite: bool
    label: LabelConfig
    obs: ObsConfig

    @model_validator(mode="after")
    def _check(self) -> PrepConfig:
        if not self.name:
            raise ValueError("name is empty")
        if self.labels is None and not self.controls_only:
            raise ValueError("labels is required unless controls_only=true")
        if self.obs.pert_id_col is not None and self.obs.pert_format != "gene":
            raise ValueError("obs.pert_id_col needs obs.pert_format=gene")
        return self


def compose_prep_config(overrides: Sequence[str]) -> PrepConfig:
    """Compose configs/prep.yaml, the `dataset=` and `label_format=` overlays and overrides."""
    data = compose_config(
        CONFIG_NAME,
        overrides,
        groups={"dataset": DATASET_GROUP, "label_format": LABEL_FORMAT_GROUP},
    )
    return PrepConfig.model_validate(data)
