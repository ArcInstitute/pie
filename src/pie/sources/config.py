"""pie-sources configuration: configs/sources.yaml and key=value overrides."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from pie.utils import StrictModel, compose_config, resolve_path

CONFIG_NAME = "sources"


class SourceOptions(StrictModel):
    """Builder settings shared by the source tools."""

    contexts_dir: Path
    on_conflict: Literal["error", "keep-prior", "replace"]
    cellosaurus_release: str
    offline: bool
    gene_info: Path | None
    drug_metadata: Path | None
    depmap_csv: Path | None
    string_release: str
    device: str
    pert_output: Path | None  # gene_text input [<output_root>/perturbation_text]

    def resolved(self) -> SourceOptions:
        """A copy with every path field through resolve_path."""
        paths = ("contexts_dir", "gene_info", "drug_metadata", "depmap_csv", "pert_output")
        update = {
            name: resolve_path(value)
            for name in paths
            if (value := getattr(self, name)) is not None
        }
        return self.model_copy(update=update)


class SourcesConfig(StrictModel):
    """The composed pie-sources config."""

    tools: list[str]
    with_deps: bool
    preprocessed_dirs: list[str]
    output_root: str
    prior_root: str | None
    overwrite: bool
    options: SourceOptions

    @model_validator(mode="after")
    def _check(self) -> SourcesConfig:
        from pie.sources.registry import TOOLS

        if not self.tools:
            raise ValueError("tools must name at least one source tool")
        if not self.preprocessed_dirs:
            raise ValueError("preprocessed_dirs must name at least one preprocessed dir")
        unknown = [name for name in self.tools if name not in TOOLS]
        if unknown:
            raise ValueError(f"unknown source tool(s) {unknown}; known: {list(TOOLS)}")
        return self


def compose_sources_config(overrides: Sequence[str]) -> SourcesConfig:
    """Compose configs/sources.yaml with overrides; validate."""
    return SourcesConfig.model_validate(compose_config(CONFIG_NAME, overrides))
