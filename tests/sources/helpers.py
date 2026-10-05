"""Shared pie-sources test helpers: SourceOptions from configs/sources.yaml."""

from __future__ import annotations

from typing import Any

from pie.sources.config import SourceOptions, compose_sources_config


def make_options(**overrides: Any) -> SourceOptions:
    """configs/sources.yaml options with `overrides` replaced."""
    base = compose_sources_config(
        ["tools=[context_text]", "preprocessed_dirs=[/x]", "output_root=/x"]
    ).options
    return base.model_copy(update=overrides)
