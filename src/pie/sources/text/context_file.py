"""The context map format: context -> Cellosaurus accession and optional stimulation.

Kept apart from contexts.py (which needs the `sources` extra) so that pie prep and the asset
resolver can validate a map with the base install.
"""

from __future__ import annotations

import re
from pathlib import Path

from omegaconf import OmegaConf
from pydantic import Field

from pie.utils import StrictModel

_ACCESSION = re.compile(r"CVCL_[A-Z0-9]{4}", re.ASCII)


class StimulationEntry(StrictModel):
    name: str
    abbreviation: str
    family: str
    receptors: list[str]
    signaling: str
    description: str


class ContextEntry(StrictModel):
    cellosaurus: str
    stimulation: str | None = None


class ContextFile(StrictModel):
    contexts: dict[str, ContextEntry]
    stimulations: dict[str, StimulationEntry] = Field(default_factory=dict)


def load_context_file(path: Path) -> ContextFile:
    """Strict load of a context map (<preprocessed dir>/contexts.yaml; format in contexts.py)."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    parsed = ContextFile.model_validate(raw)
    for key, entry in parsed.contexts.items():
        if _ACCESSION.fullmatch(entry.cellosaurus) is None:
            raise ValueError(f"{path}: context {key!r} has invalid accession {entry.cellosaurus!r}")
        if entry.stimulation is not None and entry.stimulation not in parsed.stimulations:
            raise ValueError(
                f"{path}: context {key!r} names unknown stimulation {entry.stimulation!r}"
            )
    return parsed
