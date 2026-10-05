"""pie process orchestration: filter -> normalize -> de, each stage optional."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pie.process.config import ProcessConfig
from pie.process.de import prepare_de, run_de
from pie.process.io import check_writable, resolve_inputs
from pie.process.knockdown import filter_output_paths, run_filter
from pie.process.normalize import normalize_file
from pie.utils import resolve_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessOutputs:
    """Files written by run_process, per stage (empty for a disabled stage)."""

    inputs: list[Path]
    filtered: list[Path]
    expression: list[Path]
    de: list[Path]


def _out_dir(value: str | None) -> Path:
    """A stage output_dir (repository-root relative when relative)."""
    if value is None:
        raise ValueError("output_dir is not set")
    return resolve_path(value)


def run_process(cfg: ProcessConfig) -> ProcessOutputs:
    """Run the enabled stages on `cfg.input`.

    `filter` reads the inputs. `normalize` and `de` read the filtered counts when the filter is
    enabled, else the inputs; `de` never reads the normalized expression. gpudge and the device are
    checked, and existing filter / normalize outputs are refused, before any stage runs.
    """
    inputs = resolve_inputs(cfg.input)
    stages = [name for name in ("filter", "normalize", "de") if getattr(cfg, name).enabled]
    logger.info("pie process: %d input file(s); stages: %s", len(inputs), ", ".join(stages))
    gpudge = prepare_de(cfg.de) if cfg.de.enabled else None
    planned: list[Path] = []
    if cfg.filter.enabled:
        planned += filter_output_paths(inputs, cfg.filter)
    if cfg.normalize.enabled:
        planned += [_out_dir(cfg.normalize.output_dir) / path.name for path in inputs]
    for path in planned:
        check_writable(path, cfg.overwrite)

    counts = inputs
    filtered: list[Path] = []
    if cfg.filter.enabled:
        filtered = run_filter(inputs, cfg.filter, cfg.overwrite)
        counts = filtered
    expression: list[Path] = []
    if cfg.normalize.enabled:
        out_dir = _out_dir(cfg.normalize.output_dir)
        expression = [
            normalize_file(path, out_dir / path.name, cfg.normalize.target_sum, cfg.overwrite)
            for path in counts
        ]
    de: list[Path] = []
    if cfg.de.enabled:
        de = run_de(counts, cfg.de, cfg.overwrite, gpudge=gpudge)
    logger.info(
        "pie process done: %d filtered, %d expression, %d de file(s)",
        len(filtered),
        len(expression),
        len(de),
    )
    return ProcessOutputs(inputs=inputs, filtered=filtered, expression=expression, de=de)
