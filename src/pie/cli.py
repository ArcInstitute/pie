"""Command-line entry points. Every CLI takes Hydra `key=value` overrides (see `_launch`)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from pie.utils import load_common_env, require_env, setup_logging


def _launch(
    argv: Sequence[str] | None,
    usage: str,
    run: Callable[[list[str]], object],
    *,
    require: Sequence[str] = (),
) -> int:
    """`--help` first; then common.sh, logging and the `require`d env; then run(overrides)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in ("-h", "--help") for arg in args):
        print(usage, end="")
        return 0
    load_common_env()
    setup_logging()
    require_env(*require)
    run(args)
    return 0


_PROCESS_USAGE = """\
usage: pie-process [dataset=<name>] input=<glob|[path,...]> [<key>=<value> ...]

Turn raw count h5ads into knockdown-filtered counts (filter), log1p expression h5ads
(normalize) and per-context DE parquets (de). Settings come from configs/process.yaml,
the overlay configs/process/dataset/<name>.yaml and key=value overrides such as
filter.enabled=false, normalize.output_dir=<dir>, de.output_dir=<dir>, de.device=cuda
or overwrite=true. Every enabled stage needs its output_dir. Relative paths are relative
to the repository root.
Datasets: arc_vcc_25, jiang, orion, replogle, tahoe.
The de stage needs the optional extra: uv sync --extra process.
"""


def process_main(argv: list[str] | None = None) -> int:
    """pie-process: raw count h5ads -> filtered counts, log1p expression h5ads, DE parquets."""

    def run(overrides: list[str]) -> None:
        from pie.process.config import compose_process_config
        from pie.process.run import run_process

        run_process(compose_process_config(overrides))

    return _launch(argv, _PROCESS_USAGE, run)
