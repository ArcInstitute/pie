"""The `pie <command>` entry point. Every command takes Hydra `key=value` overrides (`_launch`)."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence

from pie.utils import ENV_ROOTS, load_common_env, require_env, setup_logging

_PREP_USAGE = """\
usage: pie prep dataset=<name> [label_format=table|pie_process] labels=<glob> h5ad=<glob>
                output_dir=<dir> [genes=<file>] [controls_only=true] [overwrite=true]
                [<key>=<value> ...]

Build one preprocessed dataset dir from DE label tables and expression h5ads. Settings come
from pie/configs/prep.yaml, pie/configs/prep/dataset/<name>.yaml (obs.* columns, label rewrites) and
pie/configs/prep/label_format/<name>.yaml (label.* columns; table = PIE label tables, pie_process =
pie process DE parquets). Relative paths are relative to the repository root.
Datasets: arc_vcc_25, jiang, orion, replogle, tahoe.
"""


def prep_main(argv: list[str] | None = None) -> int:
    """pie prep: DE label tables + expression h5ads -> one preprocessed dataset dir."""

    def run(overrides: list[str]) -> None:
        from pie.prep.config import compose_prep_config
        from pie.prep.labels import run_prep

        run_prep(compose_prep_config(overrides))

    return _launch(argv, _PREP_USAGE, run)


_TRAIN_USAGE = """\
usage: pie train experiment=<name> [<key>=<value> ...]

Train PIE from pie/configs/train.yaml plus an experiment overlay
(pie/configs/experiment/<name>.yaml) and key=value overrides, e.g. vars.fold=k562,
logger.enabled=false, resume=true or overwrite=true. Relative paths are relative to the
repository root.
"""


def train_main(argv: list[str] | None = None) -> int:
    """pie train: fit the delta-p grid and evidence at setup, then train."""

    def run(overrides: list[str]) -> None:
        from pie import train
        from pie.config import compose_train_config

        train.run_train(compose_train_config(overrides))

    return _launch(argv, _TRAIN_USAGE, run, require=ENV_ROOTS)


_EVAL_USAGE = """\
usage: pie eval experiment_name=<name> split_path=<split.json> row_set=<name> [<key>=<value> ...]

Score a checkpoint (ckpt=best_auprc|last) on a split file; writes
<run_dir>/eval/<row_set>/metrics_<ckpt>.csv and granular_<ckpt>.csv. Settings come from
pie/configs/eval.yaml; save_predictions=true also writes the predictions, overwrite=true replaces
existing tables. Relative paths are relative to the repository root.
"""

_INFER_USAGE = """\
usage: pie infer experiment_name=<name> rows_kind=query|split rows_path=<file> [<key>=<value> ...]

Predict p_de, lfc_pred and delta_p_pred per (context, perturbation) with a checkpoint and write
one parquet (output_path). Settings come from pie/configs/infer.yaml; preprocessed_dirs=[...] points
at other preprocessed dirs (for example a controls-only dir), overwrite=true replaces the file.
"""


def eval_main(argv: list[str] | None = None) -> int:
    """pie eval: score a checkpoint on a split file; writes <run_dir>/eval/<row_set>/."""

    def run(overrides: list[str]) -> None:
        from pie.config import compose_eval_config
        from pie.evaluate import run_eval

        run_eval(compose_eval_config(overrides))

    return _launch(argv, _EVAL_USAGE, run, require=ENV_ROOTS)


def infer_main(argv: list[str] | None = None) -> int:
    """pie infer: predict query rows or a split file; writes one predictions parquet."""

    def run(overrides: list[str]) -> None:
        from pie.config import compose_infer_config
        from pie.infer import run_infer

        run_infer(compose_infer_config(overrides))

    return _launch(argv, _INFER_USAGE, run, require=ENV_ROOTS)


_SOURCES_USAGE = """\
usage: pie sources tools=[<name>,...] preprocessed_dirs=[<dir>,...] output_root=<dir>
                   [<key>=<value> ...]

Build knowledge sources into <output_root>/<name>/. Settings come from pie/configs/sources.yaml:
with_deps=false builds only the named tools, prior_root=<dir> extends earlier text sources,
options.<key>=<value> sets builder settings (depmap_csv, drug_metadata, gene_info, device, ...),
overwrite=true replaces existing outputs. mode=verify checks <output_root>/<name> for every tool
against the datasets and, with verify.reference=<dir>, against reference sources.
Relative paths are relative to the repository root.
"""


def sources_main(argv: list[str] | None = None) -> int:
    """pie sources: build (prints {name: out_dir}) or verify (prints the report) as JSON."""

    def run(overrides: list[str]) -> None:
        from pie.sources.config import compose_sources_config
        from pie.sources.registry import build_sources, verify_command

        cfg = compose_sources_config(overrides)
        if cfg.mode == "verify":
            print(json.dumps(verify_command(cfg), indent=1))
            return
        outputs = build_sources(cfg)
        print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))

    return _launch(argv, _SOURCES_USAGE, run)


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
usage: pie process [dataset=<name>] input=<glob|[path,...]> [<key>=<value> ...]

Turn raw count h5ads into knockdown-filtered counts (filter), log1p expression h5ads
(normalize) and per-context DE parquets (de). Settings come from pie/configs/process.yaml,
the overlay pie/configs/process/dataset/<name>.yaml and key=value overrides such as
filter.enabled=false, normalize.output_dir=<dir>, de.output_dir=<dir>, de.device=cuda
or overwrite=true. Every enabled stage needs its output_dir. Relative paths are relative
to the repository root.
Datasets: arc_vcc_25, jiang, orion, replogle, tahoe.
The de stage needs the optional extra: uv sync --extra process.
"""


def process_main(argv: list[str] | None = None) -> int:
    """pie process: raw count h5ads -> filtered counts, log1p expression h5ads, DE parquets."""

    def run(overrides: list[str]) -> None:
        from pie.process.config import compose_process_config
        from pie.process.run import run_process

        run_process(compose_process_config(overrides))

    return _launch(argv, _PROCESS_USAGE, run)


COMMANDS: dict[str, tuple[Callable[[list[str] | None], int], str]] = {
    "process": (process_main, "raw count h5ads -> log1p expression h5ads and DE parquets"),
    "prep": (prep_main, "DE labels + expression h5ads -> one preprocessed dataset dir"),
    "sources": (sources_main, "build or verify knowledge sources"),
    "train": (train_main, "train PIE (experiment=<name> plus overrides)"),
    "eval": (eval_main, "score a checkpoint on a split file"),
    "infer": (infer_main, "predict p_de, lfc_pred and delta_p_pred without labels"),
}


def _main_usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [f"  {name.ljust(width)}  {summary}" for name, (_, summary) in COMMANDS.items()]
    return (
        "usage: pie <command> [<key>=<value> ...]\n"
        "       pie <command> --help\n"
        "       pie --version\n\n"
        "Commands:\n" + "\n".join(lines) + "\n\n"
        "Every command takes Hydra key=value overrides on its config.\n"
    )


def main(argv: list[str] | None = None) -> int:
    """`pie <command> [overrides]`: dispatch to one command; no command prints the list."""
    from pie import __version__

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(_main_usage(), end="")
        return 0
    if args[0] in ("-V", "--version"):
        print(f"pie {__version__}")
        return 0
    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        print(f"pie: unknown command {name!r}\n\n{_main_usage()}", end="", file=sys.stderr)
        return 2
    return COMMANDS[name][0](rest)


def _deprecated(name: str) -> Callable[[list[str] | None], int]:
    def alias(argv: list[str] | None = None) -> int:
        print(f"warning: 'pie-{name}' is deprecated; use 'pie {name}'", file=sys.stderr)
        return COMMANDS[name][0](argv)

    alias.__name__ = f"{name}_alias"
    return alias


process_alias = _deprecated("process")
prep_alias = _deprecated("prep")
sources_alias = _deprecated("sources")
train_alias = _deprecated("train")
eval_alias = _deprecated("eval")
infer_alias = _deprecated("infer")
