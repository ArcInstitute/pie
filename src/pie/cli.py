"""Command-line entry points. Every CLI takes Hydra `key=value` overrides (see `_launch`)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from pie.utils import ENV_ROOTS, load_common_env, require_env, setup_logging

_PREP_USAGE = """\
usage: pie-prep dataset=<name> [label_format=table|pie_process] labels=<glob> h5ad=<glob>
                output_dir=<dir> [genes=<file>] [controls_only=true] [overwrite=true]
                [<key>=<value> ...]

Build one preprocessed dataset dir from DE label tables and expression h5ads. Settings come
from configs/prep.yaml, configs/prep/dataset/<name>.yaml (obs.* columns, label rewrites) and
configs/prep/label_format/<name>.yaml (label.* columns; table = PIE label tables, pie_process =
pie-process DE parquets). Relative paths are relative to the repository root.
Datasets: arc_vcc_25, jiang, orion, replogle, tahoe.
"""


def prep_main(argv: list[str] | None = None) -> int:
    """pie-prep: DE label tables + expression h5ads -> one preprocessed dataset dir."""

    def run(overrides: list[str]) -> None:
        from pie.prep.config import compose_prep_config
        from pie.prep.labels import run_prep

        run_prep(compose_prep_config(overrides))

    return _launch(argv, _PREP_USAGE, run)


_TRAIN_USAGE = """\
usage: pie-train experiment=<name> [<key>=<value> ...]

Train PIE from configs/train.yaml plus an experiment overlay (configs/experiment/<name>.yaml)
and key=value overrides, e.g. vars.fold=k562, logger.enabled=false, resume=true or
overwrite=true. Relative paths are relative to the repository root.
"""


def train_main(argv: list[str] | None = None) -> int:
    """pie-train: fit the delta-p grid and evidence at setup, then train."""

    def run(overrides: list[str]) -> None:
        from pie import train
        from pie.config import compose_train_config

        train.run_train(compose_train_config(overrides))

    return _launch(argv, _TRAIN_USAGE, run, require=ENV_ROOTS)


_EVAL_USAGE = """\
usage: pie-eval experiment_name=<name> split_path=<split.json> row_set=<name> [<key>=<value> ...]

Score a checkpoint (ckpt=best_auprc|last) on a split file; writes
<run_dir>/eval/<row_set>/metrics_<ckpt>.csv and granular_<ckpt>.csv. Settings come from
configs/eval.yaml; save_predictions=true also writes the predictions, overwrite=true replaces
existing tables. Relative paths are relative to the repository root.
"""

_INFER_USAGE = """\
usage: pie-infer experiment_name=<name> rows_kind=query|split rows_path=<file> [<key>=<value> ...]

Predict p_de, lfc_pred and delta_p_pred per (context, perturbation) with a checkpoint and write
one parquet (output_path). Settings come from configs/infer.yaml; preprocessed_dirs=[...] points
at other preprocessed dirs (for example a controls-only dir), overwrite=true replaces the file.
"""


def eval_main(argv: list[str] | None = None) -> int:
    """pie-eval: score a checkpoint on a split file; writes <run_dir>/eval/<row_set>/."""

    def run(overrides: list[str]) -> None:
        from pie.config import compose_eval_config
        from pie.evaluate import run_eval

        run_eval(compose_eval_config(overrides))

    return _launch(argv, _EVAL_USAGE, run, require=ENV_ROOTS)


def infer_main(argv: list[str] | None = None) -> int:
    """pie-infer: predict query rows or a split file; writes one predictions parquet."""

    def run(overrides: list[str]) -> None:
        from pie.config import compose_infer_config
        from pie.infer import run_infer

        run_infer(compose_infer_config(overrides))

    return _launch(argv, _INFER_USAGE, run, require=ENV_ROOTS)


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
