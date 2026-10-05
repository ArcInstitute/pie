"""Synthetic preprocessed dirs, knowledge sources, splits and aliases shared by the test suite."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

import pie
from pie.data.preprocessed import (
    CTRL_MEANS,
    CTX_IDS,
    DELTA_P,
    FDR,
    FOLD_CHANGES,
    FORMAT_VERSION,
    LFC_TRUE,
    PERT_IDS,
    TESTED,
    PreprocessedMeta,
    write_preprocessed,
)
from pie.sources import contract

ALPHA_GENES = ["GA", "GB", "GC", "GD", "GE", "GF"]
BETA_GENES = ["GC", "GD", "GE", "GF", "GG", "GH"]
GENE_AXIS = ["GA", "GB", "GC", "GD", "GE", "GF", "GG", "GH"]
TRAIN_SPLIT: dict[str, list[str]] = {
    "alpha.a1": ["GA", "GB", "GC", "OLDX"],
    "beta.b1": ["drugA", "drugB"],
}
VAL_SPLIT: dict[str, list[str]] = {"alpha.a2": ["GA", "GB"], "beta.b1": ["drugC"]}
TEST_SPLIT: dict[str, list[str]] = {
    "alpha.a2": ["GC", "OLDX"],
    "beta.b2": ["drugA", "drugB", "drugC"],
}
ALIASES: dict[str, dict[str, str]] = {
    "esm2": {"OLDX": "GD", "GC": "GA"},
    "ncbi_text": {"OLDX": "GZ"},
    "string_space": {},
}
SOURCE_DIMS: dict[str, int] = {
    "esm2": 4,
    "ncbi_text": 3,
    "context_text": 5,
    "perturbation_text": 5,
    "smiles": 3,
}
GENE_TEXT_DIM = 6


@dataclass(frozen=True)
class TinyData:
    root: Path
    preprocessed: dict[str, Path]
    sources: dict[str, Path]
    gene_text: Path
    split_dir: Path
    aliases: Path
    cache_dir: Path


def _write_dataset(
    out: Path,
    *,
    dataset: str,
    genes: Sequence[str],
    contexts: Sequence[str],
    perts: Sequence[str],
    pert_kind: str,
    control_label: str,
    seed: int,
    no_controls: Sequence[str] = (),
    forced_tested: Sequence[tuple[str, str, str, bool]] = (),
) -> Path:
    rng = np.random.default_rng(seed)
    keys = sorted((c, p) for c in contexts for p in perts)
    n, g = len(keys), len(genes)
    tested = rng.random((n, g)) < 0.85
    for context, pert, gene, value in forced_tested:
        tested[keys.index((context, pert)), list(genes).index(gene)] = value
    fc_raw = np.exp2(rng.normal(0.0, 0.8, (n, g))).astype(np.float32)
    fdr_raw = rng.uniform(0.0, 0.2, (n, g)).astype(np.float32)
    delta_p = rng.normal(0.0, 0.3, (n, g)).astype(np.float32)
    ctrl_means = rng.uniform(0.0, 5.0, (len(contexts), g)).astype(np.float32)
    for context in no_controls:
        ctrl_means[list(contexts).index(context)] = np.nan
        for row, (row_context, _) in enumerate(keys):
            if row_context == context:
                delta_p[row] = np.nan
    context_to_id = {c: i for i, c in enumerate(contexts)}
    pert_to_id = {p: i for i, p in enumerate(sorted(perts))}
    arrays = {
        FOLD_CHANGES: np.where(tested, fc_raw, np.float32(0.0)).astype(np.float32),
        FDR: np.where(tested, fdr_raw, np.float32(1.0)).astype(np.float32),
        TESTED: tested,
        LFC_TRUE: np.where(tested, np.log2(fc_raw.astype(np.float64)), np.nan),
        DELTA_P: delta_p,
        CTRL_MEANS: ctrl_means,
        CTX_IDS: np.array([context_to_id[c] for c, _ in keys], dtype=np.int32),
        PERT_IDS: np.array([pert_to_id[p] for _, p in keys], dtype=np.int32),
    }
    meta = PreprocessedMeta(
        format_version=FORMAT_VERSION,
        dataset=dataset,
        genes=list(genes),
        context_to_id=context_to_id,
        pert_to_id=pert_to_id,
        pert_kind=pert_kind,
        control_label=control_label,
        num_rows=n,
        num_genes=g,
        num_contexts=len(contexts),
        num_perts=len(perts),
        controls_only=False,
        tool_version=pie.__version__,
        array_sha256={},
    )
    return write_preprocessed(out, meta, arrays)


def _source_meta(
    name: str, layout: str, index: str, keys: Sequence[str], dim: int, dtype: str
) -> contract.SourceMeta:
    return contract.SourceMeta(
        format_version=contract.FORMAT_VERSION,
        name=name,
        layout=layout,
        index=index,
        keys=list(keys),
        dim=dim,
        dtype=dtype,
        provenance={"tool_version": pie.__version__},
    )


def _dense_source(
    out: Path, *, name: str, index: str, keys: Sequence[str], dim: int, dtype: str, seed: int
) -> Path:
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(len(keys), dim)).astype(dtype)
    meta = _source_meta(name, "dense", index, keys, dim, dtype)
    return contract.write_source(out, meta, embeddings)


def _token_source(
    out: Path, *, name: str, keys: Sequence[str], lengths: Sequence[int], dim: int, seed: int
) -> Path:
    rng = np.random.default_rng(seed)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    embeddings = rng.normal(size=(int(offsets[-1]), dim)).astype(np.float16)
    meta = _source_meta(name, "token", "gene", keys, dim, "float16")
    return contract.write_source(out, meta, embeddings, offsets=offsets)


def build_tiny_data(root: Path) -> TinyData:
    """Write the two synthetic datasets, five model sources, gene_text, splits and aliases."""
    root.mkdir(parents=True, exist_ok=True)
    pre = root / "preprocessed"
    preprocessed = {
        "alpha": _write_dataset(
            pre / "alpha",
            dataset="alpha",
            genes=ALPHA_GENES,
            contexts=["a1", "a2"],
            perts=["GA", "GB", "GC", "OLDX"],
            pert_kind="gene",
            control_label="non-targeting",
            seed=1,
            forced_tested=(("a1", "GB", "GB", True), ("a1", "GC", "GC", False)),
        ),
        "beta": _write_dataset(
            pre / "beta",
            dataset="beta",
            genes=BETA_GENES,
            contexts=["b1", "b2"],
            perts=["drugA", "drugB", "drugC"],
            pert_kind="drug",
            control_label="DMSO",
            seed=2,
            no_controls=("b2",),
        ),
    }
    src = root / "sources"
    sources = {
        "esm2": _dense_source(
            src / "esm2", name="esm2", index="gene", keys=["GA", "GB", "GC", "GD"],
            dim=4, dtype="float32", seed=3,
        ),
        "ncbi_text": _token_source(
            src / "ncbi_text", name="ncbi_text", keys=["GA", "GB", "GD"], lengths=[2, 3, 1],
            dim=3, seed=4,
        ),
        "context_text": _dense_source(
            src / "context_text", name="context_text", index="context",
            keys=["a1", "a2", "b1", "b2"], dim=5, dtype="float32", seed=5,
        ),
        "perturbation_text": _dense_source(
            src / "perturbation_text", name="perturbation_text", index="pert",
            keys=["GA", "GB", "GC", "drugA", "drugB", "drugC"], dim=5, dtype="float32", seed=6,
        ),
        "smiles": _dense_source(
            src / "smiles", name="smiles", index="pert", keys=["drugA", "drugB"],
            dim=3, dtype="float16", seed=7,
        ),
    }
    gene_text = _dense_source(
        src / "gene_text", name="gene_text", index="gene", keys=[*GENE_AXIS, "GZ"],
        dim=GENE_TEXT_DIM, dtype="float32", seed=8,
    )
    split_dir = root / "splits"
    split_dir.mkdir()
    for file_name, split in (
        ("train.json", TRAIN_SPLIT),
        ("val.json", VAL_SPLIT),
        ("test.json", TEST_SPLIT),
    ):
        (split_dir / file_name).write_text(json.dumps(split, indent=1) + "\n")
    for name, entries in ALIASES.items():
        if entries:
            OmegaConf.save(OmegaConf.create(entries), sources[name] / "aliases.yaml")
    aliases = root / "aliases.yaml"  # the same tables in the multi-source format (extra files)
    OmegaConf.save(OmegaConf.create(ALIASES), aliases)
    return TinyData(
        root=root,
        preprocessed=preprocessed,
        sources=sources,
        gene_text=gene_text,
        split_dir=split_dir,
        aliases=aliases,
        cache_dir=root / "cache",
    )


def tiny_data_config(tiny: TinyData, **overrides: Any) -> Any:
    """A DataConfig over `tiny` (alpha then beta, all five sources); overrides replace fields."""
    from pie.data.datamodule import DataConfig
    from pie.data.delta_p import DeltaPConfig
    from pie.data.evidence import EvidenceConfig

    values: dict[str, Any] = {
        "preprocessed_dirs": [str(tiny.preprocessed["alpha"]), str(tiny.preprocessed["beta"])],
        "dataset_weights": None,
        "split_dir": str(tiny.split_dir),
        "source_dirs": {name: str(path) for name, path in tiny.sources.items()},
        "gene_text_dir": str(tiny.gene_text),
        "aliases_path": None,
        "delta_p": DeltaPConfig(bin_width_fold_change=1.5),
        "evidence": EvidenceConfig(seed=0, chunk=2, lfc_clip_percentile=95.0),
        "batch_size": 4,
        "num_workers": 0,
        "fdr_threshold": 0.05,
    }
    values.update(overrides)
    return DataConfig(**values)


TINY_MODEL_OVERRIDES: tuple[str, ...] = (
    "model.d_model=8",
    "model.n_latents=4",
    "model.n_encoder_layers=1",
    "model.n_processor_layers=1",
    "model.n_decoder_layers=1",
    "model.num_heads=2",
    "model.inference_chunk_size=3",
    "model.evidence.encoder_dim=8",
    "model.evidence.dim=8",
)


def required_train_overrides(
    tiny: TinyData,
    *,
    experiment_name: str,
    devices: int,
    num_nodes: int,
    max_steps: int,
    accumulate_grad_batches: int,
) -> list[str]:
    """Hydra overrides for exactly the `???` keys of configs/train.yaml, with data.* on `tiny`."""
    dirs = ",".join(f"'{tiny.preprocessed[name]}'" for name in ("alpha", "beta"))
    sources = ",".join(f"{name}:'{path}'" for name, path in tiny.sources.items())
    return [
        f"experiment_name={experiment_name}",
        f"data.preprocessed_dirs=[{dirs}]",
        "data.dataset_weights=null",
        f"data.split_dir='{tiny.split_dir}'",
        f"data.source_dirs={{{sources}}}",
        "data.delta_p.bin_width_fold_change=1.5",
        "data.evidence.seed=0",
        "data.evidence.chunk=2",
        f"trainer.devices={devices}",
        f"trainer.num_nodes={num_nodes}",
        f"trainer.max_steps={max_steps}",
        f"trainer.accumulate_grad_batches={accumulate_grad_batches}",
    ]


def train_overrides(
    tiny: TinyData,
    *,
    experiment_name: str = "tiny_run",
    max_steps: int = 2,
    val_every_n_steps: int = 1,
    devices: int = 1,
    accumulate_grad_batches: int = 1,
    logger: bool = False,
    extra: Sequence[str] = (),
) -> list[str]:
    """Overrides of configs/train.yaml for a tiny CPU run on `tiny` (fp32, batch 2, no workers)."""
    return [
        *required_train_overrides(
            tiny,
            experiment_name=experiment_name,
            devices=devices,
            num_nodes=1,
            max_steps=max_steps,
            accumulate_grad_batches=accumulate_grad_batches,
        ),
        f"data.gene_text_dir='{tiny.gene_text}'",
        "data.batch_size=2",
        "data.num_workers=0",
        "trainer.accelerator=cpu",
        "trainer.precision=32-true",
        f"trainer.val_every_n_steps={val_every_n_steps}",
        "trainer.log_every_n_steps=1",
        "scheduler.constant_steps=0",
        f"logger.enabled={'true' if logger else 'false'}",
        *TINY_MODEL_OVERRIDES,
        *extra,
    ]


def set_run_env(mp: Any, tiny: TinyData, runs_root: Path) -> None:
    """PIE_DATA_ROOT = the tiny root, PIE_RUNS_ROOT = `runs_root`, PIE_CACHE_DIR = the tiny cache.

    `mp` is a pytest MonkeyPatch (function fixture or `pytest.MonkeyPatch.context()`).
    """
    mp.setenv("PIE_DATA_ROOT", str(tiny.root))
    mp.setenv("PIE_RUNS_ROOT", str(runs_root))
    mp.setenv("PIE_CACHE_DIR", str(tiny.cache_dir))
