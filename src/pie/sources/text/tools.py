"""Build glue of the OpenAI-embedded text tools: describe, extend the prior, embed, write."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

import pie
from pie.sources.common import provenance_path
from pie.sources.contract import (
    DESCRIPTIONS,
    FORMAT_VERSION,
    META,
    SourceMeta,
    extend_descriptions,
    extend_embeddings,
    read_descriptions,
    read_source,
    write_source,
)
from pie.sources.embed import openai as openai_embed
from pie.sources.text import contexts, drugs, genes
from pie.utils import canonical_json, sha256_bytes, sha256_file

if TYPE_CHECKING:
    from pie.sources.registry import RunContext


def ordered_keys(groups: Iterable[Sequence[str]]) -> list[str]:
    """Each group sorted (codepoint order), groups in the given order, first occurrence kept."""
    seen: dict[str, None] = {}
    for group in groups:
        for key in sorted(group):
            seen.setdefault(key, None)
    return list(seen)


def prior_dir(ctx: RunContext, name: str) -> Path | None:
    if ctx.prior_root is None:
        return None
    path = ctx.prior_root / name
    return path if (path / META).exists() else None


def _prior_texts(ctx: RunContext, name: str) -> dict[str, str] | None:
    path = prior_dir(ctx, name)
    return read_descriptions(path) if path is not None else None


def write_text_source(
    ctx: RunContext,
    name: str,
    index: Literal["pert", "context", "gene"],
    texts: Mapping[str, str],
    params: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Extend the prior (rows verbatim), embed new or replaced texts, write <out_root>/<name>."""
    prior_path = prior_dir(ctx, name)
    prior = read_source(prior_path) if prior_path is not None else None
    prior_texts = read_descriptions(prior_path) if prior_path is not None else {}
    merged = extend_descriptions(prior_texts or None, texts, ctx.options.on_conflict)
    resume = ctx.cache_dir.parent / "embed" / name

    def embed(keys: list[str]) -> np.ndarray:
        return openai_embed.embed_texts([merged[k] for k in keys], resume_dir=resume)

    rows, keys = extend_embeddings(prior, list(merged), embed)
    changed = [k for k, text in prior_texts.items() if merged[k] != text]
    if changed:
        rows = np.array(rows, copy=True)
        fresh = openai_embed.embed_texts(
            [merged[k] for k in changed], resume_dir=resume / "replaced"
        )
        position = {k: i for i, k in enumerate(keys)}
        for key, row in zip(changed, fresh, strict=True):
            rows[position[key]] = row
    meta = SourceMeta(
        format_version=FORMAT_VERSION,
        name=name,
        layout="dense",
        index=index,
        keys=keys,
        dim=openai_embed.DIM,
        dtype="float32",
        provenance={
            "tool_version": pie.__version__,
            "inputs": {key: dict(value) for key, value in inputs.items()},
            "model": openai_embed.MODEL,
            "revision": None,  # the OpenAI API serves the model by name only
            "batch": openai_embed.BATCH,
            "max_tokens": openai_embed.MAX_TOKENS,
            "prior": provenance_path(prior_path) if prior_path is not None else None,
            "on_conflict": ctx.options.on_conflict,
            "params": dict(params),
        },
    )
    return write_source(
        ctx.out_root / name, meta, rows, descriptions=merged, overwrite=ctx.overwrite
    )


def _file_input(path: Path, url: str | None = None) -> dict[str, Any]:
    path = Path(path)
    return {"url": url, "sha256": sha256_file(path) if path.is_file() else None, "release": None}


def _tree_sha256(root: Path) -> str | None:
    """sha256 over {relative path: sha256} of every file under root (None when empty)."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
    if not files:
        return None
    digests = {p.relative_to(root).as_posix(): sha256_file(p) for p in files}
    return sha256_bytes(canonical_json(digests))


def _gene_clients(ctx: RunContext, cache_dir: Path) -> tuple[Any, Any, dict[str, dict]]:
    """(GeneInfoIndex, EsummaryClient, inputs so far) with caches under `cache_dir`."""
    offline = ctx.options.offline
    gene_info = ctx.options.gene_info or genes.fetch_gene_info(cache_dir, offline=offline)
    index = genes.GeneInfoIndex.load(gene_info)
    esummary = genes.EsummaryClient(cache_dir, os.environ.get("NCBI_API_KEY"), offline=offline)
    return index, esummary, {"ncbi_gene_info": _file_input(gene_info, genes.GENE_INFO_URL)}


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if Path(path).suffix == ".csv" else pd.read_parquet(path)


def run_context_text(ctx: RunContext) -> Path:
    opts = ctx.options
    client = contexts.CellosaurusClient(
        ctx.cache_dir / "context_text", opts.cellosaurus_release, opts.offline
    )
    described = contexts.describe_contexts(ctx.datasets, opts.contexts_dir, client)
    order = ordered_keys(d.contexts for d in ctx.datasets)
    params = {
        "cellosaurus_release": opts.cellosaurus_release,
        "contexts_dir": provenance_path(opts.contexts_dir),
    }
    record = client.provenance()
    inputs = {
        "cellosaurus": {
            "url": record["url"],
            "sha256": sha256_bytes(canonical_json(record["records"])),
            "release": record["release"],
        }
    }
    texts = {k: described[k] for k in order}
    return write_text_source(ctx, "context_text", "context", texts, params, inputs)


def _without_control(d: Any) -> list[str]:
    return sorted({key for key in d.perts if key != d.meta.control_label})


def run_perturbation_text(ctx: RunContext) -> Path:
    """Genetic perts of the first genetic dataset via describe_genetic_perts (control included).

    Every later genetic dataset takes the gene-query path: describe_gene_queries over its perts
    without the control key, with prior = the prior gene_text descriptions and pert_output = the
    first genetic dataset's texts as they land in the output (prior text first). Drug datasets
    go through PubChem. Row order: each dataset's keys sorted, datasets in the given order.
    The first genetic dataset's Ensembl ids come from its preprocessed dir.
    """
    opts = ctx.options
    gene_sets = [d for d in ctx.datasets if d.meta.pert_kind == "gene"]
    drug_sets = [d for d in ctx.datasets if d.meta.pert_kind == "drug"]
    if drug_sets and opts.drug_metadata is None:
        raise ValueError("perturbation_text for a drug dataset needs options.drug_metadata")
    cache_dir = ctx.cache_dir / "perturbation_text"
    described: dict[str, str] = {}
    inputs: dict[str, dict] = {}
    first = gene_sets[0] if gene_sets else None
    if first is not None:
        index, esummary, inputs = _gene_clients(ctx, cache_dir)
        ensembl = first.pert_ensembl
        first_texts = genes.describe_genetic_perts(
            first.perts, index, esummary, ensembl, first.meta.control_label
        )
        described.update(first_texts)
        if len(gene_sets) > 1:
            pert_output = extend_descriptions(
                _prior_texts(ctx, "perturbation_text"), first_texts, opts.on_conflict
            )
            pert_keys = _without_control(first)
            prior_genes = _prior_texts(ctx, "gene_text")
            for d in gene_sets[1:]:
                texts = genes.describe_gene_queries(
                    _without_control(d), prior_genes, pert_output, pert_keys, index, esummary
                )
                for key, text in texts.items():
                    described.setdefault(key, text)
        inputs.update(esummary.provenance())
    if drug_sets:
        assert opts.drug_metadata is not None
        metadata = _read_table(opts.drug_metadata)
        client = drugs.PubChemClient(cache_dir, offline=opts.offline)
        for d in drug_sets:
            texts = drugs.describe_drug_perts(d.perts, metadata, client, d.meta.control_label)
            for key, text in texts.items():
                described.setdefault(key, text)
        inputs["drug_metadata"] = _file_input(opts.drug_metadata)
        inputs["pubchem"] = {
            "url": drugs.BASE_URL,
            "sha256": _tree_sha256(client.cache_dir),
            "release": None,
        }
    order = ordered_keys(
        _without_control(d) if d.meta.pert_kind == "gene" and d is not first else d.perts
        for d in ctx.datasets
    )
    params = {
        "gene_info": provenance_path(opts.gene_info) if opts.gene_info else None,
        "drug_metadata": provenance_path(opts.drug_metadata) if opts.drug_metadata else None,
    }
    texts = {k: described[k] for k in order}
    return write_text_source(ctx, "perturbation_text", "pert", texts, params, inputs)


def run_gene_text(ctx: RunContext) -> Path:
    """Gene text over one codepoint-sorted gene axis (the union of every dataset's genes)."""
    pert_dir = ctx.options.pert_output or ctx.out_root / "perturbation_text"
    if not (pert_dir / META).exists():
        raise FileNotFoundError(f"gene_text needs a perturbation_text source at {pert_dir}")
    pert_output = read_descriptions(pert_dir)
    pert_keys = ordered_keys(d.perts for d in ctx.datasets if d.meta.pert_kind == "gene")
    vocab = sorted({g for d in ctx.datasets for g in d.genes})
    prior = _prior_texts(ctx, "gene_text")
    index, esummary, inputs = _gene_clients(ctx, ctx.cache_dir / "gene_text")
    described = genes.describe_gene_queries(vocab, prior, pert_output, pert_keys, index, esummary)
    inputs.update(esummary.provenance())
    inputs["perturbation_text"] = _file_input(Path(pert_dir) / DESCRIPTIONS)
    params = {"pert_output": provenance_path(pert_dir)}
    texts = {g: described[g] for g in vocab}
    return write_text_source(ctx, "gene_text", "gene", texts, params, inputs)
