"""ESM2 protein embeddings: mean of the per-residue final hidden states.

A sequence longer than WINDOW residues is encoded in windows starting every STRIDE residues;
each residue's states are averaged over the windows that contain it before the global mean.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from pie.sources import uniprot
from pie.sources.common import base_provenance, environment_record, input_record
from pie.sources.contract import FORMAT_VERSION, SourceMeta, write_source
from pie.sources.text.ncbi import FTP_URLS, fetch_ncbi_ftp

if TYPE_CHECKING:
    from pie.sources.registry import RunContext

log = logging.getLogger(__name__)

MODEL = "facebook/esm2_t33_650M_UR50D"
REVISION = "08e4846e537177426273712802403f7ba8261b6c"
WINDOW = 1022
STRIDE = 511
ATTN_IMPLEMENTATION = "sdpa"
LOG_EVERY = 1000


def _load_model(model_id: str, revision: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        attn_implementation=ATTN_IMPLEMENTATION,
    )
    return tokenizer, model.to(device).eval()


def _hidden(sequence: str, tokenizer: Any, model: Any, device: str) -> torch.Tensor:
    encoded = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=WINDOW + 2)
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.no_grad():
        return model(**encoded).last_hidden_state


def _embed_short(sequence: str, tokenizer: Any, model: Any, device: str) -> torch.Tensor:
    # Positions 1..len are the residues (0 is <cls>, len + 1 is <eos>).
    residues = _hidden(sequence, tokenizer, model, device)[0, 1 : len(sequence) + 1]
    return residues.mean(dim=0).float().cpu()


def _embed_long(sequence: str, tokenizer: Any, model: Any, device: str, dim: int) -> torch.Tensor:
    length = len(sequence)
    total = torch.zeros(length, dim, dtype=torch.float32)
    count = torch.zeros(length, dtype=torch.float32)
    for start in range(0, length, STRIDE):
        end = min(start + WINDOW, length)
        states = _hidden(sequence[start:end], tokenizer, model, device)
        total[start:end] += states[0, 1 : end - start + 1].float().cpu()
        count[start:end] += 1.0
    return (total / count.unsqueeze(-1).clamp(min=1)).mean(dim=0)


def embed_proteins(
    sequences: Sequence[str],
    *,
    device: str,
    model_id: str = MODEL,
    revision: str = REVISION,
) -> np.ndarray:
    """(N, hidden) float32: one mean-pooled vector per sequence, each encoded alone."""
    if any(not sequence for sequence in sequences):
        raise ValueError("empty protein sequence")
    tokenizer, model = _load_model(model_id, revision, device)
    dim = int(model.config.hidden_size)
    out = np.empty((len(sequences), dim), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        if len(sequence) <= WINDOW:
            vector = _embed_short(sequence, tokenizer, model, device)
        else:
            vector = _embed_long(sequence, tokenizer, model, device, dim)
        out[i] = vector.numpy()
        if (i + 1) % LOG_EVERY == 0:
            log.info("esm2: embedded %d/%d proteins", i + 1, len(sequences))
    if not np.isfinite(out).all():
        raise ValueError("ESM2 produced non-finite embeddings")
    return out


def run_esm2(ctx: RunContext) -> Path:
    """esm2: UniProt reviewed human proteome + NCBI protein-coding genes -> ESM2 means."""
    cache = ctx.cache_dir / "esm2"
    stream = uniprot.fetch_reviewed_human(cache, offline=ctx.options.offline)
    offline = ctx.options.offline
    pinned = ctx.options.gene_info
    gene_info = (
        Path(pinned)
        if pinned is not None
        else fetch_ncbi_ftp(cache, names=("gene_info",), offline=offline)["gene_info"]
    )
    table = uniprot.build_proteins_table(stream, gene_info)
    embeddings = embed_proteins(table["sequence"].tolist(), device=ctx.options.device)
    provenance = base_provenance(
        {
            "uniprot_reviewed_human": input_record(stream, uniprot.STREAM_URL),
            "gene_info": input_record(gene_info, FTP_URLS["gene_info"]),
        },
        {
            "window": WINDOW,
            "stride": STRIDE,
            "pooling": "mean",
            "model_dtype": "float16",
            "attn_implementation": ATTN_IMPLEMENTATION,
            "uniprot_query": uniprot.QUERY,
        },
        model=MODEL,
        revision=REVISION,
        proteins_sha256=uniprot.proteins_sha256(table),
        environment=environment_record(ctx.options.device),
    )
    meta = SourceMeta(
        format_version=FORMAT_VERSION,
        name="esm2",
        layout="dense",
        index="pert",
        keys=table["symbol"].tolist(),
        dim=int(embeddings.shape[1]),
        dtype="float32",
        provenance=provenance,
    )
    return write_source(ctx.out_root / "esm2", meta, embeddings, overwrite=ctx.overwrite)
