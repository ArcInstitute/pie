"""ChemBERTa SMILES embeddings: attention-mask weighted mean of the final hidden states.

Every drug-dose key ``<drug>_<dose>uM`` gets its drug's vector, so all doses of a drug share one
row value. SMILES strings are embedded verbatim; a drug without a SMILES string is left out.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch

from pie.sources.chem_profiles import drug_keys, parse_perturbation, read_drug_metadata
from pie.sources.common import base_provenance, environment_record, input_record
from pie.sources.contract import FORMAT_VERSION, SourceMeta, write_source

if TYPE_CHECKING:
    from pie.sources.registry import RunContext

log = logging.getLogger(__name__)

MODEL = "DeepChem/ChemBERTa-77M-MTR"
REVISION = "66b895cab8adebea0cb59a8effa66b2020f204ca"
MAX_LENGTH = 256
BATCH = 64
POOLING = "attention-mask-weighted mean of final hidden states"


def _load_model(model_id: str, revision: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision)
    return tokenizer, model.eval().to(device)


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the positions where `mask` is 1."""
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(1) / weights.sum(1)


def embed_smiles(
    smiles: Sequence[str],
    *,
    device: str,
    model_id: str = MODEL,
    revision: str = REVISION,
) -> np.ndarray:
    """(N, hidden) float32, in `smiles` order, batches of BATCH padded to their longest string."""
    if not smiles:
        raise ValueError("no SMILES strings to embed")
    tokenizer, model = _load_model(model_id, revision, device)
    lengths = [len(ids) for ids in tokenizer(list(smiles))["input_ids"]]
    truncated = sum(n > MAX_LENGTH for n in lengths)
    if truncated:
        log.warning("chemberta: %d SMILES strings exceed %d tokens", truncated, MAX_LENGTH)
    chunks = []
    for start in range(0, len(smiles), BATCH):
        tokens = tokenizer(
            list(smiles[start : start + BATCH]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokens = {name: value.to(device) for name, value in tokens.items()}
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
        chunks.append(masked_mean(hidden, tokens["attention_mask"]).float().cpu())
    vectors = torch.cat(chunks).numpy().astype(np.float32)
    if vectors.shape[0] != len(smiles) or not np.isfinite(vectors).all():
        raise ValueError("ChemBERTa returned an invalid embedding matrix")
    return vectors


@dataclass(frozen=True)
class SmilesInputs:
    keys: list[str]  # covered drug-dose keys, sorted
    smiles: list[str]  # distinct SMILES strings, sorted: the embedding order
    rows: list[int]  # row of `smiles` for each key
    omitted: list[str]  # drugs without a SMILES string


def smiles_inputs(drug_keys: Sequence[str], drug_metadata: pd.DataFrame) -> SmilesInputs:
    """Map drug-dose keys to distinct SMILES strings; names match after stripping whitespace."""
    if not {"drug", "canonical_smiles"} <= set(drug_metadata.columns):
        raise ValueError("drug metadata needs drug and canonical_smiles columns")
    perts = [parse_perturbation(key) for key in sorted(drug_keys)]
    names = drug_metadata["drug"].astype(str).str.strip()
    if names.duplicated().any():
        raise ValueError("duplicate drug names in the drug metadata")
    strings = drug_metadata["canonical_smiles"].fillna("").astype(str)
    smiles_by_drug = dict(zip(names, strings, strict=True))
    drugs = sorted({pert["drug"] for pert in perts})
    missing = [drug for drug in drugs if drug.strip() not in smiles_by_drug]
    if missing:
        raise ValueError(f"missing drug metadata for {missing}")
    smiles = {drug: smiles_by_drug[drug.strip()] for drug in drugs}
    distinct = sorted({smiles[drug] for drug in drugs if smiles[drug]})
    if not distinct:
        raise ValueError("no SMILES strings to embed")
    row_of = {value: i for i, value in enumerate(distinct)}
    covered = [pert for pert in perts if smiles[pert["drug"]]]
    return SmilesInputs(
        keys=[pert["pert"] for pert in covered],
        smiles=distinct,
        rows=[row_of[smiles[pert["drug"]]] for pert in covered],
        omitted=[drug for drug in drugs if not smiles[drug]],
    )


def run_smiles(ctx: RunContext) -> Path:
    """smiles: the drug datasets' keys + a drug table -> one ChemBERTa vector per key."""
    metadata_path = ctx.options.drug_metadata
    if metadata_path is None:
        raise ValueError(
            "smiles needs options.drug_metadata (a table with drug and canonical_smiles columns)"
        )
    inputs = smiles_inputs(drug_keys(ctx.datasets), read_drug_metadata(metadata_path))
    vectors = embed_smiles(inputs.smiles, device=ctx.options.device)
    embeddings = np.ascontiguousarray(vectors[inputs.rows])
    provenance = base_provenance(
        {"drug_metadata": input_record(metadata_path, None)},
        {
            "pooling": POOLING,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH,
            "n_distinct_smiles": len(inputs.smiles),
            "omitted_drugs": inputs.omitted,
        },
        model=MODEL,
        revision=REVISION,
        environment=environment_record(ctx.options.device),
    )
    meta = SourceMeta(
        format_version=FORMAT_VERSION,
        name="smiles",
        layout="dense",
        index="pert",
        keys=inputs.keys,
        dim=int(embeddings.shape[1]),
        dtype="float32",
        provenance=provenance,
    )
    return write_source(ctx.out_root / "smiles", meta, embeddings, overwrite=ctx.overwrite)
