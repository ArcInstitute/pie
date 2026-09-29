"""Tiny synthetic model inputs shared by the model tests."""

from __future__ import annotations

import torch

from pie.data.datamodule import DataStats
from pie.data.dataset import Batch, GeneGroup
from pie.data.delta_p import DeltaPGrid
from pie.data.evidence import BLOCK_KEYS, CONTEXT_BLOCK_KEYS, HAVE_COL
from pie.model.pie import ModelConfig, PieModel

SOURCE_DIMS: dict[str, int] = {"esm2": 6, "context_text": 5}
GENE_QUERY_DIM = 7
N_AXIS = 10
N_BINS = 5
D_MODEL = 16
GroupSpec = tuple[int, tuple[int, ...], tuple[int, ...]]  # (dir_index, rows, gene_ids)
ONE_GROUP: tuple[GroupSpec, ...] = ((0, (0, 1, 2), (0, 1, 2, 3, 4, 5)),)
TWO_GROUPS: tuple[GroupSpec, ...] = (
    (0, (0, 2), (0, 1, 2, 3, 4, 5)),
    (1, (1,), (3, 4, 5, 6, 7, 8, 9)),
)


def tiny_config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "d_model": D_MODEL,
        "n_latents": 4,
        "n_encoder_layers": 1,
        "n_processor_layers": 1,
        "n_decoder_layers": 1,
        "num_heads": 2,
        "ff_mult": 2,
        "dropout": 0.1,
        "drop_src": 0.05,
        "inference_chunk_size": 3,
        "class_weight_cap": 10.0,
        "class_weight_cap_delta_p": 5.0,
        "lfc_huber_delta": 1.0,
        "lfc_target_gene_alpha": 0.001,
        "lfc_direction_temperature": 0.25,
        "evidence": {"encoder_dim": 8, "dim": 8, "dropout": 0.1, "response_dropout": 0.25},
        "temperature": {"de": 4.0, "delta_p": 4.5},
    }
    values.update(overrides)
    return ModelConfig.model_validate(values)


def tiny_grid() -> DeltaPGrid:
    return DeltaPGrid(n_bins=N_BINS, max_delta=0.5, width=0.2)


def tiny_stats(n_donor_datasets: int = 1) -> DataStats:
    return DataStats(
        format_version=1,
        datasets=["dir0", "dir1"],
        delta_p=tiny_grid(),
        per_dir_percentile={"dir0": 0.5, "dir1": None},
        evidence_key="0" * 64,
        evidence_datasets=[f"donor{i}" for i in range(n_donor_datasets)],
        train_json_sha256="0" * 64,
        source_dims=dict(SOURCE_DIMS),
        gene_query_dim=GENE_QUERY_DIM,
    )


def gene_query_text(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(N_AXIS, GENE_QUERY_DIM, generator=generator)


def tiny_model(n_donor_datasets: int = 1, seed: int = 0, **overrides: object) -> PieModel:
    cfg = tiny_config(**overrides)
    stats = tiny_stats(n_donor_datasets)
    queries = gene_query_text()
    torch.manual_seed(seed)
    return PieModel(cfg, stats, dict(SOURCE_DIMS), queries)


def _group(
    spec: GroupSpec, n_donor_datasets: int, generator: torch.Generator
) -> GeneGroup:
    dir_index, rows, gene_ids = spec
    b, n = len(rows), len(gene_ids)
    evidence: dict[str, torch.Tensor] = {}
    for key in (*BLOCK_KEYS, *CONTEXT_BLOCK_KEYS):
        width = 3 * n_donor_datasets if key == "evidence_prov" else 4
        block = torch.rand(b, n, width, generator=generator)
        if key != "evidence_prov":
            block[..., HAVE_COL] = (torch.rand(b, n, generator=generator) > 0.3).float()
        evidence[key] = block
    fold_changes = torch.rand(b, n, generator=generator) * 3.0 + 0.1
    return GeneGroup(
        rows=torch.tensor(rows, dtype=torch.int64),
        dir_index=dir_index,
        gene_ids=torch.tensor(gene_ids, dtype=torch.int64),
        ctrl_means=torch.rand(b, n, generator=generator),
        evidence=evidence,
        fold_changes=fold_changes,
        de_mask=torch.rand(b, n, generator=generator) > 0.5,
        tested=torch.ones(b, n, dtype=torch.bool),
        delta_p=torch.randn(b, n, generator=generator) * 0.2,
        lfc_true=torch.log2(fold_changes).double(),
    )


def tiny_batch(
    groups: tuple[GroupSpec, ...] = ONE_GROUP,
    n_donor_datasets: int = 1,
    seed: int = 0,
    sources: tuple[str, ...] = ("context_text", "esm2"),
) -> Batch:
    generator = torch.Generator().manual_seed(seed)
    b = sum(len(rows) for _, rows, _ in groups)
    names = sorted(sources)
    tokens = {n: torch.randn(b, 2, SOURCE_DIMS[n], generator=generator) for n in names}
    masks = {n: torch.tensor([[True, i % 2 == 0] for i in range(b)]) for n in names}
    dataset_ids = torch.empty(b, dtype=torch.int64)
    for dir_index, rows, _ in groups:
        dataset_ids[list(rows)] = dir_index
    return Batch(
        source_tokens=tokens,
        source_masks=masks,
        dataset_ids=dataset_ids,
        ctx_names=[f"ctx{i % 2}" for i in range(b)],
        pert_names=[f"pert{i}" for i in range(b)],
        row_index=torch.arange(b),
        target_gene=torch.full((b,), -1, dtype=torch.int64),
        target_gene_idx=torch.full((b,), -1, dtype=torch.int64),
        groups=[_group(spec, n_donor_datasets, generator) for spec in groups],
    )
