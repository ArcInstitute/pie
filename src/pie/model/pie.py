"""PIE: knowledge-source tokens -> Perceiver trunk -> gene queries with evidence -> heads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pie.data.datamodule import DataStats
from pie.data.dataset import NO_SOURCE, Batch, GeneGroup
from pie.data.evidence import HAVE_COL
from pie.model import evidence as ev
from pie.model.layers import RMSNorm
from pie.model.trunk import TransformerBlock
from pie.utils import StrictModel

# Rows of the source-type embedding, in canonical order.
SOURCE_REGISTRY: tuple[str, ...] = (
    "esm2",
    "ncbi_text",
    "string_space",
    "depmap_gene_effect",
    "context_text",
    "perturbation_text",
    "smiles",
    "l1000_tas",
    "prism_secondary",
    "jump_morphology",
)
N_DE_CLASSES = 2


class EvidenceModelConfig(StrictModel):
    """model.evidence."""

    encoder_dim: int
    dim: int
    dropout: float
    response_dropout: float


class TemperatureConfig(StrictModel):
    """model.temperature: divisors for DE and delta-p logits outside training."""

    de: float
    delta_p: float


class ModelConfig(StrictModel):
    """model.*: numbers only; structural switches are fixed in code."""

    d_model: int
    n_latents: int
    n_encoder_layers: int
    n_processor_layers: int
    n_decoder_layers: int
    num_heads: int
    ff_mult: int
    dropout: float
    drop_src: float
    inference_chunk_size: int
    class_weight_cap: float
    class_weight_cap_delta_p: float
    lfc_huber_delta: float
    lfc_target_gene_alpha: float
    lfc_direction_temperature: float
    evidence: EvidenceModelConfig
    temperature: TemperatureConfig


@dataclass
class GroupOutput:
    """Raw head outputs of one gene group, on that group's local gene axis."""

    rows: Tensor  # (b,) positions in the batch (= GeneGroup.rows)
    de_logits: Tensor  # (b, G_d, 2)
    lfc: Tensor  # (b, G_d)
    dp_logits: Tensor  # (b, G_d, n_bins)


def _with_kv_carrier(x: Tensor, kv_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Append a zero token, attendable only by rows whose every KV token is masked."""
    empty = ~kv_mask.any(dim=1)
    if not bool(empty.any()):
        return x, kv_mask
    carrier = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
    return torch.cat([x, carrier], dim=1), torch.cat([kv_mask, empty[:, None]], dim=1)


class PieModel(nn.Module):
    """Sources -> Perceiver trunk -> gene queries (+ evidence) -> DE / LFC / delta-p heads.

    Module names and creation order follow the reference model, so its weights load by
    name and the init and dropout RNG streams are unchanged.
    """

    gene_query_text: Tensor

    def __init__(
        self,
        cfg: ModelConfig,
        stats: DataStats,
        source_dims: dict[str, int],
        gene_query_text: Tensor,
    ) -> None:
        super().__init__()
        registry = SOURCE_REGISTRY
        unknown = [name for name in source_dims if name not in registry]
        if unknown:
            raise ValueError(f"sources missing from SOURCE_REGISTRY: {unknown}")
        if gene_query_text.dim() != 2 or gene_query_text.shape[1] != stats.gene_query_dim:
            raise ValueError(
                f"gene_query_text must be (genes, {stats.gene_query_dim}), "
                f"got {tuple(gene_query_text.shape)}"
            )
        self.cfg = cfg
        self.n_bins = stats.delta_p.n_bins
        self.n_donor_datasets = stats.n_donor_datasets
        d = cfg.d_model

        self.source_projections = nn.ModuleDict(
            {
                name: nn.Sequential(RMSNorm(dim), nn.Linear(dim, d, bias=False))
                for name, dim in source_dims.items()
            }
        )
        self.source_k_norms = nn.ModuleDict({name: RMSNorm(d) for name in source_dims})
        self.source_type_embedding = nn.Embedding(len(registry), d)
        self.source_name_to_id = {name: i for i, name in enumerate(registry)}
        self.source_dropout = nn.Dropout(cfg.drop_src)

        self.latents = nn.Parameter(torch.randn(cfg.n_latents, d) * 0.02)
        self.latent_prenorm = RMSNorm(d)
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(d, cfg.num_heads, cfg.ff_mult, cfg.dropout, is_cross=True)
                for _ in range(cfg.n_encoder_layers)
            ]
        )
        self.processor = nn.ModuleList(
            [
                TransformerBlock(d, cfg.num_heads, cfg.ff_mult, cfg.dropout, is_cross=False)
                for _ in range(cfg.n_processor_layers)
            ]
        )
        self.register_buffer(
            "gene_query_text", gene_query_text.to(torch.float32), persistent=False
        )
        d_text = int(gene_query_text.shape[1])
        self.gene_query_mlp = nn.Sequential(
            RMSNorm(d_text),
            nn.Linear(d_text, d, bias=False),
            nn.GELU(),
            nn.Linear(d, d, bias=False),
        )
        self.gene_norm = RMSNorm(d)
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(d, cfg.num_heads, cfg.ff_mult, cfg.dropout, is_cross=True)
                for _ in range(cfg.n_decoder_layers)
            ]
        )

        self.output_head = nn.Linear(d, N_DE_CLASSES)
        self.lfc_head = nn.Linear(d, 1)
        self.delta_p_head = nn.Linear(d, self.n_bins)

        width, dim, p = cfg.evidence.encoder_dim, cfg.evidence.dim, cfg.evidence.dropout
        self.evidence_ctrl_enc = ev.encoder(ev.N_CTRL_FEATURES, width)
        self.evidence_dp_enc = ev.encoder(ev.BLOCK_WIDTH, width)
        self.evidence_de_enc = ev.encoder(ev.BLOCK_WIDTH, width)
        self.evidence_lfc_enc = ev.encoder(ev.BLOCK_WIDTH, width)
        n_modalities = 7
        if self.n_donor_datasets > 1:
            self.evidence_prov_enc = ev.encoder(3 * self.n_donor_datasets, width)
            n_modalities += 1
        self.evidence_ctx_dp_enc = ev.encoder(ev.BLOCK_WIDTH, width)
        self.evidence_ctx_de_enc = ev.encoder(ev.BLOCK_WIDTH, width)
        self.evidence_ctx_lfc_enc = ev.encoder(ev.BLOCK_WIDTH, width)
        self.evidence_fuse = ev.fuse_block(width * n_modalities, dim, p)
        self.evidence_query_adapter = ev.query_adapter(dim, d)
        self.evidence_output_adapters = nn.ModuleDict()
        for head in ("de", "lfc", "delta_p"):
            self.evidence_output_adapters[head] = ev.output_adapter(dim, d, p)

    def forward(self, batch: Batch) -> list[GroupOutput]:
        """Encode the sources once, then decode every gene group on its own gene ids."""
        memory = self._encode(batch)
        return [self._decode_group(memory, group) for group in batch.groups]

    def _encode(self, batch: Batch) -> Tensor:
        x, kv_mask = self._build_kv(
            batch.source_tokens, batch.source_masks, int(batch.dataset_ids.shape[0])
        )
        z = self.latent_prenorm(self.latents.unsqueeze(0).expand(x.shape[0], -1, -1))
        for block in self.encoder:
            z = block(z, kv=x, kv_mask=kv_mask)
        for block in self.processor:
            z = block(z)
        return z

    def _build_kv(
        self, source_tokens: Mapping[str, Tensor], source_masks: Mapping[str, Tensor], b: int
    ) -> tuple[Tensor, Tensor]:
        projected: list[Tensor] = []
        masks: list[Tensor] = []
        for name, projection in self.source_projections.items():
            if name not in source_tokens:
                continue
            tokens = projection(source_tokens[name])
            type_id = torch.tensor(self.source_name_to_id[name], device=tokens.device)
            type_emb = self.source_dropout(self.source_type_embedding(type_id))
            tokens = self.source_k_norms[name](tokens + type_emb)
            projected.append(tokens)
            masks.append(source_masks[name])
        if NO_SOURCE in source_tokens:
            projected.append(torch.zeros(b, 1, self.cfg.d_model, device=self.latents.device))
            masks.append(source_masks[NO_SOURCE])
        return _with_kv_carrier(torch.cat(projected, dim=1), torch.cat(masks, dim=1))

    def _decode_group(self, memory: Tensor, group: GeneGroup) -> GroupOutput:
        device = memory.device
        mem = memory[group.rows.to(device)]
        gene_ids = group.gene_ids.to(device)
        total = int(gene_ids.shape[0])
        chunk = max(total, 1) if self.training else self.cfg.inference_chunk_size
        evidence = ev.drop_evidence(
            group.evidence, self.cfg.evidence.response_dropout, self.training
        )
        adapters = self.evidence_output_adapters
        de_chunks: list[Tensor] = []
        lfc_chunks: list[Tensor] = []
        dp_chunks: list[Tensor] = []
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            feats, fused = self._decode_chunk(
                mem,
                gene_ids[start:end],
                group.ctrl_means[:, start:end].to(device),
                {key: value[:, start:end] for key, value in evidence.items()},
            )
            output_input = torch.cat([fused, feats], dim=-1)
            de_chunks.append(self.output_head(feats + adapters["de"](output_input)))
            lfc_chunks.append(self.lfc_head(feats + adapters["lfc"](output_input)).squeeze(-1))
            dp_chunks.append(self.delta_p_head(feats + adapters["delta_p"](output_input)))
        return GroupOutput(
            rows=group.rows,
            de_logits=torch.cat(de_chunks, dim=1),
            lfc=torch.cat(lfc_chunks, dim=1),
            dp_logits=torch.cat(dp_chunks, dim=1),
        )

    def _decode_chunk(
        self,
        memory: Tensor,
        gene_ids: Tensor,
        ctrl_means: Tensor,
        evidence: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Decoded features (b, g, d) and fused evidence (b, g, dim) for one gene chunk."""
        b = memory.shape[0]
        q = self.gene_query_mlp(self.gene_query_text[gene_ids])
        gene_queries = q.unsqueeze(0).expand(b, -1, -1)
        fused = self._fuse_evidence(ctrl_means, evidence, b, gene_queries.shape[1])
        gene_queries = self.gene_norm(gene_queries + self.evidence_query_adapter(fused))
        for block in self.decoder:
            gene_queries = block(gene_queries, kv=memory, kv_mask=None)
        return gene_queries, fused

    def _fuse_evidence(
        self, ctrl_means: Tensor, evidence: Mapping[str, Tensor], b: int, g: int
    ) -> Tensor:
        device = self.latents.device

        def block(key: str) -> Tensor:
            if key not in evidence:
                raise KeyError(f"evidence block {key!r} missing from the batch")
            x = evidence[key].to(device).float()
            if tuple(x.shape[:2]) != (b, g):
                raise ValueError(
                    f"{key} covers {tuple(x.shape[:2])} rows x genes but {(b, g)} are decoded"
                )
            return x

        def gated(encoder: nn.Module, key: str) -> Tensor:
            x = block(key)
            return encoder(x) * x[..., HAVE_COL : HAVE_COL + 1]

        parts = [
            self.evidence_ctrl_enc(ev.ctrl_query_features(ctrl_means.to(device))),
            gated(self.evidence_dp_enc, "evidence_dp"),
            gated(self.evidence_de_enc, "evidence_de"),
            gated(self.evidence_lfc_enc, "evidence_lfc"),
        ]
        if self.n_donor_datasets > 1:
            prov = block("evidence_prov")
            any_donor = block("evidence_dp")[..., HAVE_COL : HAVE_COL + 1]
            parts.append(self.evidence_prov_enc(prov) * any_donor)
        parts.append(gated(self.evidence_ctx_dp_enc, "evidence_ctx_dp"))
        parts.append(gated(self.evidence_ctx_de_enc, "evidence_ctx_de"))
        parts.append(gated(self.evidence_ctx_lfc_enc, "evidence_ctx_lfc"))
        return self.evidence_fuse(torch.cat(parts, dim=-1))
