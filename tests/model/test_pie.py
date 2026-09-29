import pytest
import torch
from torch import nn

from pie.model import pie as pie_model_module
from pie.model.pie import SOURCE_REGISTRY, PieModel
from tests.model.helpers import (
    D_MODEL,
    N_BINS,
    ONE_GROUP,
    SOURCE_DIMS,
    TWO_GROUPS,
    gene_query_text,
    tiny_batch,
    tiny_config,
    tiny_model,
    tiny_stats,
)


def _block_names(prefix: str, is_cross: bool) -> list[str]:
    names = [f"{prefix}.norm_attn.weight"]
    if is_cross:
        names.append(f"{prefix}.norm_kv.weight")
    names += [
        f"{prefix}.attn.{p}"
        for p in (
            "attn_scale",
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "out_proj.weight",
            "q_norm.weight",
            "k_norm.weight",
        )
    ]
    return [
        *names,
        f"{prefix}.norm_ff.weight",
        f"{prefix}.ff.fc1.weight",
        f"{prefix}.ff.fc2.weight",
    ]


def _enc(prefix: str) -> list[str]:
    return [f"{prefix}.{p}" for p in ("0.weight", "0.bias", "2.weight", "2.bias", "3.weight")]


def _expected_names(with_prov: bool) -> list[str]:
    names = [
        "latents",
        "source_projections.esm2.0.weight",
        "source_projections.esm2.1.weight",
        "source_projections.context_text.0.weight",
        "source_projections.context_text.1.weight",
        "source_k_norms.esm2.weight",
        "source_k_norms.context_text.weight",
        "source_type_embedding.weight",
        "latent_prenorm.weight",
        *_block_names("encoder.0", True),
        *_block_names("processor.0", False),
        "gene_query_mlp.0.weight",
        "gene_query_mlp.1.weight",
        "gene_query_mlp.3.weight",
        "gene_norm.weight",
        *_block_names("decoder.0", True),
        "output_head.weight",
        "output_head.bias",
        "lfc_head.weight",
        "lfc_head.bias",
        "delta_p_head.weight",
        "delta_p_head.bias",
        *_enc("evidence_ctrl_enc"),
        *_enc("evidence_dp_enc"),
        *_enc("evidence_de_enc"),
        *_enc("evidence_lfc_enc"),
    ]
    if with_prov:
        names += _enc("evidence_prov_enc")
    names += [
        *_enc("evidence_ctx_dp_enc"),
        *_enc("evidence_ctx_de_enc"),
        *_enc("evidence_ctx_lfc_enc"),
        "evidence_fuse.0.weight",
        "evidence_fuse.1.weight",
        "evidence_fuse.1.bias",
        "evidence_fuse.4.weight",
        "evidence_fuse.4.bias",
        "evidence_fuse.5.weight",
        "evidence_query_adapter.0.weight",
        "evidence_query_adapter.0.bias",
        "evidence_query_adapter.2.weight",
    ]
    for head in ("de", "lfc", "delta_p"):
        names += [
            f"evidence_output_adapters.{head}.{p}"
            for p in ("0.weight", "1.weight", "1.bias", "4.weight")
        ]
    return names


def test_source_registry_is_the_ten_canonical_slots():
    assert SOURCE_REGISTRY == (
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


@pytest.mark.parametrize("k", [1, 2])
def test_parameter_names_follow_reference_order(k):
    names = [name for name, _ in tiny_model(n_donor_datasets=k).named_parameters()]
    assert names == _expected_names(with_prov=k > 1)


def test_initialisation_consumes_rng_in_reference_order():
    model = tiny_model(seed=0)
    torch.manual_seed(0)
    esm2 = nn.Linear(6, D_MODEL, bias=False)
    context_text = nn.Linear(5, D_MODEL, bias=False)
    type_embedding = nn.Embedding(len(SOURCE_REGISTRY), D_MODEL)
    latents = torch.randn(4, D_MODEL) * 0.02
    assert torch.equal(model.source_projections["esm2"][1].weight, esm2.weight)
    assert torch.equal(model.source_projections["context_text"][1].weight, context_text.weight)
    assert torch.equal(model.source_type_embedding.weight, type_embedding.weight)
    assert torch.equal(model.latents, latents)


def test_construction_is_deterministic_per_seed():
    a, b = tiny_model(seed=3).state_dict(), tiny_model(seed=3).state_dict()
    assert all(torch.equal(a[key], b[key]) for key in a)
    c = tiny_model(seed=4).state_dict()
    assert not torch.equal(a["latents"], c["latents"])


def test_gene_query_text_is_a_non_persistent_buffer():
    model = tiny_model()
    assert "gene_query_text" not in model.state_dict()
    assert "gene_query_text" in dict(model.named_buffers())
    assert torch.equal(model.gene_query_text, gene_query_text())


def test_type_embedding_rows_follow_the_patched_registry(monkeypatch):
    names = (*(f"slot{i}" for i in range(19)), "esm2", "context_text")
    monkeypatch.setattr(pie_model_module, "SOURCE_REGISTRY", names)
    model = tiny_model()
    assert model.source_type_embedding.num_embeddings == 21
    assert model.source_name_to_id["esm2"] == 19


def test_unknown_source_and_bad_gene_queries_raise():
    with pytest.raises(ValueError, match="not_a_source"):
        PieModel(tiny_config(), tiny_stats(), {"esm2": 6, "not_a_source": 3}, gene_query_text())
    with pytest.raises(ValueError, match="gene_query_text"):
        PieModel(tiny_config(), tiny_stats(), dict(SOURCE_DIMS), torch.zeros(10, 3))


@pytest.mark.parametrize("training", [False, True])
def test_forward_shapes_one_group(training):
    model = tiny_model().train(training)
    outs = model(tiny_batch(ONE_GROUP))
    assert len(outs) == 1
    assert torch.equal(outs[0].rows, torch.tensor([0, 1, 2]))
    assert outs[0].de_logits.shape == (3, 6, 2)
    assert outs[0].lfc.shape == (3, 6)
    assert outs[0].dp_logits.shape == (3, 6, N_BINS)


@pytest.mark.parametrize("training", [False, True])
def test_forward_shapes_two_ragged_groups(training):
    model = tiny_model().train(training)
    first, second = model(tiny_batch(TWO_GROUPS))
    assert first.de_logits.shape == (2, 6, 2)
    assert second.de_logits.shape == (1, 7, 2)
    assert second.lfc.shape == (1, 7)
    assert second.dp_logits.shape == (1, 7, N_BINS)
    assert torch.isfinite(first.dp_logits).all() and torch.isfinite(second.lfc).all()


def test_training_decodes_the_whole_axis_in_one_chunk_and_eval_chunks():
    model = tiny_model()
    calls = []

    def count(*_: object) -> None:
        calls.append(1)

    model.gene_query_mlp.register_forward_hook(count)
    batch = tiny_batch(ONE_GROUP)
    model.train()(batch)
    assert len(calls) == 1
    calls.clear()
    model.eval()(batch)
    assert len(calls) == 2  # 6 genes, inference_chunk_size 3


def test_eval_chunking_does_not_change_outputs():
    model = tiny_model().eval()
    batch = tiny_batch(TWO_GROUPS)
    with torch.no_grad():
        chunked = model(batch)
        model.cfg = tiny_config(inference_chunk_size=100)
        whole = model(batch)
    for a, b in zip(chunked, whole, strict=True):
        torch.testing.assert_close(a.de_logits, b.de_logits)
        torch.testing.assert_close(a.lfc, b.lfc)
        torch.testing.assert_close(a.dp_logits, b.dp_logits)


def test_provenance_encoder_exists_iff_more_than_one_donor_dataset():
    one = tiny_model(n_donor_datasets=1)
    assert not hasattr(one, "evidence_prov_enc")
    assert one.evidence_fuse[1].in_features == 7 * 8
    three = tiny_model(n_donor_datasets=3)
    assert three.evidence_prov_enc[0].in_features == 9
    assert three.evidence_fuse[1].in_features == 8 * 8
    outs = three.eval()(tiny_batch(TWO_GROUPS, n_donor_datasets=3))
    assert outs[1].dp_logits.shape == (1, 7, N_BINS)


class _Zero(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(*x.shape[:-1], D_MODEL)


def test_zero_init_adapters_equal_no_adapter_at_init():
    model = tiny_model().eval()
    batch = tiny_batch(TWO_GROUPS)
    with torch.no_grad():
        ref = model(batch)
        model.evidence_query_adapter = _Zero()
        for head in ("de", "lfc", "delta_p"):
            model.evidence_output_adapters[head] = _Zero()
        out = model(batch)
    for a, b in zip(ref, out, strict=True):
        assert torch.equal(a.de_logits, b.de_logits)
        assert torch.equal(a.lfc, b.lfc)
        assert torch.equal(a.dp_logits, b.dp_logits)


def test_outputs_ignore_evidence_values_at_init():
    model = tiny_model().eval()
    batch = tiny_batch(TWO_GROUPS)
    with torch.no_grad():
        ref = model(batch)
        for group in batch.groups:
            group.evidence = {k: torch.rand_like(v) for k, v in group.evidence.items()}
            group.ctrl_means = torch.rand_like(group.ctrl_means)
        out = model(batch)
    for a, b in zip(ref, out, strict=True):
        assert torch.equal(a.de_logits, b.de_logits)
        assert torch.equal(a.lfc, b.lfc)


def test_two_training_forwards_with_the_same_seed_are_identical():
    model = tiny_model().train()
    batch = tiny_batch(TWO_GROUPS)
    torch.manual_seed(123)
    first = model(batch)
    torch.manual_seed(123)
    second = model(batch)
    torch.manual_seed(124)
    third = model(batch)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a.de_logits, b.de_logits)
        assert torch.equal(a.lfc, b.lfc)
        assert torch.equal(a.dp_logits, b.dp_logits)
    assert not torch.equal(first[0].lfc, third[0].lfc)


def test_absent_source_and_no_source_carrier():
    model = tiny_model().eval()
    outs = model(tiny_batch(ONE_GROUP, sources=("esm2",)))
    assert torch.isfinite(outs[0].lfc).all()
    batch = tiny_batch(ONE_GROUP, sources=())
    batch.source_tokens["__no_source__"] = torch.zeros(3, 1, 1)
    batch.source_masks["__no_source__"] = torch.ones(3, 1, dtype=torch.bool)
    outs = model(batch)
    assert torch.isfinite(outs[0].de_logits).all()
