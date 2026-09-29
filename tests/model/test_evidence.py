import math

import torch
from torch import nn

from pie.data.evidence import BLOCK_KEYS, CONTEXT_BLOCK_KEYS
from pie.model import evidence as ev


def test_ctrl_query_features_values():
    z = torch.tensor([[0.0, 0.01, 1.0]])
    expected = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.01, math.log1p(1.0), 0.0], [1.0, math.log1p(100.0), 0.0]]]
    )
    torch.testing.assert_close(ev.ctrl_query_features(z), expected)


def test_ctrl_query_features_blank_a_whole_row_with_nan():
    z = torch.tensor([[0.5, float("nan")], [0.5, 0.2]])
    feats = ev.ctrl_query_features(z)
    assert torch.equal(feats[0], torch.zeros(2, 3))
    assert torch.isfinite(feats[1]).all()
    assert feats[1].abs().sum() > 0


def _blocks(b: int, g: int = 3) -> dict[str, torch.Tensor]:
    return {key: torch.ones(b, g, 4) for key in (*BLOCK_KEYS, *CONTEXT_BLOCK_KEYS)}


def test_drop_evidence_is_identity_outside_training():
    blocks = _blocks(4)
    assert ev.drop_evidence(blocks, 0.25, training=False) is blocks
    assert ev.drop_evidence(blocks, 0.0, training=True) is blocks


def test_drop_evidence_draws_donor_then_context_row_masks():
    blocks = _blocks(64)
    torch.manual_seed(7)
    out = ev.drop_evidence(blocks, 0.5, training=True)
    torch.manual_seed(7)
    keep = (torch.rand(64) >= 0.5).float()
    keep_ctx = (torch.rand(64) >= 0.5).float()
    assert not torch.equal(keep, keep_ctx)
    for key in BLOCK_KEYS:
        assert torch.equal(out[key][:, 0, 0], keep)
        assert torch.equal(out[key][:, 2, 3], keep)
    for key in CONTEXT_BLOCK_KEYS:
        assert torch.equal(out[key][:, 1, 1], keep_ctx)


def test_builder_parameter_names():
    names = [n for n, _ in ev.encoder(3, 8).named_parameters()]
    assert names == ["0.weight", "0.bias", "2.weight", "2.bias", "3.weight"]
    names = [n for n, _ in ev.fuse_block(56, 8, 0.1).named_parameters()]
    assert names == ["0.weight", "1.weight", "1.bias", "4.weight", "4.bias", "5.weight"]
    assert ev.fuse_block(56, 8, 0.1)[1].out_features == 16


def test_adapters_end_in_zero_initialised_bias_free_linear():
    q = ev.query_adapter(8, 16)
    o = ev.output_adapter(8, 16, 0.1)
    assert (q[0].in_features, q[0].out_features) == (8, 16)
    assert q[2].bias is None and torch.count_nonzero(q[2].weight) == 0
    assert (o[1].in_features, o[1].out_features) == (24, 8)
    assert o[4].bias is None and torch.count_nonzero(o[4].weight) == 0


def test_zeroed_projection_draws_init_rng_first():
    torch.manual_seed(0)
    out_adapter = ev.output_adapter(8, 16, 0.1)
    torch.manual_seed(0)
    nn.Linear(8, 16, bias=False)
    expected = nn.Linear(24, 8)
    assert torch.equal(out_adapter[1].weight, expected.weight)
    assert torch.equal(out_adapter[1].bias, expected.bias)

    torch.manual_seed(1)
    q_adapter = ev.query_adapter(8, 16)
    torch.manual_seed(1)
    nn.Linear(16, 16, bias=False)
    expected = nn.Linear(8, 16)
    assert torch.equal(q_adapter[0].weight, expected.weight)
    assert torch.equal(q_adapter[0].bias, expected.bias)
