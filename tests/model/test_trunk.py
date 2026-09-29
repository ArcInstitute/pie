import math

import torch
import torch.nn.functional as F

from pie.model.trunk import FeedForward, MultiHeadAttention, TransformerBlock


def test_cross_block_shape_and_parameter_names():
    torch.manual_seed(0)
    block = TransformerBlock(16, 2, 2, 0.1, is_cross=True)
    x = torch.randn(3, 4, 16)
    kv = torch.randn(3, 5, 16)
    assert block(x, kv=kv).shape == (3, 4, 16)
    assert [name for name, _ in block.named_parameters()] == [
        "norm_attn.weight",
        "norm_kv.weight",
        "attn.attn_scale",
        "attn.q_proj.weight",
        "attn.k_proj.weight",
        "attn.v_proj.weight",
        "attn.out_proj.weight",
        "attn.q_norm.weight",
        "attn.k_norm.weight",
        "norm_ff.weight",
        "ff.fc1.weight",
        "ff.fc2.weight",
    ]


def test_self_block_has_no_kv_norm():
    block = TransformerBlock(16, 2, 2, 0.1, is_cross=False)
    assert not hasattr(block, "norm_kv")
    assert block(torch.randn(2, 3, 16)).shape == (2, 3, 16)


def test_attention_scale_initialised_to_inverse_sqrt_head_dim():
    attn = MultiHeadAttention(16, 4, 0.0)
    torch.testing.assert_close(attn.attn_scale.detach(), torch.full((4,), 1.0 / math.sqrt(4)))
    assert all(
        lin.bias is None for lin in (attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj)
    )


def test_masked_kv_tokens_do_not_affect_output():
    torch.manual_seed(1)
    block = TransformerBlock(16, 2, 2, 0.1, is_cross=True).eval()
    x = torch.randn(2, 3, 16)
    kv = torch.randn(2, 5, 16)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    kv_changed = kv.clone()
    kv_changed[0, 3:] = 100.0 * torch.randn(2, 16)
    torch.testing.assert_close(
        block(x, kv=kv, kv_mask=mask), block(x, kv=kv_changed, kv_mask=mask)
    )


def test_eval_is_deterministic_and_train_uses_global_rng():
    torch.manual_seed(2)
    block = TransformerBlock(16, 2, 2, 0.5, is_cross=False)
    x = torch.randn(2, 3, 16)
    block.eval()
    assert torch.equal(block(x), block(x))
    block.train()
    torch.manual_seed(3)
    first = block(x)
    torch.manual_seed(3)
    second = block(x)
    torch.manual_seed(4)
    third = block(x)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_feed_forward_is_bias_free_gelu_mlp():
    ff = FeedForward(4, 2, 0.0).eval()
    x = torch.randn(2, 4)
    assert ff.fc1.bias is None and ff.fc2.bias is None
    assert torch.equal(ff(x), ff.fc2(F.gelu(ff.fc1(x))))
