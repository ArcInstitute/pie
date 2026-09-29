import pytest
import torch
import torch.nn.functional as F
from torch import nn

from pie.model import layers
from pie.model.layers import RMSNorm, rms_norm


def test_rms_norm_matches_formula_with_eps():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 8)
    w = torch.randn(8)
    eps = 1e-5
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w
    torch.testing.assert_close(rms_norm(x, w, eps), expected, rtol=1e-6, atol=1e-6)


def test_rms_norm_default_eps_is_dtype_epsilon():
    x = torch.tensor([[1e-4, -2e-4, 3e-4, 0.0]])
    w = torch.ones(4)
    eps = torch.finfo(torch.float32).eps
    expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    torch.testing.assert_close(rms_norm(x, w, None), expected, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_forward_and_grads_equal_torch(dtype):
    torch.manual_seed(2)
    x = torch.randn(4, 6, 16).to(dtype)
    w = torch.randn(16)
    upstream = torch.randn(4, 6, 16).to(dtype)
    results = []
    for fn in (rms_norm, lambda x, w, eps: F.rms_norm(x, (x.shape[-1],), w, eps)):
        xi = x.clone().requires_grad_(True)
        wi = w.clone().requires_grad_(True)
        out = fn(xi, wi, None)
        out.backward(upstream)
        results.append((out, xi.grad, wi.grad))
    for ours, ref in zip(*results, strict=True):
        assert ours.dtype == ref.dtype
        assert torch.equal(ours, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_rms_norm_saves_only_inputs_on_cuda():
    x = torch.randn(8, 32, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(64, device="cuda", requires_grad=True)
    saved = []
    hooks = torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t)
    with hooks, torch.autocast("cuda", dtype=torch.bfloat16):
        rms_norm(x, w, None)
    assert sum(t.nbytes for t in saved) == x.nbytes + w.nbytes


def test_rmsnorm_module_equals_torch_rmsnorm():
    torch.manual_seed(1)
    ours = RMSNorm(8)
    ref = nn.RMSNorm(8)
    with torch.no_grad():
        ours.weight.copy_(torch.randn(8))
        ref.weight.copy_(ours.weight)
    x = torch.randn(4, 8)
    assert torch.equal(ours(x), ref(x))
    assert [name for name, _ in ours.named_parameters()] == ["weight"]


def test_rmsnorm_forward_calls_module_level_seam(monkeypatch):
    calls = []

    def fake(x, weight, eps):
        calls.append(eps)
        return torch.zeros_like(x)

    monkeypatch.setattr(layers, "rms_norm", fake)
    out = RMSNorm(4, eps=1e-3)(torch.ones(2, 4))
    assert calls == [1e-3]
    assert torch.equal(out, torch.zeros(2, 4))
