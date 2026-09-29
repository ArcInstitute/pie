import math

import torch

from pie.data.delta_p import DeltaPGrid
from pie.model.heads import readout, temper
from pie.model.pie import GroupOutput
from tests.model.helpers import tiny_config

LN3 = math.log(3.0)


def _output() -> GroupOutput:
    return GroupOutput(
        rows=torch.tensor([0]),
        de_logits=torch.tensor([[[0.0, 4.0 * LN3], [0.0, 0.0], [float("nan"), 0.0]]]),
        lfc=torch.tensor([[1.5, -2.0, 0.25]]),
        dp_logits=torch.tensor([[[0.0, 1.0, 0.5], [3.0, 0.0, 0.0], [0.0, 0.0, 9.0]]]),
    )


def test_temper_divides_only_the_classification_logits():
    cfg = tiny_config()
    out = _output()
    tempered = temper(out, cfg)
    torch.testing.assert_close(tempered.de_logits[0, :2], out.de_logits[0, :2] / 4.0)
    torch.testing.assert_close(tempered.dp_logits, out.dp_logits / 4.5)
    assert tempered.lfc is out.lfc
    assert tempered.rows is out.rows


def test_readout_probability_lfc_and_bin_centres():
    cfg = tiny_config()
    grid = DeltaPGrid(n_bins=3, max_delta=0.3, width=0.2)
    got = readout(_output(), cfg, grid)
    # softmax([0, 4 ln 3] / 4) = [1/4, 3/4]; NaN logits read out as 0.
    torch.testing.assert_close(got.p_de, torch.tensor([[0.75, 0.5, 0.0]]))
    torch.testing.assert_close(got.lfc, torch.tensor([[1.5, -2.0, 0.25]]))
    # Bin centres for 3 bins over +-0.3: -0.2, 0.0, 0.2.
    torch.testing.assert_close(got.delta_p, torch.tensor([[0.0, -0.2, 0.2]]))
    assert got.p_de.dtype == got.lfc.dtype == got.delta_p.dtype == torch.float32


def test_readout_returns_float32_for_bfloat16_outputs():
    # Under bf16 autocast the heads emit bf16; the readout must still hand float32 to metrics.
    cfg = tiny_config()
    grid = DeltaPGrid(n_bins=3, max_delta=0.3, width=0.2)
    out = _output()
    half = GroupOutput(
        rows=out.rows,
        de_logits=out.de_logits.bfloat16(),
        lfc=out.lfc.bfloat16(),
        dp_logits=out.dp_logits.bfloat16(),
    )
    got = readout(half, cfg, grid)
    assert got.p_de.dtype == got.lfc.dtype == got.delta_p.dtype == torch.float32
    torch.testing.assert_close(got.p_de, torch.tensor([[0.75, 0.5, 0.0]]), atol=1e-2, rtol=0)
    torch.testing.assert_close(got.lfc, torch.tensor([[1.5, -2.0, 0.25]]))
    torch.testing.assert_close(got.delta_p, torch.tensor([[0.0, -0.2, 0.2]]))
