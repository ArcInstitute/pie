import math

import pytest
import torch

from pie.data.delta_p import two_hot
from pie.model.heads import temper
from pie.model.losses import (
    LOSS_KEYS,
    compute_losses,
    de_loss,
    dp_loss,
    lfc_direction_loss,
    lfc_loss,
    lfc_targets,
)
from pie.model.pie import GroupOutput
from tests.model.helpers import (
    N_BINS,
    TWO_GROUPS,
    tiny_batch,
    tiny_config,
    tiny_grid,
    tiny_model,
)

LN2 = math.log(2.0)
LN3 = math.log(3.0)
LN4_3 = math.log(4.0 / 3.0)


def test_de_loss_is_a_class_weighted_mean_over_tested_cells():
    logits = torch.tensor([[[0.0, LN3], [0.0, 0.0], [LN3, 0.0], [5.0, -5.0]]])
    targets = torch.tensor([[1, 0, 0, 0]])
    valid = torch.tensor([[True, True, True, False]])
    # Tested cells: one positive, two negatives, n = 3 -> weights 3/(2*1) = 1.5, 3/(2*2) = 0.75.
    expected = (1.5 * LN4_3 + 0.75 * LN2 + 0.75 * LN4_3) / 3
    assert de_loss(logits, targets, valid, cap=10.0).item() == pytest.approx(expected, rel=1e-6)


def test_de_loss_caps_class_weights_and_handles_no_tested_cells():
    logits = torch.zeros(1, 11, 2)
    targets = torch.zeros(1, 11, dtype=torch.long)
    targets[0, 0] = 1
    valid = torch.ones(1, 11, dtype=torch.bool)
    # Uncapped weights: positive 11/2 = 5.5, negatives 11/20 = 0.55.
    assert de_loss(logits, targets, valid, cap=10.0).item() == pytest.approx(LN2, rel=1e-6)
    capped = (2.0 + 10 * 0.55) * LN2 / 11
    assert de_loss(logits, targets, valid, cap=2.0).item() == pytest.approx(capped, rel=1e-6)
    assert de_loss(logits, targets, torch.zeros_like(valid), cap=10.0).item() == 0.0


def test_lfc_targets_are_log2_on_de_finite_positive_cells():
    fc = torch.tensor([[2.0, 0.5, 1.0, 8.0, 0.0, 3.0, float("inf")]])
    de = torch.tensor([[True, True, True, True, True, False, True]])
    lfc, mask = lfc_targets(fc, de)
    assert mask.tolist() == [[True, True, True, True, False, False, False]]
    assert lfc[mask].tolist() == [1.0, -1.0, 0.0, 3.0]
    assert torch.isnan(lfc[~mask]).all()


def test_lfc_targets_match_the_zero_filled_reference_on_the_mask():
    generator = torch.Generator().manual_seed(0)
    fc = torch.rand(4, 50, generator=generator) * 4.0
    de = torch.rand(4, 50, generator=generator) > 0.5
    lfc, mask = lfc_targets(fc, de)
    zero_filled = torch.where(de, fc, torch.zeros_like(fc))
    ref_mask = de & torch.isfinite(zero_filled) & (zero_filled > 0)
    ref = torch.zeros_like(fc)
    ref[ref_mask] = torch.log2(zero_filled[ref_mask].clamp(min=1e-8))
    assert torch.equal(mask, ref_mask)
    assert torch.equal(lfc[mask], ref[ref_mask])


def test_lfc_loss_huber_with_target_gene_split():
    pred = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    lfc = torch.tensor([[1.0, -1.0, 0.0, 3.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    # Huber(delta 1) per cell: 0.125, 0.5, 0.0, 2.5.
    no_target = lfc_loss(pred, lfc, mask, torch.tensor([-1]), 0.001, 1.0)
    assert no_target.item() == pytest.approx(3.125 / 4, rel=1e-6)
    split = lfc_loss(pred, lfc, mask, torch.tensor([3]), 0.001, 1.0)
    assert split.item() == pytest.approx(0.625 / 3 + 0.001 * 2.5, rel=1e-6)
    off_mask = torch.tensor([[True, True, True, False]])
    target_not_de = lfc_loss(pred, lfc, off_mask, torch.tensor([3]), 0.001, 1.0)
    assert target_not_de.item() == pytest.approx(0.625 / 3, rel=1e-6)
    empty = lfc_loss(pred, lfc, torch.zeros_like(mask), torch.tensor([-1]), 0.001, 1.0)
    assert empty.item() == 0.0


def test_lfc_loss_target_cells_are_per_row():
    pred = torch.zeros(2, 2)
    lfc = torch.tensor([[2.0, 0.5], [0.5, 0.5]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    # Row 0 targets gene 0 (Huber 1.5); the other three cells have Huber 0.125 each.
    got = lfc_loss(pred, lfc, mask, torch.tensor([0, -1]), 0.001, 1.0)
    assert got.item() == pytest.approx(0.125 + 0.001 * 1.5, rel=1e-6)


def test_lfc_direction_hinge_skips_zero_targets_and_masked_cells():
    pred = torch.tensor([[0.5, 0.5, -1.0, 2.0, float("nan")]], requires_grad=True)
    lfc = torch.tensor([[1.0, -1.0, 0.0, -3.0, float("nan")]])
    mask = torch.tensor([[True, True, True, True, False]])
    # Eligible cells 0, 1, 3: relu(-sign(y) * yhat / 0.25) = 0, 2, 8.
    got = lfc_direction_loss(pred, lfc, mask, 0.25)
    assert got.item() == pytest.approx(10.0 / 3, rel=1e-6)
    empty = lfc_direction_loss(pred, lfc, torch.zeros_like(mask), 0.25)
    empty.backward()
    assert empty.item() == 0.0
    assert torch.equal(pred.grad, torch.zeros_like(pred))


def test_dp_loss_soft_class_weighted_cross_entropy():
    logits = torch.tensor([[[LN2, 0.0, 0.0], [0.0, 0.0, 0.0], [9.0, 9.0, 9.0]]])
    probs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 0.0]]])
    valid = torch.tensor([[True, True, False]])
    # CE: cell 0 ln 2, cell 1 ln 3. Soft counts 1, .5, .5 over n = 2 -> weights 2/3, 4/3.
    expected = (2.0 / 3.0 * LN2 + 4.0 / 3.0 * LN3) / 2
    assert dp_loss(logits, probs, valid, cap=5.0).item() == pytest.approx(expected, rel=1e-6)
    capped = (2.0 / 3.0 * LN2 + 1.0 * LN3) / 2
    assert dp_loss(logits, probs, valid, cap=1.0).item() == pytest.approx(capped, rel=1e-6)


def _outputs(batch, seed: int = 5) -> list[GroupOutput]:
    generator = torch.Generator().manual_seed(seed)
    outputs = []
    for group in batch.groups:
        b, g = len(group.rows), len(group.gene_ids)
        outputs.append(
            GroupOutput(
                rows=group.rows,
                de_logits=torch.randn(b, g, 2, generator=generator),
                lfc=torch.randn(b, g, generator=generator),
                dp_logits=torch.randn(b, g, N_BINS, generator=generator),
            )
        )
    return outputs


def test_compute_losses_row_weights_groups_and_sums_terms():
    cfg = tiny_config()
    grid = tiny_grid()
    batch = tiny_batch(TWO_GROUPS)
    batch.target_gene = torch.tensor([-1, 2, 0])
    outputs = _outputs(batch)
    got = compute_losses(outputs, batch, cfg, grid, temperatures=False)
    assert list(got) == list(LOSS_KEYS)
    expected = {key: torch.zeros(()) for key in LOSS_KEYS[1:]}
    targets = (torch.tensor([-1, 0]), torch.tensor([2]))
    for out, group, target in zip(outputs, batch.groups, targets, strict=True):
        w = len(group.rows) / 3
        lfc, mask = lfc_targets(group.fold_changes, group.de_mask)
        targets_de = group.de_mask.long()
        expected["de_loss"] += w * de_loss(out.de_logits, targets_de, group.tested, 10.0)
        expected["lfc_loss"] += w * lfc_loss(out.lfc, lfc, mask, target, 0.001, 1.0)
        expected["lfc_dir_loss"] += w * lfc_direction_loss(out.lfc, lfc, mask, 0.25)
        probs, valid = two_hot(group.delta_p, grid)
        expected["dp_loss"] += w * dp_loss(out.dp_logits, probs, valid, 5.0)
    for key, value in expected.items():
        torch.testing.assert_close(got[key], value)
    total = got["de_loss"] + got["lfc_loss"] + got["lfc_dir_loss"] + got["dp_loss"]
    assert torch.equal(got["loss"], total)


def test_compute_losses_with_temperatures_scores_tempered_logits():
    cfg = tiny_config()
    batch = tiny_batch(TWO_GROUPS)
    outputs = _outputs(batch)
    got = compute_losses(outputs, batch, cfg, tiny_grid(), temperatures=True)
    tempered = [temper(out, cfg) for out in outputs]
    ref = compute_losses(tempered, batch, cfg, tiny_grid(), temperatures=False)
    raw = compute_losses(outputs, batch, cfg, tiny_grid(), temperatures=False)
    for key in LOSS_KEYS:
        assert torch.equal(got[key], ref[key])
    assert torch.equal(got["lfc_loss"], raw["lfc_loss"])
    assert not torch.equal(got["de_loss"], raw["de_loss"])


def test_compute_losses_rejects_unlabelled_groups():
    batch = tiny_batch(TWO_GROUPS)
    batch.groups[1].fold_changes = None
    with pytest.raises(ValueError, match="no labels"):
        compute_losses(_outputs(batch), batch, tiny_config(), tiny_grid(), temperatures=False)


def test_losses_backpropagate_to_every_parameter():
    model = tiny_model(n_donor_datasets=2).train()
    batch = tiny_batch(TWO_GROUPS, n_donor_datasets=2)
    batch.target_gene = torch.tensor([1, -1, 0])
    losses = compute_losses(model(batch), batch, model.cfg, tiny_grid(), temperatures=False)
    losses["loss"].backward()
    assert [name for name, p in model.named_parameters() if p.grad is None] == []
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
