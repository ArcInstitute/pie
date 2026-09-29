"""Tests for pie.data.samplers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import lightning as L
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler

from pie.data import samplers
from pie.data.samplers import BlendedDistributedSampler, ValLoader


def _group_ids(*sizes: int) -> np.ndarray:
    """Concatenate [0]*sizes[0], [1]*sizes[1], ... as an int64 array."""
    return np.concatenate([np.full(n, g, dtype=np.int64) for g, n in enumerate(sizes)])


def _draw(sampler: BlendedDistributedSampler, epoch: int = 0) -> list[int]:
    sampler.set_epoch(epoch)
    return list(iter(sampler))


def _reference_draw(
    group_ids: np.ndarray,
    weights: dict[int, float],
    *,
    seed: int,
    epoch: int,
    num_replicas: int,
    rank: int,
) -> list[int]:
    """Independent restatement of the sampler's RNG protocol, one element at a time."""
    groups = sorted(g for g, w in weights.items() if w > 0)
    members = {g: np.where(group_ids == g)[0] for g in groups}
    n_eligible = sum(m.size for m in members.values())
    per_rank = -(-n_eligible // num_replicas)
    total = per_rank * num_replicas
    ws = [weights[g] for g in groups]
    raw = [w / sum(ws) * total for w in ws]
    counts = [int(x) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    slots = np.repeat(np.array(groups, dtype=np.int64), counts)
    np.random.default_rng([seed, 0, epoch]).shuffle(slots)
    draw = np.empty(total, dtype=np.int64)
    for g, count in zip(groups, counts, strict=True):
        m = members[g]
        stream = []
        for k in range(count):
            cycle, offset = divmod(epoch * count + k, m.size)
            perm = np.random.default_rng([seed, 1, g, cycle]).permutation(m.size)
            stream.append(m[perm[offset]])
        if count:
            draw[slots == g] = np.array(stream, dtype=np.int64)
    return draw[rank * per_rank : (rank + 1) * per_rank].tolist()


def test_epoch_composition_is_exact() -> None:
    ids = _group_ids(600, 300, 100)
    s = BlendedDistributedSampler(ids, {0: 0.5, 1: 0.3, 2: 0.2}, seed=0)
    counts = Counter(int(ids[i]) for i in _draw(s))
    assert counts == {0: 500, 1: 300, 2: 200}


def test_len_and_total_size_round_up_to_replicas() -> None:
    ids = _group_ids(101)
    s = BlendedDistributedSampler(ids, {0: 1.0}, num_replicas=4, rank=0)
    assert len(s) == 26
    assert s.total_size == 104


def test_ranks_partition_the_epoch_draw() -> None:
    ids = _group_ids(600, 200)
    parts = [
        BlendedDistributedSampler(ids, {0: 0.75, 1: 0.25}, num_replicas=4, rank=r, seed=42)
        for r in range(4)
    ]
    slices = [_draw(s) for s in parts]
    assert all(len(sl) == 200 for sl in slices)
    flat = [i for sl in slices for i in sl]
    assert len(set(flat)) == 800
    assert Counter(int(ids[i]) for i in flat) == {0: 600, 1: 200}


def test_within_epoch_no_duplicates_when_quota_fits() -> None:
    ids = _group_ids(100, 100)
    drawn = _draw(BlendedDistributedSampler(ids, {0: 0.5, 1: 0.5}, seed=1))
    assert len(drawn) == 200
    assert len(set(drawn)) == 200


def test_cross_epoch_stream_is_without_replacement() -> None:
    ids = _group_ids(100, 100)
    s = BlendedDistributedSampler(ids, {0: 0.25, 1: 0.75}, seed=7)
    g0 = [i for e in (0, 1) for i in _draw(s, e) if ids[i] == 0]
    assert len(g0) == 100
    assert len(set(g0)) == 100


def test_upweighted_group_wraps_with_reshuffle() -> None:
    ids = _group_ids(100, 200)
    s = BlendedDistributedSampler(ids, {0: 2 / 3, 1: 1 / 3}, seed=3)
    g0 = [i for i in _draw(s) if ids[i] == 0]
    assert len(g0) == 200
    assert Counter(g0) == {i: 2 for i in range(100)}
    assert g0[:100] != g0[100:]


def test_set_epoch_changes_order_not_composition() -> None:
    ids = _group_ids(300, 100)
    s = BlendedDistributedSampler(ids, {0: 0.6, 1: 0.4}, seed=5)
    e0 = _draw(s, 0)
    e1 = _draw(s, 1)
    assert s.epoch == 1
    assert e0 != e1
    comp0 = Counter(int(ids[i]) for i in e0)
    comp1 = Counter(int(ids[i]) for i in e1)
    assert comp0 == comp1 == {0: 240, 1: 160}


def test_deterministic_for_same_seed_and_epoch() -> None:
    ids = _group_ids(200, 100)
    a = BlendedDistributedSampler(ids, {0: 0.5, 1: 0.5}, seed=11)
    b = BlendedDistributedSampler(ids, {0: 0.5, 1: 0.5}, seed=11)
    assert _draw(a, 2) == _draw(b, 2)


def test_slots_are_shuffled_across_the_epoch() -> None:
    ids = _group_ids(100, 100)
    groups = [int(ids[i]) for i in _draw(BlendedDistributedSampler(ids, {0: 0.5, 1: 0.5}, seed=13))]
    windows = [set(groups[k : k + 20]) for k in range(0, 200, 20)]
    assert sum(1 for w in windows if w == {0, 1}) >= 8


def test_single_row_group_with_large_quota() -> None:
    ids = _group_ids(1, 99)
    s = BlendedDistributedSampler(ids, {0: 0.5, 1: 0.5}, seed=17)
    assert [i for i in _draw(s) if ids[i] == 0] == [0] * 50


def test_zero_weight_group_is_excluded() -> None:
    ids = _group_ids(500, 500)
    drawn = _draw(BlendedDistributedSampler(ids, {0: 1.0, 1: 0.0}, seed=0))
    assert len(drawn) == 500
    assert {int(ids[i]) for i in drawn} == {0}


def test_single_group_weight_one_covers_every_row_per_epoch() -> None:
    ids = np.zeros(10, dtype=np.int64)
    for epoch in (0, 1):
        ranks = [
            _draw(BlendedDistributedSampler(ids, {0: 1.0}, num_replicas=2, rank=r, seed=42), epoch)
            for r in range(2)
        ]
        assert sorted(ranks[0] + ranks[1]) == list(range(10))


def test_raises_on_invalid_inputs() -> None:
    ids = _group_ids(10, 10)
    with pytest.raises(ValueError):
        BlendedDistributedSampler(ids, {0: 1.0, 1: -1.0})
    with pytest.raises(ValueError):
        BlendedDistributedSampler(ids, {0: 0.0, 1: 0.0})
    with pytest.raises(ValueError):
        BlendedDistributedSampler(ids, {0: 1.0, 5: 1.0})
    with pytest.raises(ValueError):
        BlendedDistributedSampler(np.array([], dtype=np.int64), {0: 1.0})
    with pytest.raises(ValueError):
        BlendedDistributedSampler(ids, {0: 1.0, 1: 1.0}, num_replicas=0)
    with pytest.raises(ValueError):
        BlendedDistributedSampler(ids, {0: 1.0, 1: 1.0}, num_replicas=2, rank=2)


@pytest.mark.parametrize("num_replicas", [1, 3])
def test_matches_reference_rng_protocol(num_replicas: int) -> None:
    ids = np.array([1, 0, 2, 1, 0, 0, 2, 1, 1, 0, 2, 2, 0, 1] * 5, dtype=np.int64)
    weights = {0: 0.68, 1: 0.3, 2: 0.02}
    for seed in (0, 42):
        for epoch in (0, 1, 2):
            for rank in range(num_replicas):
                s = BlendedDistributedSampler(
                    ids, weights, num_replicas=num_replicas, rank=rank, seed=seed
                )
                expected = _reference_draw(
                    ids, weights, seed=seed, epoch=epoch, num_replicas=num_replicas, rank=rank
                )
                assert _draw(s, epoch) == expected


def test_xdataset_shape_weight_zero_first_group_keeps_ids() -> None:
    # xdataset shape: dataset id 0 has weight 0; groups 1..4 keep their ids in the RNG seed.
    ids = np.array([0, 1, 2, 3, 4, 0, 1, 1, 2, 4, 0, 3, 1, 4, 2] * 6, dtype=np.int64)
    weights = {0: 0.0, 1: 0.68, 2: 0.01, 3: 0.01, 4: 0.3}
    for rank in range(2):
        s = BlendedDistributedSampler(ids, weights, num_replicas=2, rank=rank, seed=42)
        drawn = _draw(s, 1)
        assert all(ids[i] != 0 for i in drawn)
        assert drawn == _reference_draw(
            ids, weights, seed=42, epoch=1, num_replicas=2, rank=rank
        )


def test_build_train_sampler_is_the_blended_sampler() -> None:
    ids = _group_ids(30, 10)
    s = samplers.build_train_sampler(ids, {0: 0.7, 1: 0.3}, num_replicas=2, rank=1, seed=5)
    assert isinstance(s, BlendedDistributedSampler)
    ref = BlendedDistributedSampler(ids, {0: 0.7, 1: 0.3}, num_replicas=2, rank=1, seed=5)
    assert _draw(s, 3) == _draw(ref, 3)


def _collate_list(xs: list[int]) -> list[int]:
    return list(xs)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_build_val_loader_is_unsharded() -> None:
    out = samplers.build_val_loader(
        list(range(10)),
        batch_size=4,
        num_workers=0,
        collate_fn=_collate_list,
        num_replicas=4,
        rank=3,
    )
    assert isinstance(out, ValLoader)
    assert out.sharded is False
    assert isinstance(out.loader.sampler, SequentialSampler)
    assert out.loader.drop_last is False
    assert out.loader.pin_memory is True
    assert list(out.loader) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


class _RecordingSampler(BlendedDistributedSampler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)
        super().set_epoch(epoch)


class _TinyModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.layer(batch.float().unsqueeze(-1)).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_lightning_fit_loop_calls_set_epoch(tmp_path: Path) -> None:
    sampler = _RecordingSampler(np.zeros(8, dtype=np.int64), {0: 1.0}, seed=0)
    loader = DataLoader(torch.arange(8), batch_size=4, sampler=sampler, drop_last=True)
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        use_distributed_sampler=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(_TinyModule(), train_dataloaders=loader)
    assert sampler.epochs == [0, 1, 2]
