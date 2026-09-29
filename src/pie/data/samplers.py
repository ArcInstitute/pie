"""Train sampler (blended data mix, DDP-sharded) and the validation loader."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import NamedTuple

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler


def _largest_remainder(weights: list[float], total: int) -> list[int]:
    """Split ``total`` into integer counts proportional to ``weights``."""
    s = sum(weights)
    raw = [w / s * total for w in weights]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


class BlendedDistributedSampler(Sampler[int]):
    """Megatron-style blended data-mix sampler, DDP-sharded.

    ``group_ids[i]`` is the dataset-group index of train sample ``i``;
    ``group_weights`` maps a group index to its relative mix weight. Groups
    with weight ``0`` are excluded. Each epoch draws ``total_size`` indices
    (the eligible-row count, rounded up to a multiple of ``num_replicas``),
    split across groups by largest-remainder rounding of the normalized
    weights — so per-epoch composition matches the mix exactly (a weight
    below ``1 / total_size`` can round to zero slots and is then never
    drawn). Slot order is reshuffled each epoch (seeded by
    ``seed``/``epoch``), so each batch is a random cut of the stream whose
    composition fluctuates around the mix instead of following a fixed
    per-batch quota. Within a group, rows come from an infinite
    without-replacement stream of shuffled cycles — cycle ``j`` is a
    permutation seeded by ``(seed, group, j)`` — and epoch ``e`` resumes
    that stream at offset ``e * count_g``, so full coverage carries across
    epoch boundaries and resume needs no sampler state beyond
    ``set_epoch``. All ranks build the same draw; each yields its
    contiguous shard.

    :meth:`set_epoch` must be called each epoch; Lightning's fit loop does so
    for the train dataloader's sampler.
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        group_weights: Mapping[int, float],
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ) -> None:
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid (num_replicas={num_replicas}, rank={rank}).")
        group_ids = np.asarray(group_ids)
        if group_ids.size == 0:
            raise ValueError("group_ids must be non-empty.")
        if any(w < 0 for w in group_weights.values()):
            raise ValueError("blend group weights must be non-negative.")
        if sum(group_weights.values()) <= 0:
            raise ValueError("blend group weights sum to zero.")
        present = {int(g) for g in np.unique(group_ids)}
        rowless = [g for g, w in group_weights.items() if w > 0 and g not in present]
        if rowless:
            raise ValueError(f"groups {rowless} have positive weight but no rows.")

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self._epoch = 0

        self._groups = sorted(g for g, w in group_weights.items() if w > 0)
        self._members: dict[int, np.ndarray] = {
            g: np.where(group_ids == g)[0] for g in self._groups
        }
        n_eligible = sum(m.size for m in self._members.values())
        self.num_samples_per_rank = (n_eligible + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples_per_rank * num_replicas
        counts = _largest_remainder(
            [group_weights[g] for g in self._groups], self.total_size
        )
        self._counts: dict[int, int] = dict(zip(self._groups, counts, strict=True))

    @property
    def epoch(self) -> int:
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _stream_slice(self, g: int, start: int, length: int) -> np.ndarray:
        """Rows ``[start, start + length)`` of group ``g``'s shuffled stream."""
        members = self._members[g]
        n = members.size
        if n == 1:
            # One row has one possible permutation; skip the per-cycle rng
            # construction, which would otherwise run once per slot.
            return np.full(length, members[0], dtype=members.dtype)
        chunks: list[np.ndarray] = []
        pos = start
        remaining = length
        while remaining > 0:
            cycle, offset = divmod(pos, n)
            perm = np.random.default_rng([self.seed, 1, g, cycle]).permutation(n)
            k = min(remaining, n - offset)
            chunks.append(members[perm[offset : offset + k]])
            pos += k
            remaining -= k
        return np.concatenate(chunks)

    def __iter__(self) -> Iterator[int]:
        slots = np.repeat(
            np.array(self._groups, dtype=np.int64),
            [self._counts[g] for g in self._groups],
        )
        np.random.default_rng([self.seed, 0, self._epoch]).shuffle(slots)
        draw = np.empty(self.total_size, dtype=np.int64)
        for g, count in self._counts.items():
            if count > 0:
                draw[slots == g] = self._stream_slice(g, self._epoch * count, count)
        start = self.rank * self.num_samples_per_rank
        end = start + self.num_samples_per_rank
        yield from draw[start:end].tolist()

    def __len__(self) -> int:
        return self.num_samples_per_rank


def build_train_sampler(
    group_ids: np.ndarray,
    weights: Mapping[int, float],
    num_replicas: int,
    rank: int,
    seed: int,
) -> Sampler[int]:
    """Train sampler for the datamodule (looked up as a module attribute at call time)."""
    return BlendedDistributedSampler(
        group_ids, weights, num_replicas=num_replicas, rank=rank, seed=seed
    )


class ValLoader(NamedTuple):
    loader: DataLoader
    sharded: bool  # True means each rank sees a shard and outputs must be all-gathered


def build_val_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    collate_fn: Callable,
    num_replicas: int,
    rank: int,
) -> ValLoader:
    """Validation loader: every rank iterates the full val set in order (unsharded)."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return ValLoader(loader=loader, sharded=False)
