#!/usr/bin/env python3
"""
grouped_placement_sampler.py — eval2_pathC_lora_placement_grouped_v1

GroupedPlacementBatchSampler:
  - Loads placement_index.csv and layout_groups.csv.
  - Each batch contains `groups_per_batch` layout groups × 3 slots = batch_size.
  - For each selected layout group: one random left, one middle, one right
    placement-phase frame index is drawn.
  - Groups are reshuffled each epoch; within each pool, sampling is with
    replacement so a small pool can still yield diverse batches.

Usage:
    from grouped_placement_sampler import GroupedPlacementBatchSampler

    sampler = GroupedPlacementBatchSampler(
        placement_index_path="sampler_indices/placement_index.csv",
        layout_groups_path="sampler_indices/layout_groups.csv",
        batch_size=30,          # must be divisible by 3
        shuffle=True,
        seed=42,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=...,
    )
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch.utils.data


class GroupedPlacementBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that forms same-layout, different-target-slot groups.

    Each batch = `groups_per_batch` layout groups × 3 target slots.
    Batch size must be divisible by 3.

    Args:
        placement_index_path: Path to placement_index.csv.
        layout_groups_path:   Path to layout_groups.csv.
        batch_size:           Total batch size (must be divisible by 3).
        shuffle:              Whether to shuffle layout group order each epoch.
        seed:                 Random seed for reproducibility.
        drop_incomplete:      Drop the last batch if it doesn't fill completely.
    """

    SLOTS = ("left", "middle", "right")

    def __init__(
        self,
        placement_index_path: str | Path,
        layout_groups_path: str | Path,
        batch_size: int = 30,
        shuffle: bool = True,
        seed: int = 42,
        drop_incomplete: bool = True,
    ) -> None:
        assert batch_size % 3 == 0, f"batch_size must be divisible by 3, got {batch_size}"
        self.groups_per_batch = batch_size // 3
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_incomplete = drop_incomplete

        # ── Load placement index: episode → list of valid global frame indices ──
        placement_df = pd.read_csv(placement_index_path)
        # Build: episode_index → list of global_idx (for that episode)
        self._ep_to_frames: dict[int, list[int]] = defaultdict(list)
        for _, row in placement_df.iterrows():
            self._ep_to_frames[int(row["episode_index"])].append(int(row["global_idx"]))
        # Freeze into sorted lists
        self._ep_to_frames = {ep: sorted(frames) for ep, frames in self._ep_to_frames.items()}

        # ── Load layout groups ────────────────────────────────────────────────
        layout_df = pd.read_csv(layout_groups_path)
        # Build: layout_id → {slot: [episode_indices]}
        self._layout_pools: dict[str, dict[str, list[int]]] = {}
        for _, row in layout_df.iterrows():
            lid = str(row["layout_id"])
            pools: dict[str, list[int]] = {}
            for slot in self.SLOTS:
                raw = str(row[f"{slot}_pool"])
                pools[slot] = [int(x) for x in raw.split(";") if x.strip()]
            # Only keep layouts where every slot has at least one episode with placement frames
            valid = True
            for slot in self.SLOTS:
                ep_list = [ep for ep in pools[slot] if ep in self._ep_to_frames and self._ep_to_frames[ep]]
                pools[slot] = ep_list
                if not ep_list:
                    valid = False
                    break
            if valid:
                self._layout_pools[lid] = pools

        self._layout_ids = sorted(self._layout_pools.keys())
        self._rng = random.Random(seed)

        n_valid = len(self._layout_ids)
        n_batches = n_valid // self.groups_per_batch
        print(
            f"[GroupedPlacementBatchSampler] "
            f"{n_valid} valid layouts → "
            f"{n_batches} complete batches/epoch "
            f"(groups_per_batch={self.groups_per_batch}, batch_size={batch_size})"
        )

    # ── Sampling helpers ──────────────────────────────────────────────────────

    def _sample_frame_for_episode(self, ep_idx: int) -> int:
        """Sample a random placement frame from an episode."""
        frames = self._ep_to_frames[ep_idx]
        return self._rng.choice(frames)

    def _sample_group(self, layout_id: str) -> list[int]:
        """Sample left, middle, right global frame indices from one layout group."""
        pools = self._layout_pools[layout_id]
        indices = []
        for slot in self.SLOTS:
            ep = self._rng.choice(pools[slot])
            frame = self._sample_frame_for_episode(ep)
            indices.append(frame)
        return indices  # [left_idx, middle_idx, right_idx]

    # ── Iterator ──────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[list[int]]:
        layout_ids = list(self._layout_ids)
        if self.shuffle:
            self._rng.shuffle(layout_ids)

        batch: list[int] = []
        n_groups_in_batch = 0

        for layout_id in layout_ids:
            group_indices = self._sample_group(layout_id)
            batch.extend(group_indices)
            n_groups_in_batch += 1

            if n_groups_in_batch == self.groups_per_batch:
                yield batch
                batch = []
                n_groups_in_batch = 0

        # Yield incomplete last batch unless drop_incomplete
        if batch and not self.drop_incomplete:
            yield batch

    def __len__(self) -> int:
        n = len(self._layout_ids) // self.groups_per_batch
        return n

    # ── Debug / audit ─────────────────────────────────────────────────────────

    def print_sample_batches(self, n_batches: int = 5) -> None:
        """Print n_batches sample batches for audit. Does not consume the iterator."""
        # We need the placement_df to map global_idx back to metadata.
        # This method is for interactive audit — call it with a copy of placement_df.
        print(f"\n{'='*70}")
        print(f"  GROUPED PLACEMENT SAMPLER AUDIT — {n_batches} sample batches")
        print(f"{'='*70}")
        layout_ids = list(self._layout_ids)
        if self.shuffle:
            layout_ids_shuffled = list(layout_ids)
            self._rng.shuffle(layout_ids_shuffled)
        else:
            layout_ids_shuffled = layout_ids

        for b in range(min(n_batches, len(layout_ids_shuffled) // self.groups_per_batch)):
            print(f"\n  Batch {b}:")
            print(f"  {'layout_id':15s} {'slot':8s} {'episode':10s} {'global_idx':12s}")
            print(f"  {'-'*50}")
            for g in range(self.groups_per_batch):
                lid = layout_ids_shuffled[b * self.groups_per_batch + g]
                pools = self._layout_pools[lid]
                for slot in self.SLOTS:
                    ep = self._rng.choice(pools[slot])
                    frame = self._sample_frame_for_episode(ep)
                    print(f"  {lid:15s} {slot:8s} ep={ep:<8d} gidx={frame}")
        print(f"\n{'='*70}\n")


def audit_sampler(
    placement_index_path: str | Path,
    layout_groups_path: str | Path,
    batch_size: int = 30,
    n_batches: int = 5,
) -> None:
    """
    Standalone audit function — call from train script with --audit_only.
    Loads the sampler and prints n_batches sample batches.
    """
    placement_df = pd.read_csv(placement_index_path)

    sampler = GroupedPlacementBatchSampler(
        placement_index_path=placement_index_path,
        layout_groups_path=layout_groups_path,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )

    # Build reverse map: global_idx → row metadata
    idx_to_meta = {}
    for _, row in placement_df.iterrows():
        idx_to_meta[int(row["global_idx"])] = row

    print(f"\n{'='*75}")
    print(f"  SAMPLER AUDIT — {n_batches} batches, batch_size={batch_size}")
    print(f"{'='*75}")

    for b, batch in enumerate(sampler):
        if b >= n_batches:
            break
        print(f"\n  Batch {b}:")
        print(f"  {'layout_id':15s} {'slot':8s} {'ep_idx':8s} {'local_f':8s} {'task[:60]'}")
        print(f"  {'-'*80}")
        for gidx in batch:
            if gidx in idx_to_meta:
                row = idx_to_meta[gidx]
                print(f"  {row['layout_id']:15s} {row['target_slot']:8s} "
                      f"{int(row['episode_index']):<8d} f{int(row['local_frame']):<7d} "
                      f"{str(row['task'])[:60]}")
            else:
                print(f"  gidx={gidx} NOT IN INDEX")
    print(f"\n{'='*75}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement_index", required=True)
    ap.add_argument("--layout_groups", required=True)
    ap.add_argument("--batch_size", type=int, default=30)
    ap.add_argument("--n_batches", type=int, default=5)
    args = ap.parse_args()
    audit_sampler(
        placement_index_path=args.placement_index,
        layout_groups_path=args.layout_groups,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
    )
