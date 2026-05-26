#!/usr/bin/env python3
"""Same-layout left/middle/right batch sampler for target routing."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch.utils.data


class GroupedTargetRoutingBatchSampler(torch.utils.data.Sampler[list[int]]):
    SLOTS = ("left", "middle", "right")

    def __init__(
        self,
        target_index_path: str | Path,
        layout_groups_path: str | Path,
        batch_size: int = 30,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        if batch_size % 3 != 0:
            raise ValueError(f"batch_size must be divisible by 3, got {batch_size}")
        self.target_df = pd.read_csv(target_index_path).reset_index(drop=True)
        self.groups_df = pd.read_csv(layout_groups_path)
        self.batch_size = batch_size
        self.groups_per_batch = batch_size // 3
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.drop_last = drop_last

        ep_to_rows: dict[int, list[int]] = defaultdict(list)
        for row_idx, row in self.target_df.iterrows():
            ep_to_rows[int(row["episode_index"])].append(row_idx)
        self.ep_to_rows = {ep: rows for ep, rows in ep_to_rows.items() if rows}

        self.layout_pools: dict[str, dict[str, list[int]]] = {}
        for _, row in self.groups_df.iterrows():
            layout_id = str(row["layout_id"])
            pools = {}
            ok = True
            for slot in self.SLOTS:
                episodes = [int(x) for x in str(row[f"{slot}_pool"]).split(";") if x.strip()]
                row_pool = [idx for ep in episodes for idx in self.ep_to_rows.get(ep, [])]
                if not row_pool:
                    ok = False
                pools[slot] = row_pool
            if ok:
                self.layout_pools[layout_id] = pools
        self.layout_ids = sorted(self.layout_pools)

    def __len__(self) -> int:
        n = len(self.layout_ids) // self.groups_per_batch
        if not self.drop_last and len(self.layout_ids) % self.groups_per_batch:
            n += 1
        return n

    def __iter__(self) -> Iterator[list[int]]:
        layout_ids = list(self.layout_ids)
        if self.shuffle:
            self.rng.shuffle(layout_ids)
        batch = []
        for layout_id in layout_ids:
            pools = self.layout_pools[layout_id]
            for slot in self.SLOTS:
                batch.append(self.rng.choice(pools[slot]))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def audit(self, n_batches: int = 3) -> None:
        print(f"GroupedTargetRoutingBatchSampler: {len(self.layout_ids)} complete layouts")
        for b, batch in zip(range(n_batches), iter(self)):
            print(f"\nBatch {b}")
            print(f"{'layout':12s} {'slot':7s} {'episode':8s} {'frame':7s} prompt")
            for idx in batch:
                row = self.target_df.iloc[idx]
                print(
                    f"{str(row['layout_id']):12s} {str(row['target_slot']):7s} "
                    f"{int(row['episode_index']):<8d} {int(row['local_frame']):<7d} "
                    f"{str(row['prompt'])[:80]}"
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-index", required=True)
    parser.add_argument("--layout-groups", required=True)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--n-batches", type=int, default=3)
    args = parser.parse_args()

    sampler = GroupedTargetRoutingBatchSampler(args.target_index, args.layout_groups, args.batch_size)
    sampler.audit(args.n_batches)
