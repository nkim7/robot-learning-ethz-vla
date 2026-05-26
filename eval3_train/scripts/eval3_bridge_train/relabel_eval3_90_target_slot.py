#!/usr/bin/env python3
"""Generate sidecar JSON mapping episode_index → target_slot for eval3_90_permutation.

Label rule:
    within_block = episode_index % 30
    0-9   → "left"
    10-19 → "middle"
    20-29 → "right"

Outputs:
    data/eval3_90_permutation_spatial_labels/target_slot_by_episode.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SLOT_BOUNDARIES = [(0, 9, "left"), (10, 19, "middle"), (20, 29, "right")]


def get_target_slot(episode_index: int) -> str:
    within_block = episode_index % 30
    for lo, hi, slot in SLOT_BOUNDARIES:
        if lo <= within_block <= hi:
            return slot
    raise ValueError(f"Unexpected within_block={within_block} for episode_index={episode_index}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="data/eval3_90_permutation",
        help="Root of the eval3_90_permutation dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval3_90_permutation_spatial_labels",
        help="Directory for the sidecar file",
    )
    parser.add_argument("--n-episodes", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_file = out_dir / "target_slot_by_episode.json"

    if out_file.exists() and not args.overwrite:
        print(f"Sidecar already exists: {out_file}  (use --overwrite to replace)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, str] = {}
    for ep in range(args.n_episodes):
        labels[str(ep)] = get_target_slot(ep)

    out_file.write_text(json.dumps(labels, indent=2))
    print(f"Wrote {len(labels)} episode labels → {out_file}")

    # Quick summary
    from collections import Counter
    counts = Counter(labels.values())
    for slot in ["left", "middle", "right"]:
        print(f"  {slot:6s}: {counts[slot]} episodes")


if __name__ == "__main__":
    main()
