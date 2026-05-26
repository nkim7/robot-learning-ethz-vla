#!/usr/bin/env python3
"""Validate the target_slot sidecar and print distribution statistics."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar",
        default="data/eval3_90_permutation_spatial_labels/target_slot_by_episode.json",
    )
    parser.add_argument("--n-episodes", type=int, default=90)
    args = parser.parse_args()

    path = Path(args.sidecar)
    if not path.exists():
        raise FileNotFoundError(f"Sidecar not found: {path}\nRun relabel_eval3_90_target_slot.py first.")

    labels: dict[str, str] = json.loads(path.read_text())
    print(f"Loaded {len(labels)} labels from {path}")

    errors = []
    for ep in range(args.n_episodes):
        key = str(ep)
        if key not in labels:
            errors.append(f"  Missing episode_index={ep}")
            continue
        slot = labels[key]
        if slot not in ("left", "middle", "right"):
            errors.append(f"  Bad slot '{slot}' for episode_index={ep}")

    if errors:
        print("ERRORS:")
        for e in errors:
            print(e)
    else:
        print("All episodes present and valid.")

    counts = Counter(labels.values())
    print("\nDistribution:")
    for slot in ["left", "middle", "right"]:
        n = counts.get(slot, 0)
        bar = "#" * n
        print(f"  {slot:6s}: {n:3d}  {bar}")

    # Cross-check arithmetic rule
    print("\nCross-check against label rule (within_block = ep % 30):")
    mismatches = 0
    expected = {"left": set(range(0, 90, 30)) | set(range(1, 90, 30)) | set(),  # dummy
    }
    slot_map = {}
    for ep in range(args.n_episodes):
        wb = ep % 30
        if wb <= 9:
            expected_slot = "left"
        elif wb <= 19:
            expected_slot = "middle"
        else:
            expected_slot = "right"
        actual_slot = labels.get(str(ep), "MISSING")
        if actual_slot != expected_slot:
            mismatches += 1
            print(f"  ep={ep:3d}  within_block={wb:2d}  expected={expected_slot}  got={actual_slot}")

    if mismatches == 0:
        print("  All labels match the arithmetic rule.")
    else:
        print(f"  {mismatches} mismatches found!")


if __name__ == "__main__":
    main()
