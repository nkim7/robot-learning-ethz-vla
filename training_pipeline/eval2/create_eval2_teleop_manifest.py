#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LAYOUTS = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")
COLOR = {"b": "blue", "r": "red", "g": "green"}
POSITIONS = ("left", "middle", "right")


ORDINAL_PROMPTS = {
    "left": [
        "Put the banana into the 1st bowl from the left.",
        "Put the banana into the leftmost bowl.",
        "Put the banana into the 3rd bowl from the right.",
    ],
    "middle": [
        "Put the banana into the 2nd bowl from the left.",
        "Put the banana into the middle bowl.",
        "Put the banana into the 2nd bowl from the right.",
    ],
    "right": [
        "Put the banana into the 3rd bowl from the left.",
        "Put the banana into the rightmost bowl.",
        "Put the banana into the 1st bowl from the right.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fresh Eval 2 teleop collection manifest.")
    parser.add_argument("--out", type=Path, default=Path("data/eval2_fresh_manifest/teleop_manifest.csv"))
    return parser.parse_args()


def layout_colors(layout: str) -> list[str]:
    return [COLOR[c] for c in layout]


def negation_prompt(target: str) -> str:
    excluded = [c for c in ("blue", "red", "green") if c != target]
    return f"Put the banana into the bowl that is not {excluded[0]} and not {excluded[1]}."


def rows_for_layout(layout: str) -> list[dict]:
    colors = layout_colors(layout)
    rows: list[dict] = []

    for pos_idx, position in enumerate(POSITIONS):
        target = colors[pos_idx]
        for prompt in ORDINAL_PROMPTS[position]:
            rows.append(
                {
                    "layout": layout,
                    "prompt_type": "ordinal",
                    "target_position": position,
                    "target_color": target,
                    "prompt": prompt,
                }
            )

    for anchor_idx, anchor in enumerate(colors):
        if anchor_idx > 0:
            target_idx = anchor_idx - 1
            rows.append(
                {
                    "layout": layout,
                    "prompt_type": "relative",
                    "target_position": POSITIONS[target_idx],
                    "target_color": colors[target_idx],
                    "prompt": f"Put the banana into the bowl on the left of the {anchor} bowl.",
                }
            )
        if anchor_idx < 2:
            target_idx = anchor_idx + 1
            rows.append(
                {
                    "layout": layout,
                    "prompt_type": "relative",
                    "target_position": POSITIONS[target_idx],
                    "target_color": colors[target_idx],
                    "prompt": f"Put the banana into the bowl on the right of the {anchor} bowl.",
                }
            )

    for pos_idx, position in enumerate(POSITIONS):
        target = colors[pos_idx]
        rows.append(
            {
                "layout": layout,
                "prompt_type": "negation",
                "target_position": position,
                "target_color": target,
                "prompt": negation_prompt(target),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for layout in LAYOUTS:
        rows.extend(rows_for_layout(layout))

    fieldnames = ["layout", "prompt_type", "target_position", "target_color", "prompt"]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    jsonl = args.out.with_suffix(".jsonl")
    with jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows)} prompts to {args.out} and {jsonl}")


if __name__ == "__main__":
    main()
