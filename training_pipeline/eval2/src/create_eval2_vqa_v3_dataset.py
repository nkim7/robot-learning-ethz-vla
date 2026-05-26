#!/usr/bin/env python3
"""
Create Eval2 VQA v3 dataset — same as v2 (smolvlm2) but adds position-output questions.

v2 questions (all answer with a COLOR: "The blue bowl"):
  - ordinal_left:  "Which bowl is 1st/2nd/3rd from the left?"
  - ordinal_right: "Which bowl is 1st/2nd/3rd from the right?"
  - relative:      "Which bowl is to the left of the red bowl?"
  - negation:      "Which bowl is not blue and not red?"

v3 adds (answer with a POSITION: "left" / "middle" / "right"):
  - color_to_pos:  "Where is the blue bowl?"  →  "left"

Usage:
    python scripts/create_eval2_vqa_v3_dataset.py \
        --photos-root /home/shadeform/augmented/augmented \
        --out data/eval2_vqa_v3/vqa.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

COLORS = ("blue", "red", "green")
LAYOUTS = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# ── Existing v2 question templates (answer = color) ───────────────────────────

ORDINAL_LEFT = {
    0: [
        "Which bowl is 1st from the left?",
        "Which bowl is first from the left?",
        "Which bowl is leftmost?",
        "What color is the first bowl from the left?",
        "What color is the left bowl?",
        "Name the bowl on the far left.",
        "Which color is in the left position?",
    ],
    1: [
        "Which bowl is 2nd from the left?",
        "Which bowl is second from the left?",
        "Which bowl is in the middle?",
        "What color is the center bowl?",
        "What color is the middle bowl?",
        "Name the bowl between the other two bowls.",
        "Which color is in the middle position?",
    ],
    2: [
        "Which bowl is 3rd from the left?",
        "Which bowl is third from the left?",
        "Which bowl is rightmost?",
        "What color is the last bowl from the left?",
        "What color is the right bowl?",
        "Name the bowl on the far right.",
        "Which color is in the right position?",
    ],
}

ORDINAL_RIGHT = {
    0: [
        "Which bowl is 1st from the right?",
        "Which bowl is first from the right?",
        "What color is the first bowl from the right?",
        "Starting from the right, what color is bowl number one?",
    ],
    1: [
        "Which bowl is 2nd from the right?",
        "Which bowl is second from the right?",
        "What color is the second bowl from the right?",
        "Starting from the right, what color is bowl number two?",
    ],
    2: [
        "Which bowl is 3rd from the right?",
        "Which bowl is third from the right?",
        "What color is the third bowl from the right?",
        "Starting from the right, what color is bowl number three?",
    ],
}

# ── New v3 question templates (answer = position) ─────────────────────────────

COLOR_TO_POS = {
    color: [
        f"Where is the {color} bowl?",
        f"Which position is the {color} bowl in?",
        f"The {color} bowl is on which side?",
        f"Is the {color} bowl on the left, in the middle, or on the right?",
        f"What position does the {color} bowl occupy?",
        f"Which side has the {color} bowl?",
        f"Where can I find the {color} bowl?",
        f"What is the location of the {color} bowl?",
    ]
    for color in COLORS
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def color_answer(color: str) -> str:
    return f"The {color} bowl"

def position_answer(pos: str) -> str:
    return pos   # "left" / "middle" / "right"

def color_layout(layout: str) -> list[str]:
    return [{"b": "blue", "r": "red", "g": "green"}[c] for c in layout]

def image_files(root: Path, layout: str) -> list[Path]:
    """Find all images for a given layout code.
    Handles both:
      - flat files: root/bgr.jpg
      - flat files with suffix: root/bgr_2_aug1.jpeg  (augmented server format)
      - nested: root/bgr/*.jpg
    """
    # Flat exact match (e.g. bgr.jpg)
    flat_exact = sorted(
        p for p in root.glob(f"{layout}.*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    # Flat with underscore suffix (e.g. bgr_2_aug1.jpeg)
    flat_aug = sorted(
        p for p in root.glob(f"{layout}_*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    # Nested folder (e.g. root/bgr/*.jpg)
    nested = []
    folder = root / layout
    if folder.exists():
        nested = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return flat_exact + flat_aug + nested


def make_questions(layout: str) -> list[dict]:
    colors = color_layout(layout)     # [left_color, middle_color, right_color]
    slots  = ["left", "middle", "right"]
    rows: list[dict] = []

    # ── v2: ordinal from left (answer = color) ────────────────────────────────
    for idx, prompts in ORDINAL_LEFT.items():
        for q in prompts:
            rows.append({
                "type": "ordinal_left",
                "target": colors[idx],
                "question": q,
                "answer": color_answer(colors[idx]),
            })

    # ── v2: ordinal from right (answer = color) ───────────────────────────────
    for idx_from_right, prompts in ORDINAL_RIGHT.items():
        idx = 2 - idx_from_right
        for q in prompts:
            rows.append({
                "type": "ordinal_right",
                "target": colors[idx],
                "question": q,
                "answer": color_answer(colors[idx]),
            })

    # ── v2: relative (answer = color) ─────────────────────────────────────────
    for idx, anchor in enumerate(colors):
        if idx > 0:
            target = colors[idx - 1]
            for q in [
                f"Which bowl is to the left of the {anchor} bowl?",
                f"What color is the bowl immediately left of the {anchor} bowl?",
                f"Which color bowl is directly to the left of {anchor}?",
                f"If I look immediately left of the {anchor} bowl, which color do I find?",
            ]:
                rows.append({"type": "relative", "target": target, "question": q, "answer": color_answer(target)})
        if idx < 2:
            target = colors[idx + 1]
            for q in [
                f"Which bowl is to the right of the {anchor} bowl?",
                f"What color is the bowl immediately right of the {anchor} bowl?",
                f"Which color bowl is directly to the right of {anchor}?",
                f"If I look immediately right of the {anchor} bowl, which color do I find?",
            ]:
                rows.append({"type": "relative", "target": target, "question": q, "answer": color_answer(target)})

    # ── v2: negation (answer = color) ─────────────────────────────────────────
    for target in COLORS:
        excluded = [c for c in COLORS if c != target]
        for q in [
            f"Which bowl is not {excluded[0]} and not {excluded[1]}?",
            f"What color is the bowl that is neither {excluded[0]} nor {excluded[1]}?",
            f"Identify the bowl that is not {excluded[0]} or {excluded[1]}.",
            f"Only one bowl is neither {excluded[0]} nor {excluded[1]}; what color is it?",
        ]:
            rows.append({"type": "negation", "target": target, "question": q, "answer": color_answer(target)})

    # ── v3 NEW: color → position (answer = "left"/"middle"/"right") ───────────
    for idx, color in enumerate(colors):
        pos = slots[idx]
        for q in COLOR_TO_POS[color]:
            rows.append({
                "type": "color_to_pos",
                "target": pos,            # "left" / "middle" / "right"
                "question": q,
                "answer": position_answer(pos),
            })

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create Eval2 VQA v3 JSONL (v2 + position-output questions)."
    )
    p.add_argument("--photos-root", type=Path, required=True,
                   help="Directory containing layout images (e.g. /home/shadeform/augmented/augmented)")
    p.add_argument("--out", type=Path, default=Path("data/eval2_vqa_v3/vqa.jsonl"))
    p.add_argument("--seed", type=int, default=47)
    p.add_argument("--val-frac", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for layout in LAYOUTS:
        imgs = image_files(args.photos_root, layout)
        if not imgs:
            print(f"[warn] no images found for layout '{layout}' in {args.photos_root}")
            continue
        print(f"  {layout}: {len(imgs)} images")
        questions = make_questions(layout)
        for img in imgs:
            for row in questions:
                records.append({
                    "image":  str(img.resolve()),
                    "layout": layout,
                    "left":   color_layout(layout)[0],
                    "middle": color_layout(layout)[1],
                    "right":  color_layout(layout)[2],
                    **row,
                })

    if not records:
        raise SystemExit("No VQA records generated — check --photos-root and layout directories.")

    random.shuffle(records)
    val_n = int(round(len(records) * args.val_frac))
    for i, rec in enumerate(records):
        rec["split"] = "val" if i < val_n else "train"

    with args.out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # Count by type
    type_counts: dict[str, int] = {}
    for r in records:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1

    summary = {
        "records": len(records),
        "train":   sum(r["split"] == "train" for r in records),
        "val":     sum(r["split"] == "val"   for r in records),
        "by_type": type_counts,
        "photos_root": str(args.photos_root),
        "out":     str(args.out),
    }
    (args.out.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
