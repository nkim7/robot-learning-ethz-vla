#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


COLORS = ("blue", "red", "green")
LAYOUTS = ("bgr", "brg", "gbr", "grb", "rbg", "rgb")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create fresh Eval 2 VQA JSONL from real setup photos. "
            "Expected input: one subdirectory per layout code, e.g. photos/bgr/*.jpg."
        )
    )
    parser.add_argument("--photos-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/eval2_vqa_fresh/vqa.jsonl"))
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--val-frac", type=float, default=0.15)
    return parser.parse_args()


def answer(color: str) -> str:
    return f"The {color} bowl"


def color_layout(layout: str) -> list[str]:
    return [{"b": "blue", "r": "red", "g": "green"}[c] for c in layout]


def image_files(root: Path, layout: str) -> list[Path]:
    flat_matches = sorted(
        p for p in root.glob(f"{layout}.*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    folder = root / layout
    nested_matches = []
    if folder.exists():
        nested_matches = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return flat_matches + nested_matches


def make_questions(layout: str) -> list[dict]:
    colors = color_layout(layout)
    rows: list[dict] = []

    for idx, prompts in ORDINAL_LEFT.items():
        for question in prompts:
            rows.append({"type": "ordinal", "target": colors[idx], "question": question, "answer": answer(colors[idx])})

    for idx_from_right, prompts in ORDINAL_RIGHT.items():
        idx = 2 - idx_from_right
        for question in prompts:
            rows.append({"type": "ordinal", "target": colors[idx], "question": question, "answer": answer(colors[idx])})

    for idx, anchor in enumerate(colors):
        if idx > 0:
            target = colors[idx - 1]
            rows.extend(
                [
                    {
                        "type": "relative",
                        "target": target,
                        "question": f"Which bowl is to the left of the {anchor} bowl?",
                        "answer": answer(target),
                    },
                    {
                        "type": "relative",
                        "target": target,
                        "question": f"What color is the bowl immediately left of the {anchor} bowl?",
                        "answer": answer(target),
                    },
                    {
                        "type": "relative",
                        "target": target,
                        "question": f"Which color bowl is directly to the left of {anchor}?",
                        "answer": answer(target),
                    },
                    {
                        "type": "relative",
                        "target": target,
                        "question": f"If I look immediately left of the {anchor} bowl, which color do I find?",
                        "answer": answer(target),
                    },
                ]
            )
        if idx < 2:
            target = colors[idx + 1]
            rows.extend(
                [
                    {
                        "type": "relative",
                        "target": target,
                        "question": f"Which bowl is to the right of the {anchor} bowl?",
                        "answer": answer(target),
                    },
                    {
                        "type": "relative",
                        "target": target,
                        "question": f"What color is the bowl immediately right of the {anchor} bowl?",
                        "answer": answer(target),
                    },
                    {
                        "type": "relative",
                        "target": target,
                        "question": f"Which color bowl is directly to the right of {anchor}?",
                        "answer": answer(target),
                    },
                    {
                        "type": "relative",
                        "target": target,
                        "question": f"If I look immediately right of the {anchor} bowl, which color do I find?",
                        "answer": answer(target),
                    },
                ]
            )

    for target in COLORS:
        excluded = [c for c in COLORS if c != target]
        rows.extend(
            [
                {
                    "type": "negation",
                    "target": target,
                    "question": f"Which bowl is not {excluded[0]} and not {excluded[1]}?",
                    "answer": answer(target),
                },
                {
                    "type": "negation",
                    "target": target,
                    "question": f"What color is the bowl that is neither {excluded[0]} nor {excluded[1]}?",
                    "answer": answer(target),
                },
                {
                    "type": "negation",
                    "target": target,
                    "question": f"Identify the bowl that is not {excluded[0]} or {excluded[1]}.",
                    "answer": answer(target),
                },
                {
                    "type": "negation",
                    "target": target,
                    "question": f"Only one bowl is neither {excluded[0]} nor {excluded[1]}; what color is it?",
                    "answer": answer(target),
                },
            ]
        )
    return rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for layout in LAYOUTS:
        imgs = image_files(args.photos_root, layout)
        if not imgs:
            print(f"[warn] no images found for layout {layout} in {args.photos_root / layout}")
            continue
        questions = make_questions(layout)
        for img in imgs:
            for row in questions:
                records.append(
                    {
                        "image": str(img.resolve()),
                        "layout": layout,
                        "left": color_layout(layout)[0],
                        "middle": color_layout(layout)[1],
                        "right": color_layout(layout)[2],
                        **row,
                    }
                )

    if not records:
        raise SystemExit("No VQA records generated. Check --photos-root and layout subdirectories.")

    random.shuffle(records)
    val_n = int(round(len(records) * args.val_frac))
    for i, rec in enumerate(records):
        rec["split"] = "val" if i < val_n else "train"

    with args.out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    summary = {
        "records": len(records),
        "train": sum(r["split"] == "train" for r in records),
        "val": sum(r["split"] == "val" for r in records),
        "photos_root": str(args.photos_root),
        "out": str(args.out),
    }
    (args.out.parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
