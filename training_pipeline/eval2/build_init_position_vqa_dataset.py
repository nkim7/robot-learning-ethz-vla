#!/usr/bin/env python3
"""Build initial-frame Eval2 target-grounding VQA data.

Data sources:
- permutation_init_position local images, including augmented variants.
- Optional one clean initial frame per Eval2 layout/permutation from
  robot-learning-group47/eval2_compositional_augmented.

Inputs stay as the original Eval2 natural-language prompts. Outputs are compact
target grounding labels:

target color: green
target slot: right
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from eval2_layout_utils import COLORS, normalize_layout, prompt_type, resolve_prompt_target_color, target_slot

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CODE_TO_COLOR = {"b": "blue", "g": "green", "r": "red"}

EVAL2_PROMPTS = [
    "Put the banana in the blue colored bowl.",
    "Put the banana into the 1st bowl from the left from the robot perspective.",
    "Put the banana into the 3rd bowl from the right from the robot perspective.",
    "Put the banana into the bowl on the left of the green bowl from the robot perspective.",
    "Put the banana into the bowl that is not green and not red.",
    "Put the banana into the bowl that is not red and not green.",
    "Put the banana in the green colored bowl.",
    "Put the banana into the 2nd bowl from the left from the robot perspective.",
    "Put the banana into the 2nd bowl from the right from the robot perspective.",
    "Put the banana into the middle bowl from the robot perspective.",
    "Put the banana into the bowl on the right of the blue bowl from the robot perspective.",
    "Put the banana into the bowl on the left of the red bowl from the robot perspective.",
    "Put the banana into the bowl that is not blue and not red.",
    "Put the banana into the bowl that is not red and not blue.",
    "Put the banana in the red colored bowl.",
    "Put the banana into the 3rd bowl from the left from the robot perspective.",
    "Put the banana into the 1st bowl from the right from the robot perspective.",
    "Put the banana into the bowl on the right of the green bowl from the robot perspective.",
    "Put the banana into the bowl that is not blue and not green.",
    "Put the banana into the bowl that is not green and not blue.",
    "Put the banana into the bowl on the right of the red bowl from the robot perspective.",
    "Put the banana into the bowl on the left of the blue bowl from the robot perspective.",
]


def layout_from_code(code: str) -> list[str]:
    return [CODE_TO_COLOR[ch] for ch in code]


def code_from_layout(layout: list[str]) -> str:
    inv = {v: k for k, v in CODE_TO_COLOR.items()}
    return "".join(inv[color] for color in layout)


def answer_text(color: str, slot: str) -> str:
    return f"target color: {color}\ntarget slot: {slot}"


def parse_perm_image(path: Path) -> tuple[str, str, bool]:
    stem = path.stem.lower()
    parts = stem.split("_")
    layout_code = parts[0]
    if sorted(layout_code) != ["b", "g", "r"]:
        raise ValueError(f"Cannot parse permutation from filename: {path.name}")
    source_id = layout_code if len(parts) == 1 or parts[1].startswith("aug") else f"{layout_code}_{parts[1]}"
    is_augmented = path.parent.name == "augmented" or "aug" in parts
    return layout_code, source_id, is_augmented


def source_split(source_id: str) -> str:
    parts = source_id.split("_")
    variant = "1" if len(parts) == 1 else parts[1]
    if variant in {"1", "2"}:
        return "train"
    if variant == "3":
        return "val"
    return "test"


def valid_prompt_rows(layout: list[str]) -> list[tuple[str, str, str]]:
    rows = []
    for prompt in EVAL2_PROMPTS:
        try:
            color = resolve_prompt_target_color(prompt, layout)
        except ValueError:
            continue
        rows.append((prompt, color, target_slot(layout, color)))
    return rows


def select_prompt_rows(rows: list[tuple[str, str, str]], limit: int) -> list[tuple[str, str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return rows

    by_type: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row in rows:
        by_type[prompt_type(row[0])].append(row)

    selected: list[tuple[str, str, str]] = []

    def add(row: tuple[str, str, str]) -> None:
        if row not in selected and len(selected) < limit:
            selected.append(row)

    # Anchor every target slot with direct color and simple ordinal prompts.
    for slot in ("left", "middle", "right"):
        for row in by_type["color"]:
            if row[2] == slot:
                add(row)
                break
        for row in by_type["ordinal"]:
            if row[2] == slot:
                add(row)
                break

    # For 10 prompts, keep compositional coverage instead of letting one type dominate.
    type_quota = {"relative": 2, "negation": 2}
    for typ, quota in type_quota.items():
        added = 0
        for row in sorted(by_type[typ], key=lambda r: Counter(x[2] for x in selected)[r[2]]):
            add(row)
            added += 1
            if added >= quota or len(selected) >= limit:
                break
        if len(selected) >= limit:
            break

    # If a layout has fewer valid relative prompts, backfill deterministically.
    for typ in ("ordinal", "color", "negation", "relative"):
        for row in sorted(by_type[typ], key=lambda r: Counter(x[2] for x in selected)[r[2]]):
            add(row)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break

    return selected[:limit]


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src).convert("RGB")
    img.save(dst, quality=95)


def add_image_rows(
    rows: list[dict],
    *,
    src: Path,
    dst: Path,
    layout: list[str],
    layout_code: str,
    source_id: str,
    split: str,
    source_dataset: str,
    is_augmented: bool,
    prompts_per_image: int,
) -> None:
    copy_image(src, dst)
    prompts = select_prompt_rows(valid_prompt_rows(layout), prompts_per_image)
    for prompt, color, slot in prompts:
        rows.append(
            {
                "image": str(dst),
                "image_or_clip": str(dst),
                "phase": "initial_visible",
                "layout_code": layout_code,
                "layout_id": "-".join(layout),
                "layout_left_to_right": layout,
                "source_id": source_id,
                "split": split,
                "source_dataset": source_dataset,
                "is_augmented_image": is_augmented,
                "prompt": prompt,
                "prompt_type": prompt_type(prompt),
                "target_color": color,
                "target_slot": slot,
                "answer": answer_text(color, slot),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, default=Path("permutation_init_position"))
    parser.add_argument("--eval2-init-jsonl", type=Path)
    parser.add_argument("--eval2-init-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data/init_position_vqa"))
    parser.add_argument("--no-augmented", action="store_true")
    parser.add_argument("--one-image-per-layout-split", action="store_true")
    parser.add_argument("--prompts-per-image", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir
    image_out = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    image_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    image_files = sorted(
        p for p in args.images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    image_records = []
    for src in image_files:
        layout_code, source_id, is_augmented = parse_perm_image(src)
        if is_augmented and args.no_augmented:
            continue
        split = source_split(source_id)
        image_records.append((layout_code, split, is_augmented, source_id, src))
    if args.one_image_per_layout_split:
        picked = {}
        for layout_code, split, is_augmented, source_id, src in sorted(
            image_records, key=lambda x: (x[0], x[1], x[2], x[3], x[4].name)
        ):
            picked.setdefault((layout_code, split), (layout_code, split, is_augmented, source_id, src))
        image_records = list(picked.values())

    for layout_code, split, is_augmented, source_id, src in image_records:
        layout = layout_from_code(layout_code)
        dst = image_out / "permutation_init_position" / src.name
        add_image_rows(
            rows,
            src=src,
            dst=dst,
            layout=layout,
            layout_code=layout_code,
            source_id=source_id,
            split=split,
            source_dataset="permutation_init_position",
            is_augmented=is_augmented,
            prompts_per_image=args.prompts_per_image,
        )

    if args.eval2_init_jsonl:
        used_layouts: set[str] = set()
        for line in args.eval2_init_jsonl.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("phase") != "initial_visible":
                continue
            layout = normalize_layout("-".join(rec["layout_left_to_right"]))
            layout_code = code_from_layout(layout)
            if layout_code in used_layouts:
                continue
            used_layouts.add(layout_code)
            src = Path(rec["image"])
            dst = image_out / "eval2_video_init" / f"{layout_code}_eval2_init.jpg"
            add_image_rows(
                rows,
                src=src,
                dst=dst,
                layout=layout,
                layout_code=layout_code,
                source_id=f"{layout_code}_eval2_video_init",
                split="train",
                source_dataset="robot-learning-group47/eval2_compositional_augmented",
                is_augmented=False,
                prompts_per_image=args.prompts_per_image,
            )

    if args.eval2_init_dir:
        for src in sorted(args.eval2_init_dir.glob("*_eval2_init.*")):
            if src.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            layout_code = src.stem.split("_")[0].lower()
            layout = layout_from_code(layout_code)
            dst = image_out / "eval2_video_init" / f"{layout_code}_eval2_init{src.suffix.lower()}"
            add_image_rows(
                rows,
                src=src,
                dst=dst,
                layout=layout,
                layout_code=layout_code,
                source_id=f"{layout_code}_eval2_video_init",
                split="train",
                source_dataset="robot-learning-group47/eval2_compositional_augmented",
                is_augmented=False,
                prompts_per_image=args.prompts_per_image,
            )

    jsonl_path = out_dir / "vqa.jsonl"
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    by_source = defaultdict(set)
    for row in rows:
        by_source[row["source_id"]].add(row["image"])
    unique_images = {row["image"]: row for row in rows}.values()
    summary = {
        "num_rows": len(rows),
        "num_images": len({r["image"] for r in rows}),
        "num_source_groups": len(by_source),
        "one_image_per_layout_split": args.one_image_per_layout_split,
        "prompts_per_image": args.prompts_per_image,
        "rows_by_split": dict(Counter(r["split"] for r in rows)),
        "images_by_split": dict(Counter(r["split"] for r in unique_images)),
        "layouts_by_split": {
            split: sorted({r["layout_code"] for r in rows if r["split"] == split})
            for split in ["train", "val", "test"]
        },
        "rows_by_layout": dict(Counter(r["layout_code"] for r in rows)),
        "rows_by_prompt_type": dict(Counter(r["prompt_type"] for r in rows)),
        "rows_by_target_slot": dict(Counter(r["target_slot"] for r in rows)),
        "source_datasets": dict(Counter(r["source_dataset"] for r in rows)),
        "data_sources": [
            "permutation_init_position",
            "robot-learning-group47/eval2_compositional_augmented"
            if args.eval2_init_jsonl or args.eval2_init_dir
            else None,
        ],
        "output": str(jsonl_path),
    }
    summary["images_by_split_and_source"] = {
        f"{split}:{source}": count
        for (split, source), count in Counter((r["split"], r["source_dataset"]) for r in unique_images).items()
    }
    summary["data_sources"] = [s for s in summary["data_sources"] if s]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
