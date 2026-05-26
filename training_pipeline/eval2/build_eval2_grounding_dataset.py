#!/usr/bin/env python3
"""
Build the Eval2-Specific Grounding VLM dataset from permutation_init_position images.

Tasks:
  A - Layout recognition (left/middle/right → color)
  B - Slot-to-color  ("What color is the left bowl?" → "blue")
  C - Color-to-slot  ("Where is the blue bowl?" → "left")
  D - Direct color instruction grounding
  E - Ordinal instruction grounding
  F - Relational instruction grounding
  G - Negative instruction grounding

Output: data/eval2_grounding_vqa/vqa.jsonl
        data/eval2_grounding_vqa/metadata.json
        data/eval2_grounding_vqa/clean_images/   (filename-overlay-free crops)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw


# ── Layout encoding ───────────────────────────────────────────────────────────

COLOR_FULL = {"b": "blue", "g": "green", "r": "red"}
SLOTS = ["left", "middle", "right"]

# All 6 permutations encoded in filename stems.
PERM_MAP: dict[str, list[str]] = {
    "bgr": ["blue", "green", "red"],
    "brg": ["blue", "red", "green"],
    "gbr": ["green", "blue", "red"],
    "grb": ["green", "red", "blue"],
    "rbg": ["red", "blue", "green"],
    "rgb": ["red", "green", "blue"],
}


class Layout(NamedTuple):
    left: str
    middle: str
    right: str

    def slot_of(self, color: str) -> str:
        return SLOTS[[self.left, self.middle, self.right].index(color)]

    def color_of(self, slot: str) -> str:
        return getattr(self, slot)

    def as_dict(self) -> dict[str, str]:
        return {"left": self.left, "middle": self.middle, "right": self.right}

    def as_list(self) -> list[str]:
        return [self.left, self.middle, self.right]


# ── Image cleaning ────────────────────────────────────────────────────────────

def clean_image(src: Path, dst: Path, crop_top: int = 0, crop_bottom: int = 0) -> None:
    """Save a clean version of an image, cropping a safety strip off the top and/or bottom.

    Even if the source image appears clean, crop_top/crop_bottom pixel rows are removed
    as a conservative measure against any filename/path text overlays.
    Cropped result is resized back to the original dimensions for consistent input.
    """
    img = Image.open(src).convert("RGB")
    w, h = img.size
    top = crop_top
    bottom = h - crop_bottom if crop_bottom > 0 else h
    cropped = img.crop((0, top, w, bottom))
    # Resize back to original size so downstream code sees consistent 256×256 inputs
    clean = cropped.resize((w, h), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    clean.save(dst, quality=95)


# ── QA generators: Tasks A–G ─────────────────────────────────────────────────

SYSTEM = (
    "You are a robot vision assistant. A tabletop scene shows a robot arm, "
    "a banana, and exactly three bowls: one blue, one green, and one red. "
    "The bowls are arranged left to right from the robot's perspective. "
    "Use the image to determine the bowl order and answer concisely."
)


def task_a_layout(layout: Layout) -> list[tuple[str, str]]:
    """Task A: Layout recognition."""
    pairs = []

    # Plain text answer
    for q in [
        "What is the bowl order from left to right?",
        "Describe the bowl colors from left to right.",
        "List the bowl colors in order from left to right.",
        "What colors are the bowls from left to right?",
        "From left to right, what are the bowl colors?",
    ]:
        pairs.append((q, f"{layout.left}, {layout.middle}, {layout.right}"))

    # Structured JSON answer
    struct_q_variants = [
        "What is the bowl layout? Respond with JSON keys left, middle, right.",
        "Return the bowl positions as JSON with keys left, middle, right.",
        "Identify the bowl layout as JSON: {left, middle, right}.",
    ]
    struct_a = json.dumps({"layout_left": layout.left, "layout_middle": layout.middle, "layout_right": layout.right})
    for q in struct_q_variants:
        pairs.append((q, struct_a))

    return pairs


def task_b_slot_to_color(layout: Layout) -> list[tuple[str, str]]:
    """Task B: Slot-to-color."""
    pairs = []
    slot_questions = {
        "left": [
            "What color is the left bowl?",
            "What color is the leftmost bowl?",
            "Which color bowl is on the left?",
            "What is the color of the bowl on the left side?",
        ],
        "middle": [
            "What color is the middle bowl?",
            "What color is the center bowl?",
            "Which color bowl is in the middle?",
            "What is the color of the bowl in the center?",
        ],
        "right": [
            "What color is the right bowl?",
            "What color is the rightmost bowl?",
            "Which color bowl is on the right?",
            "What is the color of the bowl on the right side?",
        ],
    }
    for slot, questions in slot_questions.items():
        answer = layout.color_of(slot)
        for q in questions:
            pairs.append((q, answer))
    return pairs


def task_c_color_to_slot(layout: Layout) -> list[tuple[str, str]]:
    """Task C: Color-to-slot."""
    pairs = []
    color_questions = {
        "blue": [
            "Where is the blue bowl?",
            "Which position is the blue bowl in?",
            "Is the blue bowl on the left, middle, or right?",
            "What slot is the blue bowl in?",
        ],
        "green": [
            "Where is the green bowl?",
            "Which position is the green bowl in?",
            "Is the green bowl on the left, middle, or right?",
            "What slot is the green bowl in?",
        ],
        "red": [
            "Where is the red bowl?",
            "Which position is the red bowl in?",
            "Is the red bowl on the left, middle, or right?",
            "What slot is the red bowl in?",
        ],
    }
    for color, questions in color_questions.items():
        answer = layout.slot_of(color)
        for q in questions:
            pairs.append((q, answer))
    return pairs


def task_d_direct_color(layout: Layout) -> list[tuple[str, str]]:
    """Task D: Direct color instruction grounding → structured JSON."""
    pairs = []
    templates_by_color = {
        "blue": [
            "Put the banana in the blue bowl.",
            "Place the banana into the blue bowl.",
            "Move the banana to the blue bowl.",
            "Drop the banana in the blue bowl.",
            "Put the banana into the bowl that is blue.",
            "Place the banana in the bowl colored blue.",
            "The task is: put the banana in the blue bowl.",
        ],
        "green": [
            "Put the banana in the green bowl.",
            "Place the banana into the green bowl.",
            "Move the banana to the green bowl.",
            "Drop the banana in the green bowl.",
            "Put the banana into the bowl that is green.",
            "Place the banana in the bowl colored green.",
            "The task is: put the banana in the green bowl.",
        ],
        "red": [
            "Put the banana in the red bowl.",
            "Place the banana into the red bowl.",
            "Move the banana to the red bowl.",
            "Drop the banana in the red bowl.",
            "Put the banana into the bowl that is red.",
            "Place the banana in the bowl colored red.",
            "The task is: put the banana in the red bowl.",
        ],
    }
    for color, instructions in templates_by_color.items():
        slot = layout.slot_of(color)
        answer = json.dumps({"target_color": color, "target_slot": slot})
        for instr in instructions:
            q = f"Instruction: {instr}\nWhat is the target color and target slot?"
            pairs.append((q, answer))
    return pairs


def task_e_ordinal(layout: Layout) -> list[tuple[str, str]]:
    """Task E: Ordinal instruction grounding."""
    pairs = []

    # Ordinal from left: 1st=left, 2nd=middle, 3rd=right
    ordinal_from_left = [
        (1, "left"),
        (2, "middle"),
        (3, "right"),
    ]
    ordinal_templates_left = {
        1: [
            "Put the banana into the 1st bowl from the left.",
            "Put the banana into the first bowl from the left.",
            "Place the banana into the leftmost bowl.",
            "Move the banana to the first bowl counting from the left.",
            "Put the banana in the first bowl on the left.",
        ],
        2: [
            "Put the banana into the 2nd bowl from the left.",
            "Put the banana into the second bowl from the left.",
            "Place the banana into the middle bowl.",
            "Move the banana to the center bowl.",
            "Put the banana in the bowl in the middle.",
            "Put the banana in the second bowl from the left.",
            "Place the banana in the center bowl.",
        ],
        3: [
            "Put the banana into the 3rd bowl from the left.",
            "Put the banana into the third bowl from the left.",
            "Place the banana into the rightmost bowl.",
            "Move the banana to the third bowl counting from the left.",
            "Put the banana in the last bowl.",
        ],
    }
    for ordinal, slot in ordinal_from_left:
        color = layout.color_of(slot)
        answer = json.dumps({"target_color": color, "target_slot": slot})
        for instr in ordinal_templates_left[ordinal]:
            q = f"Instruction: {instr}\nWhat is the target color and target slot?"
            pairs.append((q, answer))

    # Ordinal from right: 1st=right, 2nd=middle, 3rd=left
    ordinal_from_right = [
        (1, "right"),
        (2, "middle"),
        (3, "left"),
    ]
    ordinal_templates_right = {
        1: [
            "Put the banana into the 1st bowl from the right.",
            "Put the banana into the first bowl from the right.",
            "Place the banana into the rightmost bowl.",
            "Move the banana to the first bowl counting from the right.",
        ],
        2: [
            "Put the banana into the 2nd bowl from the right.",
            "Put the banana into the second bowl from the right.",
            "Move the banana to the second bowl from the right.",
        ],
        3: [
            "Put the banana into the 3rd bowl from the right.",
            "Put the banana into the third bowl from the right.",
            "Place the banana into the leftmost bowl (from right perspective).",
            "Move the banana to the third bowl counting from the right.",
        ],
    }
    for ordinal, slot in ordinal_from_right:
        color = layout.color_of(slot)
        answer = json.dumps({"target_color": color, "target_slot": slot})
        for instr in ordinal_templates_right[ordinal]:
            q = f"Instruction: {instr}\nWhat is the target color and target slot?"
            pairs.append((q, answer))

    return pairs


def task_f_relational(layout: Layout) -> list[tuple[str, str]]:
    """Task F: Relational instruction grounding (only valid relations generated)."""
    pairs = []
    colors = layout.as_list()  # [left_color, middle_color, right_color]

    def _gen_right_of(ref_color: str) -> list[tuple[str, str]]:
        """Generate 'right of ref_color' queries (only when ref_color is not rightmost)."""
        ref_idx = colors.index(ref_color)
        if ref_idx >= 2:
            return []  # ref is rightmost, no bowl to its right
        target_slot = SLOTS[ref_idx + 1]
        target_color = colors[ref_idx + 1]
        answer = json.dumps({"target_color": target_color, "target_slot": target_slot})
        instructions = [
            f"Put the banana into the bowl right of the {ref_color} bowl.",
            f"Place the banana in the bowl to the right of the {ref_color} bowl.",
            f"Move the banana to the bowl immediately right of the {ref_color} bowl.",
            f"Put the banana in the bowl next to the {ref_color} bowl on its right side.",
        ]
        return [(f"Instruction: {instr}\nWhat is the target color and target slot?", answer)
                for instr in instructions]

    def _gen_left_of(ref_color: str) -> list[tuple[str, str]]:
        """Generate 'left of ref_color' queries (only when ref_color is not leftmost)."""
        ref_idx = colors.index(ref_color)
        if ref_idx <= 0:
            return []
        target_slot = SLOTS[ref_idx - 1]
        target_color = colors[ref_idx - 1]
        answer = json.dumps({"target_color": target_color, "target_slot": target_slot})
        instructions = [
            f"Put the banana into the bowl left of the {ref_color} bowl.",
            f"Place the banana in the bowl to the left of the {ref_color} bowl.",
            f"Move the banana to the bowl immediately left of the {ref_color} bowl.",
        ]
        return [(f"Instruction: {instr}\nWhat is the target color and target slot?", answer)
                for instr in instructions]

    def _gen_between(c_left: str, c_right: str) -> list[tuple[str, str]]:
        """Generate 'between c_left and c_right' (only valid when middle bowl is between them)."""
        idx_l = colors.index(c_left)
        idx_r = colors.index(c_right)
        if abs(idx_l - idx_r) != 2:
            return []  # not left–right neighbors with one between
        mid_idx = (idx_l + idx_r) // 2
        target_slot = SLOTS[mid_idx]
        target_color = colors[mid_idx]
        answer = json.dumps({"target_color": target_color, "target_slot": target_slot})
        instructions = [
            f"Put the banana into the bowl between the {c_left} bowl and the {c_right} bowl.",
            f"Place the banana in the bowl that is between {c_left} and {c_right}.",
            f"Move the banana to the bowl sandwiched between the {c_left} and {c_right} bowls.",
        ]
        return [(f"Instruction: {instr}\nWhat is the target color and target slot?", answer)
                for instr in instructions]

    for c in ["blue", "green", "red"]:
        pairs.extend(_gen_right_of(c))
        pairs.extend(_gen_left_of(c))

    # Between queries: only (leftmost, rightmost) is guaranteed valid
    left_c, right_c = colors[0], colors[2]
    pairs.extend(_gen_between(left_c, right_c))
    pairs.extend(_gen_between(right_c, left_c))

    return pairs


def task_g_negative(layout: Layout) -> list[tuple[str, str]]:
    """Task G: Negative instruction grounding ('not X and not Y' → remaining color)."""
    pairs = []
    colors = ["blue", "green", "red"]

    for excluded1, excluded2 in [
        ("green", "red"),    # → blue
        ("blue", "red"),     # → green
        ("blue", "green"),   # → red
    ]:
        target_color = [c for c in colors if c != excluded1 and c != excluded2][0]
        target_slot = layout.slot_of(target_color)
        answer = json.dumps({"target_color": target_color, "target_slot": target_slot})

        instructions = [
            f"Put the banana into the bowl that is not {excluded1} and not {excluded2}.",
            f"Place the banana in the bowl that is neither {excluded1} nor {excluded2}.",
            f"Move the banana to the only bowl that is not {excluded1} or {excluded2}.",
            f"Put the banana in the bowl that is not {excluded2} and not {excluded1}.",
            f"Place the banana in the bowl that is neither {excluded2} nor {excluded1}.",
        ]
        for instr in instructions:
            q = f"Instruction: {instr}\nWhat is the target color and target slot?"
            pairs.append((q, answer))

    return pairs


# ── Dataset assembly ──────────────────────────────────────────────────────────

TASK_GENERATORS = [
    ("A", task_a_layout),
    ("B", task_b_slot_to_color),
    ("C", task_c_color_to_slot),
    ("D", task_d_direct_color),
    ("E", task_e_ordinal),
    ("F", task_f_relational),
    ("G", task_g_negative),
]


def generate_records(image_rel: str, layout: Layout, split: str) -> list[dict]:
    """Generate all QA records for one (image, layout) pair."""
    records = []
    for task_id, gen_fn in TASK_GENERATORS:
        pairs = gen_fn(layout)
        for question, answer in pairs:
            records.append({
                "image": image_rel,
                "question": question,
                "answer": answer,
                "task": task_id,
                "layout": layout.as_dict(),
                "split": split,
            })
    return records


def parse_perm_from_stem(stem: str) -> tuple[str, str] | tuple[None, None]:
    """Extract (perm_code, suffix) from a filename stem.

    Examples:
        'bgr'    → ('bgr', '')       → train
        'bgr_2'  → ('bgr', '_2')     → train
        'bgr_3'  → ('bgr', '_3')     → val
        'bgr_4'  → ('bgr', '_4')     → benchmark  (never shown during training)
    """
    stem_lower = stem.lower()
    for perm in PERM_MAP:
        if stem_lower == perm:
            return perm, ""
        if stem_lower.startswith(perm + "_"):
            suffix = stem_lower[len(perm):]  # e.g. '_2', '_3', '_4'
            return perm, suffix
    return None, None


# Split assignment rules:
#   suffix '_4'  → val    (validation / early-stopping; one held-out image per permutation)
#   everything else (bare, _2, _3) → train
_VAL_SUFFIX = "_4"


def assign_split(suffix: str) -> str:
    if suffix == _VAL_SUFFIX:
        return "val"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path,
                        default=Path("permutation_init_position"),
                        help="Directory containing raw JPEG images.")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/eval2_grounding_vqa"),
                        help="Output directory for dataset.")
    parser.add_argument("--crop-top", type=int, default=0,
                        help="Pixels to crop from top (removes filename overlay if present).")
    parser.add_argument("--crop-bottom", type=int, default=0,
                        help="Pixels to crop from bottom (removes path overlay if present).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    images_dir = args.images_dir
    out_dir = args.out_dir
    clean_dir = out_dir / "clean_images"
    clean_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect images grouped by permutation ─────────────────────────────────
    # Each entry: (path, perm_code, suffix, split)
    image_entries: list[tuple[Path, str, str, str]] = []
    perm_seen: set[str] = set()

    for img_path in sorted(images_dir.glob("*.jpeg")) + sorted(images_dir.glob("*.jpg")):
        perm, suffix = parse_perm_from_stem(img_path.stem)
        if perm is None:
            print(f"  [skip] cannot parse perm from: {img_path.name}")
            continue
        split = assign_split(suffix)
        image_entries.append((img_path, perm, suffix, split))
        perm_seen.add(perm)

    print("\n=== Image inventory ===")
    print(f"  {'PERM':<6}  {'SUFFIX':<6}  SPLIT")
    print(f"  {'-'*6}  {'-'*6}  -----")
    for img_path, perm, suffix, split in image_entries:
        print(f"  {perm:<6}  {(suffix or 'bare'):<6}  {split}")

    missing_perms = [p for p in PERM_MAP if p not in perm_seen]
    if missing_perms:
        print(f"\nWARNING: missing permutations entirely: {missing_perms}")
        print("Please collect initial-position images for these permutations.\n")

    splits_without_benchmark = {s for _, _, _, s in image_entries if s != "benchmark"}
    print(f"\nSplit summary:")
    from collections import Counter
    split_counts = Counter(s for _, _, _, s in image_entries)
    for spl in ["train", "val", "benchmark"]:
        print(f"  {spl}: {split_counts.get(spl, 0)} images")

    # ── Clean images and build QA records ────────────────────────────────────
    metadata = []
    all_records = []

    for src_path, perm, suffix, split_name in image_entries:
        layout = Layout(*PERM_MAP[perm])

        # Clean image (crop overlay strips if specified)
        clean_name = src_path.stem + "_clean" + src_path.suffix
        clean_path = clean_dir / clean_name
        clean_image(src_path, clean_path,
                    crop_top=args.crop_top,
                    crop_bottom=args.crop_bottom)

        # Relative path for JSONL (relative to out_dir)
        image_rel = str(Path("clean_images") / clean_name)

        # Metadata entry
        meta_entry = {
            "image": clean_name,
            "source": src_path.name,
            "permutation": perm,
            "suffix": suffix or "bare",
            "layout": layout.as_dict(),
            "split": split_name,
        }
        metadata.append(meta_entry)

        # QA records — generated for ALL splits so benchmark can be evaluated
        records = generate_records(image_rel, layout, split_name)
        all_records.extend(records)

    # ── Write outputs ─────────────────────────────────────────────────────────
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nMetadata written: {meta_path} ({len(metadata)} images)")

    jsonl_path = out_dir / "vqa.jsonl"
    with jsonl_path.open("w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    from collections import Counter
    split_qa = Counter(r["split"] for r in all_records)
    train_n = split_qa["train"]
    val_n   = split_qa["val"]
    print(f"JSONL written: {jsonl_path}")
    print(f"  train={train_n}  val={val_n}  total={len(all_records)}")

    # Per-task breakdown (train only)
    task_counts = Counter(r["task"] for r in all_records if r["split"] == "train")
    print("\nTrain QA counts by task:")
    for task_id, cnt in sorted(task_counts.items()):
        print(f"  Task {task_id}: {cnt}")

    print(f"\nVal images (*_4):        6 images — one per permutation, used for early stopping.")
    print("Train images (bare, *_2, *_3): 18 images — 3 per permutation.")
    print("\nDone. Next step: python train_eval2_grounding_vlm.py")


if __name__ == "__main__":
    main()
