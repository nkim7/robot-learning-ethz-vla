#!/usr/bin/env python3
"""
VQA backbone ceiling check.

Loads smolvlm2-500m-vqa-lora-merged and probes it on all 6 bowl permutations
using images from permutation_init_position/.  Ground truth is derived from the
filename (e.g. bgr.jpeg → blue=left, green=centre, red=right).

For each permutation we ask 20 eval2 questions and compare against the known
correct answer.  This tells us whether the reasoning is already in the backbone
weights — if yes, any VLA failure is purely an action-path problem.

Usage:
    python probe_vqa_backbone.py
    python probe_vqa_backbone.py --model robot-learning-group47/smolvlm2-500m-vqa-lora-merged
    python probe_vqa_backbone.py --images_dir permutation_init_position --per_perm 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

# ── Ground-truth derivation ───────────────────────────────────────────────────

COLOR_NAMES = {"b": "blue", "g": "green", "r": "red"}

def derive_gt(order: str) -> dict:
    """
    Given a 3-char ordering string like 'bgr', return a dict of correct answers
    for all 20 eval2 prompts.

    order[0] = leftmost bowl colour initial
    order[1] = centre bowl colour initial
    order[2] = rightmost bowl colour initial
    """
    c = [COLOR_NAMES[x] for x in order]   # ['blue','green','red'] for 'bgr'

    gt = {}

    # ── COLOR (3) ─────────────────────────────────────────────────────────────
    gt["Put the banana in the red colored bowl."]   = c.index("red")   + 1  # position 1-indexed
    gt["Put the banana in the green colored bowl."] = c.index("green") + 1
    gt["Put the banana in the blue colored bowl."]  = c.index("blue")  + 1

    # ── ORDINAL FROM LEFT (3) ──────────────────────────────────────────────────
    gt["Put the banana into the 1st bowl from the left from the robot perspective."]  = 1
    gt["Put the banana into the 2nd bowl from the left from the robot perspective."]  = 2
    gt["Put the banana into the 3rd bowl from the left from the robot perspective."]  = 3

    # ── ORDINAL FROM RIGHT (3) ─────────────────────────────────────────────────
    gt["Put the banana into the 1st bowl from the right from the robot perspective."] = 3
    gt["Put the banana into the 2nd bowl from the right from the robot perspective."] = 2
    gt["Put the banana into the 3rd bowl from the right from the robot perspective."] = 1

    # ── MIDDLE (1) ────────────────────────────────────────────────────────────
    gt["Put the banana into the middle bowl from the robot perspective."] = 2

    # ── RELATIVE (4) ─────────────────────────────────────────────────────────
    red_pos   = c.index("red")   # 0-indexed
    green_pos = c.index("green")
    # left of red  = bowl whose 0-indexed pos < red_pos (there's exactly one)
    left_of_red   = [i for i in range(3) if i < red_pos]
    right_of_red  = [i for i in range(3) if i > red_pos]
    left_of_green  = [i for i in range(3) if i < green_pos]
    right_of_green = [i for i in range(3) if i > green_pos]

    def _pos(lst): return lst[0] + 1 if lst else None  # 1-indexed, None if impossible

    gt["Put the banana into the bowl on the left of the red bowl from the robot perspective."]   = _pos(left_of_red)
    gt["Put the banana into the bowl on the right of the red bowl from the robot perspective."]  = _pos(right_of_red)
    gt["Put the banana into the bowl on the left of the green bowl from the robot perspective."] = _pos(left_of_green)
    gt["Put the banana into the bowl on the right of the green bowl from the robot perspective."]= _pos(right_of_green)

    # ── NEGATION (6) — "not A and not B" = C ──────────────────────────────────
    gt["Put the banana into the bowl that is not red and not green."]  = c.index("blue")  + 1
    gt["Put the banana into the bowl that is not red and not blue."]   = c.index("green") + 1
    gt["Put the banana into the bowl that is not green and not blue."] = c.index("red")   + 1
    gt["Put the banana into the bowl that is not green and not red."]  = c.index("blue")  + 1
    gt["Put the banana into the bowl that is not blue and not red."]   = c.index("green") + 1
    gt["Put the banana into the bowl that is not blue and not green."] = c.index("red")   + 1

    return gt


PROMPTS = list(derive_gt("bgr").keys())   # order matches eval2

# ── Prompt → VQA question ─────────────────────────────────────────────────────

SYSTEM = (
    "You are looking at a robot's wrist/overhead camera view. "
    "There are exactly three bowls in a row: one on the left, one in the centre, one on the right. "
    "Answer questions about which bowl to target. "
    "Reply with only ONE of: 'left bowl', 'centre bowl', or 'right bowl'."
)

def prompt_to_question(prompt: str) -> str:
    """Turn an action prompt into a VQA question."""
    q = prompt.replace("Put the banana", "Where should the robot put the banana")
    q = q.replace("in the ", "— into the ").replace("into the ", "— into the ")
    q = q.rstrip(".")
    q += "? Answer with only 'left bowl', 'centre bowl', or 'right bowl'."
    return q


def pos_to_word(pos: int | None) -> str:
    if pos == 1: return "left bowl"
    if pos == 2: return "centre bowl"
    if pos == 3: return "right bowl"
    return "N/A"


def parse_answer(raw: str) -> str | None:
    raw = raw.lower()
    if "left" in raw:   return "left bowl"
    if "centre" in raw or "center" in raw or "middle" in raw: return "centre bowl"
    if "right" in raw:  return "right bowl"
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="robot-learning-group47/smolvlm2-500m-vqa-lora-merged",
                        help="HF model ID  OR  path to a SmolVLA checkpoint dir (step4000 etc.)")
    parser.add_argument("--images_dir", default="permutation_init_position")
    parser.add_argument("--per_perm", type=int, default=1,
                        help="How many images per permutation to test (1=only baseN.jpeg, 4=all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERROR: {images_dir} not found"); sys.exit(1)

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    # Detect whether it's a SmolVLA checkpoint dir or a plain HF VLM model.
    VLM_PREFIX = "model.vlm_with_expert.vlm."
    VQA_BASE   = "robot-learning-group47/smolvlm2-500m-vqa-lora-merged"

    smolvla_path = Path(args.model)
    is_smolvla_ckpt = smolvla_path.exists() and (smolvla_path / "model.safetensors").exists()

    if is_smolvla_ckpt:
        import safetensors.torch as st
        print(f"\nSmolVLA checkpoint detected: {args.model}")
        print(f"  Loading processor from {VQA_BASE} ...")
        processor = AutoProcessor.from_pretrained(VQA_BASE)

        print(f"  Loading architecture from {VQA_BASE} ...")
        model = SmolVLMForConditionalGeneration.from_pretrained(
            VQA_BASE,
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",
        )

        print(f"  Extracting and injecting VLM weights from checkpoint ...")
        ckpt_sd = st.load_file(smolvla_path / "model.safetensors")
        vlm_sd  = {
            k.replace(VLM_PREFIX, "model."): v
            for k, v in ckpt_sd.items()
            if k.startswith(VLM_PREFIX)
        }
        missing, unexpected = model.load_state_dict(vlm_sd, strict=False)
        print(f"  Injected {len(vlm_sd)} keys  |  missing={len(missing)}  unexpected={len(unexpected)}")
        model = model.to(device)
    else:
        print(f"\nLoading {args.model} ...")
        processor = AutoProcessor.from_pretrained(args.model)
        model = SmolVLMForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",
        ).to(device)

    model.eval()
    print("Model ready.\n")

    PERMUTATIONS = ["bgr", "brg", "gbr", "grb", "rbg", "rgb"]

    # Collect results: prompt → list of (correct, answered)
    by_prompt: dict[str, list[bool]] = defaultdict(list)
    by_perm:   dict[str, list[bool]] = defaultdict(list)
    overall_results = []

    for perm in PERMUTATIONS:
        gt_map = derive_gt(perm)

        # Find images for this permutation
        img_paths = sorted(images_dir.glob(f"{perm}*.jpeg"))[:args.per_perm]
        if not img_paths:
            img_paths = sorted(images_dir.glob(f"{perm}*.jpg"))[:args.per_perm]
        if not img_paths:
            print(f"  WARNING: no images found for {perm}, skipping")
            continue

        for img_path in img_paths:
            image = Image.open(img_path).convert("RGB")

            print(f"\n{'─'*70}")
            print(f"  Perm: {perm}  ({img_path.name})  "
                  f"[L={COLOR_NAMES[perm[0]]} C={COLOR_NAMES[perm[1]]} R={COLOR_NAMES[perm[2]]}]")
            print(f"{'─'*70}")

            perm_correct = 0

            for prompt in PROMPTS:
                gt_pos  = gt_map[prompt]
                if gt_pos is None:
                    # e.g. "left of red" when red is already leftmost — skip
                    continue
                gt_word = pos_to_word(gt_pos)
                question = prompt_to_question(prompt)

                # Build chat message
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"{SYSTEM}\n\n{question}"},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
                inputs = processor(text=text, images=[image], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

                # Decode only new tokens
                n_input = inputs["input_ids"].shape[1]
                raw = processor.decode(out[0][n_input:], skip_special_tokens=True).strip()
                pred = parse_answer(raw)
                correct = (pred == gt_word)

                by_prompt[prompt].append(correct)
                by_perm[perm].append(correct)
                overall_results.append(correct)
                perm_correct += correct

                mark = "✅" if correct else "❌"
                if args.verbose or not correct:
                    print(f"  {mark}  GT={gt_word:13s}  Pred={str(pred):13s}  | {prompt[:60]}")
                elif correct:
                    print(f"  {mark}  GT={gt_word:13s}  | {prompt[:60]}")

            print(f"  → {perm} score: {perm_correct}/{len([p for p in PROMPTS if gt_map[p] is not None])}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total = len(overall_results)
    correct_total = sum(overall_results)
    print(f"\n{'═'*70}")
    print(f"  OVERALL: {correct_total}/{total} = {correct_total/total*100:.1f}%")
    print(f"{'═'*70}")

    # By prompt category
    cats = {
        "Color (3)":           PROMPTS[0:3],
        "Ordinal-left (3)":    PROMPTS[3:6],
        "Ordinal-right (3)":   PROMPTS[6:9],
        "Middle (1)":          PROMPTS[9:10],
        "Relative pos (4)":    PROMPTS[10:14],
        "Negation (6)":        PROMPTS[14:20],
    }
    print("\n  By category:")
    for cat, ps in cats.items():
        bools = [b for p in ps for b in by_prompt[p]]
        if bools:
            print(f"    {cat:22s}: {sum(bools)}/{len(bools)} = {sum(bools)/len(bools)*100:.0f}%")

    print("\n  By permutation:")
    for perm in PERMUTATIONS:
        bools = by_perm[perm]
        if bools:
            print(f"    {perm}: {sum(bools)}/{len(bools)} = {sum(bools)/len(bools)*100:.0f}%")

    print()


if __name__ == "__main__":
    main()
