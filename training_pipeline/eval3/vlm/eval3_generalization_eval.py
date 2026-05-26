#!/usr/bin/env python3
"""
eval3_generalization_eval.py
=============================
Evaluates whether robot-learning-group47/eval3_lora_fixed_Chengming
generalises to *unseen* celebrity images.

Source of unseen images:
    robot-learning-group47/celebrities_faces_all_different
    (all images in this dataset are confirmed different from the 15 training images)

Four test blocks:
  G1  Single-image recognition on unseen images  (→ compare with T1 100%)
  G2  Prompt stability on unseen images           (→ compare with T3 11/15 stable-correct)
  G3  Composite panorama grounding on unseen      (→ compare with T5a 94.4%)
  G4  Robot-view scene grounding on unseen        (NEW — pastes faces into real robot
       backgrounds at the exact portrait-box positions used in eval3_record.sh)

Usage:
    python eval3_generalization_eval.py

Options:
    --model        HF repo for the LoRA model   [robot-learning-group47/eval3_lora_fixed_Chengming]
    --base_model   HF backbone                  [HuggingFaceTB/SmolVLM2-2.2B-Instruct]
    --dataset      unseen-images HF repo        [robot-learning-group47/celebrities_faces_all_different]
    --scene_dir    directory with real robot-background scene images  [./data/scene_setup]
    --n_per_class  max unseen images to test per identity             [10]
    --output       results JSON path            [./eval3_generalization_results.json]
    --device       cuda / mps / cpu             [auto]
"""

import argparse
import itertools
import json
import os
import re
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",       default="robot-learning-group47/eval3_lora_fixed_Chengming")
parser.add_argument("--base_model",  default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--dataset",     default="robot-learning-group47/celebrities_faces_all_different")
parser.add_argument("--scene_dir",   default="./data/scene_setup")
parser.add_argument("--n_per_class", type=int, default=999)  # 999 = load all available
parser.add_argument("--output",      default="./eval3_generalization_results.json")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--device",      default=None)
parser.add_argument("--max_new_tokens", type=int, default=32)
args = parser.parse_args()

random.seed(args.seed)

if args.device is None:
    args.device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
DTYPE = torch.bfloat16 if args.device == "cuda" else torch.float32
print(f"Device: {args.device}  |  dtype: {DTYPE}")

# ── Constants (match the training-set verification) ───────────────────────────
CLASSES    = ["taylor", "obama", "yann"]
FULL_NAMES = {"taylor": "Taylor Swift", "obama": "Barack Obama", "yann": "Yann LeCun"}
HF_LABELS  = {"taylor": "Taylor Swift", "obama": "Barack Obama", "yann": "Yann LeCun"}

# Canonical prompt used in training (P0) — must match verify_eval3_training_fit.py
CANONICAL_PROMPT = (
    "Who is the person in this image? "
    "Answer with exactly one word from: taylor, obama, yann."
)

# 5 prompt variants (same as in T3 verification)
PROMPT_VARIANTS = [
    CANONICAL_PROMPT,
    "Who is this person? Give the full name.",
    "Which of the following celebrities is shown? Taylor Swift, Barack Obama, or Yann LeCun?",
    "Name the person in the photo.",
    "Which celebrity does this image show — Taylor Swift, Barack Obama, or Yann LeCun? Answer with the full name.",
]

ALIASES = {
    "taylor": "taylor", "taylor swift": "taylor", "swift": "taylor", "ms. swift": "taylor",
    "obama": "obama", "barack obama": "obama", "barack": "obama",
    "president obama": "obama", "mr. obama": "obama",
    "yann": "yann", "yann lecun": "yann", "yann le cun": "yann",
    "lecun": "yann", "le cun": "yann",
}

# ── v6 composition data (from composite_test6.py) ────────────────────────────
# Exact polygon quad corners for each card position in each real scene background.
# Used for perspective warping — much more accurate than flat PORTRAIT_BOXES.
CARD_QUADS = {
    "scene_L-Swift_C-LeCun_R-Obama_1.jpg": {
        "LEFT":   [(31,43),(93,82),(55,119),(-5,77)],
        "CENTER": [(98,16),(104,62),(153,60),(143,16)],
        "RIGHT":  [(154,79),(192,118),(249,84),(213,49)],
        "CAN_OCCLUDE": [(148,41),(136,44),(133,53),(130,66),(122,84),(115,99),(116,115),(121,116),(132,117),(138,104),(146,94),(161,71),(165,62),(167,54),(160,46),(158,42)],
    },
    "scene_L-Swift_C-LeCun_R-Obama_2.jpg": {
        "LEFT":   [(29,44),(93,80),(55,116),(-5,77)],
        "CENTER": [(97,16),(103,60),(153,60),(144,18)],
        "RIGHT":  [(215,52),(156,80),(193,119),(248,85)],
        "CAN_OCCLUDE": [(146,41),(137,43),(134,49),(132,59),(129,72),(123,84),(115,96),(113,106),(122,116),(136,116),(144,94),(166,62),(166,53),(156,42)],
    },
    "scene_L-LeCun_C-Swift_R-Obama.jpg": {
        "LEFT":   [(-5,74),(36,48),(84,85),(41,118)],
        "CENTER": [(91,13),(90,60),(136,59),(134,14)],
        "RIGHT":  [(202,47),(154,82),(202,123),(251,82)],
        "CAN_OCCLUDE": [(147,40),(137,44),(133,56),(128,69),(123,81),(115,97),(115,114),(121,116),(130,116),(138,116),(147,95),(152,83),(158,74),(164,65),(168,56),(160,46),(157,41)],
    },
    "scene_L-LeCun_C-Obama_R-Swift_1.jpg": {
        "LEFT":   [(-4,77),(30,46),(85,83),(49,119)],
        "CENTER": [(108,14),(102,62),(161,66),(156,18)],
        "RIGHT":  [(171,78),(206,124),(264,92),(232,54)],
        "CAN_OCCLUDE": [(144,41),(137,43),(131,55),(129,66),(119,86),(113,107),(119,114),(130,115),(136,115),(140,100),(152,84),(167,61),(167,51),(157,45),(155,40)],
    },
    "scene_L-LeCun_C-Obama_R-Swift_2.jpg": {
        "LEFT":   [(-5,76),(31,46),(85,81),(46,117)],
        "CENTER": [(106,14),(101,60),(160,64),(157,17)],
        "RIGHT":  [(172,84),(217,125),(268,89),(230,53)],
        "CAN_OCCLUDE": [(141,41),(133,45),(130,57),(129,69),(121,83),(114,96),(113,112),(121,118),(132,118),(138,116),(137,105),(144,100),(152,85),(156,76),(165,64),(166,54),(160,47),(157,43),(150,41)],
    },
}

CORNER_ORDER = {
    "scene_L-Swift_C-LeCun_R-Obama_1.jpg": {
        "LEFT": [3,0,1,2], "CENTER": [0,3,2,1], "RIGHT": [3,2,1,0],
    },
    "scene_L-Swift_C-LeCun_R-Obama_2.jpg": {
        "LEFT": [3,0,1,2], "CENTER": [0,3,2,1], "RIGHT": [0,3,2,1],
    },
    "scene_L-LeCun_C-Swift_R-Obama.jpg": {
        "LEFT": [0,1,2,3], "CENTER": [0,3,2,1], "RIGHT": [0,3,2,1],
    },
    "scene_L-LeCun_C-Obama_R-Swift_1.jpg": {
        "LEFT": [0,1,2,3], "CENTER": [0,3,2,1], "RIGHT": [3,2,1,0],
    },
    "scene_L-LeCun_C-Obama_R-Swift_2.jpg": {
        "LEFT": [0,1,2,3], "CENTER": [0,3,2,1], "RIGHT": [3,2,1,0],
    },
}

# position key used in CARD_QUADS → our "left/center/right" vocabulary
POS_KEY = {"left": "LEFT", "center": "CENTER", "right": "RIGHT"}

# Reference results from training-set verification (T1/T3/T5)
TRAINING_RESULTS = {
    "T1_recognition_acc":      1.000,   # 15/15
    "T3_stable_correct_rate":  11/15,   # 11/15
    "T5a_composite_acc":       17/18,   # 94.4%
    "T5b_multiimage_acc":      16/18,   # 88.9%
}


# ── Label helpers ─────────────────────────────────────────────────────────────
def parse_label(raw: str):
    text = raw.strip().lower()
    if text in ALIASES:
        return ALIASES[text]
    for alias, label in ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text):
            return label
    return None

def parse_position(raw: str):
    text = raw.strip().lower()
    for w in ["left", "center", "middle", "right"]:
        if w in text:
            return "center" if w == "middle" else w
    return None


# ── Model loader ──────────────────────────────────────────────────────────────
def load_lora_model(model_id: str, base_id: str, device: str, dtype):
    from peft import PeftModel
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None

    print(f"\n[load] Processor  : {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    print(f"[load] Base model : {base_id}")
    base = SmolVLMForConditionalGeneration.from_pretrained(
        base_id, torch_dtype=dtype, device_map=device, token=hf_token
    )
    print(f"[load] LoRA adapter: {model_id}")
    model = PeftModel.from_pretrained(base, model_id, token=hf_token)
    model.eval()
    print(f"[load] Ready.  Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────
def ask(model, processor, images: list, question: str, device: str, max_new_tokens: int = 32) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": question}],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_ids = ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True).strip()


# ── Composition helpers ───────────────────────────────────────────────────────
THUMB = (224, 224)

def make_panorama(imgs: list) -> Image.Image:
    """Side-by-side panorama (same as T5a in verify_eval3_training_fit.py)."""
    resized = [im.convert("RGB").resize(THUMB) for im in imgs]
    canvas  = Image.new("RGB", (THUMB[0] * 3, THUMB[1]))
    for i, im in enumerate(resized):
        canvas.paste(im, (i * THUMB[0], 0))
    return canvas


# ── v6 composition helpers (ported from composite_test6.py) ──────────────────

def _get_quad(poly_pts, order):
    """Return [TL, TR, BR, BL] float32 array using user-specified corner indices."""
    return np.array([poly_pts[i] for i in order], dtype=np.float32)

def _expand_quad(quad, frac=0.12):
    """Expand quad outward from centroid by frac (gives slight margin)."""
    arr = np.array(quad, np.float32)
    cx, cy = arr.mean(axis=0)
    return (arr + frac * (arr - np.array([[cx, cy]] * 4))).astype(int)

def _match_brightness(warped, base, mask):
    ref = base[mask > 128].astype(np.float32)
    src = warped[mask > 128].astype(np.float32)
    if ref.size == 0 or src.size == 0:
        return warped
    rm, rs = ref.mean(0), ref.std(0) + 1e-6
    sm, ss = src.mean(0), src.std(0) + 1e-6
    out = warped.copy().astype(np.float32)
    out[mask > 128] = np.clip((src - sm) / ss * rs + rm, 0, 255)
    return out.astype(np.uint8)

def _paste_v6(base_bgr, face_pil, poly_pts, corner_order, can_poly):
    """
    Perspective-warp face_pil into the card polygon on base_bgr.
    Preserves can occlusion and matches brightness to background.
    """
    H, W = base_bgr.shape[:2]
    pq = _expand_quad(_get_quad(poly_pts, corner_order), frac=0.12)

    # Convert PIL face to BGR numpy, resize to something reasonable
    face_bgr = cv2.cvtColor(np.array(face_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    fH, fW = face_bgr.shape[:2]

    src = np.float32([[0, 0], [fW, 0], [fW, fH], [0, fH]])
    dst = pq.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(face_bgr, M, (W, H))

    # Card mask
    mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(mask, pq.astype(np.int32), 255)

    # Coke-can occludes the portrait
    can_pts = np.array(can_poly, np.int32)
    can_pts[:, 0] = np.clip(can_pts[:, 0], 0, W - 1)
    can_pts[:, 1] = np.clip(can_pts[:, 1], 0, H - 1)
    cv2.fillPoly(mask, [can_pts], 0)

    warped = _match_brightness(warped, base_bgr, mask)
    warped = cv2.GaussianBlur(warped, (3, 3), 0)
    mask_b = cv2.GaussianBlur(mask, (5, 5), 0)

    a = mask_b.astype(np.float32)[:, :, None] / 255.0
    blended = np.clip(base_bgr * (1 - a) + warped * a, 0, 255).astype(np.uint8)
    return blended


def make_v6_scene(scene_fname: str, scene_dir: Path,
                  imgs_by_pos: dict) -> Image.Image:
    """
    Build a v6-quality robot-view composite for scene_fname.

    scene_fname  : key in CARD_QUADS (e.g. 'scene_L-LeCun_C-Obama_R-Swift_1.jpg')
    scene_dir    : directory containing the real background files
    imgs_by_pos  : {"left": PIL.Image, "center": PIL.Image, "right": PIL.Image}

    Returns a 256×256 PIL.Image.
    """
    bg_path = scene_dir / scene_fname
    if not bg_path.exists():
        raise FileNotFoundError(f"Background not found: {bg_path}")

    base_bgr = cv2.imread(str(bg_path))
    if base_bgr is None:
        raise IOError(f"cv2 could not read: {bg_path}")

    quads  = CARD_QUADS[scene_fname]
    orders = CORNER_ORDER[scene_fname]
    can    = quads["CAN_OCCLUDE"]

    for pos, face_pil in imgs_by_pos.items():
        pk = POS_KEY[pos]
        base_bgr = _paste_v6(base_bgr, face_pil, quads[pk], orders[pk], can)

    # slight noise + resize to 256×256 (matches v6 output exactly)
    noise   = np.random.normal(0, 1.5, base_bgr.shape).astype(np.int16)
    base_bgr = np.clip(base_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    base_bgr = cv2.resize(base_bgr, (256, 256), interpolation=cv2.INTER_AREA)

    return Image.fromarray(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB))


# ── Load unseen images ─────────────────────────────────────────────────────────
print(f"\nLoading unseen images from {args.dataset} …")
from datasets import load_dataset
ds = load_dataset(args.dataset, split="train")

unseen: dict[str, list] = {"taylor": [], "obama": [], "yann": []}
label_map = {v: k for k, v in HF_LABELS.items()}

for row in ds:
    key = label_map.get(row["label"])
    if key and len(unseen[key]) < args.n_per_class:
        unseen[key].append(row["image"].convert("RGB"))

print(f"Loaded unseen images:")
for cls in CLASSES:
    print(f"  {cls:8s}: {len(unseen[cls])} images (requested {args.n_per_class})")

total_unseen = sum(len(v) for v in unseen.values())

# ── Locate background scenes (for G4 v6 robot-view test) ─────────────────────
scene_dir = Path(args.scene_dir)
available_scenes = [f for f in CARD_QUADS if (scene_dir / f).exists()]
print(f"\nv6 background scenes found in {scene_dir}: {len(available_scenes)}/{len(CARD_QUADS)}")
for f in available_scenes:
    print(f"  {f}")
if not available_scenes:
    print("  WARNING: no scene backgrounds found — G4 will be skipped.")
    print(f"  Expected files like 'scene_L-LeCun_C-Obama_R-Swift_1.jpg' in {scene_dir}")


# ── Load fine-tuned model ─────────────────────────────────────────────────────
model, proc = load_lora_model(args.model, args.base_model, args.device, DTYPE)


# ─────────────────────────────────────────────────────────────────────────────
# G1 — Single-image recognition on unseen images
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("G1 — Single-image recognition on unseen images")
print(f"     Prompt: canonical  |  N = {total_unseen} images")
print("=" * 70)

g1_results = []
for cls in CLASSES:
    for i, img in enumerate(unseen[cls]):
        raw  = ask(model, proc, [img], CANONICAL_PROMPT, args.device, args.max_new_tokens)
        pred = parse_label(raw)
        ok   = pred == cls
        tag  = "✓" if ok else "✗"
        name = f"{cls}_{i+1:02d}_unseen"
        print(f"  [{tag}] {name:<25}  gt={cls:8s}  pred={str(pred):8s}  raw='{raw}'")
        g1_results.append({"name": name, "gt": cls, "pred": pred, "raw": raw, "ok": ok})

g1_correct = sum(r["ok"] for r in g1_results)
g1_acc     = g1_correct / len(g1_results)
per_class_g1 = {}
for cls in CLASSES:
    rows = [r for r in g1_results if r["gt"] == cls]
    c = sum(r["ok"] for r in rows)
    per_class_g1[cls] = {"correct": c, "total": len(rows), "acc": c/len(rows) if rows else 0.0}

print(f"\n  G1 accuracy: {g1_correct}/{len(g1_results)} = {g1_acc:.1%}")
for cls in CLASSES:
    d = per_class_g1[cls]
    bar = "█" * d["correct"] + "░" * (d["total"] - d["correct"])
    print(f"    {cls:8s}: {d['correct']}/{d['total']} {bar}  {d['acc']:.0%}")

delta_g1 = g1_acc - TRAINING_RESULTS["T1_recognition_acc"]
print(f"\n  vs. training images (100%):  {delta_g1:+.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# G2 — Prompt stability on unseen images (5 prompt variants)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("G2 — Prompt stability on unseen images (5 prompt variants each)")
print("=" * 70)

g2_results = []
for cls in CLASSES:
    for i, img in enumerate(unseen[cls]):
        preds = []
        for prompt in PROMPT_VARIANTS:
            raw  = ask(model, proc, [img], prompt, args.device, args.max_new_tokens)
            pred = parse_label(raw)
            preds.append(pred)
        all_same    = len(set(p for p in preds if p is not None)) == 1
        all_correct = all(p == cls for p in preds)
        stable_correct = all_same and preds[0] == cls
        status = "✓stable" if stable_correct else ("~varies" if not all_same else "✗wrong")
        name = f"{cls}_{i+1:02d}_unseen"
        print(f"  [{status}] {name:<25}  gt={cls:8s}  preds={preds}")
        g2_results.append({
            "name": name, "gt": cls, "preds": preds,
            "all_same": all_same, "all_correct": all_correct,
            "stable_correct": stable_correct,
        })

g2_stable   = sum(r["all_same"] for r in g2_results)
g2_all_corr = sum(r["all_correct"] for r in g2_results)
g2_both     = sum(r["stable_correct"] for r in g2_results)
print(f"\n  Stable predictions (same across all 5 prompts): {g2_stable}/{len(g2_results)}")
print(f"  All-5-correct:                                   {g2_all_corr}/{len(g2_results)}")
print(f"  Stable AND correct:                              {g2_both}/{len(g2_results)}")
print(f"  vs. training (11/15 = 73.3%):  {g2_both/len(g2_results):+.1%} raw, "
      f"delta = {g2_both/len(g2_results) - TRAINING_RESULTS['T3_stable_correct_rate']:+.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# G3 — Composite panorama grounding on unseen images
#       (side-by-side, same make_panorama as T5a)
# ─────────────────────────────────────────────────────────────────────────────

# n_rounds = shortest class list so every image is used at least once
n_rounds = min(len(unseen[c]) for c in CLASSES)
print(f"\nImage counts for scene tests: " +
      ", ".join(f"{c}={len(unseen[c])}" for c in CLASSES) +
      f"  →  {n_rounds} rounds (limited by obama)")

print("\n" + "=" * 70)
print("G3 — Composite panorama grounding on unseen images")
print(f"     {n_rounds} image-rounds × 6 permutations × 3 targets = {n_rounds*6*3} queries")
print(f"     Each round uses a different face image per identity")
print("=" * 70)

arrangements = list(itertools.permutations(CLASSES))
g3_results = []

# cycle through every available unseen image per identity
for round_i in range(n_rounds):
    imgs = {cls: unseen[cls][round_i] for cls in CLASSES}
    for arrangement in arrangements:
        left, mid, right = arrangement
        scene = make_panorama([imgs[left], imgs[mid], imgs[right]])
        for target in CLASSES:
            expected_pos = ["left", "center", "right"][arrangement.index(target)]
            q = (
                f"This image shows three celebrity portraits placed side by side. "
                f"The left third contains one person, the middle third another, the right third another. "
                f"Where is {FULL_NAMES[target]}? Answer with exactly one word: left, center, or right."
            )
            raw  = ask(model, proc, [scene], q, args.device, 16)
            pred = parse_position(raw)
            ok   = (pred == expected_pos) or (expected_pos == "center" and pred in {"center", "middle"})
            tag  = "✓" if ok else "✗"
            print(f"  [{tag}] r{round_i} L={left[:5]} M={mid[:5]} R={right[:5]} | "
                  f"target={FULL_NAMES[target]:<15s}  exp={expected_pos:6s}  got='{raw}'")
            g3_results.append({
                "round": round_i, "arrangement": list(arrangement), "target": target,
                "expected": expected_pos, "raw": raw, "pred": pred, "ok": ok,
            })

g3_acc = sum(r["ok"] for r in g3_results) / len(g3_results)
print(f"\n  G3 accuracy: {sum(r['ok'] for r in g3_results)}/{len(g3_results)} = {g3_acc:.1%}")
print(f"  vs. training T5a (94.4%):  {g3_acc - TRAINING_RESULTS['T5a_composite_acc']:+.1%}")

# per-target breakdown
print("  Per-target:")
for target in CLASSES:
    rows = [r for r in g3_results if r["target"] == target]
    c = sum(r["ok"] for r in rows)
    print(f"    {FULL_NAMES[target]:<15s}: {c}/{len(rows)}")


# ─────────────────────────────────────────────────────────────────────────────
# G4 — Robot-view scene grounding on unseen images  (v6 composition)
#       5 real backgrounds × 6 permutations × 3 targets × 2 prompts = 180 queries
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("G4 — Robot-view scene grounding on unseen images  [v6 composition]")
print("     Perspective-warped faces into real robot backgrounds (composite_test6.py)")
print(f"     {n_rounds} image-rounds × {len(available_scenes)} backgrounds × 6 permutations × 3 targets × 2 prompts")
print(f"     = {n_rounds * len(available_scenes) * 6 * 3 * 2} total queries")
print("=" * 70)

# SP0 — mirrors exact robot task language from eval3_record.sh
# SP1 — generic scene description (same wording as G3/T5a for direct comparison)
SCENE_PROMPTS = [
    (
        "SP0-task",
        "{name} is one of the three celebrities shown on cards in the robot scene. "
        "The cards are placed at the LEFT, CENTER, and RIGHT positions. "
        "Which position is {name} in? Answer with exactly one word: left, center, or right."
    ),
    (
        "SP1-scene",
        "This image shows three celebrity portraits on cards arranged left, center, and right. "
        "Where is {name}? Answer with exactly one word: left, center, or right."
    ),
]

g4_results = []

if not available_scenes:
    print("  [G4 skipped — no scene backgrounds available]")
else:
    # outer loop: image rounds (one different face image per identity per round)
    for round_i in range(n_rounds):
        imgs = {cls: unseen[cls][round_i] for cls in CLASSES}
        for scene_fname in available_scenes:
            print(f"\n  round={round_i}  bg={scene_fname}")
            for arrangement in arrangements:
                left, mid, right = arrangement
                try:
                    scene_img = make_v6_scene(
                        scene_fname, scene_dir,
                        {"left": imgs[left], "center": imgs[mid], "right": imgs[right]},
                    )
                except Exception as e:
                    print(f"    [SKIP] v6 render failed: {e}")
                    continue

                for sp_name, prompt_template in SCENE_PROMPTS:
                    for target in CLASSES:
                        expected_pos = ["left", "center", "right"][arrangement.index(target)]
                        q   = prompt_template.format(name=FULL_NAMES[target])
                        raw = ask(model, proc, [scene_img], q, args.device, 16)
                        pred = parse_position(raw)
                        ok   = (pred == expected_pos) or (
                            expected_pos == "center" and pred in {"center", "middle"}
                        )
                        tag  = "✓" if ok else "✗"
                        print(f"    [{tag}] r{round_i} {sp_name} "
                              f"L={left[:5]} M={mid[:5]} R={right[:5]} | "
                              f"target={FULL_NAMES[target]:<15s}  exp={expected_pos:6s}  got='{raw}'")
                        g4_results.append({
                            "round": round_i, "scene": scene_fname, "prompt_style": sp_name,
                            "arrangement": list(arrangement), "target": target,
                            "expected": expected_pos, "raw": raw, "pred": pred, "ok": ok,
                        })

if g4_results:
    g4_acc     = sum(r["ok"] for r in g4_results) / len(g4_results)
    g4_sp0     = [r for r in g4_results if r["prompt_style"] == "SP0-task"]
    g4_sp1     = [r for r in g4_results if r["prompt_style"] == "SP1-scene"]
    g4_acc_sp0 = sum(r["ok"] for r in g4_sp0) / len(g4_sp0) if g4_sp0 else 0.0
    g4_acc_sp1 = sum(r["ok"] for r in g4_sp1) / len(g4_sp1) if g4_sp1 else 0.0

    print(f"\n  G4 overall accuracy:              {sum(r['ok'] for r in g4_results)}/{len(g4_results)} = {g4_acc:.1%}")
    print(f"  G4 robot-task prompt (SP0-task):  {sum(r['ok'] for r in g4_sp0)}/{len(g4_sp0)} = {g4_acc_sp0:.1%}")
    print(f"  G4 scene prompt      (SP1-scene): {sum(r['ok'] for r in g4_sp1)}/{len(g4_sp1)} = {g4_acc_sp1:.1%}")

    print("  Per-target (all prompts):")
    for target in CLASSES:
        rows = [r for r in g4_results if r["target"] == target]
        c    = sum(r["ok"] for r in rows)
        print(f"    {FULL_NAMES[target]:<15s}: {c}/{len(rows)} = {c/len(rows):.0%}" if rows else f"    {target}: 0 rows")

    print("  Per-expected-position (G4):")
    for pos in ["left", "center", "right"]:
        rows = [r for r in g4_results if r["expected"] == pos]
        c    = sum(r["ok"] for r in rows)
        print(f"    {pos:6s}: {c}/{len(rows)} = {c/len(rows):.0%}" if rows else f"    {pos}: 0 rows")
else:
    g4_acc = g4_acc_sp0 = g4_acc_sp1 = None
    print("  [G4 skipped]")


# ─────────────────────────────────────────────────────────────────────────────
# Full comparison table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON: training images  vs.  unseen images")
print("=" * 70)

g2_rate = g2_both / len(g2_results)
rows = [
    ("G1 Recognition (canonical)",
     f"{TRAINING_RESULTS['T1_recognition_acc']:.1%}",
     f"{g1_acc:.1%}",
     f"{g1_acc - TRAINING_RESULTS['T1_recognition_acc']:+.1%}"),
    ("G2 Stable-and-correct (5 prompts)",
     f"{TRAINING_RESULTS['T3_stable_correct_rate']:.1%}",
     f"{g2_rate:.1%}",
     f"{g2_rate - TRAINING_RESULTS['T3_stable_correct_rate']:+.1%}"),
    ("G3 Panorama grounding",
     f"{TRAINING_RESULTS['T5a_composite_acc']:.1%}",
     f"{g3_acc:.1%}",
     f"{g3_acc - TRAINING_RESULTS['T5a_composite_acc']:+.1%}"),
    ("G4 v6 robot-view (SP0-task)",
     "N/A (new test)",
     f"{g4_acc_sp0:.1%}" if g4_acc_sp0 is not None else "skipped",
     "—"),
    ("G4 v6 robot-view (SP1-scene)",
     "N/A (new test)",
     f"{g4_acc_sp1:.1%}" if g4_acc_sp1 is not None else "skipped",
     "—"),
]

hdr = f"  {'Metric':<38}  {'Train':>8}  {'Unseen':>8}  {'Delta':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for name, tr, un, delta in rows:
    print(f"  {name:<38}  {tr:>8}  {un:>8}  {delta:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Failure analysis
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FAILURE ANALYSIS")
print("=" * 70)

g1_fail = [r for r in g1_results if not r["ok"]]
if g1_fail:
    print(f"\n  G1 failures ({len(g1_fail)}):")
    for r in g1_fail:
        print(f"    {r['name']:<25}  gt={r['gt']:8s}  pred={str(r['pred']):8s}  raw='{r['raw']}'")
    # which identity is confused with what
    conf = defaultdict(lambda: defaultdict(int))
    for r in g1_fail:
        conf[r["gt"]][r["pred"] if r["pred"] else "none"] += 1
    print(f"  Confusion in failures:")
    for gt_cls, preds in conf.items():
        print(f"    {gt_cls}: {dict(preds)}")
else:
    print("\n  G1: no failures")

g3_fail = [r for r in g3_results if not r["ok"]]
if g3_fail:
    print(f"\n  G3 grounding failures ({len(g3_fail)}):")
    for r in g3_fail:
        print(f"    arr={r['arrangement']}  target={r['target']:8s}  "
              f"exp={r['expected']:6s}  got='{r['raw']}'")
else:
    print("\n  G3: no failures")

g4_fail = [r for r in g4_results if not r["ok"]]
if not g4_results:
    print("\n  G4: skipped")
elif g4_fail:
    print(f"\n  G4 robot-view failures ({len(g4_fail)}/{len(g4_results)}):")
    for r in g4_fail[:20]:
        print(f"    scene={Path(r['scene']).stem[-20:]}  {r['prompt_style']}  "
              f"arr={[c[:5] for c in r['arrangement']]}  "
              f"target={r['target']:8s}  exp={r['expected']:6s}  got='{r['raw']}'")
    if len(g4_fail) > 20:
        print(f"    ... and {len(g4_fail) - 20} more")
    fail_pos = defaultdict(int)
    fail_tgt = defaultdict(int)
    fail_sp  = defaultdict(int)
    for r in g4_fail:
        fail_pos[r["expected"]] += 1
        fail_tgt[r["target"]]   += 1
        fail_sp[r["prompt_style"]] += 1
    print(f"  Failures by expected position: {dict(fail_pos)}")
    print(f"  Failures by target identity:   {dict(fail_tgt)}")
    print(f"  Failures by prompt style:      {dict(fail_sp)}")
else:
    print("\n  G4: no failures")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis & Conclusion
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DIAGNOSIS & CONCLUSION")
print("=" * 70)

issues = []

if g1_acc >= 0.90:
    print(f"\n  ✅ G1 ({g1_acc:.1%}): Single-image recognition generalises well.")
elif g1_acc >= 0.70:
    print(f"\n  ⚠️  G1 ({g1_acc:.1%}): Partial generalisation — some identities fail.")
    worst = min(CLASSES, key=lambda c: per_class_g1[c]["acc"])
    print(f"       Worst identity: {worst} ({per_class_g1[worst]['acc']:.0%})")
    issues.append(f"G1 partial: {worst} recognition drops to {per_class_g1[worst]['acc']:.0%} on unseen images")
else:
    print(f"\n  ❌ G1 ({g1_acc:.1%}): Recognition fails on unseen images — backbone does not generalise.")
    issues.append(f"G1 fail: {g1_acc:.1%} — backbone memorised training images")

if g2_rate >= 0.70:
    print(f"  ✅ G2 ({g2_rate:.1%}): Prompt-stable predictions on unseen images.")
else:
    print(f"  ⚠️  G2 ({g2_rate:.1%}): Prompt sensitivity on unseen images — "
          f"predictions vary with wording.")
    issues.append(f"G2: prompt instability on unseen ({g2_rate:.1%})")

if g3_acc >= 0.80:
    print(f"  ✅ G3 ({g3_acc:.1%}): Panorama grounding generalises well.")
elif g3_acc >= 0.55:
    print(f"  ⚠️  G3 ({g3_acc:.1%}): Partial grounding — some position errors.")
    issues.append(f"G3 partial: {g3_acc:.1%}")
else:
    print(f"  ❌ G3 ({g3_acc:.1%}): Grounding fails on unseen-image panoramas.")
    issues.append(f"G3 fail: {g3_acc:.1%}")

if g4_acc_sp0 is None:
    print("  — G4 skipped (no scene backgrounds found)")
elif g4_acc_sp0 >= 0.80:
    print(f"  ✅ G4 robot-task prompt SP0 ({g4_acc_sp0:.1%}): Robot-view v6 grounding works on unseen images.")
elif g4_acc_sp0 >= 0.55:
    print(f"  ⚠️  G4 robot-task prompt SP0 ({g4_acc_sp0:.1%}): Partial robot-view grounding.")
    issues.append(f"G4 partial: {g4_acc_sp0:.1%} on SP0-task prompt")
else:
    print(f"  ❌ G4 robot-task prompt SP0 ({g4_acc_sp0:.1%}): Robot-view grounding fails on unseen images.")
    issues.append(f"G4 fail: {g4_acc_sp0:.1%} — further Phase 2 scene training needed")

# Overall verdict
print()
if not issues:
    print("  ━━ OVERALL: BACKBONE GENERALISES ━━")
    print("  The model correctly recognises and grounds all three identities on")
    print("  images it has never seen. You can proceed to the robot deployment test.")
else:
    print("  ━━ OVERALL: PARTIAL GENERALISATION — further training needed ━━")
    print("  Issues:")
    for iss in issues:
        print(f"    • {iss}")
    print()
    print("  Recommended next steps:")
    if any("G1" in i for i in issues):
        print("    1. Fine-tune Phase 1 with more diverse training images per identity.")
        print("       Add top-50 internet images (scripts/train_eval3_phase1_lora.py --run top50).")
    if any("G4" in i or "G3" in i for i in issues):
        print("    2. Run Phase 2 scene training with unseen-image composites as training data.")
        print("       See scripts/train_eval3_phase2.py and train_eval3_phase2_scene.py.")
    if any("G2" in i for i in issues):
        print("    3. Add prompt augmentation to Phase 1 training data.")


# ── Save results ──────────────────────────────────────────────────────────────
output = {
    "model":   args.model,
    "dataset": args.dataset,
    "n_unseen_per_class": args.n_per_class,
    "training_reference": TRAINING_RESULTS,
    "G1_recognition":     {"acc": g1_acc, "per_class": per_class_g1, "results": g1_results},
    "G2_stability":       {"stable_correct": g2_both, "total": len(g2_results),
                           "rate": g2_both/len(g2_results), "results": g2_results},
    "G3_panorama":        {"acc": g3_acc, "results": g3_results},
    "G4_robot_view_v6":   {"acc_overall": g4_acc, "acc_sp0_task": g4_acc_sp0,
                           "acc_sp1_scene": g4_acc_sp1, "results": g4_results,
                           "composition": "v6 perspective-warp (composite_test6.py)"},
    "issues": issues,
    "verdict": "GENERALISES" if not issues else "PARTIAL",
}
Path(args.output).write_text(json.dumps(output, indent=2, default=str))
print(f"\nFull results saved → {args.output}")
