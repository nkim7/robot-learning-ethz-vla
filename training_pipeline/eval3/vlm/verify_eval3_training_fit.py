#!/usr/bin/env python3
"""
verify_eval3_training_fit.py
============================
Five-test verification suite for robot-learning-group47/eval3_lora_fixed_Chengming.

Tests:
  T1  Single-image recognition on all 15 original training images
  T2  Three-way identity confusion matrix
  T3  Prompt-stability across 5 prompt variants for each image
  T4  Training-set vs. held-out comparison (baseline SmolVLM2 as control)
  T5  VLM grounding isolation — scene-level position query (composite + multi-image)

Run on GPU server:
    python verify_eval3_training_fit.py

Options:
    --model        HF repo for the LoRA model   [robot-learning-group47/eval3_lora_fixed_Chengming]
    --base_model   HF repo for the base model   [HuggingFaceTB/SmolVLM2-2.2B-Instruct]
    --image_dir    directory of TOY images       [./clelb_pic]
    --output       result JSON path              [./eval3_verification_results.json]
    --skip_base    skip baseline (saves ~4 GB)
    --skip_scene   skip scene-grounding tests
    --device       cuda / mps / cpu             [auto]
"""

import argparse
import itertools
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",       default="robot-learning-group47/eval3_lora_fixed_Chengming")
parser.add_argument("--base_model",  default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--image_dir",   default="./clelb_pic")
parser.add_argument("--output",      default="./eval3_verification_results.json")
parser.add_argument("--skip_base",   action="store_true", help="Skip baseline model (saves time/memory)")
parser.add_argument("--skip_scene",  action="store_true", help="Skip scene-grounding tests T5")
parser.add_argument("--device",      default=None)
parser.add_argument("--max_new_tokens", type=int, default=32)
args = parser.parse_args()

if args.device is None:
    args.device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
DTYPE = torch.bfloat16 if args.device == "cuda" else torch.float32

# ── Label helpers ─────────────────────────────────────────────────────────────
CLASSES = ["taylor", "obama", "yann"]
FULL_NAMES = {
    "taylor": "Taylor Swift",
    "obama":  "Barack Obama",
    "yann":   "Yann LeCun",
}
ALIASES = {
    "taylor":          "taylor",
    "taylor swift":    "taylor",
    "swift":           "taylor",
    "ms. swift":       "taylor",
    "obama":           "obama",
    "barack obama":    "obama",
    "barack":          "obama",
    "president obama": "obama",
    "mr. obama":       "obama",
    "yann":            "yann",
    "yann lecun":      "yann",
    "yann le cun":     "yann",
    "lecun":           "yann",
    "le cun":          "yann",
}

def parse_label(raw: str) -> str | None:
    text = raw.strip().lower()
    if text in ALIASES:
        return ALIASES[text]
    for alias, label in ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text):
            return label
    return None

# ── Five prompt variants per image (T3: prompt stability) ─────────────────────
PROMPT_VARIANTS = [
    # P0 — canonical (matches training format)
    "Who is the person in this image? Answer with exactly one word from: taylor, obama, yann.",
    # P1 — free-form name
    "Who is this person? Give the full name.",
    # P2 — yes/no style (indirect)
    "Which of the following celebrities is shown? Taylor Swift, Barack Obama, or Yann LeCun?",
    # P3 — context-free minimal
    "Name the person in the photo.",
    # P4 — robot task phrasing (matches eval3_record.sh)
    "Which celebrity does this image show — Taylor Swift, Barack Obama, or Yann LeCun? Answer with the full name.",
]

POSITION_WORDS = {"left", "center", "middle", "right"}

# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(model_id: str, device: str, dtype):
    from peft import PeftModel
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
    print(f"\n[load] Processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    # Try to load as a PEFT / LoRA model first; fall back to plain HF model
    print(f"[load] Model: {model_id}  (may download several GB) …")
    try:
        # If the repo contains adapter_config.json it's a PEFT adapter
        base = SmolVLMForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=dtype, device_map=device, token=hf_token
        )
        model = PeftModel.from_pretrained(base, model_id, token=hf_token)
        print("[load] Loaded as PEFT/LoRA adapter on top of base model.")
    except Exception as peft_err:
        print(f"[load] PEFT load failed ({peft_err}); trying plain HF load …")
        model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device, token=hf_token
        )
        print("[load] Loaded as plain HF model.")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[load] Ready. Parameters: {n_params:.2f}B  |  device: {device}")
    return model, processor


def load_base_model(base_id: str, device: str, dtype):
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
    print(f"\n[load-base] Loading baseline: {base_id}")
    processor = AutoProcessor.from_pretrained(base_id, token=hf_token)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        base_id, torch_dtype=dtype, device_map=device, token=hf_token
    )
    model.eval()
    print(f"[load-base] Baseline ready.")
    return model, processor


# ── Inference helper ──────────────────────────────────────────────────────────
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
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    new_ids = ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_ids, skip_special_tokens=True).strip()


# ── Image catalogue ───────────────────────────────────────────────────────────
IMAGE_DIR = Path(args.image_dir)
if not IMAGE_DIR.exists():
    sys.exit(f"ERROR: image_dir {IMAGE_DIR!r} does not exist. "
             f"Pass --image_dir pointing to the clelb_pic folder.")

SAMPLES: list[tuple[Path, str]] = []
for label in CLASSES:
    paths = sorted(IMAGE_DIR.glob(f"{label}_*.png"))
    if not paths:
        # also try without underscore (e.g. taylor01.png)
        paths = sorted(IMAGE_DIR.glob(f"{label}*.png"))
    for p in paths:
        SAMPLES.append((p, label))

if not SAMPLES:
    sys.exit(f"ERROR: No images found under {IMAGE_DIR}. "
             "Expected files like taylor_01.png, obama_02.png, yann_05.png.")

print(f"\nFound {len(SAMPLES)} images across {len(set(s[1] for s in SAMPLES))} classes:")
for label in CLASSES:
    cnt = sum(1 for _, l in SAMPLES if l == label)
    print(f"  {label:8s}: {cnt} images")

# ── Load fine-tuned model ──────────────────────────────────────────────────────
model_ft, proc_ft = load_model(args.model, args.device, DTYPE)

# ── T1: Single-image recognition on all 15 training images ───────────────────
print("\n" + "=" * 70)
print("TEST 1 — Single-image recognition on all 15 training images")
print("         Prompt: canonical (matches training format)")
print("=" * 70)

t1_results = []
CANONICAL_PROMPT = PROMPT_VARIANTS[0]

for img_path, gt in SAMPLES:
    img  = Image.open(img_path).convert("RGB")
    raw  = ask(model_ft, proc_ft, [img], CANONICAL_PROMPT, args.device, args.max_new_tokens)
    pred = parse_label(raw)
    ok   = pred == gt
    tag  = "✓" if ok else "✗"
    print(f"  [{tag}] {img_path.name:<22}  gt={gt:8s}  pred={str(pred):8s}  raw='{raw}'")
    t1_results.append({"file": img_path.name, "gt": gt, "pred": pred, "raw": raw, "ok": ok})

t1_correct = sum(r["ok"] for r in t1_results)
t1_acc     = t1_correct / len(t1_results)
print(f"\n  T1 accuracy: {t1_correct}/{len(t1_results)} = {t1_acc:.1%}")
per_class_t1 = {}
for label in CLASSES:
    rows = [r for r in t1_results if r["gt"] == label]
    c    = sum(r["ok"] for r in rows)
    per_class_t1[label] = {"correct": c, "total": len(rows), "acc": c / len(rows) if rows else 0.0}
    print(f"    {label:8s}: {c}/{len(rows)} = {per_class_t1[label]['acc']:.0%}")

# ── T2: Three-way confusion matrix ────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST 2 — Three-way confusion matrix  (which class does the model")
print("         confuse each identity with?)")
print("=" * 70)

confusion = defaultdict(lambda: defaultdict(int))
for r in t1_results:
    pred_key = r["pred"] if r["pred"] else "none"
    confusion[r["gt"]][pred_key] += 1

_header_label = "gt / pred"
header = f"  {_header_label:12s}" + "".join(f" {c:>8s}" for c in CLASSES) + f" {'none':>8s}"
print(header)
print("  " + "-" * (12 + 9 * (len(CLASSES) + 1)))
for gt_class in CLASSES:
    row = f"  {gt_class:12s}"
    for pred_class in CLASSES:
        row += f" {confusion[gt_class][pred_class]:>8d}"
    row += f" {confusion[gt_class]['none']:>8d}"
    print(row)

# Check if model is collapsing to one class
pred_counts = defaultdict(int)
for r in t1_results:
    pred_counts[r["pred"] if r["pred"] else "none"] += 1
print(f"\n  Predicted-label distribution: {dict(pred_counts)}")
dominant = max(pred_counts, key=pred_counts.get)
if pred_counts[dominant] > 0.6 * len(t1_results):
    print(f"  ⚠  Model may be collapsing → '{dominant}' ({pred_counts[dominant]}/{len(t1_results)} predictions)")

# ── T3: Prompt stability (5 prompts × each image) ────────────────────────────
print("\n" + "=" * 70)
print("TEST 3 — Prompt stability (5 prompt variants per image)")
print("=" * 70)

t3_results = []
for img_path, gt in SAMPLES:
    img    = Image.open(img_path).convert("RGB")
    preds  = []
    for pi, prompt in enumerate(PROMPT_VARIANTS):
        raw  = ask(model_ft, proc_ft, [img], prompt, args.device, args.max_new_tokens)
        pred = parse_label(raw)
        preds.append(pred)
    all_same = len(set(p for p in preds if p)) == 1
    all_correct = all(p == gt for p in preds)
    stable_correct = all_same and preds[0] == gt
    status = "✓stable" if stable_correct else ("~varies" if not all_same else "✗wrong")
    print(f"  [{status}] {img_path.name:<22}  gt={gt:8s}  preds={preds}")
    t3_results.append({
        "file": img_path.name, "gt": gt, "preds": preds,
        "all_same": all_same, "all_correct": all_correct,
    })

t3_stable   = sum(r["all_same"] for r in t3_results)
t3_all_corr = sum(r["all_correct"] for r in t3_results)
t3_both     = sum(r["all_same"] and r["all_correct"] for r in t3_results)
print(f"\n  Stable predictions (same label all 5 prompts): {t3_stable}/{len(t3_results)}")
print(f"  All-5-correct:                                  {t3_all_corr}/{len(t3_results)}")
print(f"  Stable AND correct:                             {t3_both}/{len(t3_results)}")

# ── T4: Baseline comparison (base SmolVLM2, no LoRA) ─────────────────────────
t4_results = []
if not args.skip_base:
    print("\n" + "=" * 70)
    print("TEST 4 — Baseline comparison: base SmolVLM2 vs. fine-tuned LoRA")
    print(f"         (canonical prompt, same 15 images)")
    print("=" * 70)

    model_base, proc_base = load_base_model(args.base_model, args.device, DTYPE)
    for img_path, gt in SAMPLES:
        img     = Image.open(img_path).convert("RGB")
        raw_ft  = next(r["raw"]  for r in t1_results if r["file"] == img_path.name)
        pred_ft = next(r["pred"] for r in t1_results if r["file"] == img_path.name)
        raw_b   = ask(model_base, proc_base, [img], CANONICAL_PROMPT, args.device, args.max_new_tokens)
        pred_b  = parse_label(raw_b)
        ok_ft   = pred_ft == gt
        ok_base = pred_b == gt
        delta   = ("→gained" if ok_ft and not ok_base else
                   "→lost"   if not ok_ft and ok_base else
                   "same")
        print(f"  {img_path.name:<22}  gt={gt:6s}  base={str(pred_b):8s}  ft={str(pred_ft):8s}  {delta}")
        t4_results.append({
            "file": img_path.name, "gt": gt,
            "base_pred": pred_b, "base_ok": ok_base,
            "ft_pred": pred_ft, "ft_ok": ok_ft,
            "delta": delta,
        })
    base_acc = sum(r["base_ok"] for r in t4_results) / len(t4_results)
    ft_acc   = sum(r["ft_ok"]   for r in t4_results) / len(t4_results)
    gained = sum(1 for r in t4_results if r["delta"] == "→gained")
    lost   = sum(1 for r in t4_results if r["delta"] == "→lost")
    print(f"\n  Base accuracy:        {base_acc:.1%}")
    print(f"  Fine-tuned accuracy:  {ft_acc:.1%}")
    print(f"  Net gain vs. base:    +{gained} / -{lost}")
    if ft_acc < base_acc - 0.05:
        print("  ⚠  Fine-tuning made things WORSE — possible catastrophic forgetting.")
    elif ft_acc > base_acc + 0.10:
        print("  ✓  Fine-tuning shows clear improvement over base model.")
    else:
        print("  ~  Fine-tuning shows marginal improvement — inspect individual failures.")
    # Free base model to save VRAM before scene tests
    del model_base, proc_base
    if args.device == "cuda":
        torch.cuda.empty_cache()
else:
    print("\n[T4 skipped — --skip_base flag set]")

# ── T5: VLM grounding isolation (scene-level) ─────────────────────────────────
t5_composite = []
t5_multi     = []
THUMB = (224, 224)

def make_composite(paths: list) -> Image.Image:
    imgs = [Image.open(p).convert("RGB").resize(THUMB) for p in paths]
    canvas = Image.new("RGB", (THUMB[0] * 3, THUMB[1]))
    for i, im in enumerate(imgs):
        canvas.paste(im, (i * THUMB[0], 0))
    return canvas

def parse_position(raw: str) -> str | None:
    text = raw.strip().lower()
    for w in ["left", "middle", "center", "right"]:
        if w in text:
            return "center" if w == "middle" else w
    return None

if not args.skip_scene:
    print("\n" + "=" * 70)
    print("TEST 5 — VLM grounding: scene-level position query")
    print("         (all 6 arrangements × 3 targets = 18 queries each format)")
    print("=" * 70)

    # Build per-class representative images (use first available)
    class_imgs: dict[str, Path] = {}
    for label in CLASSES:
        paths = sorted(IMAGE_DIR.glob(f"{label}_*.png"))
        if paths:
            class_imgs[label] = paths[0]

    arrangements = list(itertools.permutations(CLASSES))

    # T5a: composite (3-wide image)
    print("\n  T5a — Composite image (3-wide panorama)")
    for arrangement in arrangements:
        left, mid, right = arrangement
        scene = make_composite([class_imgs[c] for c in arrangement])
        for target in CLASSES:
            expected_pos = ["left", "center", "right"][arrangement.index(target)]
            q = (
                f"This image shows three celebrity portraits placed side by side. "
                f"The left third contains one person, the middle third another, the right third another. "
                f"Where is {FULL_NAMES[target]}? Answer with exactly one word: left, center, or right."
            )
            raw  = ask(model_ft, proc_ft, [scene], q, args.device, 16)
            pred = parse_position(raw)
            # Normalize expected
            ok   = (pred == expected_pos) or (expected_pos == "center" and pred in {"center", "middle"})
            tag  = "✓" if ok else "✗"
            print(f"  [{tag}] L={left[:5]} M={mid[:5]} R={right[:5]} | "
                  f"target={FULL_NAMES[target]:<15s}  exp={expected_pos:6s}  got='{raw}'")
            t5_composite.append({
                "arrangement": list(arrangement), "target": target,
                "expected": expected_pos, "raw": raw, "pred": pred, "ok": ok,
            })
    acc5a = sum(r["ok"] for r in t5_composite) / len(t5_composite)
    print(f"\n  T5a accuracy: {sum(r['ok'] for r in t5_composite)}/{len(t5_composite)} = {acc5a:.1%}")

    # T5b: three separate image tokens
    print("\n  T5b — Three separate images (multi-image input)")
    for arrangement in arrangements:
        left, mid, right = arrangement
        imgs = [Image.open(class_imgs[c]).convert("RGB") for c in arrangement]
        for target in CLASSES:
            expected_pos = ["left", "center", "right"][arrangement.index(target)]
            q = (
                f"I am showing you three celebrity photos. "
                f"Photo 1 = LEFT, Photo 2 = CENTER, Photo 3 = RIGHT. "
                f"Where is {FULL_NAMES[target]}? Answer with exactly one word: left, center, or right."
            )
            raw  = ask(model_ft, proc_ft, imgs, q, args.device, 16)
            pred = parse_position(raw)
            ok   = (pred == expected_pos) or (expected_pos == "center" and pred in {"center", "middle"})
            tag  = "✓" if ok else "✗"
            print(f"  [{tag}] L={left[:5]} M={mid[:5]} R={right[:5]} | "
                  f"target={FULL_NAMES[target]:<15s}  exp={expected_pos:6s}  got='{raw}'")
            t5_multi.append({
                "arrangement": list(arrangement), "target": target,
                "expected": expected_pos, "raw": raw, "pred": pred, "ok": ok,
            })
    acc5b = sum(r["ok"] for r in t5_multi) / len(t5_multi)
    print(f"\n  T5b accuracy: {sum(r['ok'] for r in t5_multi)}/{len(t5_multi)} = {acc5b:.1%}")
else:
    print("\n[T5 skipped — --skip_scene flag set]")
    acc5a = acc5b = None

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n  T1  Recognition (canonical prompt, 15 training images)")
for label in CLASSES:
    d = per_class_t1[label]
    bar = "█" * d["correct"] + "░" * (d["total"] - d["correct"])
    print(f"    {label:8s}: {d['correct']}/{d['total']} {bar}  {d['acc']:.0%}")
print(f"    TOTAL    : {t1_correct}/{len(t1_results)} = {t1_acc:.1%}")

print(f"\n  T2  Confusion")
for gt_class in CLASSES:
    main_err = [(c, n) for c, n in confusion[gt_class].items()
                if c != gt_class and c != "none" and n > 0]
    main_err.sort(key=lambda x: -x[1])
    err_str = ", ".join(f"→{c}×{n}" for c, n in main_err) if main_err else "none"
    print(f"    {gt_class:8s}: errors={err_str}")

print(f"\n  T3  Prompt stability")
print(f"    Stable & correct across all 5 prompts: {t3_both}/{len(t3_results)}")
print(f"    Stable but wrong:  "
      f"{sum(r['all_same'] and not r['all_correct'] for r in t3_results)}/{len(t3_results)}")
print(f"    Varies across prompts: "
      f"{sum(not r['all_same'] for r in t3_results)}/{len(t3_results)}")

if t4_results:
    b_acc = sum(r["base_ok"] for r in t4_results) / len(t4_results)
    f_acc = sum(r["ft_ok"]   for r in t4_results) / len(t4_results)
    print(f"\n  T4  LoRA vs. base SmolVLM2")
    print(f"    Base:          {b_acc:.1%}")
    print(f"    Fine-tuned:    {f_acc:.1%}")
    print(f"    Δ (ft - base): {f_acc - b_acc:+.1%}")
else:
    print(f"\n  T4  (skipped)")

if acc5a is not None:
    print(f"\n  T5  Scene grounding (VLM isolation)")
    print(f"    Composite image:       {acc5a:.1%}")
    print(f"    Multi-image input:     {acc5b:.1%}")
else:
    print(f"\n  T5  (skipped)")

# ── Go / no-go decision ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("GO / NO-GO DECISION")
print("=" * 70)

issues = []

if t1_acc < 0.80:
    issues.append(
        f"[FAIL] T1 recognition accuracy {t1_acc:.1%} < 80%. "
        "Model does not reliably identify its own training images. "
        "→ LoRA either under-trained, collapsed, or misapplied."
    )
elif t1_acc < 0.95:
    issues.append(
        f"[WARN] T1 recognition accuracy {t1_acc:.1%}. "
        "Some training images still fail. Inspect per-class breakdown."
    )

stable_rate = t3_both / len(t3_results)
if stable_rate < 0.70:
    issues.append(
        f"[FAIL] T3 stable-and-correct rate {stable_rate:.1%} < 70%. "
        "Predictions depend heavily on prompt wording — backbone is unreliable."
    )

if t4_results:
    delta = f_acc - b_acc
    if delta < -0.05:
        issues.append(
            f"[FAIL] T4 fine-tuning regressed by {delta:.1%}. "
            "LoRA may have caused catastrophic forgetting. "
            "→ Re-check LoRA rank, alpha, training epochs, and whether the merge was clean."
        )

if acc5a is not None and acc5a < 0.50:
    issues.append(
        f"[FAIL] T5 composite grounding {acc5a:.1%} < 50% (chance = 33%). "
        "Model cannot localize celebrities in the 3-portrait scene. "
        "→ Scene-level training data is needed before deploying."
    )
if acc5b is not None and acc5b < 0.50:
    issues.append(
        f"[FAIL] T5 multi-image grounding {acc5b:.1%} < 50%. "
        "Multi-image position reasoning fails even with separated inputs."
    )

# Confusion: check if any class is never predicted
for gt_class in CLASSES:
    if per_class_t1[gt_class]["acc"] == 0:
        issues.append(
            f"[FAIL] {gt_class} is never correctly predicted (0/{per_class_t1[gt_class]['total']}). "
            "Possible class collapse — model ignores one identity entirely."
        )

if not issues:
    print("\n  ✅  BACKBONE PASSES all checks.")
    print("  The model correctly recognises its training images and is stable")
    print("  across prompt variants. You can proceed to test on new TOY images.")
else:
    print("\n  ❌  BACKBONE FAILS — issues found:\n")
    for iss in issues:
        print(f"    • {iss}\n")
    print("  Do NOT proceed to new TOY images until the above are resolved.")
    print("  Suggested next steps:")
    if t1_acc < 0.80:
        print("    1. Check the LoRA adapter was correctly saved and loaded.")
        print("       Run: python verify_eval3_training_fit.py --model <local_checkpoint_path>")
        print("    2. Re-run scripts/train_eval3_phase1_lora.py with more epochs or lower LR.")
    if stable_rate < 0.70:
        print("    3. Try prompt ensembling: take majority vote across prompt variants.")
    if acc5a is not None and acc5a < 0.50:
        print("    4. Run scripts/train_eval3_phase2.py to add scene-grounding training.")

print()

# ── Failure gallery ───────────────────────────────────────────────────────────
failures = [r for r in t1_results if not r["ok"]]
if failures:
    print("FAILURE GALLERY (T1 wrong predictions):")
    for r in failures:
        print(f"  {r['file']:<22}  gt={r['gt']:8s}  pred={str(r['pred']):8s}  raw='{r['raw']}'")
    print()

# ── Save full results ─────────────────────────────────────────────────────────
output = {
    "model":       args.model,
    "base_model":  args.base_model,
    "image_dir":   str(IMAGE_DIR),
    "n_images":    len(SAMPLES),
    "T1_recognition": {
        "accuracy":  t1_acc,
        "per_class": per_class_t1,
        "results":   t1_results,
    },
    "T2_confusion": {label: dict(confusion[label]) for label in CLASSES},
    "T3_stability": {
        "stable_and_correct_rate": t3_both / len(t3_results),
        "results": t3_results,
    },
    "T4_baseline": {
        "base_acc":       sum(r["base_ok"] for r in t4_results) / len(t4_results) if t4_results else None,
        "finetuned_acc":  t1_acc,
        "results":        t4_results,
    } if t4_results else None,
    "T5_grounding": {
        "composite_acc":   acc5a,
        "multi_image_acc": acc5b,
        "composite":       t5_composite,
        "multi_image":     t5_multi,
    } if not args.skip_scene else None,
    "go_no_go": "PASS" if not issues else "FAIL",
    "issues": issues,
}

out_path = Path(args.output)
out_path.write_text(json.dumps(output, indent=2, default=str))
print(f"Full results saved → {out_path}")
