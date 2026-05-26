#!/usr/bin/env python3
"""
eval3_celeb_diag.py
====================
Diagnostic evaluation of celebrity recognition on unseen images.

Tests two models:
  - robot-learning-group47/eval3_phase1_celeb
  - robot-learning-group47/eval3_phase2_celeb

Dataset:
  - robot-learning-group47/celebrities_faces_all_different
  - 1920 images, 100 celebrities (~10–26 images each)

For each image: ask "Who is the person in this image? Give the full name."
Parse answer: check if the true name (or last name) appears in the response.

Reports:
  [1]  Overall accuracy per model
  [2]  Per-celebrity accuracy (sorted worst-first)
  [3]  Eval-3 target accuracy: Taylor Swift / Barack Obama / Yann LeCun
  [4]  Top wrong predictions (what does the model say when it fails?)
  [5]  Confusion pairs (true → predicted name)
  [6]  Accuracy by count-band (few-shot vs many-shot celebrities)
  [7]  Head-to-head: phase1 vs phase2 per celebrity
"""

import json, os, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODELS      = {
    "phase1": "robot-learning-group47/eval3_phase1_celeb",
    "phase2": "robot-learning-group47/eval3_phase2_celeb",
}
DATASET_ID  = "robot-learning-group47/celebrities_faces_all_different"
OUTPUT      = Path("./eval3_celeb_diag_results.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
print(f"Device: {DEVICE}")

EVAL3_TARGETS = {"Taylor Swift", "Barack Obama", "Yann LeCun"}

QUESTION = "Who is the person in this image? Give the full name."

# ── Load dataset ──────────────────────────────────────────────────────────────
print("\nLoading dataset…")
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
ds = load_dataset(DATASET_ID, split="train", token=hf_token)
print(f"  {len(ds)} images, columns: {ds.column_names}")

label_counts = Counter(r["label"] for r in ds)
print(f"  {len(label_counts)} unique celebrities")
print(f"  Eval-3 targets: " +
      ", ".join(f"{n}={label_counts[n]}" for n in EVAL3_TARGETS))

# Pre-compute last names for fuzzy matching
def last_name(full): return full.strip().split()[-1].lower()
LAST_NAMES = {n: last_name(n) for n in label_counts}

def parse_match(raw_output, true_label):
    """Return True if the model output contains the true celebrity name."""
    raw = raw_output.lower().strip()
    # Full name match (case-insensitive)
    if true_label.lower() in raw:
        return True
    # Last name match (only if last name is distinctive enough — len > 3)
    ln = LAST_NAMES[true_label]
    if len(ln) > 3 and ln in raw:
        return True
    return False

def extract_predicted_name(raw_output, all_labels):
    """Try to find which celebrity the model actually named."""
    raw = raw_output.lower()
    # full name first
    for label in all_labels:
        if label.lower() in raw:
            return label
    # last name fallback
    for label in all_labels:
        ln = LAST_NAMES[label]
        if len(ln) > 3 and ln in raw:
            return label
    return "__unmatched__"

all_labels = list(label_counts.keys())

# ── Inference helper ──────────────────────────────────────────────────────────
def ask(model, processor, image):
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": QUESTION}
    ]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    return processor.decode(
        ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ── Run each model ────────────────────────────────────────────────────────────
all_results = {}

for model_key, model_id in MODELS.items():
    print(f"\n{'='*70}")
    print(f"Model: {model_key}  ({model_id})")
    print(f"{'='*70}")

    # Load
    print("  Loading…")
    processor = AutoProcessor.from_pretrained(BASE_MODEL, token=hf_token)
    base = SmolVLMForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=DTYPE, device_map=DEVICE, token=hf_token)
    model = PeftModel.from_pretrained(base, model_id, token=hf_token)
    model.eval()
    print("  Model ready.")

    records = []
    n_correct = 0

    for i, row in enumerate(ds):
        img   = row["image"].convert("RGB")
        label = row["label"]
        raw   = ask(model, processor, img)
        ok    = parse_match(raw, label)
        pred  = extract_predicted_name(raw, all_labels)
        n_correct += ok
        records.append({
            "idx": i, "label": label, "raw": raw,
            "pred": pred, "ok": ok,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(ds)}  acc={n_correct/(i+1):.1%}", flush=True)

    all_results[model_key] = records
    overall = n_correct / len(records)
    print(f"\n  OVERALL: {n_correct}/{len(records)} = {overall:.1%}")

    # Free GPU
    del model, base
    torch.cuda.empty_cache()

# ── Reporting ─────────────────────────────────────────────────────────────────
SEP = "=" * 70
sep = "-" * 70

print(f"\n\n{SEP}")
print("CELEBRITY RECOGNITION DIAGNOSTIC")
print(SEP)

# [1] Overall
print("\n[1] Overall accuracy")
print(sep)
for mk, records in all_results.items():
    c = sum(r["ok"] for r in records)
    n = len(records)
    print(f"  {mk}: {c}/{n} = {c/n:.1%}")

# [2] Per-celebrity (worst-first for each model)
for mk, records in all_results.items():
    print(f"\n[2] Per-celebrity accuracy — {mk}  (worst first)")
    print(sep)
    by_celeb = defaultdict(list)
    for r in records:
        by_celeb[r["label"]].append(r["ok"])
    accs = {name: (sum(v), len(v)) for name, v in by_celeb.items()}
    sorted_celebs = sorted(accs.items(), key=lambda x: x[1][0]/x[1][1])
    for name, (c, n) in sorted_celebs:
        bar = "█" * c + "░" * (n - c)
        star = " ★" if name in EVAL3_TARGETS else ""
        print(f"  {name:<30}{c:3d}/{n:<3d} = {c/n:.0%}  {bar}{star}")

# [3] Eval-3 targets detail
print("\n[3] Eval-3 target celebrities — both models")
print(sep)
for name in sorted(EVAL3_TARGETS):
    print(f"\n  {name}:")
    for mk, records in all_results.items():
        rows = [r for r in records if r["label"] == name]
        c = sum(r["ok"] for r in rows)
        n = len(rows)
        wrong = [r["raw"][:60] for r in rows if not r["ok"]]
        print(f"    {mk}: {c}/{n} = {c/n:.0%}")
        for w in wrong[:4]:
            print(f"      → '{w}'")

# [4] Top wrong predictions
print("\n[4] Top wrong prediction patterns — both models")
print(sep)
for mk, records in all_results.items():
    wrong_records = [r for r in records if not r["ok"]]
    print(f"\n  {mk}: {len(wrong_records)} failures")
    # raw outputs truncated
    raw_counts = Counter(r["raw"][:60] for r in wrong_records)
    for raw, cnt in raw_counts.most_common(15):
        print(f"    {cnt:3d}×  '{raw}'")

# [5] Confusion pairs (true → pred)
print("\n[5] Confusion pairs (true label → predicted label) — phase2")
print(sep)
for mk, records in all_results.items():
    wrong = [r for r in records if not r["ok"] and r["pred"] != "__unmatched__"]
    pair_counts = Counter((r["label"], r["pred"]) for r in wrong)
    print(f"\n  {mk}: top confusions")
    for (true_n, pred_n), cnt in pair_counts.most_common(20):
        print(f"    {cnt:3d}×  {true_n:<28} → {pred_n}")

# [6] Accuracy by image count per celebrity
print("\n[6] Accuracy by celebrity image count")
print(sep)
for mk, records in all_results.items():
    by_celeb = defaultdict(list)
    for r in records:
        by_celeb[r["label"]].append(r["ok"])
    bands = {"few (<=10)": [], "mid (11-15)": [], "many (16+)": []}
    for name, vals in by_celeb.items():
        n = len(vals)
        acc = sum(vals) / n
        if n <= 10:
            bands["few (<=10)"].append(acc)
        elif n <= 15:
            bands["mid (11-15)"].append(acc)
        else:
            bands["many (16+)"].append(acc)
    print(f"\n  {mk}:")
    for band, accs in bands.items():
        if accs:
            avg = sum(accs) / len(accs)
            print(f"    {band}: avg_acc={avg:.1%} over {len(accs)} celebrities")

# [7] Head-to-head per celebrity
print("\n[7] Head-to-head: phase1 vs phase2 per celebrity")
print(sep)
by_celeb_p1 = defaultdict(list)
by_celeb_p2 = defaultdict(list)
for r in all_results.get("phase1", []):
    by_celeb_p1[r["label"]].append(r["ok"])
for r in all_results.get("phase2", []):
    by_celeb_p2[r["label"]].append(r["ok"])

diffs = []
for name in all_labels:
    if name in by_celeb_p1 and name in by_celeb_p2:
        a1 = sum(by_celeb_p1[name]) / len(by_celeb_p1[name])
        a2 = sum(by_celeb_p2[name]) / len(by_celeb_p2[name])
        diffs.append((name, a1, a2, a2 - a1))

# Phase2 better
improved = sorted([d for d in diffs if d[3] > 0], key=lambda x: -x[3])
regressed = sorted([d for d in diffs if d[3] < 0], key=lambda x: x[3])

print(f"\n  Phase2 better than phase1 ({len(improved)} celebrities):")
for name, a1, a2, delta in improved[:20]:
    star = " ★" if name in EVAL3_TARGETS else ""
    print(f"    {name:<30} p1={a1:.0%}  p2={a2:.0%}  Δ={delta:+.0%}{star}")

print(f"\n  Phase2 worse than phase1 ({len(regressed)} celebrities):")
for name, a1, a2, delta in regressed[:20]:
    star = " ★" if name in EVAL3_TARGETS else ""
    print(f"    {name:<30} p1={a1:.0%}  p2={a2:.0%}  Δ={delta:+.0%}{star}")

# Save
OUTPUT.write_text(json.dumps({
    "config": {"base": BASE_MODEL, "models": MODELS, "dataset": DATASET_ID,
               "n_images": len(ds), "n_celebs": len(label_counts)},
    "results": {mk: records for mk, records in all_results.items()},
}, indent=2, default=str))
print(f"\nSaved -> {OUTPUT}")
