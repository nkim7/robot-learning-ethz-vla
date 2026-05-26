#!/usr/bin/env python3
"""
eval3_phase1_percelebs.py
==========================
Phase1-only per-celebrity accuracy diagnostic.
Runs inference on celebrities_faces_all_different (1920 images, 100 celebs),
saves per-record JSON, and prints worst-celebrity report.

Usage:
  python3 eval3_phase1_percelebs.py [--out ./p1_percelebs.json]
"""

import json, os, sys, argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="./p1_percelebs.json")
args = parser.parse_args()

BASE_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_ID   = "robot-learning-group47/eval3_phase1_celeb"
DATASET_ID = "robot-learning-group47/celebrities_faces_all_different"
QUESTION   = "Who is the person in this image? Give the full name."
EVAL3_TARGETS = {"Taylor Swift", "Barack Obama", "Yann LeCun"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
print(f"Device: {DEVICE}")

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

print("Loading dataset…")
ds = load_dataset(DATASET_ID, split="train", token=hf_token)
print(f"  {len(ds)} images")

label_counts = Counter(r["label"] for r in ds)
all_labels   = list(label_counts.keys())

def last_name(full): return full.strip().split()[-1].lower()
LAST_NAMES = {n: last_name(n) for n in label_counts}

def parse_match(raw, true_label):
    raw = raw.lower().strip()
    if true_label.lower() in raw: return True
    ln = LAST_NAMES[true_label]
    if len(ln) > 3 and ln in raw: return True
    return False

def extract_pred(raw, all_labels):
    raw = raw.lower()
    for label in all_labels:
        if label.lower() in raw: return label
    for label in all_labels:
        ln = LAST_NAMES[label]
        if len(ln) > 3 and ln in raw: return label
    return "__unmatched__"

print("Loading model…")
processor = AutoProcessor.from_pretrained(BASE_MODEL, token=hf_token)
base = SmolVLMForConditionalGeneration.from_pretrained(
    BASE_MODEL, dtype=DTYPE, device_map=DEVICE, token=hf_token)
model = PeftModel.from_pretrained(base, MODEL_ID, token=hf_token)
model.eval()
print("Model ready.")

def ask(img):
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": QUESTION}
    ]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    return processor.decode(
        ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

records = []
n_correct = 0
for i, row in enumerate(ds):
    img   = row["image"].convert("RGB")
    label = row["label"]
    raw   = ask(img)
    ok    = parse_match(raw, label)
    pred  = extract_pred(raw, all_labels)
    n_correct += ok
    records.append({"idx": i, "label": label, "raw": raw, "pred": pred, "ok": ok})
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(ds)}  acc={n_correct/(i+1):.1%}", flush=True)

overall = n_correct / len(records)
print(f"\nOVERALL: {n_correct}/{len(records)} = {overall:.1%}")

# Per-celebrity report
by_celeb = defaultdict(list)
for r in records:
    by_celeb[r["label"]].append(r)

accs = {name: (sum(r["ok"] for r in rows), len(rows))
        for name, rows in by_celeb.items()}
sorted_celebs = sorted(accs.items(), key=lambda x: x[1][0]/x[1][1])

SEP = "=" * 70
sep = "-" * 70

print(f"\n{SEP}")
print("PER-CELEBRITY ACCURACY (worst first)")
print(SEP)
bad = []
for name, (c, n) in sorted_celebs:
    bar  = "█" * c + "░" * (n - c)
    star = " ★" if name in EVAL3_TARGETS else ""
    acc  = c / n
    flag = " ← WEAK" if acc < 0.80 else (" ← OK" if acc < 0.95 else "")
    print(f"  {name:<32}{c:3d}/{n:<3d} = {acc:.0%}  {bar}{star}{flag}")
    if acc < 0.80:
        bad.append((name, c, n, acc))

print(f"\n{sep}")
print(f"SUMMARY: {overall:.1%} overall | {len(bad)} celebrities below 80%")
for name, c, n, acc in bad:
    wrongs = [r["raw"][:70] for r in by_celeb[name] if not r["ok"]]
    print(f"  {name}: {acc:.0%}")
    for w in wrongs[:4]:
        print(f"    → '{w}'")

print(f"\nEval-3 targets:")
for name in sorted(EVAL3_TARGETS):
    c, n = accs.get(name, (0, 0))
    if n > 0:
        print(f"  {name}: {c}/{n} = {c/n:.0%}")

# Save JSON
Path(args.out).write_text(json.dumps({
    "overall": overall,
    "n_correct": n_correct,
    "n_total": len(records),
    "per_celebrity": {name: {"correct": c, "total": n, "acc": c/n}
                      for name, (c, n) in accs.items()},
    "records": records,
}, indent=2, default=str))
print(f"\nSaved → {args.out}")
