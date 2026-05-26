#!/usr/bin/env python3
"""
Live inference test for the LoRA VQA adapter.
Tests on:
  1. The 6 training images (should reproduce 100%)
  2. OOD contact-sheet frames (real robot rollout images)
  3. bowls.jpg (standalone OOD image)
"""
import re, json, textwrap
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

ADAPTER = Path("models/eval2_fresh_smolvlm_vqa_lora_r32_online_parsefix/best")
BASE    = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SYSTEM  = (
    "You answer questions about a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Use the image to identify the left, middle, and right bowls, then answer with only the target bowl color."
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

print("Loading base model...")
processor = AutoProcessor.from_pretrained(str(ADAPTER))
base = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.float32)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, str(ADAPTER))
model = model.to(DEVICE)
model.eval()
print("Model ready.\n")


def parse_color(text: str) -> str:
    text = text.strip().lower()
    for color in ("blue", "red", "green"):
        if re.search(rf"\b{color}\b", text):
            return color
    return text  # return raw if no match


def ask(image_path: str | Path, question: str) -> tuple[str, str]:
    img = Image.open(image_path).convert("RGB")
    # Match exactly the training format: system prompt + question concatenated in user message
    prompt_text = f"{SYSTEM}\n\nQuestion: {question}"
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[prompt], images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=8, do_sample=False)
    raw = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return parse_color(raw), raw.strip()


def run_section(title: str, cases: list[dict]):
    print(f"{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    correct = 0
    for c in cases:
        pred, raw = ask(c["image"], c["question"])
        ok = pred == c["target"]
        correct += ok
        mark = "OK" if ok else "FAIL"
        img_name = Path(c["image"]).name
        print(f"  [{mark}] {img_name:35s} | Q: {c['question'][:50]:<50s} | target={c['target']} pred={pred!r} raw={raw!r}")
    print(f"  --> {correct}/{len(cases)} = {correct/len(cases)*100:.1f}%\n")
    return correct, len(cases)


# ── 1. Training images (sample, one per layout per question type) ────────────
TRAIN_DIR = Path("permutation_init_position")

# Pull representative questions from val split
with open("data/eval2_vqa_fresh/vqa.jsonl") as f:
    all_rows = [json.loads(l) for l in f]

val_rows = [r for r in all_rows if r.get("split") == "val"]
# Pick first of each (layout, type) combo
seen = set()
train_cases = []
for r in val_rows:
    key = (r["layout"], r["type"])
    if key not in seen:
        seen.add(key)
        train_cases.append({"image": r["image"], "question": r["question"], "target": r["target"]})

t_ok, t_tot = run_section("1. TRAINING IMAGES (val split sample)", train_cases)


# ── 2. OOD: contact-sheet frames (real robot rollout, never seen in training) ─
SHEETS = Path("eval2_180_split_inspection_posebased/contact_sheets")
# Layout encoded in filename: eval2_<layout>_<color>_trimmed.jpg
# We'll ask a simple "which color is the leftmost bowl?" style question
# but actually we don't know exact positions from filename alone.
# Use a set of well-defined questions matched to known layouts.

# Layout definitions (left, middle, right)
LAYOUT_DEFS = {
    "bgr": ("blue",  "green", "red"),
    "brg": ("blue",  "red",   "green"),
    "gbr": ("green", "blue",  "red"),
    "grb": ("green", "red",   "blue"),
    "rbg": ("red",   "blue",  "green"),
    "rgb": ("red",   "green", "blue"),
}

ood_cases = []
for layout, (left, mid, right) in LAYOUT_DEFS.items():
    for target_color in ("r", "g", "b"):
        fname = SHEETS / f"eval2_{layout}_{target_color}_trimmed.jpg"
        if not fname.exists():
            continue
        color_map = {"r": "red", "g": "green", "b": "blue"}
        target = color_map[target_color]
        # Use an ordinal question appropriate to the layout
        if target == left:
            q = "Which bowl is leftmost?"
        elif target == mid:
            q = "Which bowl is in the middle?"
        else:
            q = "Which bowl is rightmost?"
        ood_cases.append({"image": str(fname), "question": q, "target": target})

run_section("2. OOD: REAL ROBOT CONTACT-SHEET FRAMES", ood_cases)


# ── 3. OOD: bowls.jpg standalone ─────────────────────────────────────────────
if Path("bowls.jpg").exists():
    print("="*60)
    print("  3. OOD: bowls.jpg (standalone image)")
    print("="*60)
    for q, _ in [
        ("Which bowl is leftmost?", None),
        ("Which bowl is in the middle?", None),
        ("Which bowl is rightmost?", None),
        ("Identify the bowl that is not blue or red.", None),
        ("Which bowl is first from the left?", None),
    ]:
        pred, raw = ask("bowls.jpg", q)
        print(f"  Q: {q:<50s} -> pred={pred!r}  raw={raw!r}")
    print()

print("Done.")
