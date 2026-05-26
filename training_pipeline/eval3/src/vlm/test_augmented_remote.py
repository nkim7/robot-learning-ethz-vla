#!/usr/bin/env python3
import re, random, torch
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

ADAPTER = Path("/root/adapter")
BASE    = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SYSTEM  = (
    "You answer questions about a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Use the image to identify the left, middle, and right bowls, then answer with only the target bowl color."
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

LAYOUTS = {
    "bgr": ("blue",  "green", "red"),
    "brg": ("blue",  "red",   "green"),
    "gbr": ("green", "blue",  "red"),
    "grb": ("green", "red",   "blue"),
    "rbg": ("red",   "blue",  "green"),
    "rgb": ("red",   "green", "blue"),
}

QUESTIONS = [
    ("ordinal",  "Which bowl is leftmost?",                              0),
    ("ordinal",  "Which bowl is in the middle?",                         1),
    ("ordinal",  "Which bowl is rightmost?",                             2),
    ("relative", "Which bowl is immediately to the right of {left}?",    1),
    ("negation", "Identify the bowl that is not {mid} and not {right}.", 0),
]

print("Loading model...")
processor = AutoProcessor.from_pretrained(str(ADAPTER))
base = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, str(ADAPTER)).to(DEVICE)
model.eval()
print("Ready.\n")

def parse_color(text):
    t = text.strip().lower()
    for c in ("blue", "red", "green"):
        if re.search(rf"\b{c}\b", t):
            return c
    return t

def ask(image_path, question):
    img = Image.open(image_path).convert("RGB")
    prompt_text = f"{SYSTEM}\n\nQuestion: {question}"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[prompt], images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=8, do_sample=False)
    raw = processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return parse_color(raw), raw.strip()

aug_images = sorted(Path("/root/augmented").glob("*.jpeg"))
print(f"Testing {len(aug_images)} augmented images × {len(QUESTIONS)} questions = {len(aug_images)*len(QUESTIONS)} predictions\n")

correct = total = 0
failures = []
by_layout = {}
by_qtype = {}

for aug_path in aug_images:
    stem = aug_path.stem.replace("_aug1","").replace("_aug2","").replace("_aug3","").replace("_aug4","").replace("_aug5","")
    layout = stem.split("_")[0]
    left, mid, right = LAYOUTS[layout]

    for qtype, qtpl, idx in QUESTIONS:
        target = (left, mid, right)[idx]
        question = qtpl.format(left=left, mid=mid, right=right)
        pred, raw = ask(aug_path, question)
        ok = pred == target
        correct += ok
        total += 1
        by_layout.setdefault(layout, []).append(ok)
        by_qtype.setdefault(qtype, []).append(ok)
        if not ok:
            failures.append((aug_path.name, layout, qtype, target, pred, question))

print(f"{'='*65}")
print(f"AUGMENTED IMAGES: {correct}/{total} = {correct/total*100:.1f}%")
print(f"{'='*65}")
print(f"\nBy layout:")
for layout in sorted(by_layout):
    r = by_layout[layout]
    print(f"  {layout}: {sum(r)}/{len(r)} = {sum(r)/len(r)*100:.1f}%")
print(f"\nBy question type:")
for qt in sorted(by_qtype):
    r = by_qtype[qt]
    print(f"  {qt:8s}: {sum(r)}/{len(r)} = {sum(r)/len(r)*100:.1f}%")
print(f"\nFailures ({len(failures)}):")
for f in failures[:30]:
    print(f"  [{f[1]}] {f[0]:28s} {f[2]:8s} target={f[3]:6s} pred={f[4]:6s} | {f[5][:55]}")
if len(failures) > 30:
    print(f"  ... and {len(failures)-30} more")
