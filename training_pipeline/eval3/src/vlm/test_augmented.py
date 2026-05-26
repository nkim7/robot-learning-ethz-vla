#!/usr/bin/env python3
"""
Generate hard-augmented versions of all 24 layout images, then test the model on them.
Augmentations: rotation, brightness, contrast, color saturation, hue shift, blur, noise.
"""
import re, random, torch
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

ADAPTER = Path("models/eval2_fresh_smolvlm_vqa_lora_r32_online_parsefix/best")
BASE    = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SYSTEM  = (
    "You answer questions about a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Use the image to identify the left, middle, and right bowls, then answer with only the target bowl color."
)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
AUG_DIR = Path("permutation_init_position/augmented")
AUG_DIR.mkdir(exist_ok=True)
N_AUG = 5  # augmented copies per image
random.seed(42)

LAYOUTS = {
    "bgr": ("blue",  "green", "red"),
    "brg": ("blue",  "red",   "green"),
    "gbr": ("green", "blue",  "red"),
    "grb": ("green", "red",   "blue"),
    "rbg": ("red",   "blue",  "green"),
    "rgb": ("red",   "green", "blue"),
}

QUESTIONS = [
    ("ordinal",  "Which bowl is leftmost?",                               0),
    ("ordinal",  "Which bowl is in the middle?",                          1),
    ("ordinal",  "Which bowl is rightmost?",                              2),
    ("relative", "Which bowl is immediately to the right of {left}?",     1),
    ("negation", "Identify the bowl that is not {mid} and not {right}.",  0),
]


def augment(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    # Rotation: up to ±20 degrees
    angle = rng.uniform(-20, 20)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(200, 200, 200))
    # Brightness: simulate lighting changes — quite wide range
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.45, 1.9))
    # Contrast
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.6))
    # Saturation: conservative — colors must stay recognisable as red/green/blue
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.7, 1.4))
    # Sharpness
    img = ImageEnhance.Sharpness(img).enhance(rng.uniform(0.5, 2.0))
    # Blur (occasional)
    if rng.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5)))
    # Gaussian noise
    if rng.random() < 0.5:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.default_rng(seed).normal(0, rng.uniform(5, 15), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img


# ── Generate augmented images ─────────────────────────────────────────────────
img_dir = Path("permutation_init_position")
source_images = sorted([f for f in img_dir.glob("*.jpeg") if f.parent == img_dir])

print(f"Generating {N_AUG} augmentations × {len(source_images)} images = {N_AUG * len(source_images)} total...")
aug_images = []
for src in source_images:
    for i in range(1, N_AUG + 1):
        img = Image.open(src).convert("RGB")
        aug = augment(img, seed=hash(src.name + str(i)) & 0xFFFFFF)
        out_path = AUG_DIR / f"{src.stem}_aug{i}.jpeg"
        aug.save(out_path, quality=92)
        aug_images.append((out_path, src.stem))

print(f"Saved {len(aug_images)} augmented images to {AUG_DIR}\n")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
processor = AutoProcessor.from_pretrained(str(ADAPTER))
base = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.float32)
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


# ── Run inference ─────────────────────────────────────────────────────────────
correct = total = 0
failures = []
by_layout = {}
by_qtype = {}

for aug_path, stem in aug_images:
    # stem is e.g. "bgr", "bgr_2", "rgb_4" — extract base layout
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
for f in failures[:20]:
    print(f"  [{f[1]:3s}] {f[0]:25s} {f[2]:8s} target={f[3]:6s} pred={f[4]:6s} | {f[5][:55]}")
if len(failures) > 20:
    print(f"  ... and {len(failures)-20} more")
