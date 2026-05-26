#!/usr/bin/env python3
"""
eval3_test_recognition.py
=========================
Test whether the wall-x VLM backbone can correctly identify the people
in data/eval3_toy_images (taylor / obama / yann).

Wall-X is based on Qwen2.5-VL.  This script uses its text-generation
mode directly — no robot state, no action tokens, just image + question.

Usage (from repo root, with venv active):
    python data/eval3_test_recognition.py

Options:
    --model    HF model ID or local path  [x-square-robot/wall-oss-flow]
    --images   toy-image directory        [data/eval3_toy_images]
    --device   cuda / mps / cpu           [auto]
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# ── locate repo root and add lerobot to path ─────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent          # data/ lives one level below repo root
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

# ── load HF token from .env if present ───────────────────────────────────────
_env_file = REPO_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import torch
from PIL import Image

# ── label helpers ─────────────────────────────────────────────────────────────
LABEL_ALIASES: dict[str, str] = {
    "taylor":         "taylor",
    "taylor swift":   "taylor",
    "swift":          "taylor",
    "obama":          "obama",
    "barack obama":   "obama",
    "barack":         "obama",
    "president obama":"obama",
    "yann":           "yann",
    "yann lecun":     "yann",
    "yann le cun":    "yann",
    "lecun":          "yann",
}

QUESTION = (
    "Who is the person in this image? "
    "Answer with exactly one word from: taylor, obama, yann."
)


def parse_prediction(raw: str) -> str | None:
    text = raw.strip().lower()
    if text in LABEL_ALIASES:
        return LABEL_ALIASES[text]
    for alias, label in LABEL_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text):
            return label
    return None


# ── model loading ─────────────────────────────────────────────────────────────
def load_wall_x(model_id: str, device: str):
    """Load the wall-x Qwen2_5_VLMoEForAction model and its processor."""
    from lerobot.policies.wall_x.modeling_wall_x import Qwen2_5_VLMoEForAction
    from transformers import AutoProcessor

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None

    print(f"[info] Loading wall-x processor from: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True, token=hf_token)

    print(f"[info] Loading wall-x model from: {model_id}  (this may take a minute…)")
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        pretrained_name_or_path=model_id,
        attn_implementation="eager",
        token=hf_token,
    )
    model = model.to(device)
    model.eval()
    print("[info] Model ready.")
    return model, processor


# ── build a single VQA input ──────────────────────────────────────────────────
def build_input(processor, image: Image.Image, device: str) -> dict:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": QUESTION},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in inputs.items()}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="x-square-robot/wall-oss-flow")
    parser.add_argument("--images", default=str(SCRIPT_DIR / "eval3_toy_images"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    # device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else \
                 "mps"  if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"[info] device: {device}")

    # load labels
    images_dir  = Path(args.images)
    labels_path = images_dir / "labels.csv"
    samples: list[tuple[Path, str]] = []
    with open(labels_path) as f:
        for row in csv.DictReader(f):
            p = images_dir / row["filename"]
            if p.exists():
                samples.append((p, row["label"]))
    print(f"[info] {len(samples)} images loaded")

    model, processor = load_wall_x(args.model, device)

    results, correct = [], 0
    for img_path, gt in samples:
        img    = Image.open(img_path).convert("RGB")
        inputs = build_input(processor, img, device)

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        raw     = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        pred    = parse_prediction(raw)
        ok      = pred == gt
        if ok:
            correct += 1

        results.append(dict(filename=img_path.name, gt=gt, raw=raw.strip(), pred=pred, ok=ok))
        print(f"  {'✓' if ok else '✗'}  {img_path.name:25s}  gt={gt:8s}  pred={str(pred):8s}  raw='{raw.strip()}'")

    total = len(samples)
    acc   = correct / total * 100
    print(f"\n  Accuracy: {correct}/{total} = {acc:.1f}%")

    from collections import defaultdict
    per_class = defaultdict(lambda: {"c": 0, "t": 0})
    for r in results:
        per_class[r["gt"]]["t"] += 1
        if r["ok"]:
            per_class[r["gt"]]["c"] += 1
    print("  Per-class:")
    for lbl in sorted(per_class):
        c, t = per_class[lbl]["c"], per_class[lbl]["t"]
        print(f"    {lbl:10s}: {c}/{t} = {c/t*100:.0f}%")

    if acc < 80:
        print(
            f"\n[→] Accuracy {acc:.1f}% is below 80% — run the finetune script:\n"
            "      python data/eval3_finetune_recognition.py"
        )
    else:
        print(f"\n[→] wall-x already recognises the toy images well ({acc:.1f}%).")

    out = SCRIPT_DIR / "eval3_recognition_results.json"
    out.write_text(json.dumps({"model": args.model, "accuracy": acc,
                               "per_class": {k: v for k, v in per_class.items()},
                               "details": results}, indent=2))
    print(f"[info] Results saved → {out}")


if __name__ == "__main__":
    main()
