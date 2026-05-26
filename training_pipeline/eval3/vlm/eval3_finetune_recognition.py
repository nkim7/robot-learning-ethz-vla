#!/usr/bin/env python3
"""
eval3_finetune_recognition.py
==============================
Fine-tune wall-x (Qwen2.5-VL backbone, LoRA) to identify the toy-image
celebrities: taylor / obama / yann.

We frame it as a VQA task:
  input : image + "Who is this person? Answer with: taylor, obama, or yann."
  target: single-word answer

Only the answer tokens contribute to the loss (prompt tokens are masked).
The vision tower is frozen — only LM + LoRA layers train.

Usage (from repo root, venv active, on GPU machine):
    python data/eval3_finetune_recognition.py

Options:
    --model       HF model ID or local path   [x-square-robot/wall-oss-flow]
    --images      toy-image directory         [data/eval3_toy_images]
    --output_dir  save fine-tuned model here  [data/eval3_finetuned_recognition]
    --epochs      training epochs             [60]
    --lr          learning rate               [2e-4]
    --lora_r      LoRA rank                   [16]
    --lora_alpha  LoRA alpha                  [32]
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

_env_file = REPO_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

QUESTION = (
    "Who is the person in this image? "
    "Answer with exactly one word from: taylor, obama, yann."
)


# ── augmentation ──────────────────────────────────────────────────────────────
def augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    return img


# ── build one training sample ─────────────────────────────────────────────────
def build_train_sample(processor, image: Image.Image, answer: str, device: str) -> dict:
    from qwen_vl_utils import process_vision_info

    messages_full = [
        {"role": "user",      "content": [{"type": "image", "image": image},
                                           {"type": "text",  "text": QUESTION}]},
        {"role": "assistant", "content": [{"type": "text",  "text": answer}]},
    ]
    messages_prompt = messages_full[:1]

    full_text   = processor.apply_chat_template(messages_full,   tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

    image_inputs, _ = process_vision_info(messages_full)

    full_enc   = processor(text=[full_text],   images=image_inputs, padding=False, return_tensors="pt")
    prompt_enc = processor(text=[prompt_text], images=image_inputs, padding=False, return_tensors="pt")

    input_ids = full_enc["input_ids"][0]
    labels    = input_ids.clone()
    labels[:prompt_enc["input_ids"].shape[1]] = -100   # mask prompt

    return {
        "input_ids":      input_ids.to(device),
        "attention_mask": full_enc["attention_mask"][0].to(device),
        "labels":         labels.to(device),
        "pixel_values":   full_enc["pixel_values"].to(device),
        "image_grid_thw": full_enc["image_grid_thw"].to(device),
    }


# ── model loading + LoRA ──────────────────────────────────────────────────────
def load_wall_x_with_lora(model_id: str, device: str, lora_r: int, lora_alpha: int):
    from lerobot.policies.wall_x.modeling_wall_x import Qwen2_5_VLMoEForAction
    from transformers import AutoProcessor

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None

    print(f"[info] Loading processor: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True, token=hf_token)

    print(f"[info] Loading model: {model_id}  (this may take a minute…)")
    model = Qwen2_5_VLMoEForAction.from_pretrained(
        pretrained_name_or_path=model_id,
        attn_implementation="eager",
        token=hf_token,
    )

    print(f"[info] Adding LoRA (r={lora_r}, alpha={lora_alpha})")
    model.add_lora(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
    )

    model = model.to(device)
    return model, processor


# ── training ──────────────────────────────────────────────────────────────────
def train(model, processor, samples, device, epochs, lr):
    # freeze vision tower — only LM + LoRA adapters train
    frozen = 0
    for name, param in model.named_parameters():
        if "visual" in name and "lora" not in name.lower():
            param.requires_grad_(False)
            frozen += param.numel()
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    print(f"[info] Frozen vision params: {frozen/1e6:.1f}M  |  Trainable: {trainable_count/1e6:.2f}M")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = epochs * len(samples)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.05)

    print(f"[info] Training: {epochs} epochs × {len(samples)} samples = {total_steps} steps")
    print("─" * 60)
    model.train()

    for epoch in range(1, epochs + 1):
        random.shuffle(samples)
        epoch_loss = 0.0

        for img_path, label in samples:
            img    = augment(Image.open(img_path).convert("RGB"))
            sample = build_train_sample(processor, img, label, device)

            optimizer.zero_grad()

            # wall-x train_step_forward returns a dataclass with .loss (language loss included)
            outputs = model.train_step_forward(
                input_ids      = sample["input_ids"].unsqueeze(0),
                attention_mask = sample["attention_mask"].unsqueeze(0),
                labels         = sample["labels"].unsqueeze(0),
                pixel_values   = sample["pixel_values"],
                image_grid_thw = sample["image_grid_thw"],
                use_cache      = False,
            )

            # pull out the LM cross-entropy loss
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            else:
                # fallback: compute from logits
                logits = outputs.logits
                loss   = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    sample["labels"].unsqueeze(0)[:, 1:].reshape(-1),
                    ignore_index=-100,
                )

            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss at epoch {epoch}, skipping step")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            avg = epoch_loss / len(samples)
            print(f"  epoch {epoch:4d}/{epochs}  loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    print("─" * 60)
    print("[info] Training complete.")


# ── evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, processor, samples, device):
    from eval3_test_recognition import build_input, parse_prediction

    model.eval()
    correct = 0
    for img_path, gt in samples:
        img    = Image.open(img_path).convert("RGB")
        inputs = build_input(processor, img, device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        raw     = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        pred    = parse_prediction(raw)
        ok      = pred == gt
        if ok:
            correct += 1
        print(f"  {'✓' if ok else '✗'}  {img_path.name:25s}  gt={gt:8s}  pred={str(pred):8s}  raw='{raw.strip()}'")

    acc = correct / len(samples) * 100
    print(f"\n  Fine-tuned accuracy: {correct}/{len(samples)} = {acc:.1f}%")
    return acc


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="x-square-robot/wall-oss-flow")
    parser.add_argument("--images",     default=str(SCRIPT_DIR / "eval3_toy_images"))
    parser.add_argument("--output_dir", default=str(SCRIPT_DIR / "eval3_finetuned_recognition"))
    parser.add_argument("--device",     default=None)
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lora_r",     type=int,   default=16)
    parser.add_argument("--lora_alpha", type=int,   default=32)
    args = parser.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[info] device: {device}")

    # load samples
    images_dir = Path(args.images)
    samples: list[tuple[Path, str]] = []
    with open(images_dir / "labels.csv") as f:
        for row in csv.DictReader(f):
            p = images_dir / row["filename"]
            if p.exists():
                samples.append((p, row["label"]))
    print(f"[info] {len(samples)} training samples")

    model, processor = load_wall_x_with_lora(args.model, device, args.lora_r, args.lora_alpha)
    train(model, processor, samples, device, args.epochs, args.lr)

    # save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(out_dir))
    except Exception:
        torch.save(model.state_dict(), out_dir / "model_state_dict.pt")
    processor.save_pretrained(str(out_dir))
    print(f"[info] Model saved → {out_dir}")

    # eval on training set to confirm learning
    print("\n[info] Evaluating fine-tuned model on training images…")
    evaluate(model, processor, samples, device)

    print(f"\n[done] To test on its own:\n  python data/eval3_test_recognition.py --model {out_dir}")


if __name__ == "__main__":
    main()
