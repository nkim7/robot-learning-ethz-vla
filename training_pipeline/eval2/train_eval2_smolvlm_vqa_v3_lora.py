#!/usr/bin/env python3
"""
Fine-tune SmolVLM2 on Eval2 VQA v3 to produce smolvlm3.
Fast version — same tricks as eval3 training, adapted for small in-RAM dataset:

  1. Pre-tokenize ALL 10,200 samples through processor at startup → stored as CPU
     tensors in RAM. Collate just pads+stacks — near-zero overhead per batch.
  2. TF32 on matmul + cuDNN — free tensor-core speedup on A100.
  3. Fused AdamW — single CUDA kernel optimizer step.
  4. non_blocking=True on .to(device) — async H→D DMA.
  5. pixel_values cast to float16 on device transfer to match model dtype.

No .pt files needed (dataset fits entirely in RAM).
Base model: robot-learning-group47/smolvlm2-500m-vqa-lora-merged
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

VALID_COLORS    = {"blue", "red", "green"}
VALID_POSITIONS = {"left", "middle", "right"}
VALID_TARGETS   = VALID_COLORS | VALID_POSITIONS

SYSTEM = (
    "You answer questions about a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Use the image to identify the left, middle, and right bowls. "
    "Answer with only the target bowl color (blue / red / green) "
    "or its position (left / middle / right) — whichever the question asks for."
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data",        type=Path, default=Path("data/eval2_vqa_v3/vqa.jsonl"))
    p.add_argument("--model",       default="robot-learning-group47/smolvlm2-500m-vqa-lora-merged")
    p.add_argument("--output-dir",  type=Path, default=Path("models/smolvlm3_vqa_lora"))
    p.add_argument("--device",      default=None)
    p.add_argument("--epochs",      type=int,   default=12)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--num-workers", type=int,   default=4)
    p.add_argument("--lora-r",      type=int,   default=32)
    p.add_argument("--lora-alpha",  type=int,   default=64)
    p.add_argument("--seed",        type=int,   default=47)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--target-accuracy", type=float, default=0.995)
    p.add_argument("--wandb-enable",   action="store_true")
    p.add_argument("--wandb-project",  default="lerobot")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--push-to-hub",    action="store_true")
    p.add_argument("--hub-repo",       default="robot-learning-group47/smolvlm3-500m-vqa-lora-merged")
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def load_records(path: Path, split: str) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    selected = [r for r in rows if r.get("split", "train") == split]
    if split == "train" and not selected:
        selected = rows
    return selected


def resolve_image_path(path: str, data_path: Path | None = None) -> Path:
    p = Path(path)
    if p.exists():
        return p
    candidates = [
        Path("permutation_init_position") / p.name,
        Path("augmented/augmented") / p.name,
    ]
    if data_path is not None:
        candidates += [
            data_path.parent.parent / "permutation_init_position" / p.name,
            data_path.parent.parent / "augmented" / "augmented" / p.name,
        ]
    for c in candidates:
        if c.exists():
            return c
    return p


def target_text(record: dict) -> str:
    target = record["target"].strip().lower()
    if target not in VALID_TARGETS:
        raise ValueError(f"Unexpected target '{target}'")
    return target


def parse_answer(text: str) -> str:
    text = text.strip().lower()
    for word in VALID_TARGETS:
        if re.search(rf"\b{word}\b", text):
            return word
    return ""


# ── Pre-tokenize all samples at startup ──────────────────────────────────────
# Eval3 trick: run processor (image→tiles + tokenizer) once for all samples.
# Collate then just pads+stacks pre-computed tensors — near-zero overhead.

def pretokenize(records: list[dict], processor, data_path: Path) -> tuple[list[dict], dict]:
    """
    Two-part pre-computation to keep RAM under control:

    1. pixel_values for the 120 UNIQUE images only (~360MB total).
       Each image is processed through the vision processor once and cached.

    2. input_ids + labels for all N samples (text only, ~100MB).
       The tokenizer runs once per sample; results stored as CPU tensors.

    Returns: (samples list, pixel_cache dict)
      samples: [{input_ids, attention_mask, labels, target, image_key}, ...]
      pixel_cache: {image_path -> FloatTensor[T,C,H,W]}
    """
    # ── Step 1: pixel_values for unique images ────────────────────────────────
    unique_paths = {r["image"] for r in records}
    print(f"  Processing {len(unique_paths)} unique images through vision encoder...")
    pixel_cache: dict[str, torch.Tensor] = {}
    dummy_text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q"}]}],
        tokenize=False, add_generation_prompt=True,
    )
    for raw_path in unique_paths:
        resolved = resolve_image_path(raw_path, data_path)
        try:
            img = Image.open(resolved).convert("RGB")
            enc = processor(text=[dummy_text], images=[img], return_tensors="pt")
            pixel_cache[raw_path] = enc["pixel_values"][0]   # [T, C, H, W]
        except Exception as e:
            print(f"  [warn] {resolved}: {e}")
    print(f"  Cached pixel_values for {len(pixel_cache)} images.")

    # ── Step 2: tokenize text for all samples ─────────────────────────────────
    print(f"  Tokenizing text for {len(records)} samples...")
    samples = []
    skipped = 0
    # Use a single dummy image for text-only tokenization (pixel_values ignored)
    dummy_img = Image.new("RGB", (256, 256))
    for i, rec in enumerate(records):
        if rec["image"] not in pixel_cache:
            skipped += 1
            continue

        answer = target_text(rec)
        prompt = f"{SYSTEM}\n\nQuestion: {rec['question']}"

        msgs_full   = [
            {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text",  "text": answer}]},
        ]
        msgs_prompt = msgs_full[:1]

        full_text   = processor.apply_chat_template(msgs_full,   tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)

        full_enc   = processor(text=[full_text],   images=[dummy_img], return_tensors="pt")
        prompt_enc = processor(text=[prompt_text], images=[dummy_img], return_tensors="pt")

        ids    = full_enc["input_ids"][0]
        labels = ids.clone()
        labels[: prompt_enc["input_ids"].shape[1]] = -100

        samples.append({
            "input_ids":      ids,
            "attention_mask": full_enc["attention_mask"][0],
            "labels":         labels,
            "target":         answer,
            "image_key":      rec["image"],   # lookup key into pixel_cache
        })

        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(records)}", flush=True)

    print(f"  Done. {len(samples)} samples ready ({skipped} skipped).")
    return samples, pixel_cache


# ── Dataset wrapping pre-tokenized tensors ────────────────────────────────────

class VQADataset(Dataset):
    def __init__(self, samples: list[dict], pixel_cache: dict[str, torch.Tensor]):
        self.samples     = samples
        self.pixel_cache = pixel_cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {**s, "pixel_values": self.pixel_cache[s["image_key"]]}


# ── Collate: pad + stack pre-computed tensors (near-zero overhead) ─────────────

def collate_fn(batch: list[dict]) -> dict:
    pad_id    = 0
    B         = len(batch)
    max_len   = max(s["input_ids"].shape[0] for s in batch)
    max_tiles = max(s["pixel_values"].shape[0] for s in batch)
    pv_shape  = batch[0]["pixel_values"].shape[1:]   # (C, H, W)

    batched_ids    = torch.full((B, max_len),  pad_id, dtype=torch.long)
    batched_mask   = torch.zeros(B, max_len,            dtype=torch.long)
    batched_labels = torch.full((B, max_len),  -100,   dtype=torch.long)
    batched_pixels = torch.zeros(B, max_tiles, *pv_shape, dtype=batch[0]["pixel_values"].dtype)

    for i, s in enumerate(batch):
        L = s["input_ids"].shape[0]
        T = s["pixel_values"].shape[0]
        batched_ids[i,    :L] = s["input_ids"]
        batched_mask[i,   :L] = s["attention_mask"]
        batched_labels[i, :L] = s["labels"]
        batched_pixels[i, :T] = s["pixel_values"]

    return {
        "input_ids":      batched_ids,
        "attention_mask": batched_mask,
        "labels":         batched_labels,
        "pixel_values":   batched_pixels,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def to_device(batch: dict, device: torch.device) -> dict:
    """Move batch to device. pixel_values cast to float16, non_blocking for async DMA."""
    out = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            out[k] = v
        elif k == "pixel_values":
            out[k] = v.to(device, dtype=torch.float16, non_blocking=True)
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


def evaluate(model, val_ds: VQADataset, device: torch.device, batch_size: int) -> float:
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out   = model(**{k: v for k, v in batch.items() if k != "labels"},
                          labels=batch["labels"])
            total_loss += out.loss.item()
            n += 1
    return total_loss / max(n, 1)


def accuracy_probe(model, processor, val_ds: VQADataset, device: torch.device,
                   max_new_tokens: int, n: int = 20) -> float:
    """generate() on a small subset — uses pre-tokenized input_ids directly."""
    model.eval()
    indices = random.sample(range(len(val_ds)), min(n, len(val_ds)))
    correct = 0
    with torch.no_grad():
        for idx in indices:
            s   = val_ds[idx]
            enc = {
                "input_ids":      s["input_ids"].unsqueeze(0).to(device, non_blocking=True),
                "attention_mask": s["attention_mask"].unsqueeze(0).to(device, non_blocking=True),
                "pixel_values":   s["pixel_values"].unsqueeze(0).to(device, dtype=torch.float16, non_blocking=True),
            }
            prompt_len = (s["labels"] == -100).sum().item()
            ids  = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            pred = processor.decode(ids[0][prompt_len:], skip_special_tokens=True)
            if parse_answer(pred) == s["target"]:
                correct += 1
    return correct / max(len(indices), 1)


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, processor, train_ds: VQADataset, val_ds: VQADataset,
          args, device: torch.device, wandb_run=None) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]

    # Fused AdamW — single CUDA kernel, avoids Python loop over params (eval3 trick)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=0.01,
        fused=(device.type == "cuda"),
    )
    steps_per_epoch = max(1, len(train_ds) // args.batch_size)
    total_steps     = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_acc      = 0.0

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for batch in loader:
            batch = to_device(batch, device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out  = model(**{k: v for k, v in batch.items() if k != "labels"},
                             labels=batch["labels"])
                loss = out.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train = epoch_loss / max(n_batches, 1)
        val_loss  = evaluate(model, val_ds, device, args.batch_size)
        acc       = accuracy_probe(model, processor, val_ds, device, args.max_new_tokens)

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  acc={acc:.3f}",
              flush=True)

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch+1, "train_loss": avg_train,
                           "val_loss": val_loss, "accuracy": acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(str(output_dir / "best"))
            processor.save_pretrained(str(output_dir / "best"))
            print(f"  → saved best (val_loss={val_loss:.4f})", flush=True)

        if acc > best_acc:
            best_acc = acc
            if acc >= args.target_accuracy:
                print(f"  ✅ Target accuracy {args.target_accuracy:.3f} reached — stopping early.")
                break

    model.save_pretrained(str(output_dir / "last"))
    processor.save_pretrained(str(output_dir / "last"))
    print(f"\nDone. Best val_loss={best_val_loss:.4f}  Best acc={best_acc:.3f}")
    print(f"Best checkpoint: {output_dir}/best")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Base model: {args.model}")
    print(f"Data: {args.data}  batch_size={args.batch_size}  num_workers={args.num_workers}")

    # TF32 — uses A100 tensor cores at bfloat16 speed with float32 range (eval3 trick)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    train_rows = load_records(args.data, "train")
    val_rows   = load_records(args.data, "val")
    if not val_rows:
        val_rows = train_rows[:max(1, len(train_rows) // 10)]
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}")

    by_type: dict[str, int] = {}
    for r in train_rows:
        by_type[r.get("type", "?")] = by_type.get(r.get("type", "?"), 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t:20s}: {n}")

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model

    print(f"\nLoading processor from {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model)

    # Pre-tokenize once: pixel_values for 120 unique images + text tokens for all samples
    print("\nPre-tokenizing all samples (runs once)...")
    all_rows = train_rows + val_rows
    all_samples, pixel_cache = pretokenize(all_rows, processor, args.data)
    train_samples = all_samples[:len(train_rows)]
    val_samples   = all_samples[len(train_rows):]

    train_ds = VQADataset(train_samples, pixel_cache)
    val_ds   = VQADataset(val_samples,   pixel_cache)

    print(f"\nLoading model {args.model} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=str(device),
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    wandb_run = None
    if args.wandb_enable:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"smolvlm3-vqa-v3-r{args.lora_r}",
            config=vars(args),
        )

    train(model, processor, train_ds, val_ds, args, device, wandb_run)

    if wandb_run is not None:
        wandb_run.finish()

    if args.push_to_hub:
        best_dir = str(args.output_dir / "best")
        print(f"\nPushing to {args.hub_repo} ...")
        from peft import PeftModel
        model_to_push = type(model).__bases__[0].from_pretrained(
            args.model, torch_dtype=torch.float32, device_map="cpu"
        )
        model_to_push = PeftModel.from_pretrained(model_to_push, best_dir)
        model_to_push = model_to_push.merge_and_unload()
        model_to_push.push_to_hub(
            args.hub_repo, private=False,
            commit_message="Merge smolvlm3 VQA v3 LoRA (r=32, +position outputs)"
        )
        processor.push_to_hub(args.hub_repo, private=False,
                               commit_message="Add processor for smolvlm3")
        print(f"Done: https://huggingface.co/{args.hub_repo}")


if __name__ == "__main__":
    main()
