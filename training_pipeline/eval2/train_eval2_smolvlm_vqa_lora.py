#!/usr/bin/env python3
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
from transformers.utils import logging as transformers_logging


SYSTEM = (
    "You answer questions about a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Use the image to identify the left, middle, and right bowls, then answer with only the target bowl color."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLM2 on fresh Eval 2 bowl-composition VQA.")
    parser.add_argument("--data", type=Path, default=Path("data/eval2_vqa_fresh/vqa.jsonl"))
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("/data/resolver_outputs/eval2_fresh_smolvlm_vqa_lora"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--wandb-enable", action="store_true")
    parser.add_argument("--wandb-project", default="lerobot")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--target-accuracy", type=float, default=0.998)
    return parser.parse_args()


def load_records(path: Path, split: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    selected = [r for r in rows if r.get("split", "train") == split]
    if split == "train" and not selected:
        selected = rows
    return selected


def augment(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.15))
    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    return img


def target_text(record: dict) -> str:
    target = record["target"].strip().lower()
    if target not in {"blue", "red", "green"}:
        raise ValueError(f"unexpected target: {target}")
    return target


def parse_color(text: str) -> str:
    text = text.strip().lower()
    for color in ("blue", "red", "green"):
        if re.search(rf"\b{color}\b", text):
            return color
    return ""


def resolve_image_path(path: str, data_path: Path) -> Path:
    image_path = Path(path)
    if image_path.exists():
        return image_path
    candidates = [
        Path("permutation_init_position") / image_path.name,
        data_path.parent.parent.parent / "permutation_init_position" / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return image_path


def build_sample(processor, record: dict, device: torch.device, train: bool) -> dict[str, torch.Tensor]:
    image = Image.open(resolve_image_path(record["image"], build_sample.data_path)).convert("RGB")
    if train:
        image = augment(image)

    prompt = f"{SYSTEM}\n\nQuestion: {record['question']}"
    answer = target_text(record)
    messages_full = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    messages_prompt = messages_full[:1]

    full_text = processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

    full_enc = processor(text=[full_text], images=[image], return_tensors="pt")
    prompt_enc = processor(text=[prompt_text], images=[image], return_tensors="pt")

    labels = full_enc["input_ids"].clone()
    labels[:, : prompt_enc["input_ids"].shape[1]] = -100
    full_enc["labels"] = labels
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in full_enc.items()}


def train(model, processor, train_rows: list[dict], val_rows: list[dict], args, device: torch.device, wandb_run=None) -> None:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = max(1, args.epochs * len(train_rows) // args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.05)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    step = 0
    best_acc = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0

        for i, record in enumerate(train_rows, start=1):
            sample = build_sample(processor, record, device, train=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type in {"cuda", "mps"}):
                out = model(**sample, use_cache=False)
                loss = out.loss
                if loss is None:
                    logits = out.logits
                    loss = F.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        sample["labels"][:, 1:].reshape(-1),
                        ignore_index=-100,
                    )
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            running += float(loss.detach().cpu()) * args.grad_accum

            if i % args.grad_accum == 0 or i == len(train_rows):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

        train_loss = running / max(1, len(train_rows))
        acc = evaluate(model, processor, val_rows, args, device) if val_rows else 0.0
        lr = scheduler.get_last_lr()[0]
        print(f"epoch={epoch} loss={train_loss:.4f} val_acc={acc:.3f} lr={lr:.2e}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/accuracy": acc,
                    "optimizer/lr": lr,
                    "best/val_accuracy": max(best_acc, acc),
                    "global_step": step,
                },
                step=step,
            )
        if acc >= best_acc:
            best_acc = acc
            save_dir = args.output_dir / "best"
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)

    model.save_pretrained(args.output_dir / "last")
    processor.save_pretrained(args.output_dir / "last")


@torch.no_grad()
def evaluate(model, processor, rows: list[dict], args, device: torch.device) -> float:
    if not rows:
        return 0.0
    was_training = model.training
    model.eval()
    correct = 0
    for record in rows:
        sample = build_sample(processor, record, device, train=False)
        labels = sample.pop("labels")
        generated = model.generate(**sample, max_new_tokens=args.max_new_tokens, do_sample=False)
        raw = processor.batch_decode(generated[:, sample["input_ids"].shape[1] :], skip_special_tokens=True)[0]
        pred = parse_color(raw)
        correct += int(pred == target_text(record))
        sample["labels"] = labels
    if was_training:
        model.train()
    return correct / len(rows)


def main() -> None:
    args = parse_args()
    transformers_logging.set_verbosity_error()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    train_rows = load_records(args.data, "train")
    val_rows = load_records(args.data, "val")
    build_sample.data_path = args.data
    print(f"records train={len(train_rows)} val={len(val_rows)} device={device}")

    wandb_run = None
    if args.wandb_enable:
        import wandb

        run_name = args.wandb_run_name or args.output_dir.name
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model": args.model,
                "data": str(args.data),
                "output_dir": str(args.output_dir),
                "epochs": args.epochs,
                "lr": args.lr,
                "grad_accum": args.grad_accum,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "train_records": len(train_rows),
                "val_records": len(val_rows),
                "seed": args.seed,
                "target_accuracy": args.target_accuracy,
            },
        )
        wandb_run.summary["target_accuracy"] = args.target_accuracy

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, token=token)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
        token=token,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    for name, param in model.named_parameters():
        if "vision_model" in name and "lora" not in name.lower():
            param.requires_grad_(False)
    model.print_trainable_parameters()
    model.to(device)

    train(model, processor, train_rows, val_rows, args, device, wandb_run=wandb_run)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
