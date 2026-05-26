#!/usr/bin/env python3
"""
Train Eval2 Grounding VLM – simple & fast.

Same structure as the pathC VQA training (one sample at a time, no pre-cache).
Key fix over previous attempts: validation uses teacher-forced NLL loss,
NOT generate(). generate() is only called in an optional probe every N epochs.

Training  : one sample at a time, grad accumulation, augmentation
Val/epoch : teacher-forced loss on all val records (forward pass only, ~seconds)
Probe/N   : candidate scoring on 2 samples per task type  (log-likelihood, no generate)
Gen/M     : generate() on 12 samples every M epochs to inspect text output
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter

# ── Constants ─────────────────────────────────────────────────────────────────

COLORS = ["blue", "green", "red"]
SLOTS  = ["left", "middle", "right"]
COLOR_SLOT_CANDIDATES = [f"{c} {s}" for c in COLORS for s in SLOTS]
LAYOUT_CANDIDATES = [
    "blue green red", "blue red green", "green blue red",
    "green red blue", "red blue green", "red green blue",
]
TASK_CANDIDATES: dict[str, list[str]] = {
    "A": LAYOUT_CANDIDATES,
    "B": COLORS,
    "C": SLOTS,
    "D": COLOR_SLOT_CANDIDATES,
    "E": COLOR_SLOT_CANDIDATES,
    "F": COLOR_SLOT_CANDIDATES,
    "G": COLOR_SLOT_CANDIDATES,
}

SYSTEM = (
    "You are a robot vision assistant. A tabletop scene shows a robot arm, "
    "a banana, and exactly three bowls: one blue, one green, and one red. "
    "The bowls are arranged left to right from the robot's perspective. "
    "Use the image to determine the bowl order and answer concisely."
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data",        type=Path, default=Path("data/eval2_grounding_vqa/vqa.jsonl"))
    p.add_argument("--images-root", type=Path, default=Path("data/eval2_grounding_vqa"))
    p.add_argument("--model",       default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    p.add_argument("--output-dir",  type=Path, default=Path("models/eval2_grounding_vlm_lora"))
    p.add_argument("--device",      default=None)
    p.add_argument("--epochs",      type=int,   default=3)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--grad-accum",  type=int,   default=8)
    p.add_argument("--lora-r",      type=int,   default=16)
    p.add_argument("--lora-alpha",  type=int,   default=32)
    p.add_argument("--lora-dropout",type=float, default=0.05)
    p.add_argument("--probe-every", type=int,   default=1,
                   help="Candidate-scoring probe every N epochs.")
    p.add_argument("--gen-every",   type=int,   default=5,
                   help="generate() probe every N epochs (expensive).")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed",        type=int,   default=47)
    p.add_argument("--wandb-enable",   action="store_true")
    p.add_argument("--wandb-project",  default="lerobot")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--push-to-hub",    action="store_true")
    p.add_argument("--hub-repo", default="robot-learning-group47/smolvlm2-500m-eval2-grounding")
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def load_records(path: Path, split: str) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()
            if l.strip() and json.loads(l).get("split") == split]


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(img: Image.Image) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.80, 1.20))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.90, 1.10))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.92, 1.08))
    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))
    return img


# ── Sample building (identical pattern to pathC VQA script) ──────────────────

def build_sample(processor, record: dict, images_root: Path,
                 device: torch.device, train: bool) -> dict:
    img = Image.open(images_root / record["image"]).convert("RGB")
    if train:
        img = augment(img)

    q, a = record["question"], record["answer"]
    msgs_full   = [
        {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": f"{SYSTEM}\n\n{q}"}]},
        {"role": "assistant", "content": [{"type": "text",  "text": a}]},
    ]
    msgs_prompt = msgs_full[:1]

    full_text   = processor.apply_chat_template(msgs_full,   tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)

    # ONE processor call (image processing is the expensive part).
    # For prompt length: tokenizer-only on prompt_text (no image re-processing).
    # Image tokens are identical in both full and prompt sequences (same image,
    # same tile count), so the text-only token count gives the correct offset.
    full_enc   = processor(text=[full_text], images=[img], return_tensors="pt")
    prompt_len = processor.tokenizer(prompt_text, return_tensors="pt",
                                     add_special_tokens=False)["input_ids"].shape[1]

    labels = full_enc["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full_enc["labels"] = labels

    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in full_enc.items()}


# ── Answer normalisation for candidate scoring ────────────────────────────────

def normalize_answer(answer: str) -> str:
    a = answer.strip()
    m = re.search(r'"target_color"\s*:\s*"(\w+)".*?"target_slot"\s*:\s*"(\w+)"', a, re.S)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m = re.search(r'"layout_left"\s*:\s*"(\w+)".*?"layout_middle"\s*:\s*"(\w+)".*?"layout_right"\s*:\s*"(\w+)"', a, re.S)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"
    if "," in a:
        return " ".join(p.strip() for p in a.split(","))
    return a.lower().strip()


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, processor, records: list[dict], args,
                optimizer, scaler, scheduler, device) -> float:
    model.train()
    random.shuffle(records)
    optimizer.zero_grad(set_to_none=True)
    total_loss, n = 0.0, 0

    for i, rec in enumerate(records, 1):
        sample = build_sample(processor, rec, args.images_root, device, train=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type in {"cuda", "mps"}):
            out  = model(**sample, use_cache=False)
            loss = out.loss / args.grad_accum

        scaler.scale(loss).backward()
        total_loss += float(loss.detach().cpu()) * args.grad_accum
        n += 1

        if i % args.grad_accum == 0 or i == len(records):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, n)


# ── Validation: teacher-forced loss (no generate) ────────────────────────────

@torch.no_grad()
def val_loss_epoch(model, processor, records: list[dict], args, device) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    for rec in records:
        sample = build_sample(processor, rec, args.images_root, device, train=False)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type in {"cuda", "mps"}):
            out = model(**sample, use_cache=False)
        total_loss += float(out.loss.detach().cpu())
        n += 1
    return total_loss / max(1, n)


# ── Candidate scoring probe ───────────────────────────────────────────────────

@torch.no_grad()
def candidate_probe(model, processor, records: list[dict], args,
                    device, n_per_task: int = 2) -> dict[str, float]:
    """Score each candidate answer by log-likelihood, pick best. No generate()."""
    model.eval()

    # Pick n_per_task records per task type
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r.get("task", "?")].append(r)

    task_correct: dict[str, int] = defaultdict(int)
    task_total:   dict[str, int] = defaultdict(int)

    for task, recs in by_task.items():
        candidates = TASK_CANDIDATES.get(task, [])
        if not candidates:
            continue
        sample_recs = random.sample(recs, min(n_per_task, len(recs)))

        for rec in sample_recs:
            gold = normalize_answer(rec["answer"])
            img  = Image.open(args.images_root / rec["image"]).convert("RGB")
            q    = rec["question"]

            best_loss, best_cand = float("inf"), None
            for cand in candidates:
                # Build a fake record with this candidate as the answer
                fake = {**rec, "answer": cand}
                sample = build_sample(processor, fake, args.images_root, device, train=False)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type in {"cuda", "mps"}):
                    out = model(**sample, use_cache=False)
                score = float(out.loss.detach().cpu())
                if score < best_loss:
                    best_loss, best_cand = score, cand

            task_total[task]   += 1
            task_correct[task] += int((best_cand or "").strip() == gold.strip())

    results = {}
    for t in sorted(task_total):
        results[t] = task_correct[t] / task_total[t]
    results["overall"] = sum(task_correct.values()) / max(1, sum(task_total.values()))
    return results


# ── Generate probe (infrequent) ───────────────────────────────────────────────

@torch.no_grad()
def gen_probe(model, processor, records: list[dict], args, device, n: int = 12) -> None:
    model.eval()
    orig_pad = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    sample_recs = random.sample(records, min(n, len(records)))

    print("\n── generate() probe ────────────────────────────────────────")
    for rec in sample_recs:
        sample = build_sample(processor, rec, args.images_root, device, train=False)
        labels = sample.pop("labels")
        gen = model.generate(**sample, max_new_tokens=args.max_new_tokens, do_sample=False)
        pred = processor.batch_decode(
            gen[:, sample["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        gold = normalize_answer(rec["answer"])
        pred_n = normalize_answer(pred)
        match = "✓" if pred_n == gold else "✗"
        print(f"  [{rec.get('task','?')}] {match}  gold={gold!r:25s}  pred={pred!r}")
    print("────────────────────────────────────────────────────────────\n")
    processor.tokenizer.padding_side = orig_pad


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        args.device or (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else "cpu"
        )
    )
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    train_recs = load_records(args.data, "train")
    val_recs   = load_records(args.data, "val")
    print(f"Device: {device}  train={len(train_recs)}  val={len(val_recs)}")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.utils import logging as tlog
    tlog.set_verbosity_error()

    processor = AutoProcessor.from_pretrained(args.model, token=token)
    # Disable tiling: robot camera images don't need 17 tiles.
    # SmolVLM2 default tiles 17×512×512 → slow CPU bottleneck (~550ms/sample).
    # With do_image_splitting=False we get 1 tile → ~10× faster preprocessing.
    processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type != "cpu" else torch.float32,
        low_cpu_mem_usage=True, token=token,
    )
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))
    for name, param in model.named_parameters():
        if "vision_model" in name and "lora_" not in name:
            param.requires_grad_(False)
    model.print_trainable_parameters()
    model.to(device)

    trainable   = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = max(1, args.epochs * len(train_recs) // args.grad_accum)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    wandb_run = None
    if args.wandb_enable:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            config=vars(args),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, processor, train_recs, args,
                             optimizer, scaler, scheduler, device)
        v_loss = val_loss_epoch(model, processor, val_recs, args, device)
        lr_now = scheduler.get_last_lr()[0]

        log = (f"epoch={epoch:3d}  train_loss={t_loss:.4f}  "
               f"val_loss={v_loss:.4f}  lr={lr_now:.2e}")

        probe = {}
        if epoch % args.probe_every == 0:
            probe = candidate_probe(model, processor, val_recs, args, device)
            task_str = "  ".join(f"{k}={v:.2f}" for k, v in probe.items() if k != "overall")
            log += f"  cand_acc={probe.get('overall', 0):.3f}  [{task_str}]"

        print(log)

        if epoch % args.gen_every == 0:
            gen_probe(model, processor, val_recs, args, device, n=12)

        if wandb_run:
            wd = {"epoch": epoch, "train/loss": t_loss, "val/loss": v_loss,
                  "optimizer/lr": lr_now, "best/val_loss": min(best_val_loss, v_loss)}
            for t, a in probe.items():
                wd[f"val/cand_acc_{t}"] = a
            wandb_run.log(wd, step=epoch)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_dir = args.output_dir / "best"
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"  ✓ best val_loss={best_val_loss:.4f} → {save_dir}")

    model.save_pretrained(args.output_dir / "last")
    processor.save_pretrained(args.output_dir / "last")
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")

    if wandb_run:
        wandb_run.finish()
    if args.push_to_hub:
        model.push_to_hub(args.hub_repo, commit_message="Eval2 grounding LoRA best")
        processor.push_to_hub(args.hub_repo, commit_message="Processor")


if __name__ == "__main__":
    main()
