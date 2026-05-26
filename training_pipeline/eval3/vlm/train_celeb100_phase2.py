#!/usr/bin/env python3
"""
train_celeb100_phase2.py
=========================
Train Phase 2 (spatial grounding) for all 100 celebrities.

Starts from eval3_phase1_celeb (already knows all 100 faces),
adds a fresh LoRA for left/center/right spatial grounding.

Usage:
  python3 train_celeb100_phase2.py \
      --manifest ./data/celeb100_phase2/manifest.jsonl \
      --output_dir ./checkpoints/celeb100_phase2 \
      --push_to_hub robot-learning-group47/eval3_phase2_celeb100
"""

import argparse, json, os, random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--manifest",       default="./data/celeb100_phase2/manifest.jsonl")
parser.add_argument("--output_dir",     default="./checkpoints/celeb100_phase2")
parser.add_argument("--base_model_id",  default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--base_lora",      default="robot-learning-group47/eval3_phase1_celeb",
                    help="Start from phase1 (already knows all 100 faces)")
parser.add_argument("--resume_lora",    default=None,
                    help="Optional phase2 checkpoint to merge after base_lora (continue training)")
parser.add_argument("--epochs",         type=int,   default=3)
parser.add_argument("--batch_size",     type=int,   default=8)
parser.add_argument("--image_longest_edge", type=int, default=384,
                    help="Cap processor image size to avoid multi-tile explosion (default 1536→17 tiles)")
parser.add_argument("--grad_accum",     type=int,   default=4)
parser.add_argument("--lr",             type=float, default=2e-5)
parser.add_argument("--lora_rank",      type=int,   default=16)
parser.add_argument("--max_length",     type=int,   default=512)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--val_split",      type=float, default=0.05)
parser.add_argument("--save_every_steps", type=int, default=200)
parser.add_argument("--push_to_hub",    default=None)
parser.add_argument("--disable_wandb",  action="store_true")
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

try:
    import wandb
    if args.disable_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    wandb.init(
        entity="nagyungecole-eth-z-rich",
        project="smolvla-eval3",
        name="celeb100-phase2-spatial",
        config=vars(args),
    )
    USE_WANDB = True
except Exception:
    USE_WANDB = False
    print("wandb not available, skipping")

POSITIONS = ["left", "center", "right"]
EVAL3_TARGETS = ["Taylor Swift", "Barack Obama", "Yann LeCun"]

P0 = ("There are three celebrity cards in this robot scene arranged in left, center, "
      "and right slots. Which slot is {name} in? Answer with exactly one word: left, center, or right.")
P1 = ("Put the coke can on the image of {name}. The cards are arranged left, center, "
      "and right. Which slot contains {name}? Answer with exactly one word: left, center, or right.")

# ── Dataset ───────────────────────────────────────────────────────────────────
class CelebSpatialDataset(Dataset):
    def __init__(self, records):
        self.records = records
        print(f"Dataset: {len(records)} QA pairs")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["image_path"]).convert("RGB")
        return img, r["question"], r["answer"]


def collate_fn(batch, processor, device, dtype, max_length):
    images, questions, answers = zip(*batch)
    all_input_ids, all_attention_mask, all_pixel_values, all_labels = [], [], [], []

    for img, q, a in zip(images, questions, answers):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": q}
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        full   = prompt + a

        enc = processor(text=full, images=[img], return_tensors="pt")
        enc_prompt = processor(text=prompt, images=[img], return_tensors="pt")

        input_ids      = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        pixel_values   = enc["pixel_values"][0]
        prompt_len     = enc_prompt["input_ids"].shape[1]
        labels         = input_ids.clone()
        labels[:prompt_len] = -100

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_pixel_values.append(pixel_values)
        all_labels.append(labels)

    max_len = max(t.shape[0] for t in all_input_ids)
    pad_id  = processor.tokenizer.pad_token_id or 0

    def pad(ts, val):
        return torch.stack([
            torch.cat([t, torch.full((max_len - t.shape[0],), val, dtype=t.dtype)])
            for t in ts
        ])

    return {
        "input_ids":      pad(all_input_ids, pad_id).to(device),
        "attention_mask": pad(all_attention_mask, 0).to(device),
        "pixel_values":   torch.stack(all_pixel_values).to(device, dtype=dtype),
        "labels":         pad(all_labels, -100).to(device),
    }

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading manifest: {args.manifest}")
with open(args.manifest) as f:
    records = [json.loads(l) for l in f]

# Shuffle and split
random.shuffle(records)
n_val   = max(1, int(len(records) * args.val_split))
val_rec = records[:n_val]
trn_rec = records[n_val:]
print(f"  Train: {len(trn_rec)}  Val: {len(val_rec)}")

# Build monitor records: one scene per (celeb, slot) for forced-choice eval
# Prioritize eval-3 targets, then sample others
def build_monitor_records(records, max_per_combo=2):
    seen = defaultdict(int)
    monitor = []
    # Eval-3 targets first
    priority = set(EVAL3_TARGETS)
    for r in records:
        name = r.get("target")
        pos  = r.get("target_pos")
        if name in priority and r.get("prompt_type") == "P0":
            key = (name, pos)
            if seen[key] < max_per_combo:
                monitor.append({
                    "image_path": r["image_path"],
                    "target": name,
                    "answer": pos,
                })
                seen[key] += 1
    # Fill with other celebs
    for r in records:
        name = r.get("target")
        pos  = r.get("target_pos")
        if r.get("prompt_type") == "P0":
            key = (name, pos)
            if seen[key] < 1:
                monitor.append({
                    "image_path": r["image_path"],
                    "target": name,
                    "answer": pos,
                })
                seen[key] += 1
        if len(monitor) >= 600:
            break
    return monitor

monitor_rec = build_monitor_records(val_rec)
print(f"Monitor records: {len(monitor_rec)} (covering {len(set(r['target'] for r in monitor_rec))} celebs × 3 slots)")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading base model + phase1 LoRA…")
processor = AutoProcessor.from_pretrained(args.base_model_id, token=hf_token)
base = SmolVLMForConditionalGeneration.from_pretrained(
    args.base_model_id, dtype=DTYPE, device_map=DEVICE, token=hf_token)
print(f"  Merging phase1 LoRA: {args.base_lora}")
model = PeftModel.from_pretrained(base, args.base_lora, token=hf_token)
model = model.merge_and_unload()  # bake face-ID knowledge into weights

if args.resume_lora:
    print(f"  Merging phase2 checkpoint: {args.resume_lora}")
    model = PeftModel.from_pretrained(model, args.resume_lora, token=hf_token)
    model = model.merge_and_unload()  # bake previous spatial grounding into weights

# Fresh LoRA for spatial grounding
lora_cfg = LoraConfig(
    r=args.lora_rank,
    lora_alpha=args.lora_rank * 2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# Cap image tiling — default longest_edge=1536 creates 17 tiles (1430 tokens)
# Setting to 384 uses 1 tile (~84 tokens): 17× fewer tokens, much faster
processor.image_processor.size = {"longest_edge": args.image_longest_edge}
print(f"Image longest_edge capped to {args.image_longest_edge} (was 1536, was 17 tiles=1430 tokens → now ~84 tokens)")

# ── Training setup ────────────────────────────────────────────────────────────
train_ds = CelebSpatialDataset(trn_rec)
val_ds   = CelebSpatialDataset(val_rec)

_collate = lambda b: collate_fn(b, processor, DEVICE, DTYPE, args.max_length)
train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, collate_fn=_collate)
val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=0, collate_fn=_collate)

optimizer   = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr, weight_decay=0.01)
total_steps = (len(train_loader) // args.grad_accum) * args.epochs
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

print(f"\nTraining: {args.epochs} epochs | {len(train_loader)} batches/epoch | "
      f"grad_accum={args.grad_accum} | effective_bs={args.batch_size * args.grad_accum} | lr={args.lr}")
print(f"Total optimizer steps: {total_steps}")

# Forced-choice token IDs
CHOICES = ["left", "center", "right"]
choice_tids = {}
for c in CHOICES:
    for variant in [c, " " + c, c.capitalize(), " " + c.capitalize()]:
        ids = processor.tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            choice_tids[c] = ids[0]
            break
print(f"Choice token IDs: {choice_tids}")

@torch.no_grad()
def run_monitor(model, records):
    model.eval()
    stats = defaultdict(lambda: [0, 0])
    for r in records:
        img  = Image.open(r["image_path"]).convert("RGB")
        name = r["target"]
        for pk, tmpl in [("P0", P0), ("P1", P1)]:
            q = tmpl.format(name=name)
            msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}]
            prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
            logits = model(**inputs).logits[0, -1, :]
            scores = {c: logits[tid].item() for c, tid in choice_tids.items()}
            pred   = max(scores, key=scores.get)
            ok     = int(pred == r["answer"])
            stats[pk][0] += ok; stats[pk][1] += 1
            # Per eval-3 target
            if name in EVAL3_TARGETS:
                k2 = f"{pk}/{name.split()[-1]}"
                stats[k2][0] += ok; stats[k2][1] += 1
    model.train()
    out = {}
    for k, (c, n) in stats.items():
        out[k] = c / n if n else 0.0
        print(f"  {k}: {c}/{n} = {out[k]:.1%}")
    return out

# ── Training loop ─────────────────────────────────────────────────────────────
model.train()
global_step = 0
best_val_loss = float("inf")
patience_count = 0

def save_checkpoint(tag):
    ckpt = OUTPUT_DIR / tag
    model.save_pretrained(ckpt)
    processor.save_pretrained(ckpt)
    print(f"  Saved: {ckpt}")

for epoch in range(args.epochs):
    epoch_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        outputs = model(**batch)
        loss = outputs.loss / args.grad_accum
        loss.backward()
        epoch_loss += outputs.loss.item()

        if (batch_idx + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            avg = epoch_loss / (batch_idx + 1)
            lr_n = scheduler.get_last_lr()[0]
            print(f"  epoch={epoch+1} step={global_step} loss={avg:.4f} lr={lr_n:.2e}", flush=True)

            if USE_WANDB:
                import wandb
                wandb.log({"train/loss": avg, "train/lr": lr_n,
                           "train/epoch": epoch + 1}, step=global_step)

            if args.save_every_steps and global_step % args.save_every_steps == 0:
                save_checkpoint(f"step_{global_step:05d}")

    # End-of-epoch val loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            val_loss += model(**batch).loss.item()
    val_loss /= len(val_loader)
    print(f"\n--- Epoch {epoch+1} val_loss={val_loss:.4f} ---")

    # Forced-choice monitor
    print("Forced-choice monitor:")
    fc_stats = run_monitor(model, monitor_rec[:120])
    if USE_WANDB:
        import wandb
        wandb.log({"val/loss": val_loss, **{f"val/{k}": v for k, v in fc_stats.items()}},
                  step=global_step)

    save_checkpoint(f"epoch_{epoch+1}")

    # Early stopping
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        patience_count = 0
        save_checkpoint("best")
    else:
        patience_count += 1
        if patience_count >= 2:
            print("Early stopping.")
            break

    model.train()

# Final save & push
save_checkpoint("final")

if args.push_to_hub:
    print(f"\nPushing to HF: {args.push_to_hub}")
    model.push_to_hub(args.push_to_hub, token=hf_token)
    processor.push_to_hub(args.push_to_hub, token=hf_token)
    print("Done.")

print(f"\nAll done. Checkpoints in {OUTPUT_DIR}")
