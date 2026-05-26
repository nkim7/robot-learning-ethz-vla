#!/usr/bin/env python3
"""
train_eval3_repair.py
=====================
Targeted LoRA repair for Eval 3 VLA backbone spatial grounding.

Starts from the existing checkpoint:
  robot-learning-group47/eval3_lora_fixed_Chengming

Fine-tunes with spatial grounding repair data (natural VLA prompts + diagnostic),
focusing on identity → spatial slot grounding, especially Yann-left under
VLA-style action framing while preserving Yann-right, Taylor, and Obama.

Usage:
  python3 train_eval3_repair.py \
      --manifest ./data/eval3_repair/manifest.jsonl \
      --output_dir ./checkpoints/eval3_repair
"""

import argparse, json, os, random
from collections import defaultdict
from pathlib import Path

import torch
import wandb
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--manifest",       default="./data/eval3_repair/manifest.jsonl")
parser.add_argument("--output_dir",     default="./checkpoints/eval3_repair")
parser.add_argument("--base_lora",      default="robot-learning-group47/eval3_lora_fixed_Chengming")
parser.add_argument("--base_model_id",  default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--epochs",         type=int,   default=4)
parser.add_argument("--batch_size",     type=int,   default=4)
parser.add_argument("--grad_accum",     type=int,   default=4)
parser.add_argument("--lr",             type=float, default=2e-5)  # low LR for repair
parser.add_argument("--lora_rank",      type=int,   default=16)
parser.add_argument("--max_length",     type=int,   default=256)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--val_split",      type=float, default=0.05)
parser.add_argument("--early_stop_patience", type=int, default=2)
parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
parser.add_argument("--monitor_limit",  type=int, default=240,
                    help="max unique scene/target records for canonical P0/P1 monitoring")
parser.add_argument("--max_train_steps", type=int, default=0,
                    help="optional cap on optimizer steps for quick targeted repair runs")
parser.add_argument("--save_every_steps", type=int, default=0,
                    help="optional optimizer-step interval for saving in-progress repair adapters")
parser.add_argument("--direct_loop", action="store_true",
                    help="iterate records directly instead of through DataLoader; useful for batch_size=1 repair runs")
parser.add_argument("--disable_wandb",  action="store_true")
parser.add_argument("--push_to_hub",    default=None, help="HF repo to push final model to")
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

if args.disable_wandb:
    os.environ["WANDB_MODE"] = "disabled"
wandb.init(
    entity="nagyungecole-eth-z-rich",
    project="smolvla-eval3",
    name="eval3-spatial-repair-yann-left",
    config=vars(args),
)

FULL_NAMES = {"taylor": "Taylor Swift", "obama": "Barack Obama", "yann": "Yann LeCun"}
POSITIONS = ["left", "center", "right"]
P0 = ("There are three celebrity cards in this robot scene, arranged in the left, "
      "center, and right slots. Which slot is {name} in? Answer with exactly one "
      "word: left, center, or right.")
P1 = ("Put the coke can on the image of {name}. The cards are arranged left, center, "
      "and right. Which slot contains {name}? Answer with exactly one word: left, "
      "center, or right.")

# ── Dataset ───────────────────────────────────────────────────────────────────
class RepairDataset(Dataset):
    def __init__(self, records):
        self.records = records
        print(f"RepairDataset: {len(records)} QA pairs")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["image"]).convert("RGB")
        return img, r["question"], r["answer"]


def collate_fn(batch, processor, device, dtype, max_length):
    images, questions, answers = zip(*batch)
    all_input_ids, all_attention_mask, all_pixel_values, all_labels = [], [], [], []

    for img, q, a in zip(images, questions, answers):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": q}
        ]}]
        prompt   = processor.apply_chat_template(messages, add_generation_prompt=True)
        full     = prompt + a

        enc = processor(text=full, images=[img], return_tensors="pt",
                        truncation=True, max_length=max_length)
        enc_prompt = processor(text=prompt, images=[img], return_tensors="pt",
                               truncation=True, max_length=max_length)

        input_ids      = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        pixel_values   = enc["pixel_values"][0]
        prompt_len     = enc_prompt["input_ids"].shape[1]
        labels         = input_ids.clone()
        labels[:prompt_len] = -100   # only supervise on the answer token(s)

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
random.shuffle(records)

n_val   = max(1, int(len(records) * args.val_split))
val_rec = records[:n_val]
trn_rec = records[n_val:]
print(f"  Train: {len(trn_rec)}  Val: {len(val_rec)}")

def build_monitor_records(records, limit):
    unique = {}
    for r in records:
        key = (r.get("scene_id"), r.get("target"), r.get("true_pos"))
        if key not in unique and r.get("target") in FULL_NAMES and r.get("true_pos") in POSITIONS:
            unique[key] = {
                "image": r["image"],
                "target": r["target"],
                "answer": r["true_pos"],
                "scene_id": r.get("scene_id"),
                "bg": r.get("bg"),
            }
        if len(unique) >= limit:
            break
    return list(unique.values())

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading base model: {args.base_model_id}")
processor = AutoProcessor.from_pretrained(args.base_model_id, token=hf_token)

print(f"Merging existing LoRA: {args.base_lora}")
base = SmolVLMForConditionalGeneration.from_pretrained(
    args.base_model_id, torch_dtype=DTYPE, device_map=DEVICE, token=hf_token)
model = PeftModel.from_pretrained(base, args.base_lora, token=hf_token)
model = model.merge_and_unload()   # bake existing LoRA into weights

# Apply fresh LoRA adapter for repair
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

# ── Training ──────────────────────────────────────────────────────────────────
train_ds = RepairDataset(trn_rec)
val_ds   = RepairDataset(val_rec)

_collate = lambda b: collate_fn(b, processor, DEVICE, DTYPE, args.max_length)
train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, collate_fn=_collate)
val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=0, collate_fn=_collate)

optimizer    = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr, weight_decay=0.01)
total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

print(f"\nTraining: {args.epochs} epochs | {len(train_loader)} batches | "
      f"grad_accum={args.grad_accum} | lr={args.lr}")

CHOICES = ["left", "center", "right"]
choice_tids = {}
for c in CHOICES:
    for variant in [c, " " + c, c.capitalize(), " " + c.capitalize()]:
        ids = processor.tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            choice_tids[c] = ids[0]
            break

monitor_rec = build_monitor_records(val_rec, args.monitor_limit)
print(f"Canonical P0/P1 monitor records: {len(monitor_rec)}")

@torch.no_grad()
def monitor_forced_choice(model, records):
    model.eval()
    stats = defaultdict(lambda: [0, 0])
    prompts = {"P0": P0, "P1": P1}
    for r in records:
        img = Image.open(r["image"]).convert("RGB")
        name = FULL_NAMES[r["target"]]
        for pk, tmpl in prompts.items():
            q = tmpl.format(name=name)
            msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}]
            prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
            logits = model(**inputs).logits[0, -1, :]
            scores = {c: logits[tid].item() for c, tid in choice_tids.items()}
            pred = max(scores, key=scores.get)
            ok = pred == r["answer"]
            for key in [pk, f"{pk}/{r['target']}", f"{pk}/{r['target']}_{r['answer']}"]:
                stats[key][0] += int(ok)
                stats[key][1] += 1
    model.train()
    return {k: (v[0] / v[1] if v[1] else 0.0, v[0], v[1]) for k, v in stats.items()}

model.train()
global_step = 0
best_val_loss = float("inf")
epochs_without_improvement = 0
best_dir = OUTPUT_DIR / "best"

def save_step_checkpoint(step_num):
    step_ckpt = OUTPUT_DIR / f"step_{step_num:05d}"
    model.save_pretrained(step_ckpt)
    processor.save_pretrained(step_ckpt)
    print(f"  Saved step checkpoint: {step_ckpt}")

def optimizer_step(epoch, step, epoch_loss):
    global global_step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    global_step += 1

    avg = epoch_loss / max(1, step + 1)
    lr_n = scheduler.get_last_lr()[0]
    print(f"  epoch={epoch+1} step={global_step} loss={avg:.4f} lr={lr_n:.2e}")
    wandb.log({"train/loss": avg, "train/lr": lr_n,
               "train/epoch": epoch + 1}, step=global_step)
    if args.save_every_steps and global_step % args.save_every_steps == 0:
        save_step_checkpoint(global_step)
    return bool(args.max_train_steps and global_step >= args.max_train_steps)

for epoch in range(args.epochs):
    epoch_loss = 0.0
    optimizer.zero_grad()
    stop_training = False

    train_iter = range(len(trn_rec)) if args.direct_loop else enumerate(train_loader)
    n_train_steps = len(trn_rec) if args.direct_loop else len(train_loader)

    for step_item in train_iter:
        if args.direct_loop:
            step = step_item
            batch = _collate([train_ds[step]])
        else:
            step, batch = step_item
        loss = model(**batch).loss / args.grad_accum
        loss.backward()
        epoch_loss += loss.item() * args.grad_accum

        if (step + 1) % args.grad_accum == 0:
            if optimizer_step(epoch, step, epoch_loss):
                print(f"  Reached max_train_steps={args.max_train_steps}")
                stop_training = True
                break

    if stop_training:
        print("  Stopping training loop early for capped repair run.")

    avg_train = epoch_loss / n_train_steps

    # Validation loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            val_loss += model(**batch).loss.item()
    val_loss /= len(val_loader)
    model.train()

    print(f"Epoch {epoch+1}/{args.epochs} — train={avg_train:.4f}  val={val_loss:.4f}")
    wandb.log({"epoch/train_loss": avg_train, "epoch/val_loss": val_loss,
               "train/epoch": epoch + 1}, step=global_step)

    if monitor_rec:
        mon = monitor_forced_choice(model, monitor_rec)
        for key in ["P0", "P1", "P1/yann_left", "P1/yann_right", "P1/taylor", "P1/obama"]:
            if key in mon:
                acc, c, n = mon[key]
                print(f"  monitor {key:<16} {c}/{n} = {acc:.1%}")
                wandb.log({f"monitor/{key}": acc}, step=global_step)

    ckpt = OUTPUT_DIR / f"epoch_{epoch+1}"
    model.save_pretrained(ckpt)
    processor.save_pretrained(ckpt)
    print(f"  Saved: {ckpt}")

    if val_loss < best_val_loss - args.early_stop_min_delta:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        model.save_pretrained(best_dir)
        processor.save_pretrained(best_dir)
        print(f"  New best checkpoint: {best_dir} (val={best_val_loss:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"  No val improvement for {epochs_without_improvement}/{args.early_stop_patience} epochs")
        if epochs_without_improvement >= args.early_stop_patience:
            print("  Early stopping.")
            break
    if stop_training:
        break

final = OUTPUT_DIR / "final"
model.save_pretrained(final)
processor.save_pretrained(final)
print(f"\nFinal checkpoint: {final}")

# ── Quick forced-choice diagnostic ────────────────────────────────────────────
print("\n=== Post-repair backbone diagnostic (val set, forced-choice) ===")
model.eval()
correct = defaultdict(int)
total   = defaultdict(int)

for r in val_rec[:min(60, len(val_rec))]:
    img = Image.open(r["image"]).convert("RGB")
    q   = r["question"]
    ans = r["answer"]
    msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":q}]}]
    prompt = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    scores = {c: logits[tid].item() for c, tid in choice_tids.items()}
    pred   = max(scores, key=scores.get)
    ok     = (pred == ans)
    tgt    = r.get("target", "?")
    pos    = r.get("true_pos", "?")
    key    = tgt + "_" + pos
    correct[key] += ok
    total[key]   += 1

print("\n  Post-repair per-identity × per-position (val sample):")
for cls in ["taylor","obama","yann"]:
    for pos in POSITIONS:
        k = cls + "_" + pos
        if total[k]:
            c = correct[k]; n = total[k]
            print(f"    {cls:<8} {pos:<8}: {c}/{n} = {c/n:.0%}")

# Push to hub if requested
if args.push_to_hub:
    print(f"\nPushing to HF Hub: {args.push_to_hub}")
    model.push_to_hub(args.push_to_hub, token=hf_token)
    processor.push_to_hub(args.push_to_hub, token=hf_token)
    print("Done.")

wandb.finish()
