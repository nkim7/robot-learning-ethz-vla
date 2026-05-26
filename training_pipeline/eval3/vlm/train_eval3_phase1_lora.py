"""
Phase 1: LoRA fine-tuning of SmolVLM2 backbone for celebrity recognition.

Task:  "Who is the person in this image? Give the full name."
       → "This is [Name]."

Usage:
    # After TOY download:
    python3 train_eval3_phase1_lora.py --data_dir ./data/celeb_faces --run toy

    # After top50 download (continue from TOY checkpoint):
    python3 train_eval3_phase1_lora.py --data_dir ./data/celeb_faces --run top50 \
        --resume ./checkpoints/phase1_toy/final

    # After rest download (continue from top50 checkpoint):
    python3 train_eval3_phase1_lora.py --data_dir ./data/celeb_faces --run rest \
        --resume ./checkpoints/phase1_top50/final
"""

import argparse
import os
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./data/celeb_faces")
parser.add_argument("--run", default="all", help="toy | top50 | rest | all")
parser.add_argument("--resume", default=None, help="path to previous LoRA checkpoint")
parser.add_argument("--output_dir", default="./checkpoints/phase1")
parser.add_argument("--model_id", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--grad_accum", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--max_images_per_celeb", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
OUTPUT_DIR = Path(args.output_dir + f"_{args.run}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
QUESTION = "Who is the person in this image? Give the full name."

class CelebFaceDataset(Dataset):
    def __init__(self, data_dir: Path, max_per_celeb: int = 20):
        self.samples = []  # (image_path, celeb_name)
        for celeb_dir in sorted(data_dir.iterdir()):
            if not celeb_dir.is_dir():
                continue
            name = celeb_dir.name
            imgs = list(celeb_dir.glob("*.jpg")) + list(celeb_dir.glob("*.png"))
            random.shuffle(imgs)
            imgs = imgs[:max_per_celeb]
            for p in imgs:
                self.samples.append((p, name))
        random.shuffle(self.samples)
        print(f"Dataset: {len(self.samples)} images across "
              f"{len(set(s[1] for s in self.samples))} celebrities")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, name = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, name


def collate_fn(batch, processor, device, dtype):
    images, names = zip(*batch)

    # Process each sample individually then pad manually
    all_input_ids, all_attention_mask, all_pixel_values, all_labels = [], [], [], []

    for img, name in zip(images, names):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": QUESTION}
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        full_text = prompt + f"This is {name}."

        enc = processor(
            text=full_text,
            images=[img],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        pixel_values = enc["pixel_values"][0]

        # Mask prompt tokens — only supervise on answer tokens
        prompt_ids = processor(
            text=prompt, images=[img], return_tensors="pt", truncation=True, max_length=512
        )["input_ids"][0]
        prompt_len = prompt_ids.shape[0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_pixel_values.append(pixel_values)
        all_labels.append(labels)

    # Pad sequences to same length
    max_len = max(t.shape[0] for t in all_input_ids)
    pad_id = processor.tokenizer.pad_token_id or 0

    def pad(tensors, pad_val, max_l):
        return torch.stack([
            torch.cat([t, torch.full((max_l - t.shape[0],), pad_val, dtype=t.dtype)])
            for t in tensors
        ])

    result = {
        "input_ids": pad(all_input_ids, pad_id, max_len).to(device),
        "attention_mask": pad(all_attention_mask, 0, max_len).to(device),
        "pixel_values": torch.stack(all_pixel_values).to(device, dtype=dtype),
        "labels": pad(all_labels, -100, max_len).to(device),
    }
    return result


# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading model: {args.model_id}")
processor = AutoProcessor.from_pretrained(args.model_id)

if args.resume:
    print(f"Resuming from LoRA checkpoint: {args.resume}")
    base = SmolVLMForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=DTYPE, device_map=DEVICE
    )
    model = PeftModel.from_pretrained(base, args.resume)
    model = model.merge_and_unload()  # merge previous LoRA, apply fresh one below
else:
    model = SmolVLMForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=DTYPE, device_map=DEVICE
    )

# Apply LoRA to language model attention layers only
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

# ── Data ──────────────────────────────────────────────────────────────────────
data_dir = Path(args.data_dir)
dataset = CelebFaceDataset(data_dir, max_per_celeb=args.max_images_per_celeb)
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=lambda b: collate_fn(b, processor, DEVICE, DTYPE),
)

# ── Training ──────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=0.01,
)
total_steps = (len(loader) // args.grad_accum) * args.epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

print(f"\nTraining: {args.epochs} epochs, {len(loader)} batches/epoch, "
      f"grad_accum={args.grad_accum}")
print(f"Output: {OUTPUT_DIR}\n")

model.train()
global_step = 0
for epoch in range(args.epochs):
    epoch_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        outputs = model(**batch)
        loss = outputs.loss / args.grad_accum
        loss.backward()
        epoch_loss += outputs.loss.item()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 20 == 0:
                avg = epoch_loss / (step + 1)
                print(f"  epoch={epoch+1} step={global_step} loss={avg:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

    avg_loss = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}/{args.epochs} done — avg loss: {avg_loss:.4f}")

    ckpt = OUTPUT_DIR / f"epoch_{epoch+1}"
    model.save_pretrained(ckpt)
    processor.save_pretrained(ckpt)
    print(f"  Saved checkpoint: {ckpt}")

# Save final
final = OUTPUT_DIR / "final"
model.save_pretrained(final)
processor.save_pretrained(final)
print(f"\nFinal checkpoint saved: {final}")

# ── Quick validation ──────────────────────────────────────────────────────────
print("\nQuick validation (TOY set):")
model.eval()
TOY = {"Taylor Swift": "taylor", "Barack Obama": "obama", "Yann LeCun": "yann"}
for name, prefix in TOY.items():
    img_dir = data_dir / name
    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    if not imgs:
        print(f"  {name}: no images found")
        continue
    img = Image.open(imgs[0]).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": QUESTION}
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    answer = processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    ok = name.split()[-1].lower() in answer.lower()
    print(f"  [{'✓' if ok else '✗'}] {name}: {answer.strip()}")
