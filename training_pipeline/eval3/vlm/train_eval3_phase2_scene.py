"""
Phase 2: Fine-tune on real robot-setup scene images.
Teaches the model which portrait is at which position (left/center/right).

Runs after Phase 1 — loads Phase 1 LoRA checkpoint and continues training.

Usage:
    python3 train_eval3_phase2_scene.py \
        --phase1_ckpt ./checkpoints/phase1_toy/final \
        --scene_dir   ./data/scene_setup \
        --output_dir  ./checkpoints/phase2_scene
"""

import argparse
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--phase1_ckpt", required=True)
parser.add_argument("--scene_dir", default="./data/scene_setup")
parser.add_argument("--output_dir", default="./checkpoints/phase2_scene")
parser.add_argument("--model_id", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Annotations: (filename, left, center, right) ─────────────────────────────
ANNOTATIONS = [
    ("WhatsApp Image 2026-05-23 at 00.35.44.jpeg",    "Taylor Swift", "Yann LeCun",   "Barack Obama"),
    ("WhatsApp Image 2026-05-23 at 00.35.44 (1).jpeg","Taylor Swift", "Yann LeCun",   "Barack Obama"),
    ("WhatsApp Image 2026-05-23 at 00.35.44 (2).jpeg","Yann LeCun",   "Taylor Swift",  "Barack Obama"),
    ("WhatsApp Image 2026-05-23 at 00.35.45.jpeg",    "Yann LeCun",   "Barack Obama",  "Taylor Swift"),
    ("WhatsApp Image 2026-05-23 at 00.35.45 (1).jpeg","Yann LeCun",   "Barack Obama",  "Taylor Swift"),
]

# ── QA pair generator ─────────────────────────────────────────────────────────
def make_qa_pairs(left, center, right):
    """Return list of (question, answer) for one scene image."""
    pairs = []
    layout = f"Left: {left}. Center: {center}. Right: {right}."

    # 1. Full layout description
    pairs.append((
        "There are three celebrity portraits on the wall arranged left, center, and right. "
        "Identify each one.",
        layout,
    ))

    # 2. Where-is questions for each person
    for pos, name in [("left", left), ("center", center), ("right", right)]:
        pairs.append((
            f"Which celebrity portrait is on the {pos}? Answer with just the full name.",
            name,
        ))

    # 3. Position-of questions (what the robot controller will actually ask)
    for pos, name in [("left", left), ("center", center), ("right", right)]:
        pairs.append((
            f"Where is the portrait of {name}? Answer: left, center, or right.",
            pos,
        ))

    # 4. Place-command → position (mirrors real eval prompt style)
    for pos, name in [("left", left), ("center", center), ("right", right)]:
        pairs.append((
            f"I want to place the coke can on {name}'s portrait. "
            "Which position is that — left, center, or right?",
            pos,
        ))

    return pairs


# ── Dataset ───────────────────────────────────────────────────────────────────
class SceneDataset(Dataset):
    def __init__(self, scene_dir: Path):
        self.samples = []  # (image_path, question, answer)
        for fname, left, center, right in ANNOTATIONS:
            p = scene_dir / fname
            if not p.exists():
                print(f"  WARNING: {fname} not found, skipping")
                continue
            for q, a in make_qa_pairs(left, center, right):
                self.samples.append((p, q, a))
        random.shuffle(self.samples)
        print(f"Scene dataset: {len(self.samples)} QA pairs from "
              f"{len([a for a in ANNOTATIONS if (scene_dir/a[0]).exists()])} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, question, answer = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return img, question, answer


def collate_fn(batch, processor, device, dtype):
    all_input_ids, all_attention_mask, all_pixel_values, all_labels = [], [], [], []

    for img, question, answer in batch:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        full_text = prompt + answer

        enc = processor(text=full_text, images=[img], return_tensors="pt",
                        truncation=True, max_length=512)
        prompt_enc = processor(text=prompt, images=[img], return_tensors="pt",
                               truncation=True, max_length=512)

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        pixel_values = enc["pixel_values"][0]
        prompt_len = prompt_enc["input_ids"][0].shape[0]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_pixel_values.append(pixel_values)
        all_labels.append(labels)

    max_len = max(t.shape[0] for t in all_input_ids)
    pad_id = processor.tokenizer.pad_token_id or 0

    def pad(tensors, pad_val):
        return torch.stack([
            torch.cat([t, torch.full((max_len - t.shape[0],), pad_val, dtype=t.dtype)])
            for t in tensors
        ])

    return {
        "input_ids":      pad(all_input_ids, pad_id).to(device),
        "attention_mask": pad(all_attention_mask, 0).to(device),
        "pixel_values":   torch.stack(all_pixel_values).to(device, dtype=dtype),
        "labels":         pad(all_labels, -100).to(device),
    }


# ── Model: load Phase 1 checkpoint and add fresh LoRA ─────────────────────────
print(f"Loading base model: {args.model_id}")
base = SmolVLMForConditionalGeneration.from_pretrained(
    args.model_id, torch_dtype=DTYPE, device_map=DEVICE
)
processor = AutoProcessor.from_pretrained(args.model_id)

print(f"Loading Phase 1 LoRA: {args.phase1_ckpt}")
model = PeftModel.from_pretrained(base, args.phase1_ckpt)
model = model.merge_and_unload()  # bake Phase 1 weights in

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
dataset = SceneDataset(Path(args.scene_dir))
loader = DataLoader(
    dataset,
    batch_size=1,        # only 5 scene images → small batches
    shuffle=True,
    num_workers=0,
    collate_fn=lambda b: collate_fn(b, processor, DEVICE, DTYPE),
)

# ── Training ──────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr, weight_decay=0.01,
)
total_steps = len(loader) * args.epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

print(f"\nPhase 2 training: {args.epochs} epochs × {len(loader)} steps")
print(f"Output: {OUTPUT_DIR}\n")

model.train()
for epoch in range(args.epochs):
    epoch_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()
    avg = epoch_loss / len(loader)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1}/{args.epochs}  loss={avg:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

# Save
model.save_pretrained(OUTPUT_DIR / "final")
processor.save_pretrained(OUTPUT_DIR / "final")
print(f"\nPhase 2 saved: {OUTPUT_DIR / 'final'}")

# ── Quick eval ─────────────────────────────────────────────────────────────────
print("\nQuick eval — position queries:")
model.eval()
scene_dir = Path(args.scene_dir)
fname, left, center, right = ANNOTATIONS[0]
p = scene_dir / fname
if p.exists():
    img = Image.open(p).convert("RGB")
    for pos, name in [("left", left), ("center", center), ("right", right)]:
        q = f"Where is the portrait of {name}? Answer: left, center, or right."
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        ans = processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ok = pos.lower() in ans.lower()
        print(f"  [{'✓' if ok else '✗'}] {name} → expected={pos}, got={ans.strip()!r}")
