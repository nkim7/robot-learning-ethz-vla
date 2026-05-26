"""
Phase 1 v2: improved face recognition using robot-learning-group47/celebrities_faces_min10_v2
- 102 celebrities from HF dataset (including Obama + Swift, 20 each)
- + Yann LeCun from existing celeb_faces pool
- Total: ~103 celebrities, ~2000+ images
- QA format: "Who is this person?" -> "Name"
- LoRA on SmolVLM2-2.2B-Instruct
"""

import argparse, json, math, random, io
from pathlib import Path
from collections import defaultdict

import torch
import wandb
from PIL import Image
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, SmolVLMForConditionalGeneration
from facenet_pytorch import MTCNN
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--celeb_dir",    default="./data/celeb_faces")
parser.add_argument("--output_dir",   default="./checkpoints/phase1_v2")
parser.add_argument("--model_id",     default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--hf_dataset",   default="robot-learning-group47/celebrities_faces_min10_v2")
parser.add_argument("--epochs",       type=int,   default=5)
parser.add_argument("--batch_size",   type=int,   default=4)
parser.add_argument("--grad_accum",   type=int,   default=4)
parser.add_argument("--lr",           type=float, default=2e-4)
parser.add_argument("--lora_rank",    type=int,   default=16)
parser.add_argument("--seed",         type=int,   default=42)
parser.add_argument("--mtcnn_conf",   type=float, default=0.97, help="MTCNN confidence threshold")
parser.add_argument("--skip_mtcnn",   action="store_true", help="Skip MTCNN filtering (use all images)")
args = parser.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
OUTPUT_DIR = Path(args.output_dir); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MTCNN face filter ─────────────────────────────────────────────────────────
mtcnn = MTCNN(keep_all=False, device=DEVICE, min_face_size=20,
              thresholds=[0.6, 0.7, 0.8], post_process=False)

def has_valid_face(pil_img):
    if args.skip_mtcnn:
        return True
    try:
        boxes, probs = mtcnn.detect(pil_img)
        if boxes is None: return False
        best = int(np.argmax(probs))
        return probs[best] >= args.mtcnn_conf
    except:
        return False

# ── Load HF dataset ───────────────────────────────────────────────────────────
print(f"Loading {args.hf_dataset}...", flush=True)
hf_ds = load_dataset(args.hf_dataset, split="train")
print(f"  {len(hf_ds)} images, filtering faces...", flush=True)

samples = []  # list of (PIL.Image, label_str)
skipped = 0
for row in hf_ds:
    img = row["image"]
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img).convert("RGB")
    else:
        img = img.convert("RGB")
    label = row["label"]
    if has_valid_face(img):
        samples.append((img, label))
    else:
        skipped += 1

print(f"  HF dataset: {len(samples)} passed MTCNN ({skipped} skipped)", flush=True)

# ── Load LeCun from local pool ─────────────────────────────────────────────────
lecun_dir = Path(args.celeb_dir) / "Yann LeCun"
lecun_added = 0
if lecun_dir.exists():
    for p in list(lecun_dir.glob("*.jpg")) + list(lecun_dir.glob("*.png")):
        try:
            img = Image.open(p).convert("RGB")
            if has_valid_face(img):
                samples.append((img, "Yann LeCun"))
                lecun_added += 1
        except:
            pass
    print(f"  LeCun from local pool: {lecun_added} images", flush=True)
else:
    print(f"  WARNING: LeCun dir not found at {lecun_dir}", flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
from collections import Counter
label_counts = Counter(s[1] for s in samples)
print(f"\nTotal: {len(samples)} images, {len(label_counts)} identities", flush=True)
for name in ["Taylor Swift", "Barack Obama", "Yann LeCun"]:
    print(f"  {name}: {label_counts.get(name, 0)}", flush=True)
print(f"  Other celebs: {len(label_counts)-3}", flush=True)

random.shuffle(samples)

# ── Dataset ───────────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return img, label


def collate_fn(batch, processor, device, dtype):
    images, labels = zip(*batch)
    all_input_ids, all_attention_mask, all_pixel_values, all_labels_t = [], [], [], []

    for img, label in zip(images, labels):
        q = "Who is this person?"
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": q}
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        full   = prompt + label

        enc = processor(text=full, images=[img], return_tensors="pt",
                        truncation=True, max_length=2048)
        enc_prompt = processor(text=prompt, images=[img], return_tensors="pt",
                               truncation=True, max_length=2048)

        input_ids      = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        pixel_values   = enc["pixel_values"][0]
        prompt_len     = enc_prompt["input_ids"].shape[1]
        lbl            = input_ids.clone(); lbl[:prompt_len] = -100

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_pixel_values.append(pixel_values)
        all_labels_t.append(lbl)

    max_len   = max(t.shape[0] for t in all_input_ids)
    max_tiles = max(pv.shape[0] for pv in all_pixel_values)
    pad_id    = processor.tokenizer.pad_token_id or 0

    def pad(ts, val):
        return torch.stack([torch.cat([t, torch.full((max_len - t.shape[0],), val, dtype=t.dtype)])
                            for t in ts])

    padded_pv = []
    for pv in all_pixel_values:
        if pv.shape[0] < max_tiles:
            pad_tiles = torch.zeros(max_tiles - pv.shape[0], *pv.shape[1:], dtype=pv.dtype)
            pv = torch.cat([pv, pad_tiles], dim=0)
        padded_pv.append(pv)

    return {
        "input_ids":      pad(all_input_ids, pad_id).to(device),
        "attention_mask": pad(all_attention_mask, 0).to(device),
        "pixel_values":   torch.stack(padded_pv).to(device, dtype=dtype),
        "labels":         pad(all_labels_t, -100).to(device),
    }

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"\nLoading {args.model_id}...", flush=True)
processor = AutoProcessor.from_pretrained(args.model_id)
base = SmolVLMForConditionalGeneration.from_pretrained(
    args.model_id, dtype=DTYPE, device_map=DEVICE)

lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, lora_cfg)
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

# ── WandB ─────────────────────────────────────────────────────────────────────
wandb.init(
    entity="nagyungecole-eth-z-rich",
    project="smolvla-eval3",
    name="phase1-v2-103celebs",
    config=vars(args),
)

# ── Training ──────────────────────────────────────────────────────────────────
dataset = FaceDataset(samples)
loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                     collate_fn=lambda b: collate_fn(b, processor, DEVICE, DTYPE))

optimizer   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, weight_decay=0.01)
total_steps = (len(loader) // args.grad_accum) * args.epochs

def lr_lambda(step):
    warmup = 50
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"\nTraining: {args.epochs} epochs | {len(loader)} batches/epoch | "
      f"grad_accum={args.grad_accum} | output={OUTPUT_DIR}\n", flush=True)

model.train()
global_step = 0
for epoch in range(args.epochs):
    epoch_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        loss = model(**batch).loss / args.grad_accum
        loss.backward()
        epoch_loss += loss.item() * args.grad_accum

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            global_step += 1
            if global_step % 20 == 0:
                avg = epoch_loss / (step + 1)
                lr_now = scheduler.get_last_lr()[0]
                print(f"  epoch={epoch+1} step={global_step} loss={avg:.4f} lr={lr_now:.2e}", flush=True)
                wandb.log({"train/loss": avg, "train/lr": lr_now, "train/epoch": epoch+1},
                          step=global_step)

    avg = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}/{args.epochs} — avg loss: {avg:.4f}", flush=True)
    wandb.log({"epoch/avg_loss": avg, "train/epoch": epoch+1})
    ckpt = OUTPUT_DIR / f"epoch_{epoch+1}"
    model.save_pretrained(ckpt); processor.save_pretrained(ckpt)
    print(f"  Saved: {ckpt}", flush=True)

# ── Final + quick smoke test ───────────────────────────────────────────────────
final = OUTPUT_DIR / "final"
model.save_pretrained(final); processor.save_pretrained(final)
print(f"\nFinal checkpoint: {final}", flush=True)

print("\n=== Smoke check (3 random images per target celeb) ===", flush=True)
model.eval()
TARGET_CELEBS = ["Taylor Swift", "Barack Obama", "Yann LeCun"]
correct = total = 0
with torch.no_grad():
    for celeb in TARGET_CELEBS:
        celeb_imgs = [(img, lbl) for img, lbl in samples if lbl == celeb]
        test_imgs  = random.sample(celeb_imgs, min(5, len(celeb_imgs)))
        for img, lbl in test_imgs:
            messages = [{"role": "user", "content": [{"type": "image"},
                         {"type": "text", "text": "Who is this person?"}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
            ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            ans = processor.decode(ids[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
            ok = lbl.lower() in ans.lower()
            correct += int(ok); total += 1
            print(f"  [{'OK' if ok else 'XX'}] {lbl} -> {ans}", flush=True)

print(f"\nSmoke: {correct}/{total} = {correct/total*100:.1f}%", flush=True)
wandb.log({"smoke/accuracy": correct/total})
wandb.finish()
