"""
Phase 2: LoRA fine-tuning of SmolVLM2 for scene-level position recognition.

Changes vs v1:
  - --resume_phase2: resume from a saved epoch_N checkpoint (loads phase1+phase2 adapters)
  - --start_epoch:   skip already-completed epochs (0-indexed)
  - --stop_accuracy: benchmark eval after each epoch; stop when accuracy >= threshold
  - --bench_dir:     directory with v6_scene_*.jpg benchmark images
  - flush=True on all prints so logs aren't buffered

Task:  "Where is [celebrity]? Answer LEFT, CENTER, or RIGHT."  ->  "LEFT"
       "Who is on the LEFT card? Give the full name."          ->  "Taylor Swift"
"""

import argparse, json, math, random, re, sys
from pathlib import Path

import torch
import wandb
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--manifest",      default="./data/phase2_manifest.jsonl")
parser.add_argument("--output_dir",    default="./checkpoints/phase2")
parser.add_argument("--resume",        default=None, help="Phase1 adapter path (frozen)")
parser.add_argument("--resume_phase2", default=None, help="Checkpoint dir with phase1/ + phase2/ to resume from")
parser.add_argument("--start_epoch",   type=int, default=0, help="Skip this many epochs (0-indexed, set to N to start at epoch N+1)")
parser.add_argument("--model_id",      default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
parser.add_argument("--epochs",        type=int,   default=8)
parser.add_argument("--warmup_steps",  type=int,   default=30)
parser.add_argument("--batch_size",    type=int,   default=4)
parser.add_argument("--grad_accum",    type=int,   default=4)
parser.add_argument("--lr",            type=float, default=2e-4)
parser.add_argument("--lora_rank",     type=int,   default=16)
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--stop_accuracy", type=float, default=0.998, help="Stop training when benchmark >= this")
parser.add_argument("--bench_dir",     default="./data/composite_test")
args = parser.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if DEVICE == "cuda" else torch.float32
OUTPUT_DIR = Path(args.output_dir); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

run_name = "phase2-scene-lora"
if args.resume_phase2:
    run_name += f"-resume-e{args.start_epoch}"
elif args.resume:
    run_name += "-from-p1"

wandb.init(
    entity="nagyungecole-eth-z-rich",
    project="smolvla-eval2",
    name=run_name,
    config=vars(args),
)

# ── Dataset ───────────────────────────────────────────────────────────────────
class Phase2Dataset(Dataset):
    def __init__(self, manifest_path):
        with open(manifest_path) as f:
            self.samples = [json.loads(l) for l in f]
        random.shuffle(self.samples)
        print(f"Phase2Dataset: {len(self.samples)} QA pairs", flush=True)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image"]).convert("RGB")
        return img, s["question"], s["answer"]


def collate_fn(batch, processor, device, dtype):
    images, questions, answers = zip(*batch)
    all_input_ids, all_attention_mask, all_pixel_values, all_labels = [], [], [], []

    for img, q, a in zip(images, questions, answers):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": q}
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        full   = prompt + a

        enc = processor(text=full, images=[img], return_tensors="pt",
                        truncation=True, max_length=2048)
        enc_prompt = processor(text=prompt, images=[img], return_tensors="pt",
                               truncation=True, max_length=2048)

        input_ids      = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        pixel_values   = enc["pixel_values"][0]
        prompt_len     = enc_prompt["input_ids"].shape[1]
        labels         = input_ids.clone(); labels[:prompt_len] = -100

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_pixel_values.append(pixel_values)
        all_labels.append(labels)

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
        "labels":         pad(all_labels, -100).to(device),
    }


# ── Benchmark ─────────────────────────────────────────────────────────────────
SHORTNAME = {"Taylor": "Taylor Swift", "Barack": "Barack Obama", "Yann": "Yann LeCun"}

def run_benchmark(model, processor):
    bench_dir = Path(args.bench_dir)
    images = sorted(bench_dir.glob("v6_scene_*.jpg"))
    if not images:
        print("  [benchmark] No v6_scene_*.jpg found — skipping", flush=True)
        return None, 0, 0

    correct = total = 0
    model.eval()
    with torch.no_grad():
        for img_path in images:
            m = re.search(r"__L-(\w+)_C-(\w+)_R-(\w+)", img_path.stem)
            if not m:
                continue
            layout = {
                "LEFT":   SHORTNAME.get(m.group(1), m.group(1)),
                "CENTER": SHORTNAME.get(m.group(2), m.group(2)),
                "RIGHT":  SHORTNAME.get(m.group(3), m.group(3)),
            }
            rev = {v: k for k, v in layout.items()}
            img = Image.open(img_path).convert("RGB")

            qas = [
                ("Where is Taylor Swift? Answer LEFT, CENTER, or RIGHT.",  rev.get("Taylor Swift", "?")),
                ("Where is Barack Obama? Answer LEFT, CENTER, or RIGHT.",  rev.get("Barack Obama", "?")),
                ("Where is Yann LeCun? Answer LEFT, CENTER, or RIGHT.",    rev.get("Yann LeCun",   "?")),
                ("Who is on the LEFT card? Give the full name.",   layout["LEFT"]),
                ("Who is on the CENTER card? Give the full name.", layout["CENTER"]),
                ("Who is on the RIGHT card? Give the full name.",  layout["RIGHT"]),
            ]

            for q, expected in qas:
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
                ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
                ans = processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                ok = expected.lower() in ans.lower()
                correct += int(ok)
                total += 1

    acc = correct / total if total else 0.0
    model.train()
    return acc, correct, total


# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading {args.model_id}", flush=True)
processor = AutoProcessor.from_pretrained(args.model_id)

base = SmolVLMForConditionalGeneration.from_pretrained(
    args.model_id, dtype=DTYPE, device_map=DEVICE)

lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

if args.resume_phase2:
    p1 = str(Path(args.resume_phase2) / "phase1")
    p2 = str(Path(args.resume_phase2) / "phase2")
    print(f"Resuming phase2 from {args.resume_phase2}", flush=True)
    model = PeftModel.from_pretrained(base, p1, adapter_name="phase1")
    for name, param in model.named_parameters():
        if "phase1" in name:
            param.requires_grad = False
    model.load_adapter(p2, adapter_name="phase2")
    model.set_adapter("phase2")
    model.enable_adapter_layers()
elif args.resume:
    print(f"Loading phase1 adapter (frozen): {args.resume}", flush=True)
    model = PeftModel.from_pretrained(base, args.resume, adapter_name="phase1")
    for name, param in model.named_parameters():
        if "phase1" in name:
            param.requires_grad = False
    model.add_adapter("phase2", lora_cfg)
    model.set_adapter("phase2")
    model.enable_adapter_layers()
else:
    model = get_peft_model(base, lora_cfg)

model.print_trainable_parameters()

# ── Training ──────────────────────────────────────────────────────────────────
dataset = Phase2Dataset(args.manifest)
loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                     collate_fn=lambda b: collate_fn(b, processor, DEVICE, DTYPE))

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=0.01)
total_steps = (len(loader) // args.grad_accum) * args.epochs

def lr_lambda(step):
    if step < args.warmup_steps:
        return step / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print(f"\nTraining: epochs {args.start_epoch+1}-{args.epochs} | {len(loader)} batches/epoch | "
      f"grad_accum={args.grad_accum} | output={OUTPUT_DIR}\n", flush=True)

model.train()
global_step = args.start_epoch * (len(loader) // args.grad_accum)
# Advance scheduler to match start_epoch
for _ in range(global_step):
    scheduler.step()

stopped_early = False
for epoch in range(args.start_epoch, args.epochs):
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
                avg_loss = epoch_loss / (step + 1)
                lr_now   = scheduler.get_last_lr()[0]
                print(f"  epoch={epoch+1} step={global_step} loss={avg_loss:.4f} lr={lr_now:.2e}", flush=True)
                wandb.log({"train/loss": avg_loss, "train/lr": lr_now,
                           "train/epoch": epoch + 1}, step=global_step)

    avg = epoch_loss / len(loader)
    print(f"Epoch {epoch+1}/{args.epochs} — avg loss: {avg:.4f}", flush=True)
    wandb.log({"epoch/avg_loss": avg, "train/epoch": epoch + 1})
    ckpt = OUTPUT_DIR / f"epoch_{epoch+1}"
    model.save_pretrained(ckpt); processor.save_pretrained(ckpt)
    print(f"  Saved: {ckpt}", flush=True)

    # Benchmark
    print(f"  Running benchmark on {args.bench_dir}...", flush=True)
    acc, correct, total_q = run_benchmark(model, processor)
    if acc is not None:
        print(f"  Benchmark: {correct}/{total_q} = {acc:.4f} ({acc*100:.2f}%)", flush=True)
        wandb.log({"benchmark/accuracy": acc, "benchmark/correct": correct,
                   "benchmark/total": total_q, "train/epoch": epoch + 1})
        if acc >= args.stop_accuracy:
            print(f"\n>>> Reached {acc*100:.2f}% >= {args.stop_accuracy*100:.2f}% — stopping early. <<<", flush=True)
            stopped_early = True
            break

# Always save final
final = OUTPUT_DIR / "final"
model.save_pretrained(final); processor.save_pretrained(final)
if stopped_early:
    print(f"Early-stop checkpoint saved: {final}", flush=True)
else:
    print(f"\nFinal checkpoint: {final}", flush=True)

wandb.finish()
