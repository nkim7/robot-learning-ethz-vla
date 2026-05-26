"""
Evaluate Phase 2 model on the generated benchmark.
Loads: phase1 (frozen) + phase2 adapter from checkpoint.
Reports per-question-type and overall accuracy.
"""
import json, re, torch
from pathlib import Path
from collections import defaultdict
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

DEVICE = "cuda"
DTYPE  = torch.bfloat16
CKPT   = "./checkpoints/phase2_scenes/epoch_6"
BENCH  = "./data/phase2_benchmark.jsonl"

print(f"Loading model from {CKPT}...")
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
base = SmolVLMForConditionalGeneration.from_pretrained(
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct", dtype=DTYPE, device_map=DEVICE)
model = PeftModel.from_pretrained(base, f"{CKPT}/phase1", adapter_name="phase1")
for _, p in model.named_parameters():
    if "phase1" in _: p.requires_grad = False
model.load_adapter(f"{CKPT}/phase2", adapter_name="phase2")
model.set_adapter("phase2")
model.eval()
print("Model loaded.\n")

with open(BENCH) as f:
    samples = [json.loads(l) for l in f]
print(f"Benchmark: {len(samples)} QA pairs from {len(set(s['image'] for s in samples))} images\n")

correct = total = 0
by_type = defaultdict(lambda: [0, 0])  # {type: [correct, total]}
wrong_examples = []

with torch.no_grad():
    for i, s in enumerate(samples):
        img   = Image.open(s["image"]).convert("RGB")
        q     = s["question"]
        exp   = s["answer"]
        qtype = "WHERE" if q.startswith("Where") else "WHO"

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        prompt   = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs   = processor(text=prompt, images=[img], return_tensors="pt").to(DEVICE)
        ids      = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        ans      = processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        ok = exp.lower() in ans.lower()
        correct        += int(ok)
        total          += 1
        by_type[qtype][0] += int(ok)
        by_type[qtype][1] += 1

        if not ok and len(wrong_examples) < 10:
            wrong_examples.append({"img": Path(s["image"]).name, "q": q, "exp": exp, "got": ans})

        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{len(samples)}] running acc: {correct/(i+1):.4f}", flush=True)

print(f"\n{'='*55}")
print(f"  Overall: {correct}/{total} = {correct/total:.4f} ({correct/total*100:.2f}%)")
for qt, (c, t) in sorted(by_type.items()):
    print(f"  {qt:6s}: {c}/{t} = {c/t:.4f} ({c/t*100:.2f}%)")
print(f"{'='*55}")

if wrong_examples:
    print(f"\nFirst {len(wrong_examples)} wrong answers:")
    for w in wrong_examples:
        print(f"  {w['img']}")
        print(f"    Q: {w['q']}")
        print(f"    Expected: {w['exp']}  ->  Got: {w['got']}")
