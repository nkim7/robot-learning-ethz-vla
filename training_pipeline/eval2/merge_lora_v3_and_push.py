#!/usr/bin/env python3
"""
Merge smolvlm3 VQA LoRA into smolvlm2 base weights, save locally, push to HF.

Same pattern as merge_lora_and_push.py (smolvlm2).

smolvlm2 base:  robot-learning-group47/smolvlm2-500m-vqa-lora-merged
LoRA adapter:   models/smolvlm3_vqa_lora/best
Target repo:    robot-learning-group47/smolvlm3-500m-vqa-lora-merged
"""
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

ADAPTER  = Path("models/smolvlm3_vqa_lora/best")
BASE     = "robot-learning-group47/smolvlm2-500m-vqa-lora-merged"
SAVE_DIR = Path("models/smolvlm3_500m_vqa_merged")
HF_REPO  = "robot-learning-group47/smolvlm3-500m-vqa-lora-merged"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("Step 1/4  Loading smolvlm2 base (float32, CPU)...")
base = AutoModelForImageTextToText.from_pretrained(
    BASE,
    torch_dtype=torch.float32,
    device_map="cpu",
)

print("Step 2/4  Loading smolvlm3 LoRA adapter and merging weights...")
model = PeftModel.from_pretrained(base, str(ADAPTER))
model = model.merge_and_unload()
model.eval()

print(f"Step 3/4  Saving merged model to {SAVE_DIR} ...")
model.save_pretrained(str(SAVE_DIR))

processor = AutoProcessor.from_pretrained(str(ADAPTER))
processor.save_pretrained(str(SAVE_DIR))

print(f"Step 4/4  Pushing to HuggingFace: {HF_REPO} ...")
model.push_to_hub(
    HF_REPO, private=False,
    commit_message="Merge smolvlm3 VQA v3 LoRA (r=32 α=64) into smolvlm2 — adds position outputs"
)
processor.push_to_hub(
    HF_REPO, private=False,
    commit_message="Add processor for smolvlm3"
)

print(f"\nDone! Model live at: https://huggingface.co/{HF_REPO}")
