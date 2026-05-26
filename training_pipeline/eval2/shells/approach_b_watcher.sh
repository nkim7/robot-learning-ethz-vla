#!/bin/bash
# Waits for PID 2706572 to finish, then runs Approach B weight surgery
# on the best Eval 1 SmolVLA checkpoint and pushes to HF.

LOG=/home/shadeform/approach_b_watcher.log
CKPT_BASE=/home/shadeform/outputs/eval1_smolvla/checkpoints
SCRIPT=/home/shadeform/smoke_approach_b.py
HF_REPO=robot-learning-group47/smolvla-eval2-approach-b

echo "[$(date +%H:%M:%S)] Watcher started. Waiting for PID 2706572 to finish..." >> $LOG

while kill -0 2706572 2>/dev/null; do
    echo "  [$(date +%H:%M:%S)] PID 2706572 still running..." >> $LOG
    sleep 60
done

echo "[$(date +%H:%M:%S)] PID 2706572 done. Finding best Eval 1 checkpoint..." >> $LOG

# Best checkpoint = highest numbered directory
BEST_CKPT=$(ls -d $CKPT_BASE/[0-9]* 2>/dev/null | sort -V | tail -1)
echo "[$(date +%H:%M:%S)] Best checkpoint: $BEST_CKPT" >> $LOG

# Write the final weight surgery script
cat > /home/shadeform/approach_b_final.py << 'PYEOF'
import sys, torch, shutil, json
from pathlib import Path
from safetensors.torch import load_file, save_file

CKPT_DIR = Path(sys.argv[1]) / "pretrained_model"
VQA_REPO = "robot-learning-group47/smolvlm2-500m-vqa-lora-merged"
OUT_DIR  = Path("/home/shadeform/approach_b_final")
HF_REPO  = "robot-learning-group47/smolvla-eval2-approach-b"

OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Checkpoint: {CKPT_DIR}", flush=True)

print("Step 1: Loading Eval 1 expert weights...", flush=True)
ckpt_weights = load_file(str(CKPT_DIR / "model.safetensors"), device="cpu")
expert_keys  = {k: v for k, v in ckpt_weights.items() if "vlm_with_expert.vlm" not in k}
vlm_keys     = {k for k in ckpt_weights if "vlm_with_expert.vlm" in k}
print(f"  Expert tensors: {len(expert_keys)}  VLM tensors: {len(vlm_keys)}", flush=True)

print("Step 2: Loading merged VQA backbone from HF...", flush=True)
from huggingface_hub import snapshot_download
vqa_local  = snapshot_download(VQA_REPO)
vqa_weights = load_file(f"{vqa_local}/model.safetensors", device="cpu")
vlm_prefix = "model.vlm_with_expert.vlm."
remapped   = {f"{vlm_prefix}{k}": v for k, v in vqa_weights.items() if f"{vlm_prefix}{k}" in vlm_keys}
missing    = vlm_keys - set(remapped.keys())
print(f"  Remapped: {len(remapped)}  Missing: {len(missing)}", flush=True)
if missing:
    raise RuntimeError(f"Missing VLM keys: {list(missing)[:3]}")

print("Step 3: Merging and saving...", flush=True)
new_weights = {**expert_keys, **remapped}
assert len(new_weights) == len(ckpt_weights), f"{len(new_weights)} vs {len(ckpt_weights)}"
save_file(new_weights, str(OUT_DIR / "model.safetensors"))

for f in CKPT_DIR.iterdir():
    if f.name != "model.safetensors":
        shutil.copy2(f, OUT_DIR / f.name)

cfg_path = OUT_DIR / "config.json"
with open(cfg_path) as fh: cfg = json.load(fh)
cfg["vlm_model_name"] = VQA_REPO
with open(cfg_path, "w") as fh: json.dump(cfg, fh, indent=2)
print(f"  Saved {len(new_weights)} tensors to {OUT_DIR}", flush=True)

print(f"Step 4: Pushing to {HF_REPO}...", flush=True)
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(HF_REPO, exist_ok=True, private=False)
api.upload_folder(
    folder_path=str(OUT_DIR),
    repo_id=HF_REPO,
    commit_message=f"Approach B: best Eval1 expert + VQA merged backbone ({sys.argv[1].split('/')[-1]})"
)
print(f"Done: https://huggingface.co/{HF_REPO}", flush=True)
PYEOF

echo "[$(date +%H:%M:%S)] Running Approach B weight surgery on $BEST_CKPT..." >> $LOG
python3 /home/shadeform/approach_b_final.py "$BEST_CKPT" >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] Approach B weight surgery complete." >> $LOG
