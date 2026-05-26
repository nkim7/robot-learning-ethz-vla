#!/usr/bin/env bash
# Downloads final policy checkpoints from HuggingFace into models/
# Skips any model whose local directory already contains model.safetensors
# or adapter_model.safetensors.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

python3 - << 'PYEOF'
from huggingface_hub import snapshot_download
from pathlib import Path

models = [
    ("robot-learning-group47/eval1_final_policy",    "models/eval1",       "model.safetensors"),
    ("robot-learning-group47/eval2_final_policy",    "models/eval2",       "model.safetensors"),
    ("robot-learning-group47/eval3_cotrain",         "models/eval3",       "model.safetensors"),
    ("robot-learning-group47/eval3_phase2_TOY",      "models/eval3/TOY",   "adapter_model.safetensors"),
    ("robot-learning-group47/eval3_phase2_celeb100", "models/eval3/celeb", "adapter_model.safetensors"),
]

for repo_id, local_dir, sentinel in models:
    sentinel_path = Path(local_dir) / sentinel
    if sentinel_path.exists():
        print(f"  ✓ Already present — skipping {local_dir}")
        continue
    print(f"Downloading {repo_id} → {local_dir} ...")
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local_dir)
    print(f"  ✓ Done")
PYEOF
