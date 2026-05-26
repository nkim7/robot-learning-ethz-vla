#!/usr/bin/env bash
# Downloads all final policy checkpoints from HuggingFace into models/
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

python3 - << 'PYEOF'
from huggingface_hub import snapshot_download

models = [
    ("robot-learning-group47/eval1_final_policy",    "models/eval1"),
    ("robot-learning-group47/eval2_final_policy",    "models/eval2"),
    ("robot-learning-group47/eval3_cotrain",         "models/eval3"),
    ("robot-learning-group47/eval3_phase2_TOY",      "models/eval3/TOY"),
    ("robot-learning-group47/eval3_phase2_celeb100", "models/eval3/celeb"),
]
for repo_id, local_dir in models:
    print(f"Downloading {repo_id} → {local_dir}")
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local_dir)
    print(f"  done")
PYEOF
