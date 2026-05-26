#!/usr/bin/env bash
# download_datasets.sh — fetch training datasets from HuggingFace Hub
#
# Usage:
#   ./download_datasets.sh          # download all three
#   ./download_datasets.sh eval1    # download only eval1
#   ./download_datasets.sh eval2
#   ./download_datasets.sh eval3
#
# Requires: pip install huggingface_hub
#           hf auth login  (or set HF_TOKEN env var)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

download_dataset() {
    local eval_tag="$1"
    local repo_id="$2"
    local local_dir="$3"

    echo "▶ Downloading ${repo_id} → ${local_dir}/"
    mkdir -p "${REPO_ROOT}/${local_dir}"
    python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${repo_id}",
    repo_type="dataset",
    local_dir="${REPO_ROOT}/${local_dir}",
)
print("  ✓ Done")
PYEOF
}

TARGET="${1:-all}"

# Eval 1 — fixed banana/bowl dataset (108-episode clean split)
if [[ "$TARGET" == "all" || "$TARGET" == "eval1" ]]; then
    download_dataset eval1 \
        robot-learning-group47/eval1_60_fixed_banana_fixed_bowl \
        training_pipeline/eval1/data
fi

# Eval 2 — fixed bowl / fixed banana compositional dataset (108 episodes)
if [[ "$TARGET" == "all" || "$TARGET" == "eval2" ]]; then
    download_dataset eval2 \
        robot-learning-group47/eval2_fixed_bowl_fixed_banana_108 \
        training_pipeline/eval2/data
fi

# Eval 3 — LMR 210-episode permutation dataset
if [[ "$TARGET" == "all" || "$TARGET" == "eval3" ]]; then
    download_dataset eval3 \
        robot-learning-group47/eval3_lmr_210 \
        training_pipeline/eval3/data
fi

echo ""
echo "✓ All requested datasets ready in training_pipeline/eval*/data/"
