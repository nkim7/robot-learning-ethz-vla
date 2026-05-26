#!/usr/bin/env python3
from __future__ import annotations

from huggingface_hub import snapshot_download

from hf_common import (
    CHECKPOINT_DIR,
    DATA_DIR,
    DATASET_DIR,
    DATASET_REPO_ID,
    MODEL_DIR,
    MODEL_REPO_ID,
    login_with_env_token,
    print_tree,
    require_hf_token,
)


def main() -> None:
    token = require_hf_token()
    login_with_env_token(token)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model repo {MODEL_REPO_ID} -> {MODEL_DIR}")
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        local_dir=MODEL_DIR,
        token=token,
    )

    print(f"Downloading dataset repo {DATASET_REPO_ID} -> {DATASET_DIR}")
    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=DATASET_DIR,
        token=token,
    )

    print_tree(MODEL_DIR)
    print_tree(DATASET_DIR)


if __name__ == "__main__":
    main()
