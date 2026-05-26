#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "robot-learning-group47/eval3_lora_fixed_Chengming"
FOLDER = Path("/home/shadeform/outputs/vlm_lora/eval3_phase2_TOY_position_qa_3frames")


def main() -> None:
    if not FOLDER.exists():
        raise FileNotFoundError(FOLDER)

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=str(FOLDER),
        path_in_repo=".",
        commit_message="Add fixed Eval3 grounding LoRA adapter",
    )
    print(f"Uploaded {FOLDER} to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
