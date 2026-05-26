#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "robot-learning-group47/eval3_TOY_video_images"
FOLDER = Path("/home/shadeform/data/eval3_position_qa_3frames")


def main() -> None:
    if not FOLDER.exists():
        raise FileNotFoundError(FOLDER)

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(FOLDER),
        path_in_repo=".",
        commit_message="Add Eval3 TOY video image grounding dataset",
    )
    print(f"Uploaded {FOLDER} to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
