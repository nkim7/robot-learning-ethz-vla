from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

MODEL_REPO_ID = "robot-learning-group47/eval3_phase2_TOY"
DATASET_REPO_ID = "robot-learning-group47/eval3_90_permutation"

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoint"
MODEL_DIR = CHECKPOINT_DIR / "eval3_phase2_TOY"
DATASET_DIR = DATA_DIR / "eval3_90_permutation"


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "ERROR: HF_TOKEN is missing. Export it before running this script, e.g.:\n"
            "  export HF_TOKEN=hf_...\n"
            "The token is read from the environment and is never written to these scripts.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return token


def login_with_env_token(token: str) -> None:
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def safe_repo_dir(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id.strip("/"))


def print_tree(path: Path, *, max_files: int = 200) -> None:
    if not path.exists():
        print(f"{path} does not exist")
        return

    files: list[Path] = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            files.append(item)

    print(f"Files under {path}:")
    for item in files[:max_files]:
        print(f"  {item.relative_to(path)}")
    if len(files) > max_files:
        print(f"  ... {len(files) - max_files} more files")


def has_any(path: Path, patterns: Iterable[str]) -> bool:
    return any(path.glob(pattern) for pattern in patterns)


def looks_like_remote_repo_id(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", value))
