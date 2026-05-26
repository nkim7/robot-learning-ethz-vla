from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RUN1_SCRIPT_DIR = ROOT_DIR / "scripts" / "eval3_run1_action_train"
DATASET_REPO_ID = "robot-learning-group47/eval3_90_plus_lmr_210"
DATASET_DIR = ROOT_DIR / "data" / "eval3_90_plus_lmr_210"
VLM_LORA_DIR = ROOT_DIR / "checkpoint" / "eval3_phase2_TOY"
EXPECTED_BASE_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
OUTPUT_DIR = ROOT_DIR / "outputs" / "train" / "eval3_run2_90pluslmr210_action_expert_frozen_vlm_lora_30k"
OUTPUT_REPO_ID = "robot-learning-group47/eval3_run2_90pluslmr210_action_expert_frozen_vlm_lora_smolvla"
WANDB_PROJECT = "robot-learning-group47-eval3"
WANDB_NAME = "eval3_run2_90pluslmr210_action_expert_frozen_vlm_lora_30k_lr3e5"


def configure_writable_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", "/tmp/hf-home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf-datasets-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf-home/hub")


def ensure_run1_patch_importable() -> None:
    run1_dir = str(RUN1_SCRIPT_DIR)
    if run1_dir not in sys.path:
        sys.path.insert(0, run1_dir)


def read_adapter_config(path: Path = VLM_LORA_DIR) -> dict:
    adapter_config = path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"Missing LoRA adapter config: {adapter_config}")
    return json.loads(adapter_config.read_text())


def require_vlm_lora() -> dict:
    if not VLM_LORA_DIR.exists():
        raise FileNotFoundError(f"Missing VLM LoRA checkpoint directory: {VLM_LORA_DIR}")
    adapter = read_adapter_config(VLM_LORA_DIR)
    base = adapter.get("base_model_name_or_path")
    if base != EXPECTED_BASE_MODEL:
        raise RuntimeError(
            f"Unexpected adapter base model: {base!r}. Expected {EXPECTED_BASE_MODEL!r}."
        )
    weights = VLM_LORA_DIR / "adapter_model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"Missing LoRA adapter weights: {weights}")
    return adapter


def lerobot_train_executable() -> str | None:
    on_path = shutil.which("lerobot-train")
    if on_path:
        return on_path
    in_venv = ROOT_DIR / ".venv" / "bin" / "lerobot-train"
    if in_venv.exists():
        return str(in_venv)
    return None


def die(message: str, code: int = 1) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)
