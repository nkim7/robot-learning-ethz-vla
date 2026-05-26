from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

# Resolved at runtime via EVAL3_VLM_LORA_PATH env var; this is the default
VLM_LORA_DIR = Path(
    os.environ.get(
        "EVAL3_VLM_LORA_PATH",
        str(ROOT_DIR / "ot" / "checkpoints" / "eval3_cotrain" / "cycle_05_vlm_lora"),
    )
).resolve()

EXPECTED_BASE_MODEL = os.environ.get(
    "EVAL3_EXPECTED_VLM_BASE",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
)


def configure_writable_hf_cache() -> None:
    os.environ.setdefault("HF_HOME", "/tmp/hf-home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf-datasets-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf-home/hub")


def read_adapter_config(path: Path = VLM_LORA_DIR) -> dict:
    adapter_config = path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"Missing LoRA adapter config: {adapter_config}")
    return json.loads(adapter_config.read_text())


def require_vlm_lora() -> dict:
    if not VLM_LORA_DIR.exists():
        raise FileNotFoundError(
            f"Missing VLM LoRA checkpoint directory: {VLM_LORA_DIR}\n"
            "Set EVAL3_VLM_LORA_PATH to override."
        )
    adapter = read_adapter_config(VLM_LORA_DIR)
    base = adapter.get("base_model_name_or_path")
    if base != EXPECTED_BASE_MODEL:
        raise RuntimeError(
            f"Adapter declares base model {base!r}, expected {EXPECTED_BASE_MODEL!r}.\n"
            "Set EVAL3_EXPECTED_VLM_BASE to override."
        )
    weights = VLM_LORA_DIR / "adapter_model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"Missing LoRA adapter weights: {weights}")
    return adapter
