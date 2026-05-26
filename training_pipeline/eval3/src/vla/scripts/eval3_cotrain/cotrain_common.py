from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

BASE_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

START_ACTION_CHECKPOINT = (
    ROOT_DIR
    / "outputs"
    / "train"
    / "eval3_run2_90pluslmr210_action_expert_frozen_vlm_lora_30k"
    / "checkpoints"
    / "014000"
    / "pretrained_model"
)
START_VLM_LORA = ROOT_DIR / "checkpoint" / "eval3_phase2_TOY"

ROBOT_DATASET_ROOT = ROOT_DIR / "data" / "eval3_90_plus_lmr_210"
ACTION_TOY_DATASET_ROOT = ROOT_DIR / "data" / "eval3_90_permutation"
ACTION_LMR_DATASET_ROOT = ROOT_DIR / "data" / "eval3_lmr"
PREPARED_ACTION_DATASET_ROOT = ROOT_DIR / "data" / "eval3_cotrain_action_90toy_plus_lmr"
ROBOT_DATASET_REPO_ID = "robot-learning-group47/eval3_cotrain_action_90toy_plus_lmr"

GROUNDING_DATA_DIR = ROOT_DIR / "data" / "eval3_position_qa_3frames"

OUTPUT_DIR = (
    ROOT_DIR / "outputs" / "train" / "eval3_cotrain_action_lora_auxvlm_5k"
)
HF_REPO_ID = "robot-learning-group47/eval3_cotrain_action_lora_auxvlm_5k_smolvla"

DEFAULT_CYCLES = 5
DEFAULT_ACTION_STEPS = 500
DEFAULT_VLM_STEPS = 100


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def action_checkpoint_for_cycle(cycle_dir: Path, action_steps: int) -> Path:
    return cycle_dir / "checkpoints" / f"{action_steps:06d}" / "pretrained_model"
