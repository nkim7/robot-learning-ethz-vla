from __future__ import annotations

import logging
import os
from pathlib import Path

from peft import PeftConfig, PeftModel

from cotrain_common import EXPECTED_BASE_MODEL, VLM_LORA_DIR, require_vlm_lora

_PATCHED = False


def _unwrap_vlm_model(vlm):
    if hasattr(vlm, "get_base_model"):
        base = vlm.get_base_model()
    else:
        base = vlm
    return base.model


def apply_run1_vlm_lora_patch() -> None:
    """Load the Eval 3 VLM LoRA into SmolVLA's VLM and freeze it.

    LeRobot's SmolVLA CLI has flags for freezing the VLM and training the action
    expert/state projection, but it does not have a first-class flag for loading
    an external PEFT adapter into only the VLM submodule. This patch keeps the
    upstream trainer intact and changes only SmolVLA model construction.
    """
    global _PATCHED
    if _PATCHED:
        return

    adapter_path = Path(os.environ.get("EVAL3_VLM_LORA_PATH", str(VLM_LORA_DIR))).resolve()
    expected_base = os.environ.get("EVAL3_EXPECTED_VLM_BASE", EXPECTED_BASE_MODEL)
    trainable_lora = os.environ.get("EVAL3_VLM_LORA_TRAINABLE", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    lora_lr = float(os.environ.get("EVAL3_VLM_LORA_LR", "1e-6"))
    require_vlm_lora()
    peft_config = PeftConfig.from_pretrained(adapter_path)
    if peft_config.base_model_name_or_path != expected_base:
        raise RuntimeError(
            "Refusing to load VLM LoRA because adapter_config.json declares "
            f"{peft_config.base_model_name_or_path!r}, expected {expected_base!r}."
        )

    from lerobot.policies.smolvla import smolvlm_with_expert
    from lerobot.policies.smolvla import modeling_smolvla
    import lerobot.policies.factory as policy_factory

    original_init = smolvlm_with_expert.SmolVLMWithExpertModel.__init__
    original_make_policy = policy_factory.make_policy
    original_get_optim_params = modeling_smolvla.SmolVLAPolicy.get_optim_params

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        logging.info(
            "Loading %s Eval 3 VLM LoRA adapter from %s",
            "trainable" if trainable_lora else "frozen",
            adapter_path,
        )
        self.vlm = PeftModel.from_pretrained(
            self.vlm,
            adapter_path,
            config=peft_config,
            is_trainable=trainable_lora,
        )
        self.vlm.eval()
        if trainable_lora:
            for name, param in self.vlm.named_parameters():
                param.requires_grad = "lora_" in name or ".lora" in name
        else:
            for param in self.vlm.parameters():
                param.requires_grad = False
        self.vlm_lora_adapter_path = str(adapter_path)
        self.vlm_lora_frozen = not trainable_lora
        self.vlm_lora_lr = lora_lr if trainable_lora else None

    def patched_get_vlm_model(self):
        return _unwrap_vlm_model(self.vlm)

    def patched_make_policy(cfg, *args, **kwargs):
        if getattr(cfg, "type", None) == "smolvla" and getattr(cfg, "use_peft", False):
            logging.info(
                "Run 1 requested policy.use_peft=true. Interpreting this as the external "
                "VLM LoRA patch, not a new LeRobot policy-level PEFT adapter."
            )
            cfg.use_peft = False
        return original_make_policy(cfg, *args, **kwargs)

    def patched_get_optim_params(self):
        if not trainable_lora:
            return original_get_optim_params(self)

        lora_params = []
        other_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "vlm" in name and ("lora_" in name or ".lora" in name):
                lora_params.append(param)
            else:
                other_params.append(param)

        groups = []
        if other_params:
            groups.append({"params": other_params, "name": "action_state"})
        if lora_params:
            groups.append({"params": lora_params, "lr": lora_lr, "name": "vlm_lora"})
        logging.info(
            "Eval 3 optimizer groups: action/state params=%d, VLM LoRA params=%d, VLM LoRA lr=%s",
            sum(p.numel() for p in other_params),
            sum(p.numel() for p in lora_params),
            lora_lr,
        )
        return groups

    smolvlm_with_expert.SmolVLMWithExpertModel.__init__ = patched_init
    smolvlm_with_expert.SmolVLMWithExpertModel.get_vlm_model = patched_get_vlm_model
    modeling_smolvla.SmolVLAPolicy.get_optim_params = patched_get_optim_params
    policy_factory.make_policy = patched_make_policy
    _PATCHED = True
