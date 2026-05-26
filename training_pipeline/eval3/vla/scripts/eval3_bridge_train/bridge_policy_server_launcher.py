#!/usr/bin/env python3
"""Policy server for bridge-trained SmolVLA (end-to-end celebrity prompts).

Loads:
  - Bridge action checkpoint (set via EVAL3_VLM_LORA_PATH and server --pretrained_name_or_path)
  - Cycle-05 VLM LoRA frozen (EVAL3_VLM_LORA_PATH)

No instruction rewrite: the bridge-trained action expert handles celebrity prompts directly.
Policy-reuse patch is active so the model is not reloaded between episodes.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_root_dir = _this_dir.parents[1]
_run1_dir = _root_dir / "scripts" / "eval3_run1_action_train"
_run2_dir = _root_dir / "scripts" / "eval3_run2_lmr210_action_train"

for _d in [str(_this_dir), str(_run1_dir), str(_run2_dir)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


def apply_policy_reuse_patch() -> None:
    """Reuse the loaded policy across episodes without reloading weights."""
    from lerobot.async_inference import policy_server as policy_server_module
    from lerobot.async_inference.constants import SUPPORTED_POLICIES
    from lerobot.async_inference.helpers import RemotePolicyConfig
    from lerobot.transport import services_pb2

    policy_server_class = policy_server_module.PolicyServer

    if getattr(policy_server_class, "_eval3_policy_reuse_patch", False):
        return

    original = policy_server_class.SendPolicyInstructions

    def send_policy_instructions_with_reuse(self, request, context):  # noqa: N802
        policy_specs = pickle.loads(request.data)  # nosec
        if not isinstance(policy_specs, RemotePolicyConfig):
            return original(self, request, context)

        cache_key = (
            policy_specs.policy_type,
            policy_specs.pretrained_name_or_path,
            policy_specs.device,
            tuple(sorted(policy_specs.rename_map.items())),
        )

        if (
            self.policy is not None
            and getattr(self, "_eval3_cached_policy_key", None) == cache_key
            and policy_specs.policy_type in SUPPORTED_POLICIES
        ):
            client_id = context.peer()
            self.logger.info(
                f"Reusing cached policy for {client_id} | "
                f"Pretrained: {policy_specs.pretrained_name_or_path}"
            )
            self.device = policy_specs.device
            self.policy_type = policy_specs.policy_type
            self.lerobot_features = policy_specs.lerobot_features
            self.actions_per_chunk = policy_specs.actions_per_chunk
            return services_pb2.Empty()

        response = original(self, request, context)
        self._eval3_cached_policy_key = cache_key
        return response

    policy_server_class.SendPolicyInstructions = send_policy_instructions_with_reuse
    policy_server_class._eval3_policy_reuse_patch = True


def main() -> None:
    os.environ.setdefault("HF_HOME", "/tmp/hf-home")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf-datasets-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/tmp/hf-home/hub")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("EVAL3_EXPECTED_VLM_BASE", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    os.environ.setdefault("EVAL3_RUN1_PATCH_ACTIVE", "1")
    os.environ.setdefault("EVAL3_VLM_LORA_TRAINABLE", "0")

    from run1_lora_patch import apply_run1_vlm_lora_patch
    apply_run1_vlm_lora_patch()

    apply_policy_reuse_patch()

    from lerobot.async_inference.policy_server import serve
    serve()


if __name__ == "__main__":
    main()
