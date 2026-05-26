#!/usr/bin/env python3
"""Applies VLM LoRA + first-frame probe + instruction-rewrite patches, then runs lerobot-record.

When EVAL3_INSTRUCTION_REWRITE=1 (default):
  - On the first frame of each episode, probes the VLM: "Which slot is <person> in? left/center/right"
  - Rewrites the task string from "Place the coke on Taylor Swift." -> "Put the coke on left."
  - This makes the action expert use a position-conditioned instruction it was trained on.
"""
import os
import re
import sys


def _extract_person(task: str) -> str | None:
    """Extract person name from 'Place the coke on <Person>.' task strings."""
    m = re.match(r"Place the coke on (.+)\.$", (task or "").strip())
    return m.group(1) if m else None


def _apply_patches() -> None:
    try:
        from run1_lora_patch import apply_run1_vlm_lora_patch  # type: ignore[import]
    except ImportError as e:
        print(f"ERROR: cannot import run1_lora_patch: {e}", file=sys.stderr)
        sys.exit(1)
    apply_run1_vlm_lora_patch()

    if os.environ.get("EVAL3_INSTRUCTION_REWRITE", "1").lower() not in {"0", "false", "no"}:
        _patch_predict_action_inline()


def _patch_predict_action_inline() -> None:
    """Patch the inline predict_action used by lerobot-record to add probe + rewrite."""
    from lerobot.common import control_utils

    original_predict_action = control_utils.predict_action

    # Per-episode state; reset via policy.reset() hook set up on first call.
    state: dict = {"first_frame_done": False, "position_answer": None,
                   "rewrite_logged": False, "reset_hooked": False}

    def _run_vlm_probe(policy, observation: dict, person: str) -> str | None:
        """Run one forward pass through the VLM to ask which slot the person is in."""
        import torch
        try:
            vlm_with_expert = policy.model.vlm_with_expert
            processor = vlm_with_expert.processor
            vlm = vlm_with_expert.vlm
            device = next(vlm.parameters()).device
            dtype = next((p.dtype for p in vlm.parameters() if p.is_floating_point()), torch.float32)

            img_arr = next((v for k, v in observation.items() if "image" in k), None)
            if img_arr is None:
                print("EVAL3 probe: no image key found in observation", file=sys.stderr)
                return None

            from PIL import Image
            pil_image = Image.fromarray(img_arr.astype("uint8"))

            prompt = (
                f"Place the Coke on {person}, there are three celebrity cards in this robot scene, "
                f"arranged in the left, middle, and right slots. "
                f"Which slot is {person} in? "
                f"Answer with exactly one word: left, center, or right."
            )
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[text], images=[[pil_image]], return_tensors="pt", padding=True)
            inputs = {k: (v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device))
                      for k, v in inputs.items()}

            with torch.inference_mode():
                generated = vlm.generate(**inputs, max_new_tokens=8, do_sample=False)

            prompt_len = inputs["input_ids"].shape[-1]
            raw = processor.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)[0].strip().lower()

            answer = {"center": "middle", "centre": "middle", "middle": "middle",
                      "left": "left", "right": "right"}.get(raw, raw)

            print(f"EVAL3 VLM PROBE | person={person!r} | answer={answer} | raw={raw!r}", flush=True)
            return answer
        except Exception as exc:
            print(f"EVAL3 VLM probe failed: {exc}", file=sys.stderr)
            return None

    def patched_predict_action(observation, policy, device, preprocessor, postprocessor,
                               use_amp, task=None, robot_type=None):
        # Hook policy.reset() once so per-episode state resets between episodes.
        if not state["reset_hooked"]:
            state["reset_hooked"] = True
            _orig_reset = policy.reset

            def _hooked_reset():
                state["first_frame_done"] = False
                state["position_answer"] = None
                state["rewrite_logged"] = False
                return _orig_reset()

            policy.reset = _hooked_reset

        person = _extract_person(task)

        # First frame of episode: run VLM probe.
        if not state["first_frame_done"] and person:
            state["first_frame_done"] = True
            if os.environ.get("EVAL3_FIRST_FRAME_PROBE", "1").lower() not in {"0", "false", "no"}:
                state["position_answer"] = _run_vlm_probe(policy, observation, person)

        # Rewrite task if we have a position answer.
        pos = state["position_answer"]
        if pos and pos not in ("unknown", None) and person:
            rewritten = f"Put the coke on {pos}."
            if task != rewritten:
                if not state["rewrite_logged"]:
                    state["rewrite_logged"] = True
                    print(f"EVAL3 REWRITE | {task!r} -> {rewritten!r}", flush=True)
                task = rewritten

        return original_predict_action(observation, policy, device, preprocessor, postprocessor,
                                       use_amp, task, robot_type)

    # Patch both the module-level function and the already-imported reference in lerobot_record.
    control_utils.predict_action = patched_predict_action
    try:
        from lerobot.scripts import lerobot_record
        lerobot_record.predict_action = patched_predict_action
    except Exception:
        pass


if __name__ == "__main__":
    _apply_patches()
    from lerobot.scripts.lerobot_record import main  # type: ignore[import]
    main()
