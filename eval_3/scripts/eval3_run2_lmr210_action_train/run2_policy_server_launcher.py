#!/usr/bin/env python3
from __future__ import annotations

import os
import pickle

from run2_common import configure_writable_hf_cache, ensure_run1_patch_importable


def apply_instruction_rewrite_patch() -> None:
    """Rewrite person-conditioned task instructions using the VLM probe answer.

    The action expert was trained on position-conditioned instructions like
    "Put the coke on right." as well as person-conditioned ones like
    "Place the coke on Barack Obama." When deployed with a person-conditioned
    instruction, the VLM's KV cache does not reliably encode spatial position
    because the LoRA was fine-tuned for Q&A, not for enriching task embeddings.

    Fix: after the first-frame probe determines position ("left"/"right"/"middle"),
    rewrite the task string to the position-conditioned format the action expert
    was trained on ("Put the coke on right."), so the VLM KV cache carries an
    explicit spatial signal the action expert can act on.

    Controlled by env var EVAL3_INSTRUCTION_REWRITE (default "1").
    """
    from lerobot.async_inference import policy_server as policy_server_module

    policy_server_class = policy_server_module.PolicyServer

    if getattr(policy_server_class, "_eval3_instruction_rewrite_patch", False):
        return

    original_probe = policy_server_class._eval3_run_first_frame_probe
    original_predict = policy_server_class._predict_action_chunk

    def probe_and_store(self, observation):
        original_probe(self, observation)
        # Retrieve the answer that was just written to the log by reading it from
        # the normalized result stored inside the probe method itself.  We replay
        # the normalisation on the last generated text that was logged.  Instead
        # of re-running generation, we store the result directly here.

    # Replace the probe so it stores the answer on self._eval3_position_answer.
    def patched_probe(self, observation):
        if self._eval3_first_frame_probe_done:
            return
        if os.environ.get("EVAL3_FIRST_FRAME_PROBE", "1").lower() in {"0", "false", "no"}:
            self._eval3_first_frame_probe_done = True
            return

        self._eval3_first_frame_probe_done = True
        image_items = [(key, value) for key, value in observation.items() if "image" in key]
        if not image_items:
            self.logger.warning("Eval3 first-frame VLM probe skipped: no image key in observation")
            return

        import re
        import torch
        from PIL import Image

        image_key, image_tensor = image_items[0]
        prompt = self._eval3_probe_prompt()

        try:
            vlm_with_expert = self.policy.model.vlm_with_expert
            processor = vlm_with_expert.processor
            vlm = vlm_with_expert.vlm
            device = getattr(vlm, "device", None) or next(vlm.parameters()).device
            dtype = next((p.dtype for p in vlm.parameters() if p.is_floating_point()), torch.float32)

            pil_image = self._eval3_tensor_to_pil(image_tensor)
            text = self._eval3_build_probe_text(processor, prompt)
            inputs = processor(text=[text], images=[[pil_image]], return_tensors="pt", padding=True)
            inputs = self._eval3_move_probe_inputs(inputs, device, dtype)

            with torch.inference_mode():
                generated = vlm.generate(**inputs, max_new_tokens=8, do_sample=False)

            prompt_len = inputs["input_ids"].shape[-1]
            raw_answer = processor.batch_decode(
                generated[:, prompt_len:], skip_special_tokens=True
            )[0].strip()
            answer = self._eval3_normalize_probe_answer(raw_answer)
            self._eval3_position_answer = answer  # store for instruction rewrite
            message = (
                f"EVAL3 FIRST FRAME VLM PROBE | image={image_key} | question={prompt!r} | "
                f"answer={answer} | raw={raw_answer!r}"
            )
            self.logger.warning(message)
            print(message, flush=True)
        except Exception as exc:
            self._eval3_position_answer = None
            self.logger.exception(f"Eval3 first-frame VLM probe failed: {exc}")

    def _extract_person_from_task(task: str):
        import re
        m = re.match(r"Place the coke on (.+)\.$", task.strip())
        return m.group(1) if m else None

    def patched_predict(self, observation_t):
        from lerobot.async_inference.helpers import raw_observation_to_observation
        import time

        start_prepare = time.perf_counter()
        observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )

        # Set probe person from task before running probe
        task = observation.get("task", "")
        person = _extract_person_from_task(task)
        if person:
            self._eval3_probe_person_override = person
        else:
            self._eval3_probe_person_override = None

        patched_probe(self, observation)

        rewrite_enabled = os.environ.get("EVAL3_INSTRUCTION_REWRITE", "1").lower() not in {"0", "false", "no"}
        position = getattr(self, "_eval3_position_answer", None)
        # Only rewrite person-conditioned tasks; skip if already position-conditioned
        if rewrite_enabled and position and position != "unknown" and person:
            original_task = observation["task"]
            rewritten_task = f"Put the coke on {position}."
            if original_task != rewritten_task:
                observation["task"] = rewritten_task
                if not getattr(self, "_eval3_rewrite_logged", False):
                    self._eval3_rewrite_logged = True
                    self.logger.warning(
                        f"EVAL3 INSTRUCTION REWRITE | original={original_task!r} | rewritten={rewritten_task!r}"
                    )

        prepare_time = time.perf_counter() - start_prepare

        import time as _time
        start_preprocess = _time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs = observation_t
        preprocessing_time = _time.perf_counter() - start_preprocess

        start_inference = _time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = _time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        import torch
        start_postprocess = _time.perf_counter()
        _, chunk_size, _ = action_tensor.shape
        processed_actions = []
        for i in range(chunk_size):
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")
        action_tensor = action_tensor.detach().cpu()

        from lerobot.async_inference.helpers import TimedAction
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = _time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )
        return action_chunk

    def patched_probe_prompt(self):
        person = getattr(self, "_eval3_probe_person_override", None) or self._eval3_probe_person()
        return (
            f"Place the Coke on {person}, there are three celebrity cards in this robot scene, "
            f"arranged in the left, middle, and right slots. "
            f"Which slot is {person} in? "
            f"Answer with exactly one word: left, center, or right."
        )

    policy_server_class._eval3_probe_prompt = patched_probe_prompt
    policy_server_class._eval3_run_first_frame_probe = patched_probe
    policy_server_class._predict_action_chunk = patched_predict
    policy_server_class._eval3_instruction_rewrite_patch = True


def apply_policy_reuse_patch() -> None:
    from lerobot.async_inference import policy_server as policy_server_module
    from lerobot.async_inference.constants import SUPPORTED_POLICIES
    from lerobot.async_inference.helpers import RemotePolicyConfig
    from lerobot.transport import services_pb2

    policy_server_class = policy_server_module.PolicyServer
    original = policy_server_class.SendPolicyInstructions

    if getattr(policy_server_class, "_eval3_policy_reuse_patch", False):
        return

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
                f"Policy type: {policy_specs.policy_type} | "
                f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
                f"Actions per chunk: {policy_specs.actions_per_chunk} | "
                f"Device: {policy_specs.device}"
            )
            self.device = policy_specs.device
            self.policy_type = policy_specs.policy_type
            self.lerobot_features = policy_specs.lerobot_features
            self.actions_per_chunk = policy_specs.actions_per_chunk
            self._eval3_first_frame_probe_done = False
            self._eval3_position_answer = None
            self._eval3_rewrite_logged = False
            return services_pb2.Empty()

        response = original(self, request, context)
        self._eval3_cached_policy_key = cache_key
        return response

    policy_server_class.SendPolicyInstructions = send_policy_instructions_with_reuse
    policy_server_class._eval3_policy_reuse_patch = True


def main() -> None:
    configure_writable_hf_cache()
    ensure_run1_patch_importable()
    os.environ.setdefault("EVAL3_RUN1_PATCH_ACTIVE", "1")

    from run1_lora_patch import apply_run1_vlm_lora_patch

    apply_run1_vlm_lora_patch()
    apply_policy_reuse_patch()
    apply_instruction_rewrite_patch()

    from lerobot.async_inference.policy_server import serve

    serve()


if __name__ == "__main__":
    main()
