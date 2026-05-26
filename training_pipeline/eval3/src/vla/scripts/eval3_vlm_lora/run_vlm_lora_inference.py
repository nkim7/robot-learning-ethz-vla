#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoVisionTextModel
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoVisionTextModel

from hf_common import (
    CHECKPOINT_DIR,
    MODEL_DIR,
    has_any,
    looks_like_remote_repo_id,
    print_tree,
    require_hf_token,
    safe_repo_dir,
)
from inspect_checkpoint import FULL_WEIGHT_PATTERNS


def resolve_model_source(model_id_or_path: str, token: str) -> str:
    path = Path(model_id_or_path).expanduser()
    if path.exists():
        return str(path)

    if looks_like_remote_repo_id(model_id_or_path):
        from huggingface_hub import snapshot_download

        local_dir = CHECKPOINT_DIR / "base_models" / safe_repo_dir(model_id_or_path)
        print(f"Downloading base model {model_id_or_path} -> {local_dir}")
        snapshot_download(
            repo_id=model_id_or_path,
            repo_type="model",
            local_dir=local_dir,
            token=token,
        )
        return str(local_dir)

    return model_id_or_path


def classify(path: Path) -> str:
    if (path / "adapter_config.json").exists():
        return "adapter"
    if (path / "config.json").exists() and has_any(path, FULL_WEIGHT_PATTERNS):
        return "full"
    return "unclear"


def load_image(image_path: Path) -> Image.Image:
    if not image_path.exists():
        raise SystemExit(f"Image does not exist: {image_path}")
    return Image.open(image_path).convert("RGB")


def build_prompt(processor: AutoProcessor, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            return processor.apply_chat_template(messages, add_generation_prompt=True)
    return prompt


def input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def load_vision_text_model(source: str | Path, dtype: torch.dtype, device_map: str, trust_remote_code: bool):
    try:
        return AutoVisionTextModel.from_pretrained(
            source,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        return AutoVisionTextModel.from_pretrained(
            source,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )


def load_full_model(checkpoint: Path, dtype: torch.dtype, device_map: str, trust_remote_code: bool):
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    model = load_vision_text_model(checkpoint, dtype, device_map, trust_remote_code)
    return processor, model


def load_adapter_model(
    checkpoint: Path,
    token: str,
    dtype: torch.dtype,
    device_map: str,
    trust_remote_code: bool,
    base_model: str | None,
):
    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(checkpoint)
    base = base_model or peft_config.base_model_name_or_path
    if not base:
        raise SystemExit(
            "adapter_config.json was found, but no base model was provided and "
            "base_model_name_or_path is missing. Re-run with --base-model."
        )

    base_source = resolve_model_source(base, token)
    processor_source = str(checkpoint) if (checkpoint / "preprocessor_config.json").exists() else base_source
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=trust_remote_code)
    base_loaded = load_vision_text_model(base_source, dtype, device_map, trust_remote_code)
    model = PeftModel.from_pretrained(base_loaded, checkpoint)
    return processor, model


def move_inputs(inputs, device: torch.device, dtype: torch.dtype):
    if hasattr(inputs, "to"):
        try:
            return inputs.to(device, dtype=dtype)
        except TypeError:
            return inputs.to(device)

    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif torch.is_floating_point(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def build_inputs(processor: AutoProcessor, image_path: Path, image: Image.Image, prompt: str):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as exc:
            print(f"Falling back to manual processor call after chat template failed: {exc}")

    text = build_prompt(processor, prompt)
    return processor(text=text, images=[image], return_tensors="pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SmolVLM/LoRA image+prompt inference.")
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--base-model", default=None, help="Required if LoRA adapter lacks base_model_name_or_path.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    token = require_hf_token()
    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint path does not exist: {checkpoint}")

    kind = classify(checkpoint)
    if kind == "unclear":
        print_tree(checkpoint)
        raise SystemExit(
            "Unclear checkpoint layout. Missing adapter_config.json for LoRA, "
            "or config.json plus model weights for a full model."
        )

    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = getattr(torch, args.dtype)

    if kind == "adapter":
        print("Detected PEFT/LoRA adapter checkpoint.")
        with (checkpoint / "adapter_config.json").open() as f:
            adapter = json.load(f)
        if adapter.get("base_model_name_or_path"):
            print(f"Adapter base model: {adapter['base_model_name_or_path']}")
        processor, model = load_adapter_model(
            checkpoint,
            token,
            dtype,
            args.device_map,
            args.trust_remote_code,
            args.base_model,
        )
    else:
        print("Detected full fine-tuned model checkpoint.")
        processor, model = load_full_model(checkpoint, dtype, args.device_map, args.trust_remote_code)

    model.eval()
    image = load_image(args.image)
    device = input_device(model)
    inputs = build_inputs(processor, args.image, image, args.prompt)
    inputs = move_inputs(inputs, device, dtype)

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    new_tokens = generated[:, prompt_len:] if prompt_len else generated
    output = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    print(output.strip())


if __name__ == "__main__":
    main()
