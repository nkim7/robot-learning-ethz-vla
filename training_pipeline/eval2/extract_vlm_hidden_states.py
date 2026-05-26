#!/usr/bin/env python3
"""Extract frozen SmolVLM hidden states for Stage 1B target-slot probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, SmolVLMForConditionalGeneration


MODEL_ID = "robot-learning-group47/smolvlm2-500m-vqa-init-position-strict10-color-slot-merged"
SLOTS = ("left", "middle", "right")
SLOT_TO_ID = {slot: i for i, slot in enumerate(SLOTS)}


def build_chat(processor, image: Image.Image, text: str) -> tuple[str, Image.Image]:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), image


def pool_features(hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, image_ids: set[int]):
    mask = attention_mask.bool()
    image_mask = mask & torch.zeros_like(mask, dtype=torch.bool)
    for image_id in image_ids:
        image_mask |= input_ids == image_id
    text_mask = mask & ~image_mask

    def mean_pool(pool_mask: torch.Tensor) -> torch.Tensor:
        denom = pool_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (hidden * pool_mask.unsqueeze(-1)).sum(dim=1) / denom

    text_lengths = text_mask.long().sum(dim=1) - 1
    text_lengths = text_lengths.clamp(min=0)
    last_positions = []
    for row, length in enumerate(text_lengths.tolist()):
        positions = torch.where(text_mask[row])[0]
        last_positions.append(positions[length] if len(positions) else torch.where(mask[row])[0][-1])
    last_positions = torch.stack(last_positions).to(hidden.device)

    last_text = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_positions]
    mean_text = mean_pool(text_mask)
    mean_image = mean_pool(image_mask) if image_mask.any() else torch.zeros_like(mean_text)
    action_consumed = torch.cat([last_text, mean_text, mean_image], dim=-1)
    return {
        "last_text_token": last_text.cpu(),
        "mean_text_tokens": mean_text.cpu(),
        "mean_image_tokens": mean_image.cpu(),
        "action_consumed_features": action_consumed.cpu(),
    }


@torch.inference_mode()
def extract_condition(model, processor, rows: list[dict], condition: str, device: str, batch_size: int):
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    fake_image_token_id = processor.tokenizer.convert_tokens_to_ids("<fake_token_around_image>")
    image_ids = {int(image_token_id), int(fake_image_token_id)}
    features = {"last_text_token": [], "mean_text_tokens": [], "mean_image_tokens": [], "action_consumed_features": []}
    labels = []
    metadata = []
    prompt_key = "bare_prompt" if condition == "bare" else "scaffold_prompt"
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = []
        images = []
        for row in batch:
            text, image = build_chat(processor, Image.open(row["image"]).convert("RGB"), row[prompt_key])
            texts.append(text)
            images.append([image])
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
        pooled = pool_features(
            outputs.hidden_states[-1].float(),
            inputs["input_ids"],
            inputs["attention_mask"],
            image_ids,
        )
        for key, value in pooled.items():
            features[key].append(value)
        labels.extend(SLOT_TO_ID[row["target_slot"]] for row in batch)
        metadata.extend(
            {
                k: row[k]
                for k in (
                    "image",
                    "orig_episode_index",
                    "episode_index",
                    "layout_id",
                    "layout_code",
                    "phase",
                    "bare_prompt",
                    "scaffold_prompt",
                    "prompt_type",
                    "layout_left_to_right",
                    "target_color",
                    "target_slot",
                    "split",
                    "combo",
                )
            }
            for row in batch
        )
        print(f"{condition}: {min(start + batch_size, len(rows)):4d}/{len(rows)}", end="\r", flush=True)
    print()
    return {
        "features": {key: torch.cat(value, dim=0) for key, value in features.items()},
        "labels": torch.tensor(labels, dtype=torch.long),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="data/stage1b_probe/stage1b_probe.jsonl")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--out", default="data/stage1b_probe/hidden_states.pt")
    parser.add_argument("--conditions", nargs="+", default=["scaffold", "bare"], choices=["scaffold", "bare"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.jsonl).read_text().splitlines() if line.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = SmolVLMForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        _attn_implementation="eager",
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    payload = {"model": args.model, "jsonl": args.jsonl, "conditions": {}}
    for condition in args.conditions:
        payload["conditions"][condition] = extract_condition(model, processor, rows, condition, device, args.batch_size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"Saved hidden states to {out}")


if __name__ == "__main__":
    main()
