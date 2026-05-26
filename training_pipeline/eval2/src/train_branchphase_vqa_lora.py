#!/usr/bin/env python3
"""LoRA fine-tune SmolVLM2 on visible Eval 2 target-grounding frames.

Data source for the JSONL is robot-learning-group47/eval2_compositional_augmented.
The model is trained to answer target color and target slot from a single frame
and the original Eval 2 prompt. This is representation shaping only; final VLA
inference will not use a runtime resolver.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageEnhance
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, SmolVLMForConditionalGeneration

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCAFFOLD_SYSTEM = (
    "You are looking at a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Given the image and the original robot instruction, identify the target bowl. "
    "Reply with exactly two short lines, not JSON and not prose. "
    "The exact format is: "
    "target color: green\n"
    "target slot: right"
)


def maybe_augment(img: Image.Image, train: bool) -> Image.Image:
    if not train:
        return img
    brightness = random.uniform(0.88, 1.12)
    contrast = random.uniform(0.88, 1.12)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def parse_answer(text: str) -> dict | None:
    raw = text.strip()
    text = raw.lower()

    out: dict[str, str] = {}

    color_match = re.search(
        r"(?:target\s*)?color\s*(?:is|:|=)?\s*(blue|red|green)\b", text, re.I
    )
    slot_match = re.search(
        r"(?:target\s*)?slot\s*(?:is|:|=)?\s*(left|middle|right)\b", text, re.I
    )
    if color_match:
        out["target_color"] = color_match.group(1).lower()
    if slot_match:
        out["target_slot"] = slot_match.group(1).lower()

    # Backward-compatible parser for older JSON checkpoints or outputs.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
        except Exception:
            obj = None
        if isinstance(obj, dict):
            color = obj.get("target_color") or obj.get("color")
            slot = obj.get("target_slot") or obj.get("slot")
            if isinstance(color, str) and color.lower() in {"blue", "red", "green"}:
                out["target_color"] = color.lower()
            if isinstance(slot, str) and slot.lower() in {"left", "middle", "right"}:
                out["target_slot"] = slot.lower()

    if "target_color" not in out:
        color_words = re.findall(r"\b(blue|red|green)\b", text)
        unique_colors = list(dict.fromkeys(color_words))
        if len(unique_colors) == 1:
            out["target_color"] = unique_colors[0]
    if "target_slot" not in out:
        slot_words = re.findall(r"\b(left|middle|right)\b", text)
        unique_slots = list(dict.fromkeys(slot_words))
        if len(unique_slots) == 1:
            out["target_slot"] = unique_slots[0]

    if not out:
        return None
    return out


class BranchPhaseVQADataset(Dataset):
    def __init__(self, jsonl: Path, split: str):
        rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
        if split != "all":
            rows = [row for row in rows if row["split"] == split]
        self.rows = rows
        self.split = split

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def build_prompt(processor, prompt: str, prompt_mode: str = "scaffold") -> str:
    if prompt_mode == "bare":
        text = prompt
    elif prompt_mode == "scaffold":
        text = f"{SCAFFOLD_SYSTEM}\n\nInstruction: {prompt}"
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def collate_train(batch: list[dict], processor, device: str, dtype: torch.dtype, prompt_mode: str):
    input_ids_list, masks_list, labels_list, pixel_values_list = [], [], [], []
    for row in batch:
        image = Image.open(row["image"]).convert("RGB")
        image = maybe_augment(image, train=True)
        prompt = build_prompt(processor, row["prompt"], prompt_mode)
        full = prompt + row["answer"]
        enc = processor(text=full, images=[image], return_tensors="pt", truncation=True, max_length=2048)
        enc_prompt = processor(text=prompt, images=[image], return_tensors="pt", truncation=True, max_length=2048)

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        labels = input_ids.clone()
        labels[: enc_prompt["input_ids"].shape[1]] = -100
        input_ids_list.append(input_ids)
        masks_list.append(attention_mask)
        labels_list.append(labels)
        pixel_values_list.append(enc["pixel_values"][0])

    max_len = max(x.shape[0] for x in input_ids_list)
    max_tiles = max(x.shape[0] for x in pixel_values_list)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0

    def pad_1d(items, pad_value):
        return torch.stack(
            [torch.cat([x, torch.full((max_len - x.shape[0],), pad_value, dtype=x.dtype)]) for x in items]
        )

    padded_pv = []
    for pv in pixel_values_list:
        if pv.shape[0] < max_tiles:
            pad_tiles = torch.zeros(max_tiles - pv.shape[0], *pv.shape[1:], dtype=pv.dtype)
            pv = torch.cat([pv, pad_tiles], dim=0)
        padded_pv.append(pv)

    return {
        "input_ids": pad_1d(input_ids_list, pad_id).to(device),
        "attention_mask": pad_1d(masks_list, 0).to(device),
        "labels": pad_1d(labels_list, -100).to(device),
        "pixel_values": torch.stack(padded_pv).to(device, dtype=dtype),
    }


@torch.inference_mode()
def evaluate(
    model,
    processor,
    dataset: BranchPhaseVQADataset,
    device: str,
    max_items: int = 0,
    prompt_mode: str = "scaffold",
) -> dict:
    model.eval()
    rows = dataset.rows if max_items <= 0 else dataset.rows[:max_items]
    correct_layout = correct_color = correct_slot = correct_all = 0
    by_phase = defaultdict(lambda: {"all": [0, 0], "layout": [0, 0], "color": [0, 0], "slot": [0, 0]})
    by_prompt_type = defaultdict(lambda: {"all": [0, 0], "layout": [0, 0], "color": [0, 0], "slot": [0, 0]})
    by_split = defaultdict(lambda: {"all": [0, 0], "layout": [0, 0], "color": [0, 0], "slot": [0, 0]})
    by_target_slot = defaultdict(lambda: {"all": [0, 0], "layout": [0, 0], "color": [0, 0], "slot": [0, 0]})
    failures = []
    for i, row in enumerate(rows):
        image = Image.open(row["image"]).convert("RGB")
        prompt = build_prompt(processor, row["prompt"], prompt_mode)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        raw = processor.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        pred = parse_answer(raw)
        gt_layout = row["layout_left_to_right"]
        gt_color = row["target_color"]
        gt_slot = row["target_slot"]
        ok_layout = pred is not None and (
            "layout_left_to_right" not in pred or pred.get("layout_left_to_right") == gt_layout
        )
        ok_color = pred is not None and pred.get("target_color") == gt_color
        ok_slot = pred is not None and pred.get("target_slot") == gt_slot
        ok_all = ok_color and ok_slot and ok_layout
        correct_layout += int(ok_layout)
        correct_color += int(ok_color)
        correct_slot += int(ok_slot)
        correct_all += int(ok_all)
        for bucket, key in (
            (by_phase, row["phase"]),
            (by_prompt_type, row.get("prompt_type", "unknown")),
            (by_split, row["split"]),
            (by_target_slot, row["target_slot"]),
        ):
            bucket[key]["all"][0] += int(ok_all)
            bucket[key]["layout"][0] += int(ok_layout)
            bucket[key]["color"][0] += int(ok_color)
            bucket[key]["slot"][0] += int(ok_slot)
            for field in ("all", "layout", "color", "slot"):
                bucket[key][field][1] += 1
        if not ok_all and len(failures) < 50:
            failures.append({"row": i, "phase": row["phase"], "prompt": row["prompt"], "gt": row["answer"], "raw": raw})

    n = len(rows)
    model.train()
    return {
        "n": n,
        "layout_acc": correct_layout / n,
        "target_color_acc": correct_color / n,
        "target_slot_acc": correct_slot / n,
        "all_acc": correct_all / n,
        "by_phase": summarize_breakdown(by_phase),
        "by_prompt_type": summarize_breakdown(by_prompt_type),
        "by_split": summarize_breakdown(by_split),
        "by_target_slot": summarize_breakdown(by_target_slot),
        "failures": failures,
    }


def summarize_breakdown(bucket: dict) -> dict:
    out = {}
    for key, fields in sorted(bucket.items()):
        out[key] = {}
        for field, (correct, total) in fields.items():
            out[key][field] = {"correct": correct, "total": total, "acc": correct / total if total else 0.0}
    return out


def save_merged_model(model, processor, output_dir: Path, push_repo: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}", flush=True)
    if push_repo:
        merged.push_to_hub(push_repo)
        processor.push_to_hub(push_repo)
        print(f"Pushed merged model to {push_repo}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="/data/branchphase_vqa/branchphase_vqa.jsonl")
    parser.add_argument("--base-model", default="robot-learning-group47/smolvlm2-500m-vqa-lora-merged")
    parser.add_argument("--output-dir", default="/data/lerobot_outputs/smolvlm2_branchphase_vqa_lora")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--target-acc", type=float, default=0.997)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-mode", choices=["scaffold", "bare"], default="scaffold")
    parser.add_argument("--wandb-project", default="smolvlm2-branchphase-vqa")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-name", default="smolvlm2-500m-branchphase-color-slot")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--merge-output-dir", default="/data/lerobot_outputs/smolvlm2-500m-vqa-branchphase-color-slot-merged")
    parser.add_argument("--push-merged-repo", default="robot-learning-group47/smolvlm2-500m-vqa-branchphase-color-slot-merged")
    parser.add_argument("--no-push-merged", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_name,
            config=vars(args),
        )

    train_ds = BranchPhaseVQADataset(Path(args.jsonl), "train")
    val_ds = BranchPhaseVQADataset(Path(args.jsonl), "val")
    test_ds = BranchPhaseVQADataset(Path(args.jsonl), "test")
    print(f"Rows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    print(f"Loading {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.tokenizer.padding_side = "right"
    base = SmolVLMForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        _attn_implementation="eager",
    ).to(device)
    text_qv_modules = [
        f"model.text_model.layers.{i}.self_attn.{proj}"
        for i in range(base.config.text_config.num_hidden_layers)
        for proj in ("q_proj", "v_proj")
    ]
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=text_qv_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_train(batch, processor, device, dtype, args.prompt_mode),
    )
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, math.ceil(len(loader) / args.grad_accum) * args.epochs)
    warmup = min(50, max(5, total_steps // 10))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    best_acc = -1.0
    global_step = 0
    report_path = out_dir / "train_report.jsonl"
    model.train()

    for epoch in range(args.epochs):
        optim.zero_grad()
        running = 0.0
        for step, batch in enumerate(loader):
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            running += float(loss.item()) * args.grad_accum
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1
                if global_step % args.log_every == 0:
                    print(
                        f"epoch={epoch+1} step={global_step} "
                        f"loss={running / (step + 1):.4f} lr={sched.get_last_lr()[0]:.2e}",
                        flush=True,
                    )
                if global_step % args.eval_every == 0:
                    metrics = evaluate(model, processor, val_ds, device, prompt_mode=args.prompt_mode)
                    metrics.update({"epoch": epoch + 1, "step": global_step, "train_loss": running / (step + 1)})
                    with report_path.open("a") as f:
                        f.write(json.dumps(metrics) + "\n")
                    if wandb_run is not None:
                        import wandb

                        wandb.log(
                            {
                                "val/all_acc": metrics["all_acc"],
                                "val/layout_acc": metrics["layout_acc"],
                                "val/target_color_acc": metrics["target_color_acc"],
                                "val/target_slot_acc": metrics["target_slot_acc"],
                                "train/loss": metrics["train_loss"],
                                "train/epoch": metrics["epoch"],
                            },
                            step=global_step,
                        )
                    print(
                        f"VAL step={global_step}: all={metrics['all_acc']*100:.2f}% "
                        f"layout={metrics['layout_acc']*100:.2f}% "
                        f"color={metrics['target_color_acc']*100:.2f}% "
                        f"slot={metrics['target_slot_acc']*100:.2f}%",
                        flush=True,
                    )
                    if metrics["all_acc"] > best_acc:
                        best_acc = metrics["all_acc"]
                        best = out_dir / "best"
                        model.save_pretrained(best)
                        processor.save_pretrained(best)
                    if metrics["all_acc"] >= args.target_acc:
                        final = out_dir / "reached_997"
                        model.save_pretrained(final)
                        processor.save_pretrained(final)
                        test_metrics = (
                            evaluate(model, processor, test_ds, device, prompt_mode=args.prompt_mode)
                            if len(test_ds)
                            else {"n": 0}
                        )
                        (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
                        save_merged_model(
                            model,
                            processor,
                            Path(args.merge_output_dir),
                            "" if args.no_push_merged else args.push_merged_repo,
                        )
                        print(f"Reached target accuracy {metrics['all_acc']:.4f}; saved {final}", flush=True)
                        if wandb_run is not None:
                            wandb_run.finish()
                        return

        metrics = evaluate(model, processor, val_ds, device, prompt_mode=args.prompt_mode)
        metrics.update({"epoch": epoch + 1, "step": global_step, "train_loss": running / len(loader)})
        with report_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "val/all_acc": metrics["all_acc"],
                    "val/layout_acc": metrics["layout_acc"],
                    "val/target_color_acc": metrics["target_color_acc"],
                    "val/target_slot_acc": metrics["target_slot_acc"],
                    "train/loss": metrics["train_loss"],
                    "train/epoch": metrics["epoch"],
                },
                step=global_step,
            )
        print(f"EPOCH {epoch+1} VAL all={metrics['all_acc']*100:.2f}%", flush=True)
        if metrics["all_acc"] > best_acc:
            best_acc = metrics["all_acc"]
            best = out_dir / "best"
            model.save_pretrained(best)
            processor.save_pretrained(best)
        if metrics["all_acc"] >= args.target_acc:
            final = out_dir / "reached_997"
            model.save_pretrained(final)
            processor.save_pretrained(final)
            test_metrics = (
                evaluate(model, processor, test_ds, device, prompt_mode=args.prompt_mode)
                if len(test_ds)
                else {"n": 0}
            )
            (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
            save_merged_model(
                model,
                processor,
                Path(args.merge_output_dir),
                "" if args.no_push_merged else args.push_merged_repo,
            )
            print(f"Reached target accuracy {metrics['all_acc']:.4f}; saved {final}", flush=True)
            if wandb_run is not None:
                wandb_run.finish()
            return

    final = out_dir / "final"
    model.save_pretrained(final)
    processor.save_pretrained(final)
    test_metrics = (
        evaluate(model, processor, test_ds, device, prompt_mode=args.prompt_mode) if len(test_ds) else {"n": 0}
    )
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    save_merged_model(
        model,
        processor,
        Path(args.merge_output_dir),
        "" if args.no_push_merged else args.push_merged_repo,
    )
    print(f"Done without reaching target; best={best_acc:.4f}; final={final}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
