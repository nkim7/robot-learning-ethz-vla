#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


SYSTEM = (
    "You answer questions about a robot tabletop scene with exactly three bowls: blue, red, and green. "
    "Use the image to identify the left, middle, and right bowls. "
    "Answer with only one word: blue, red, or green."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe base SmolVLM2 on Eval 2 bowl VQA examples.")
    parser.add_argument("--data", type=Path, default=Path("data/eval2_vqa_fresh/vqa.jsonl"))
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/eval2_vqa_fresh/zero_shot_results.json"))
    return parser.parse_args()


def parse_color(text: str) -> str:
    text = text.strip().lower()
    for color in ("blue", "red", "green"):
        if re.search(rf"\b{color}\b", text):
            return color
    return ""


def resolve_image_path(path: str, data_path: Path) -> Path:
    image_path = Path(path)
    if image_path.exists():
        return image_path

    candidates = [
        Path("permutation_init_position") / image_path.name,
        data_path.parent.parent.parent / "permutation_init_position" / image_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return image_path


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    rows = sorted(rows, key=lambda r: (r["split"], r["layout"], r["type"], r["question"]))
    if args.limit:
        rows = rows[: args.limit]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"loading {args.model} on {device} dtype={dtype} rows={len(rows)}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if args.adapter is not None:
        from peft import PeftModel

        print(f"loading adapter {args.adapter}")
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.to(device)
    model.eval()

    correct = 0
    by_type = defaultdict(lambda: [0, 0])
    by_layout = defaultdict(lambda: [0, 0])
    by_target = defaultdict(lambda: [0, 0])
    confusion = Counter()
    examples = []
    predictions = []
    start = time.time()

    with torch.no_grad():
        for idx, row in enumerate(rows, start=1):
            image = Image.open(resolve_image_path(row["image"], args.data)).convert("RGB")
            prompt = f"{SYSTEM}\n\nQuestion: {row['question']}"
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            raw = processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            pred = parse_color(raw)
            gt = row["target"]
            ok = pred == gt

            correct += int(ok)
            by_type[row["type"]][0] += int(ok)
            by_type[row["type"]][1] += 1
            by_layout[row["layout"]][0] += int(ok)
            by_layout[row["layout"]][1] += 1
            by_target[gt][0] += int(ok)
            by_target[gt][1] += 1
            confusion[(gt, pred or "<none>")] += 1
            pred_row = {
                "layout": row["layout"],
                "type": row["type"],
                "question": row["question"],
                "target": gt,
                "prediction": pred or "<none>",
                "raw": raw.strip(),
                "ok": ok,
            }
            predictions.append(pred_row)
            if len(examples) < 30 and (not ok or idx <= 12):
                examples.append(pred_row)
            if idx % 25 == 0:
                print(f"progress {idx}/{len(rows)} acc={correct / idx:.3f}")

    elapsed = time.time() - start
    result = {
        "model": args.model,
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "elapsed_s": elapsed,
        "by_type": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]} for k, v in by_type.items()},
        "by_layout": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]} for k, v in by_layout.items()},
        "by_target": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]} for k, v in by_target.items()},
        "confusion": {f"{gt}->{pred}": n for (gt, pred), n in confusion.items()},
        "examples": examples,
        "predictions": predictions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    print("\nRESULT")
    print(f"overall {correct}/{len(rows)} = {correct / len(rows):.3f} elapsed={elapsed:.1f}s")
    for name, bucket in (("by_type", by_type), ("by_layout", by_layout), ("by_target", by_target)):
        print(name)
        for key, (c, n) in sorted(bucket.items()):
            print(f"  {key:9s} {c:3d}/{n:3d} = {c / n:.3f}")
    print("examples")
    for row in examples:
        mark = "OK " if row["ok"] else "BAD"
        print(
            f"{mark} layout={row['layout']} type={row['type']} "
            f"gt={row['target']} pred={row['prediction']} q={row['question']} raw={row['raw']!r}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
