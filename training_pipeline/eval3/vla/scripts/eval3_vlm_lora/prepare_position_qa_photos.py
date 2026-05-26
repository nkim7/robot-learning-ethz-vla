#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image


def target_person_for_episode(episode_index: int) -> str:
    if 0 <= episode_index <= 29:
        return "Taylor Swift"
    if 30 <= episode_index <= 59:
        return "Barack Obama"
    if 60 <= episode_index <= 89:
        return "Yann LeCun"
    raise ValueError(f"Unsupported episode index: {episode_index}")


def ground_truth_for_episode(episode_index: int) -> str:
    offset = episode_index % 30
    if offset < 10:
        return "left"
    if offset < 20:
        return "middle"
    return "right"


def prompt_for_person(person: str) -> str:
    return f"Is {person} on the left, middle, or right? Answer only one word: left, middle, or right."


def read_sampled_frames(video_path: Path, seconds: float) -> list[tuple[str, int, float, Image.Image]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 15.0
    max_frames = int(seconds * fps + 1e-6)

    frames = []
    for frame_index in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_s = frame_index / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((frame_index, timestamp_s, Image.fromarray(rgb)))
    cap.release()

    if not frames:
        raise RuntimeError(f"Could not read frames from {video_path}")

    picks = [0, len(frames) // 2, len(frames) - 1]
    names = ["start", "middle", "end"]
    return [(name, *frames[index]) for name, index in zip(names, picks, strict=True)]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/home/shadeform/data/eval3_90_permutation"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/shadeform/data/eval3_position_qa_3frames"))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for episode in range(90):
        person = target_person_for_episode(episode)
        answer = ground_truth_for_episode(episode)
        prompt = prompt_for_person(person)
        video_path = args.dataset / "videos" / "observation.images.front" / "chunk-000" / f"file-{episode:03d}.mp4"

        for sample_name, frame_index, timestamp_s, image in read_sampled_frames(video_path, args.seconds):
            image_path = image_dir / f"episode_{episode:03d}_{sample_name}_frame_{frame_index:03d}.jpg"
            image.save(image_path, quality=95)
            rows.append(
                {
                    "image": str(image_path),
                    "episode_index": episode,
                    "sample": sample_name,
                    "frame_index": frame_index,
                    "timestamp_s": round(timestamp_s, 6),
                    "target_person": person,
                    "ground_truth": answer,
                    "prompt": prompt,
                    "answer": answer,
                }
            )

    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["target_person"], row["ground_truth"])].append(row)

    train_rows = []
    eval_rows = []
    for group_rows in groups.values():
        rng.shuffle(group_rows)
        eval_count = round(len(group_rows) * args.eval_ratio)
        eval_rows.extend(group_rows[:eval_count])
        train_rows.extend(group_rows[eval_count:])

    train_rows.sort(key=lambda row: (row["episode_index"], row["frame_index"]))
    eval_rows.sort(key=lambda row: (row["episode_index"], row["frame_index"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "eval.jsonl", eval_rows)
    write_jsonl(args.output_dir / "all.jsonl", rows)

    summary = {
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "seconds": args.seconds,
        "seed": args.seed,
        "total": len(rows),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "ground_truth_source": "user-provided episode blocks",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
