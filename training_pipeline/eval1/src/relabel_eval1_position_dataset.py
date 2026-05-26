#!/usr/bin/env python3
"""Create a local position-labeled copy of the fixed Eval 1 LeRobot dataset."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


LABEL_MAP = {
    "Put the banana in the red colored bowl.": "Put the banana in the middle bowl.",
    "Put the banana in the blue colored bowl.": "Put the banana in the left bowl.",
    "Put the banana in the green colored bowl.": "Put the banana in the right bowl.",
}


def hardlink_copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        dst.mkdir(exist_ok=True)
        for child in src.iterdir():
            hardlink_copy(child, dst / child.name)
        return

    if src.is_symlink():
        target = os.readlink(src)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(target, dst)
        return

    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def relabel_tasks_table(path: Path) -> None:
    table = pq.read_table(path)
    data = table.to_pydict()
    data["task"] = [LABEL_MAP.get(task, task) for task in data["task"]]
    pq.write_table(pa.table(data, schema=table.schema), path)


def relabel_episode_file(path: Path) -> None:
    table = pq.read_table(path)
    data = table.to_pydict()
    data["tasks"] = [[LABEL_MAP.get(task, task) for task in tasks] for tasks in data["tasks"]]
    pq.write_table(pa.table(data, schema=table.schema), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.src.exists():
        raise FileNotFoundError(args.src)
    if args.dst.exists():
        if not args.force:
            raise FileExistsError(f"{args.dst} already exists; pass --force to replace it")
        shutil.rmtree(args.dst)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    hardlink_copy(args.src, args.dst)

    relabel_tasks_table(args.dst / "meta" / "tasks.parquet")
    for episode_file in sorted((args.dst / "meta" / "episodes").glob("chunk-*/*.parquet")):
        relabel_episode_file(episode_file)

    tasks = pq.read_table(args.dst / "meta" / "tasks.parquet").to_pydict()
    print("Wrote", args.dst)
    for idx, task in zip(tasks["task_index"], tasks["task"]):
        print(f"  {idx}: {task}")


if __name__ == "__main__":
    main()
