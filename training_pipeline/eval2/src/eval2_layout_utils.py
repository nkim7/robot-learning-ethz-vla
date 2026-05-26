#!/usr/bin/env python3
"""Shared Eval 2 layout and prompt utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COLOR_NAMES = {"b": "blue", "g": "green", "r": "red"}
COLORS = ("blue", "red", "green")
SLOTS = ("left", "middle", "right")


def normalize_layout(value: str) -> list[str]:
    value = str(value).strip().lower()
    if "-" in value:
        parts = [p.strip() for p in value.split("-")]
    elif "," in value:
        parts = [p.strip() for p in value.split(",")]
    elif re.fullmatch(r"[bgr]{3}", value):
        parts = [COLOR_NAMES[ch] for ch in value]
    else:
        raise ValueError(f"Cannot parse layout: {value!r}")
    if sorted(parts) != ["blue", "green", "red"]:
        raise ValueError(f"Bad layout colors: {parts}")
    return parts


def layout_id(layout: list[str]) -> str:
    return "-".join(layout)


def target_slot(layout: list[str], target_color: str) -> str:
    return SLOTS[layout.index(target_color)]


def prompt_type(prompt: str) -> str:
    p = prompt.lower()
    if "not " in p:
        return "negation"
    if "left of" in p or "right of" in p:
        return "relative"
    if (
        "1st" in p
        or "2nd" in p
        or "3rd" in p
        or "first" in p
        or "second" in p
        or "third" in p
        or "middle bowl" in p
        or "center bowl" in p
    ):
        return "ordinal"
    if mentioned_colors(p):
        return "color"
    return "other"


def mentioned_colors(prompt: str) -> list[str]:
    return [c for c in COLORS if re.search(rf"\b{c}\b", prompt)]


def resolve_prompt_target_color(prompt: str, layout: list[str]) -> str:
    p = prompt.lower()
    mentioned = mentioned_colors(p)

    if "not " in p:
        excluded = {c for c in COLORS if re.search(rf"\bnot\s+{c}\b", p)}
        remaining = [c for c in COLORS if c not in excluded]
        if len(remaining) == 1:
            return remaining[0]

    if "left of" in p or "right of" in p:
        if not mentioned:
            raise ValueError(f"Relative prompt has no reference color: {prompt}")
        ref = mentioned[-1]
        ref_idx = layout.index(ref)
        if "left of" in p:
            idx = ref_idx - 1
        else:
            idx = ref_idx + 1
        if idx < 0 or idx >= len(layout):
            raise ValueError(f"Relative prompt points outside layout: {prompt=} {layout=}")
        return layout[idx]

    if "1st bowl from the left" in p or "first bowl from the left" in p or "3rd bowl from the right" in p:
        return layout[0]
    if "2nd bowl from the left" in p or "second bowl from the left" in p:
        return layout[1]
    if "2nd bowl from the right" in p or "second bowl from the right" in p:
        return layout[1]
    if "3rd bowl from the left" in p or "third bowl from the left" in p or "1st bowl from the right" in p:
        return layout[2]
    if "middle bowl" in p or "center bowl" in p:
        return layout[1]
    if "left bowl" in p:
        return layout[0]
    if "right bowl" in p:
        return layout[2]

    direct = mentioned_colors(p)
    if len(direct) == 1:
        return direct[0]
    raise ValueError(f"Cannot resolve target color from prompt: {prompt!r}")


@dataclass(frozen=True)
class EpisodeLayout:
    episode_index: int
    orig_episode_index: int
    layout_left_to_right: list[str]
    layout_id: str
    target_color: str
    target_slot: str


class Eval2ComboMap:
    def __init__(self, dataset_root: str | Path, total_episodes: int):
        root = Path(dataset_root)
        path = root / "combo_augmented_map.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing Eval2 combo map: {path}")
        self.df = pd.read_csv(path).sort_values("new_ep_start").reset_index(drop=True)
        self.records = []
        for i, row in self.df.iterrows():
            end = int(self.df.iloc[i + 1]["new_ep_start"]) if i + 1 < len(self.df) else total_episodes
            layout_code, target_code = str(row["combo"]).split("_")
            layout = normalize_layout(layout_code)
            color = COLOR_NAMES[target_code]
            self.records.append(
                {
                    "start": int(row["new_ep_start"]),
                    "end": end,
                    "orig_start": int(row["episode_start"]),
                    "layout": layout,
                    "target_color": color,
                }
            )

    def episode(self, episode_index: int) -> EpisodeLayout:
        for rec in self.records:
            if rec["start"] <= episode_index < rec["end"]:
                # The augmented dataset repeats prompt variants for each original trajectory.
                orig = int(rec["orig_start"] + ((episode_index - rec["start"]) % 6))
                layout = list(rec["layout"])
                color = str(rec["target_color"])
                return EpisodeLayout(
                    episode_index=episode_index,
                    orig_episode_index=orig,
                    layout_left_to_right=layout,
                    layout_id=layout_id(layout),
                    target_color=color,
                    target_slot=target_slot(layout, color),
                )
        raise ValueError(f"Episode {episode_index} not covered by combo_augmented_map.csv")
