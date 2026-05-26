"""
Generate compositional prompt augmentation for eval2_fixed_bowl_fixed_banana_108.
For each (arrangement, target_color) combo, generates all valid compositional prompts
that correctly identify the target bowl.

Produces:
  - eval2_augmented_tasks.jsonl   — all unique tasks (prompt strings + task_index)
  - eval2_augmented_episodes.jsonl — episode → list of valid task_indices
  - eval2_prompt_map.json         — full human-readable map for inspection
"""

import json
from pathlib import Path
from itertools import combinations

# ── Combos from combo_episode_map ────────────────────────────────────────────
COMBOS = [
    # (combo_id, bowl_order_L2R, target_color, ep_start, ep_end)
    ("bgr_b", "bgr", "blue",  0,  5),
    ("bgr_g", "bgr", "green", 6,  11),
    ("bgr_r", "bgr", "red",  12, 17),
    ("brg_b", "brg", "blue",  18, 23),
    ("brg_g", "brg", "green", 24, 29),
    ("brg_r", "brg", "red",   30, 35),
    ("gbr_b", "gbr", "blue",  36, 41),
    ("gbr_g", "gbr", "green", 42, 47),
    ("gbr_r", "gbr", "red",   48, 53),
    ("grb_b", "grb", "blue",  54, 59),
    ("grb_g", "grb", "green", 60, 65),
    ("grb_r", "grb", "red",   66, 71),
    ("rbg_b", "rbg", "blue",  72, 77),
    ("rbg_g", "rbg", "green", 78, 83),
    ("rbg_r", "rbg", "red",   84, 89),
    ("rgb_b", "rgb", "blue",  90, 95),
    ("rgb_g", "rgb", "green", 96, 101),
    ("rgb_r", "rgb", "red",  102, 107),
]

COLORS = ["blue", "green", "red"]
ORDINAL_L = ["1st", "2nd", "3rd"]
ORDINAL_R = ["3rd", "2nd", "1st"]   # position 0 = 3rd from right

PREFIX = "Put the banana into the"
SUFFIX = "from the robot perspective."


def gen_prompts(arrangement: str, target: str) -> list[str]:
    """Return all valid compositional prompts for (arrangement, target)."""
    bowls = list(arrangement)  # e.g. ['b','g','r']
    color_map = {'b': 'blue', 'g': 'green', 'r': 'red'}
    order = [color_map[c] for c in bowls]  # ['blue','green','red']
    pos = order.index(target)              # 0-based position of target
    others = [c for c in COLORS if c != target]  # the two non-target colors
    prompts = []

    # 1. Direct color
    prompts.append(f"Put the banana in the {target} colored bowl.")

    # 2. Ordinal left-to-right
    prompts.append(f"{PREFIX} {ORDINAL_L[pos]} bowl from the left {SUFFIX}")

    # 3. Ordinal right-to-left
    pos_from_right = 2 - pos
    prompts.append(f"{PREFIX} {ORDINAL_L[pos_from_right]} bowl from the right {SUFFIX}")

    # 4. "Middle bowl" special case
    if pos == 1:
        prompts.append(f"{PREFIX} middle bowl {SUFFIX}")

    # 5. Relative to left neighbor (bowl immediately to the left of target)
    if pos > 0:
        left_neighbor = order[pos - 1]
        prompts.append(
            f"{PREFIX} bowl on the right of the {left_neighbor} bowl {SUFFIX}"
        )

    # 6. Relative to right neighbor (bowl immediately to the right of target)
    if pos < 2:
        right_neighbor = order[pos + 1]
        prompts.append(
            f"{PREFIX} bowl on the left of the {right_neighbor} bowl {SUFFIX}"
        )

    # 7. Negation: "not A and not B"
    a, b = others
    prompts.append(
        f"{PREFIX} bowl that is not {a} and not {b}."
    )
    # also reversed order
    prompts.append(
        f"{PREFIX} bowl that is not {b} and not {a}."
    )

    return prompts


# ── Build full map ────────────────────────────────────────────────────────────
prompt_map = {}        # combo_id → list of prompts
all_prompts_set = []   # ordered unique list of all prompts

for combo_id, arrangement, target, ep_start, ep_end in COMBOS:
    prompts = gen_prompts(arrangement, target)
    prompt_map[combo_id] = {
        "arrangement": arrangement,
        "target": target,
        "episodes": list(range(ep_start, ep_end + 1)),
        "prompts": prompts,
    }
    for p in prompts:
        if p not in all_prompts_set:
            all_prompts_set.append(p)

# Assign global task indices
task_index_map = {p: i for i, p in enumerate(all_prompts_set)}

print(f"Total unique prompts (tasks): {len(all_prompts_set)}")
print()
for i, p in enumerate(all_prompts_set):
    print(f"  [{i:2d}] {p}")

print()
print("── Per-combo prompt assignments ──────────────────────────────────")
for combo_id, info in prompt_map.items():
    idxs = [task_index_map[p] for p in info["prompts"]]
    print(f"  {combo_id:8s}  ({info['arrangement']}, target={info['target']:5s}, "
          f"eps {info['episodes'][0]:3d}-{info['episodes'][-1]:3d})  "
          f"→ task_indices {idxs}")

# ── Write outputs ─────────────────────────────────────────────────────────────
out_dir = Path(__file__).parent.parent / "eval2_prompt_augmentation"
out_dir.mkdir(exist_ok=True)

# tasks.jsonl — one task per line: {"task_index": i, "task": "..."}
with open(out_dir / "tasks.jsonl", "w") as f:
    for prompt, idx in task_index_map.items():
        f.write(json.dumps({"task_index": idx, "task": prompt}) + "\n")

# episodes.jsonl — one episode per line: {"episode_index": i, "task_indices": [...]}
with open(out_dir / "episodes.jsonl", "w") as f:
    for combo_id, info in prompt_map.items():
        task_indices = [task_index_map[p] for p in info["prompts"]]
        for ep_idx in info["episodes"]:
            f.write(json.dumps({
                "episode_index": ep_idx,
                "task_indices": task_indices,   # all valid prompts for this episode
                "primary_task_index": task_indices[0],  # direct color prompt = fallback
                "combo_id": combo_id,
                "arrangement": info["arrangement"],
                "target": info["target"],
            }) + "\n")

# full map JSON for inspection
with open(out_dir / "prompt_map.json", "w") as f:
    json.dump({
        "task_index_map": task_index_map,
        "combos": prompt_map,
    }, f, indent=2)

print(f"\nOutputs written to {out_dir}/")
print("  tasks.jsonl       — all unique compositional prompts")
print("  episodes.jsonl    — episode → valid task_indices")
print("  prompt_map.json   — full inspection map")
