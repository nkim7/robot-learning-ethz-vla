# Robot Learning — Group 47

SO-101 robot arm with SmolVLA policies. Three evaluation tasks.

---

## Setup

**Requirements:** macOS or Linux, Python 3.12+

```bash
# 1. Clone
git clone --recursive https://github.com/lorefrigo/robot-learning-ethz-vla.git
cd robot-learning-ethz-vla

# 2. Install dependencies
brew install ffmpeg uv          # macOS
# sudo apt install ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux

# 3. Save your robot port
lerobot-find-port
echo "export FOLLOWER_PORT=/dev/tty.usbmodemXXXX" >> .env   # replace with your port
```

Everything else (Python env, camera patch, model download) is handled automatically on first run.

---

## Running Evals

### Eval 1 — *"Put the banana in the blue colored bowl."*
```bash
bash run_eval_1.sh
```

### Eval 2 — *"Put the banana into the bowl to the right of the blue bowl."*
```bash
bash run_eval_2.sh
```

### Eval 3 — TOY figures (tasks 1 & 2)
```bash
TASK="Place the coke on Taylor Swift." bash run_eval_3_TOY.sh
# Other supported figures: Obama, Yann LeCun, …
```

### Eval 3 — Celebrity cards (task 3)
```bash
TASK="Place the coke on Rihanna." bash run_eval_3_celeb.sh
# Other supported celebrities: Elon Musk, …
```

---

## How Eval 3 Works

Eval 3 uses a two-stage pipeline:

1. **VLM LoRA** — on the first camera frame, a fine-tuned SmolVLM2-2.2B identifies which slot (left / middle / right) the target person is in.
2. **Instruction rewrite** — the task `"Place the coke on Taylor Swift."` is rewritten to `"Put the coke on left."` so the action expert receives the position-conditioned instruction it was trained on.
3. **Action expert** — SmolVLA takes the rewritten instruction + camera image and outputs joint commands every frame.

Two LoRA adapters cover the two visual domains:
- `TOY/` — plastic figurines
- `celeb/` — printed celebrity cards

---

## Hardware Notes

- Follower arm charger: **green** light — Leader arm charger: **blue** light
- Connect the 12V power adapter to the follower arm before running
- Calibration files: `robot/calibration/so101_group47/`

---

## Repository Structure

```
robot-learning-ethz-vla/
├── run_eval_1.sh
├── run_eval_2.sh
├── run_eval_3_TOY.sh
├── run_eval_3_celeb.sh
│
├── robot/                     — Hardware scripts, calibration, camera patch
├── models/                    — Policy checkpoints (gitignored, auto-downloaded)
│   ├── eval1/
│   ├── eval2/
│   └── eval3/
│       ├── TOY/               — VLM LoRA for TOY figures
│       ├── celeb/             — VLM LoRA for celebrity cards
│       └── runtime/           — Inference code (git-tracked)
│
├── training_pipeline/
│   ├── eval1/
│   ├── eval2/
│   └── eval3/
│       ├── vlm/               — VLM LoRA training scripts
│       └── vla/               — VLA action model training scripts
│
├── scripts/
│   ├── download_models.sh
│   └── download_datasets.sh
└── lerobot/                   — LeRobot submodule
```

---

## Calibration (new robot only)

```bash
source .venv/bin/activate
python robot/calibrate.py \
  --robot.type=so101_follower \
  --robot.port=$FOLLOWER_PORT \
  --robot.id=my_follower \
  --robot.calibration_dir=robot/calibration/so101_group47/follower
```
