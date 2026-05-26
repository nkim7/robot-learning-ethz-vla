# Eval 3 Runtime Package

This folder contains the Eval 3 inference scripts, setup script, runtime Python launchers, and patches needed to run the submitted policies.

## File Structure

```text
eval_3/
  README.md
  setup_eval_3.sh
  run_eval_3.sh
  run_eval_3_celeb.sh
  run_eval_3_e2e.sh
  scripts/
    eval3_runtime_common.sh
    eval3_run1_action_train/
      run1_common.py
      run1_lora_patch.py
    eval3_run2_lmr210_action_train/
      run2_common.py
      run2_policy_server_launcher.py
    eval3_bridge_train/
      bridge_policy_server_launcher.py
  patches/
    lerobot/
      async_inference/
        policy_server.py
```

## Checkpoints

The setup script downloads checkpoints into the repository-local `checkpoints/` folder by default.

| Purpose | Hugging Face repo | Local path |
| :--- | :--- | :--- |
| Tasks 1/2 action expert and cycle 05 LoRA | `robot-learning-group47/eval3_cotrain` | `checkpoints/eval3_cotrain` |
| Task 3 celebrity detector LoRA | `robot-learning-group47/eval3_phase2_celeb100` | `checkpoints/eval3_phase2_celeb100` |
| End-to-end VLA baseline | `robot-learning-group47/eval3_bridge_cycle05_smolvla` | `checkpoints/eval3_bridge_cycle05_smolvla` |
| SmolVLM base model | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | `checkpoints/base_models/HuggingFaceTB__SmolVLM2-2.2B-Instruct` |

Run this once on the policy-server machine:

```bash
CHECKPOINT_ROOT="$(pwd)/checkpoints" ./eval_3/setup_eval_3.sh all
```

To match the original Eval 3 server paths, use:

```bash
CHECKPOINT_ROOT=/home/shadeform/checkpoint ./eval_3/setup_eval_3.sh all
```

After the first install/download, add `EVAL3_SKIP_INSTALL=1` to skip dependency installation:

```bash
EVAL3_SKIP_INSTALL=1 CHECKPOINT_ROOT=/home/shadeform/checkpoint ./eval_3/setup_eval_3.sh all
```

If a checkpoint folder exists but should be refreshed, add `EVAL3_FORCE_DOWNLOAD=1`.

## Tasks 1 and 2

This mode uses `robot-learning-group47/eval3_cotrain` as the action expert and LoRA. It applies the first-frame preprompt patch: the server asks where the requested celebrity is, then rewrites the task to a spatial prompt before action inference.

Start the server first:

```bash
CHECKPOINT_ROOT=/home/shadeform/checkpoint \
HOST=0.0.0.0 \
PORT=8080 \
./eval_3/run_eval_3.sh server
```

Then run the async client:

```bash
FOLLOWER_PORT="$FOLLOWER_PORT" \
SERVER_ADDRESS=216.81.248.32:8080 \
SERVER_POLICY_PATH=/home/shadeform/checkpoint/eval3_cotrain/cycle_05_action_checkpoint \
TASK="Place the coke on Taylor Swift." \
./eval_3/run_eval_3.sh client
```

## Task 3 Celebrity Selection

This mode uses `robot-learning-group47/eval3_phase2_celeb100` for celebrity/slot detection and the cotrain action checkpoint for motion.

Start the server first:

```bash
CHECKPOINT_ROOT=/home/shadeform/checkpoint \
HOST=0.0.0.0 \
PORT=8080 \
./eval_3/run_eval_3_celeb.sh server
```

Then run the async client:

```bash
FOLLOWER_PORT="$FOLLOWER_PORT" \
SERVER_ADDRESS=216.81.248.32:8080 \
SERVER_POLICY_PATH=/home/shadeform/checkpoint/eval3_cotrain/cycle_05_action_checkpoint \
TASK="Place the coke on Rihanna." \
./eval_3/run_eval_3_celeb.sh client
```

## End-to-End VLA Baseline

This mode is separate from the preprompt/rewrite path. It uses `robot-learning-group47/eval3_bridge_cycle05_smolvla` as the policy checkpoint and serves celebrity prompts directly.

Start the server first:

```bash
CHECKPOINT_ROOT=/home/shadeform/checkpoint \
HOST=0.0.0.0 \
PORT=8080 \
./eval_3/run_eval_3_e2e.sh server
```

Then run the async client:

```bash
FOLLOWER_PORT="$FOLLOWER_PORT" \
SERVER_ADDRESS=216.81.248.32:8080 \
SERVER_POLICY_PATH=/home/shadeform/checkpoint/eval3_bridge_cycle05_smolvla \
TASK="Place the coke on Taylor Swift." \
./eval_3/run_eval_3_e2e.sh client
```

## Important Runtime Notes

- Set `FOLLOWER_PORT` before running any client command.
- The default camera config is the Eval 3 cropped OpenCV front camera.
- The runtime scripts install `lerobot[feetech,smolvla,async,peft]` into `.venv`.
- `patches/lerobot/async_inference/policy_server.py` is the patched policy-server source kept for inspection/reapplication. The runnable server path uses the packaged Python launchers, which apply the needed monkey patches at startup.
