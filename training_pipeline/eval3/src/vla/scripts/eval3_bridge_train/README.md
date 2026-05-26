# eval3_bridge_train

End-to-end SmolVLA training: celebrity prompts ("Place the coke on Taylor Swift.")
→ position-aware action execution — no external VLM resolver at inference.

## Approach

Two bridge losses are added on top of the standard action prediction loss:

1. **Alignment loss** (MSE, weight `ALIGN_LAMBDA=0.1`)  
   Pulls the action expert's hidden state (suffix_out) produced from celebrity prompts
   toward the hidden state produced from spatial prompts ("Put the coke on left/middle/right.").
   Teacher forward is run with spatial prompts and `torch.no_grad()`.

2. **Spatial CE loss** (CrossEntropy, weight `SPATIAL_LAMBDA=1.0`)  
   A small linear head `spatial_head(720→3)` predicts left/middle/right from
   `suffix_out.mean(dim=1)`. Provides direct supervision for slot identity.

**VLM LoRA is frozen throughout** — position accuracy (98.5%) is preserved.

## Slot label rule

```
within_block = episode_index % 30
0–9   → left
10–19 → middle
20–29 → right
```

## Workflow

```bash
# 1. Generate sidecar JSON (one-time)
python scripts/eval3_bridge_train/relabel_eval3_90_target_slot.py

# 2. Validate labels
python scripts/eval3_bridge_train/check_eval3_spatial_labels.py

# 3. Run bridge training (default: 3000 steps, batch=8)
bash scripts/eval3_bridge_train/train_eval3_bridge.sh

# 4. Inspect checkpoint
python scripts/eval3_bridge_train/check_bridge_checkpoint.py \
  --output-dir outputs/train/eval3_bridge_cycle05_3000steps_alm0.1_slm1.0
```

## Key env vars for `train_eval3_bridge.sh`

| Variable | Default | Description |
|---|---|---|
| `STEPS` | 3000 | Training steps |
| `BATCH_SIZE` | 8 | Batch size |
| `ACTION_LR` | 1e-5 | Action expert learning rate |
| `SPATIAL_HEAD_LR` | 1e-4 | Spatial head learning rate |
| `ALIGN_LAMBDA` | 0.1 | Alignment loss weight |
| `SPATIAL_LAMBDA` | 1.0 | Spatial CE loss weight |
| `TOY_REPEATS` | 4 | Dataset repetitions |
| `START_ACTION_CHECKPOINT` | cycle_05 500-step ckpt | Starting action checkpoint |
| `START_VLM_LORA` | cycle_05 VLM LoRA | Starting VLM LoRA (frozen) |
| `BRIDGE_OVERWRITE` | 0 | Set to 1 to overwrite existing output |

## Architecture

```
Batch (celebrity prompt)          Batch (spatial prompt, no_grad)
       │                                   │
 SmolVLAPolicy.forward              SmolVLAPolicy.forward
       │                                   │
 action_out_proj hook               action_out_proj hook
       │                                   │
  student_suffix_out           teacher_suffix_out.detach()
  (B, 50, 720)                      (B, 50, 720)
       │                                   │
       └──── MSE align_loss ──────────────┘
       │
  .mean(dim=1) → (B, 720)
       │
  spatial_head(720→3)
       │
  CE spatial_loss vs target_slot_ids
```

## Files

- `relabel_eval3_90_target_slot.py` — generates `data/eval3_90_permutation_spatial_labels/target_slot_by_episode.json`
- `check_eval3_spatial_labels.py` — validates the sidecar
- `bridge_loss_patch.py` — patches `VLAFlowMatching.__init__`, `SmolVLAPolicy.get_optim_params`, `SmolVLAPolicy.forward`
- `train_eval3_bridge.py` — launcher (applies run1_lora_patch + bridge_loss_patch, calls LeRobot trainer)
- `train_eval3_bridge.sh` — full bash workflow
- `check_bridge_checkpoint.py` — post-training inspection
