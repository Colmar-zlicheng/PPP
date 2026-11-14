# Seer and Seer-PPP on LIBERO

Seer ([InternRobotics/Seer](https://github.com/InternRobotics/Seer)) is a strong published policy
with an action-free foresight transformer, a frozen MAE ViT-B/16 image encoder and CLIP text
features. The implementation in `lib/models/seer/` is a port of the original model whose
parameter names match the official state dicts. Seer-PPP keeps Seer's network and training recipe
unchanged and replaces only the action readout. A small world model `F` predicts the trunk state
`s` after a candidate command chunk `u`. A progress field `p(F(s, u), l)` is read through its
action gradient for the pose and through a finite difference for the gripper.

| Model | Config | Output |
|---|---|---|
| Seer | [`config/seer_libero.yaml`](../config/seer_libero.yaml) | Seer's action head (`MODEL.HEAD: bc`) |
| Seer-PPP | [`config/seer_ppp_libero.yaml`](../config/seer_ppp_libero.yaml) | PPP field readout on Seer's trunk (`MODEL.HEAD: ppp`) |

Both train from scratch on all 6,500 LIBERO demonstrations (130 tasks) and need two assets besides
the consolidated data (see [installation.md](installation.md)):

- `checkpoints/vit_mae/mae_pretrain_vit_base.pth`: frozen MAE encoder (`MODEL.SEER.MAE_CKPT`). It is
  not stored in the policy checkpoints and is rebuilt from this file whenever a model is built, so
  evaluation needs it too.
- `assets/seer/clip_text_libero.npz`: CLIP ViT-B/32 text features of the instructions
  (`MODEL.SEER.CLIP_TEXT`).

## 1. Training

```bash
# Seer
python scripts/train_general.py -c config/seer_libero.yaml \
    --exp_id seer_libero -g 0,1,2,3 -p 29503 -w 16 --snapshot 10 --eval_interval 5

# Seer-PPP
python scripts/train_general.py -c config/seer_ppp_libero.yaml \
    --exp_id seer_ppp_libero -g 0,1,2,3 -p 29504 -w 16 --snapshot 10 --eval_interval 1
```

- Schedule (Seer's): 50 epochs, effective batch 512 (4 GPUs x 16 per GPU x 8 gradient-accumulation
  steps), learning rate 1e-3 with 5 warmup epochs and a cosine decay stepped every optimizer step.
  Seer-PPP's readout (field, gain, world model `F`) uses 3e-4.
- **The paper reports `checkpoint_50`**, the final checkpoint of each run.
- Named runs require a clean git tree (see
  [installation.md](installation.md#6-experiment-ids-and-the-clean-git-tree)).
- Seer-PPP uses about 16 GB per GPU at 16 samples per GPU.

## 2. Closed-loop evaluation

[`runs/close_loop_seer_libero.sh`](../runs/close_loop_seer_libero.sh) runs the
same protocol as the LeWM-based models ([lewm.md](lewm.md#3-closed-loop-evaluation)): the 40
evaluation tasks of LIBERO-Goal, -Object, -Spatial and -Long, 50 official initial states each,
per-episode seeded scene resets and one command per 20 Hz control step. The released settings
need no extra flags.

```bash
bash runs/close_loop_seer_libero.sh exp/seer_libero_<timestamp> "checkpoint_50" 2 "0 1 2 3" 0
bash runs/close_loop_seer_libero.sh exp/seer_ppp_libero_<timestamp> "checkpoint_50" 2 "0 1 2 3" 0
```

The arguments are `RUN_DIR "CKPT [CKPT ...]" [JOBS_PER_GPU] [GPUS] [EP_SEED]`, with `EP_SEED=0` as in
the paper. `TASKS`, `CELL_TAG`, `EXTRA_ARGS` and `PY` work as for the LeWM script. No videos are
saved by default (`VIDEO_ARGS="--save_video"` saves them).

Results are written to `RUN_DIR/closed_loop/epseed0_[<CELL_TAG>_]<ckpt>/task<N>.json` in the same
format as for the LeWM-based models:

```bash
python scripts/summarize_official_suites.py --run_dir exp/seer_ppp_libero_<timestamp> --cell epseed0_checkpoint_50
```

To evaluate the released checkpoints, see [checkpoints.md](checkpoints.md).
