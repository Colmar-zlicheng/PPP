# LeWM-based BC and PPP on LIBERO

Training has two steps. The LeWM world model is first pretrained alone on all 6,500 LIBERO
demonstrations (130 tasks). BC and PPP then each train one multitask policy from that world model.
Each policy co-trains the world model with the same architecture and data, and the two differ only
in how actions are read out:

| Model | Config | Output |
|---|---|---|
| LeWM world model | [`config/lewm_wm_libero.yaml`](../config/lewm_wm_libero.yaml) | encoder + latent dynamics |
| BC | [`config/lewm_bc_libero.yaml`](../config/lewm_bc_libero.yaml) | pose regressed from the world model's reference latent, gripper-switching head |
| PPP | [`config/lewm_ppp_libero.yaml`](../config/lewm_ppp_libero.yaml) | pose from the action gradient of the progress field, gripper from its finite difference |

Set up the environment and data first ([installation.md](installation.md)). All commands run from
the repository root on 4 GPUs. `-p` is the DDP master port (any free port) and `-w` is the number
of data-loader workers.

## 1. World model

```bash
python scripts/train_general.py -c config/lewm_wm_libero.yaml --exp_id lewm_wm_libero \
    -g 0,1,2,3 -p 29500 -w 8
```

The run trains for 100 epochs and writes `exp/lewm_wm_libero_<timestamp>/checkpoints/checkpoint_100/`.
The BC and PPP models start from this checkpoint. The released one is
`checkpoints/lewm_wm_libero/checkpoints/checkpoint_100`.

## 2. BC and PPP

Both configs read the world model from `MODEL.LEWM_CKPT`, which points at the **released** world
model:

```yaml
MODEL:
  LEWM_CKPT: checkpoints/lewm_wm_libero/checkpoints/checkpoint_100
```

**If you trained the world model yourself**, change `MODEL.LEWM_CKPT` in both configs to your run,
for example `exp/lewm_wm_libero_<timestamp>/checkpoints/checkpoint_100`. Then commit the change
before training, because named runs require a clean git tree (see
[installation.md](installation.md#6-experiment-ids-and-the-clean-git-tree)).

```bash
# BC
python scripts/train_general.py -c config/lewm_bc_libero.yaml \
    --exp_id lewm_bc_libero -g 0,1,2,3 -p 29501 -w 8 --snapshot 10

# PPP
python scripts/train_general.py -c config/lewm_ppp_libero.yaml \
    --exp_id lewm_ppp_libero -g 0,1,2,3 -p 29502 -w 8 --snapshot 10 --eval_interval 1
```

- The world model stays frozen for epochs 0-9 and is co-trained from epoch 10 (learning rate 5e-5;
  the policy heads use 3e-4).
- `--snapshot 10` saves `checkpoints/checkpoint_<epoch>/` every 10 epochs.
- **The paper reports `checkpoint_120`.** The configs keep the original 200-epoch cosine schedule
  (`TRAIN.EPOCH: 200`). Stop the run once `checkpoint_120` is saved, or let it finish. Do not pass
  `--num_epochs 120`, because it overrides `TRAIN.EPOCH` and changes the learning-rate schedule.
- `--resume exp/<run>` continues a stopped run from its latest checkpoint (`exp/<run>/checkpoints/checkpoint`)
  in a new run directory.
- PPP uses about 28 GB per GPU at the configured batch size (512 over 4 GPUs).

## 3. Closed-loop evaluation

[`runs/close_loop_lewm_libero.sh`](../runs/close_loop_lewm_libero.sh) runs the
official LIBERO protocol on the 40 evaluation tasks. Task indices 0-9 are LIBERO-Goal, 10-19
LIBERO-Object, 20-29 LIBERO-Spatial and 30-39 LIBERO-Long. Each task runs from its 50 official
initial states (`.pruned_init`) with a per-episode seeded scene reset, so 2,000 episodes per
checkpoint. The policy replans one command per 20 Hz control step.

```bash
bash runs/close_loop_lewm_libero.sh RUN_DIR "CKPT [CKPT ...]" [JOBS_PER_GPU] [GPUS] [EP_SEED]
```

The paper uses `EP_SEED=0`. `RUN_DIR` is a training run directory (it must contain `dump_cfg.yaml`
and `checkpoints/<ckpt>/`). The script reads these environment variables:

| Variable | Meaning |
|---|---|
| `EXTRA_ARGS` | extra flags for `scripts/close_loop_lewm_libero.py` |
| `CELL_TAG` | tag added to the output folder name, to keep settings apart |
| `VIDEO_ARGS` | video flags; the default saves rollout videos (about 0.7 GB per checkpoint), `VIDEO_ARGS=""` saves none |
| `TASKS` | subset of task indices, e.g. `TASKS="0 1 2"` (default 0-39) |
| `PY` | python executable (default `python`) |

**BC**

```bash
VIDEO_ARGS="" bash runs/close_loop_lewm_libero.sh exp/lewm_bc_libero_<timestamp> "checkpoint_120" 2 "0 1 2 3" 0
```

**PPP.** The field gripper switches when the finite difference of the field between the two gripper
commands exceeds one grid step of progress (`--grip_field_eps_steps 1.0`). The step is converted to
field units with the task's median demonstration length. The paper reports PPP with and without gripper settling
(`--pose_grip_settle`, which keeps the pose query conditioned on an open gripper until the fingers
stop closing):

```bash
RUN=exp/lewm_ppp_libero_<timestamp>
# PPP w/o settling
CELL_TAG=nosettle EXTRA_ARGS="--grip_field_eps_steps 1.0" VIDEO_ARGS="" \
    bash runs/close_loop_lewm_libero.sh $RUN "checkpoint_120" 2 "0 1 2 3" 0
# PPP
CELL_TAG=settle EXTRA_ARGS="--grip_field_eps_steps 1.0 --pose_grip_settle" VIDEO_ARGS="" \
    bash runs/close_loop_lewm_libero.sh $RUN "checkpoint_120" 2 "0 1 2 3" 0
```

Each (checkpoint, task) pair is one job. `JOBS_PER_GPU` jobs run concurrently on every GPU listed
in `GPUS`. A task whose result file already holds 50 episodes is skipped, so an interrupted
evaluation can simply be relaunched.

**Results.** One JSON per task is written to
`RUN_DIR/closed_loop/epseed0_[<CELL_TAG>_]<ckpt>/task<N>.json` (per-episode outcomes in
`episode_results`, the task summary in `summary`), with each job's log in `logs/task<N>.log` next
to it. Per-suite and average success rates:

```bash
python scripts/summarize_official_suites.py --run_dir $RUN --cell epseed0_settle_checkpoint_120
```

To evaluate the released checkpoints instead of your own, see [checkpoints.md](checkpoints.md).
