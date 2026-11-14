# Released checkpoints and results

The checkpoints behind the paper's LIBERO results are hosted at
[huggingface.co/zlicheng/PPP](https://huggingface.co/zlicheng/PPP). Download them into the
repository root (see [installation.md](installation.md#3-download-data-and-checkpoints) for subsets):

```bash
hf download zlicheng/PPP --local-dir . --include "assets/*" "checkpoints/*"
```

## Layout

Each model folder is laid out like a training run directory, so the evaluation and summary scripts
take it directly as `RUN_DIR`:

```
checkpoints/
├── lewm_wm_libero/                 LeWM world model (pretrained 100 epochs)
│   ├── dump_cfg.yaml
│   └── checkpoints/checkpoint_100/LeWM.pth.tar
├── lewm_bc_libero/                 BC, epoch 120
│   ├── dump_cfg.yaml
│   ├── checkpoints/checkpoint_120/LeWMBCModel.pth.tar
│   └── closed_loop/epseed0_checkpoint_120/             40 task JSONs
├── lewm_ppp_libero/                PPP, epoch 120
│   ├── dump_cfg.yaml
│   ├── checkpoints/checkpoint_120/LeWMGGFieldCotrainModel.pth.tar
│   ├── closed_loop/epseed0_nosettle_checkpoint_120/    PPP w/o settling
│   └── closed_loop/epseed0_settle_checkpoint_120/      PPP
├── seer_libero/                    Seer, epoch 50
│   ├── dump_cfg.yaml
│   ├── checkpoints/checkpoint_50/SeerPolicyModel.pth.tar
│   └── closed_loop/epseed0_checkpoint_50/
├── seer_ppp_libero/                Seer-PPP, epoch 50
│   ├── dump_cfg.yaml
│   ├── checkpoints/checkpoint_50/SeerPolicyModel.pth.tar
│   └── closed_loop/epseed0_checkpoint_50/
└── vit_mae/mae_pretrain_vit_base.pth   MAE ViT-B/16 weights used by Seer and Seer-PPP
```

`dump_cfg.yaml` is the full training configuration of each run. The checkpoints hold model weights
only (no optimizer state). The BC and PPP checkpoints include their co-trained world model.
Building them still reads `MODEL.LEWM_CKPT`, which points at `checkpoints/lewm_wm_libero`, so keep
that folder next to them.

## Reference results

Success rates (%) on LIBERO under the official protocol: 40 tasks, 50 initial states per task
(500 rollouts per suite), per-episode seed 0.

| Model | Folder / result | Spatial | Object | Goal | Long | Avg. |
|---|---|---:|---:|---:|---:|---:|
| BC | `lewm_bc_libero` / `epseed0_checkpoint_120` | 31.4 | 53.6 | 47.0 | 20.2 | 38.1 |
| PPP w/o settling | `lewm_ppp_libero` / `epseed0_nosettle_checkpoint_120` | 40.4 | 76.2 | 52.4 | 37.0 | 51.5 |
| PPP | `lewm_ppp_libero` / `epseed0_settle_checkpoint_120` | 46.0 | 75.4 | 53.6 | 42.2 | 54.3 |
| Seer | `seer_libero` / `epseed0_checkpoint_50` | 98.8 | 99.0 | 97.8 | 93.6 | 97.3 |
| Seer-PPP | `seer_ppp_libero` / `epseed0_checkpoint_50` | 99.4 | 99.8 | 98.2 | 91.0 | 97.1 |

The per-task JSONs of these runs are included (`task<N>.json`). Each holds the per-episode outcomes
(`episode_results`) and the task summary (`summary`). Summarize them with:

```bash
python scripts/summarize_official_suites.py --run_dir checkpoints/lewm_ppp_libero \
    --cell epseed0_settle_checkpoint_120
```

## Re-running the evaluation

The evaluation scripts skip any task whose result file already holds 50 episodes. To re-evaluate a
released checkpoint, give the run a new `CELL_TAG` so that it writes to a new folder next to the
shipped results. Run from the repository root after setting up LIBERO
([installation.md](installation.md#2-libero)):

```bash
# BC
CELL_TAG=rerun VIDEO_ARGS="" \
    bash runs/close_loop_lewm_libero.sh checkpoints/lewm_bc_libero "checkpoint_120" 2 "0 1 2 3" 0

# PPP w/o settling
CELL_TAG=rerun_nosettle EXTRA_ARGS="--grip_field_eps_steps 1.0" VIDEO_ARGS="" \
    bash runs/close_loop_lewm_libero.sh checkpoints/lewm_ppp_libero "checkpoint_120" 2 "0 1 2 3" 0

# PPP
CELL_TAG=rerun_settle EXTRA_ARGS="--grip_field_eps_steps 1.0 --pose_grip_settle" VIDEO_ARGS="" \
    bash runs/close_loop_lewm_libero.sh checkpoints/lewm_ppp_libero "checkpoint_120" 2 "0 1 2 3" 0

# Seer and Seer-PPP
CELL_TAG=rerun bash runs/close_loop_seer_libero.sh checkpoints/seer_libero "checkpoint_50" 2 "0 1 2 3" 0
CELL_TAG=rerun bash runs/close_loop_seer_libero.sh checkpoints/seer_ppp_libero "checkpoint_50" 2 "0 1 2 3" 0
```

Then summarize the new folder, for example:

```bash
python scripts/summarize_official_suites.py --run_dir checkpoints/lewm_ppp_libero \
    --cell epseed0_rerun_settle_checkpoint_120
```

The arguments after the checkpoint list are jobs per GPU, the GPU ids and the episode seed. Set
`TASKS="0 1 2"` to evaluate a few tasks only (indices 0-9 LIBERO-Goal, 10-19 Object, 20-29
Spatial, 30-39 Long). Re-runs on different hardware or library versions can differ slightly from
the shipped results.
