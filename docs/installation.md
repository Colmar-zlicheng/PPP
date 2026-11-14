# Installation and data

All commands run from the repository root. Paths in the configs (`assets/...`, `checkpoints/...`,
`config/...`) are relative to it.

## 1. Python environment

Python 3.10 on Linux with an NVIDIA GPU. The pinned versions in [`requirements.txt`](../requirements.txt)
are the ones used for the paper results. PyTorch comes from the CUDA 12.8 wheel index.

**conda + pip**

```bash
conda create -n ppp python=3.10 -y
conda activate ppp
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

**uv**

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt --torch-backend=cu128
```

## 2. LIBERO

The closed-loop evaluation uses the LIBERO simulator and its official initial states. The code
expects a LIBERO checkout at `external/LIBERO`:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
git -C external/LIBERO checkout 8f1084e
pip install -e external/LIBERO --no-deps --config-settings editable_mode=compat
# uv: uv pip install -e external/LIBERO --no-deps --config-settings editable_mode=compat
```

`editable_mode=compat` is needed because LIBERO's top-level `libero/` folder has no `__init__.py`:
a default editable install exposes no modules and `import libero` fails.

On its first import LIBERO asks interactively where to keep its datasets and writes
`~/.libero/config.yaml`. Run the import once by hand before launching any evaluation job (answer
`N` to keep the defaults), otherwise the non-interactive evaluation workers stop at that prompt:

```bash
python -c "import libero.libero"
```

Rendering uses MuJoCo with EGL (`MUJOCO_GL=egl` is set by `lib/envs/libero_env.py`), so the
evaluation machine needs an NVIDIA driver with EGL support.
Evaluation jobs may print `Exception ignored in: <function MjRenderContext.__del__ ...>` with an
`EGLError` when they exit. This is robosuite releasing its EGL context after the results have been
written, and it can be ignored.

## 3. Download data and checkpoints

The processed LIBERO data, the Seer assets and all released checkpoints are hosted at
[huggingface.co/zlicheng/PPP](https://huggingface.co/zlicheng/PPP), with the same layout as this
repository (`assets/` and `checkpoints/`).

```bash
# everything (about 102 GB)
hf download zlicheng/PPP --local-dir . --include "assets/*" "checkpoints/*"
```

| Path | Size | Needed for |
|---|---|---|
| `assets/latent_extract/libero/libero_full_consolidated.h5` | 99.8 GB | all training and all closed-loop evaluation (frames, actions, instruction embeddings, task metadata) |
| `assets/seer/clip_text_libero.npz` | 0.3 MB | Seer and Seer-PPP (CLIP text features of the 130 LIBERO instructions) |
| `checkpoints/vit_mae/mae_pretrain_vit_base.pth` | 343 MB | Seer and Seer-PPP (frozen MAE ViT-B/16 image encoder) |
| `checkpoints/lewm_wm_libero/` | 0.2 GB | LeWM world model; training BC/PPP and building BC/PPP models |
| `checkpoints/lewm_bc_libero/`, `checkpoints/lewm_ppp_libero/` | 0.2 GB each + results | LeWM-based BC and PPP |
| `checkpoints/seer_libero/`, `checkpoints/seer_ppp_libero/` | 0.3 GB each + results | Seer and Seer-PPP |

Subsets, for example only what the LeWM-based evaluation needs:

```bash
hf download zlicheng/PPP --local-dir . \
    --include "assets/latent_extract/*" "checkpoints/lewm_*"
```

and only what the Seer-based evaluation needs:

```bash
hf download zlicheng/PPP --local-dir . \
    --include "assets/*" "checkpoints/seer_*" "checkpoints/vit_mae/*"
```

See [checkpoints.md](checkpoints.md) for the checkpoint layout and how to evaluate them.

## 4. Building the data yourself (optional)

Instead of downloading `assets/`, the same files can be built from the public LIBERO
demonstrations.

1. Download the LIBERO demonstrations (all five suites, 6,500 demonstrations over 130 tasks):

   ```bash
   python external/LIBERO/benchmark_scripts/download_libero_datasets.py \
       --datasets all --use-huggingface --download-dir data/LIBERO
   ```

   This gives `data/LIBERO/{libero_goal,libero_object,libero_spatial,libero_10,libero_90}/*.hdf5`.

2. Consolidate them into one HDF5 file with 128x128 agent-view and wrist images, actions, robot
   state, `all-MiniLM-L6-v2` instruction embeddings and the action normalization statistics:

   ```bash
   python prepare/prepare_libero_data.py --suite all --libero_root data/LIBERO
   # -> assets/latent_extract/libero/libero_full_consolidated.h5
   ```

3. Seer assets (the MAE weights from FAIR and the CLIP text features of the 130 instructions):

   ```bash
   python prepare/prepare_seer_assets.py
   # -> checkpoints/vit_mae/mae_pretrain_vit_base.pth, assets/seer/clip_text_libero.npz
   ```

## 5. Hardware notes

- **GPUs.** Every model was trained on 4 GPUs with 48 GB each (`-g 0,1,2,3`). The LeWM-based BC and
  PPP runs use about 28 GB per GPU.
- **Shared memory.** Training reads all frames from uint8 exports of the consolidated file in
  `/dev/shm/ppp_cache/` (about 110 GB: both camera views of the training episodes plus the
  validation subset), which all data-loader workers and DDP ranks share. The exports are created on
  first use and reused by later runs, LeWM- and Seer-based alike; delete `/dev/shm/ppp_cache/` when
  you are done.

## 6. Experiment IDs and the clean git tree

`scripts/train_general.py` writes each run to `exp/<exp_id>_<timestamp>/` (config dump, logs,
TensorBoard, `checkpoints/checkpoint_<epoch>/`). When a training run uses an `--exp_id` other than
`default` (the default) or `debug`, it records the current git commit and **refuses to start unless
the working tree is clean**: no modified, staged or untracked files (files ignored by
`.gitignore`, such as `assets/`, `checkpoints/`, `data/`, `exp/`, are fine). This means:

- commit any change to a config (for example a new `MODEL.LEWM_CKPT`) before launching a named run;
- for quick tests, use `--exp_id debug` (or omit `--exp_id`), which skips the check;
- the check reads the repository in the current directory, or in `$PROJECT_ROOT` when that is set,
  and needs the code to be a git checkout.

Evaluation scripts do not perform this check.
