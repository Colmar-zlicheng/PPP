from time import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils.builder import DATASET
from lib.utils.logger import logger

from .lewm_data import _episode_split, ensure_shm_episode_export
from .libero_timing import resolve_timing


def _assert_unsupported(cfg, name):
    for key, ok in (("LOCAL_CACHE", lambda v: not v), ("GEOM_TARGETS", lambda v: not v),
                    ("FILTER_SUCCESS", lambda v: not v), ("PROG_SOURCE", lambda v: v in (None, "clock")),
                    ("ACTION_SOURCE", lambda v: v in (None, "osc")), ("IN_MEMORY", lambda v: v in (None, True)),
                    ("SHM_CACHE", lambda v: v is True), ("NORMALIZE_ACTIONS", lambda v: v in (None, True)),
                    ("SPLIT_MODE", lambda v: v == "ratio")):
        assert ok(cfg.get(key, None)), f"{name}: DATASET.{key}={cfg.get(key, None)!r} is not supported"


@DATASET.register_module()
class LiberoLeWMDataset(Dataset):
    """Multi-step frame windows of the consolidated LIBERO file for the LeWM world model."""

    def __init__(self, cfg, data_preset=None):
        _assert_unsupported(cfg, type(self).__name__)
        self.h5_path = cfg.DATA_ROOT
        self.num_steps = cfg.NUM_STEPS      # history_size + num_preds (e.g. 4)
        self.frameskip = cfg.FRAMESKIP      # raw sim steps between consecutive window frames
        self.split = cfg.get("SPLIT", "train")
        self.cameras = list(cfg.CAMERAS)
        assert len(self.cameras) >= 1, "CAMERAS must name at least one dataset"
        self.pixel_key = self.cameras[0]
        self.action_source = "osc"

        t0 = time()
        with h5py.File(self.h5_path, "r") as f:
            self.ep_offset = f["ep_offset"][:]
            self.ep_len = f["ep_len"][:]
            ep_task_idx = f["ep_task_idx"][:]
            self.timing = resolve_timing(cfg, f)
            self.action_offset = self.timing.action_offset(self.action_source)
            self.action_dim = int(f["actions"].shape[1])
            self.action_loc = f["action_loc"][:].astype(np.float32)
            self.action_scale = f["action_scale"][:].astype(np.float32)

            self.train_ratio = cfg.get("TRAIN_RATIO", 0.9)
            self.seed = cfg.get("SEED", 3072)
            stratify = cfg.get("STRATIFY", False)
            assert stratify or self.split in ("train", "val", "test"), (
                f"SPLIT_MODE=ratio with STRATIFY=false supports only train/val/test; got "
                f"SPLIT={self.split!r}. Set STRATIFY: true for the 'trainval'/'all' unions.")
            self.ep_indices = _episode_split(
                self.ep_offset, self.train_ratio, self.seed, self.split,
                ep_task_idx=ep_task_idx, stratify=stratify,
            )
            split_desc = f"ratio={self.train_ratio} seed={self.seed} stratify={stratify}"
        self.ep_indices = np.sort(np.asarray(self.ep_indices, dtype=np.int64))
        assert len(self.ep_indices) > 0, f"LiberoLeWMDataset: SPLIT {self.split!r} selected 0 episodes"
        n_unique = len(np.unique(self.ep_indices))
        assert n_unique == len(self.ep_indices), (
            f"LiberoLeWMDataset: SPLIT {self.split!r} selected {len(self.ep_indices)} episodes but "
            f"only {n_unique} distinct ones")

        self.span = self.num_steps * self.frameskip
        counts = np.array([max(0, int(self.ep_len[ei]) - self.span) for ei in self.ep_indices],
                          dtype=np.int64)
        n_dropped = int((counts == 0).sum())
        if n_dropped:
            logger.warning(f"LiberoLeWMDataset[{self.split}]: {n_dropped}/{len(self.ep_indices)} "
                           f"episodes shorter than span={self.span}+1 raw frames, dropped")
        self._cumsum = np.concatenate([[0], np.cumsum(counts)])
        self._total = int(self._cumsum[-1])

        self._pixels = None
        self._pixel_paths = None
        self._actions = None
        self._local_base = None
        self._load_into_ram()

        logger.info(f"LiberoLeWMDataset[{self.split}]: {self._total} windows from "
                    f"{len(self.ep_indices)} episodes ({split_desc}, num_steps={self.num_steps}, "
                    f"frameskip={self.frameskip}, span={self.span}, "
                    f"timing={self.timing.contract}, action_offset={self.action_offset}) "
                    f"in {time() - t0:.1f}s")

    def _load_into_ram(self):
        """Export the split's frames to /dev/shm (shared by all ranks and workers) and read its
        actions into a contiguous process-local array."""
        n_frames = int(sum(int(self.ep_len[ei]) for ei in self.ep_indices))
        self._local_base = np.concatenate(
            [[0], np.cumsum([int(self.ep_len[ei]) for ei in self.ep_indices])[:-1]]
        ).astype(np.int64)
        episodes = np.stack([[int(self.ep_offset[ei]), int(self.ep_len[ei])]
                             for ei in self.ep_indices]).astype(np.int64)
        self._pixel_paths = [ensure_shm_episode_export(self.h5_path, cam, episodes)
                             for cam in self.cameras]
        self._pixels = None
        with h5py.File(self.h5_path, "r") as f:
            self._actions = np.empty((n_frames, self.action_dim), dtype=np.float32)
            for slot, ei in enumerate(self.ep_indices):
                base, n = int(self.ep_offset[ei]), int(self.ep_len[ei])
                w = int(self._local_base[slot])
                self._actions[w:w + n] = f["actions"][base:base + n].astype(np.float32)
        logger.info(f"LiberoLeWMDataset[{self.split}]: SHM_CACHE -- {n_frames} frames x "
                    f"{len(self.cameras)} view(s) mmapped from /dev/shm, shared across ranks")

    def _ensure_pixels(self):
        """Open the tmpfs exports in THIS process, once. See `__getstate__`."""
        if self._pixels is None:
            assert self._pixel_paths is not None, "frame storage was never initialized"
            self._pixels = [np.load(pth, mmap_mode="r") for pth in self._pixel_paths]
        return self._pixels

    def __getstate__(self):
        """Pickle (for DDP workers) without the memory maps; each process reopens them."""
        state = self.__dict__.copy()
        state["_pixels"] = None      # reopened per process by _ensure_pixels
        return state

    def __len__(self) -> int:
        return self._total

    def _locate(self, idx: int):
        """(storage base index of the window's first frame, episode index)."""
        slot = int(np.searchsorted(self._cumsum, idx, side="right") - 1)
        start_local = idx - int(self._cumsum[slot])
        ep_idx = int(self.ep_indices[slot])
        base = int(self._local_base[slot])
        return base + start_local, ep_idx

    def __getitem__(self, idx: int) -> dict:
        base, ep_idx = self._locate(idx)
        frame_abs = base + np.arange(self.num_steps, dtype=np.int64) * self.frameskip

        views = [buf[frame_abs] for buf in self._ensure_pixels()]  # each (T,H,W,C) uint8
        act_windows = self._actions[frame_abs[:, None] + self.action_offset
                                    + np.arange(self.frameskip, dtype=np.int64)[None, :]]

        stacked = views[0] if len(views) == 1 else np.stack(views, axis=1)
        pixels = np.moveaxis(stacked, -1, -3).astype(np.float32) / 255.0
        actions = act_windows.astype(np.float32)                          # (T, frameskip, 7)
        actions = (actions - self.action_loc) / self.action_scale
        actions = actions.reshape(self.num_steps, -1)                     # (T, frameskip*7)

        sample = {
            "pixels": torch.from_numpy(np.ascontiguousarray(pixels)),
            "action": torch.from_numpy(np.ascontiguousarray(actions)),
            "episode_idx": torch.tensor(ep_idx, dtype=torch.long),
        }
        return sample


@DATASET.register_module()
class LiberoLeWMGGFieldDataset(LiberoLeWMDataset):
    """(context window at an anchor, the next frame, a later frame) samples with the expert
    command, gripper labels and progress labels, for the LeWM-based BC and PPP models."""

    def __init__(self, cfg, data_preset=None):
        self.history_size = int(cfg.get("HISTORY_SIZE", 3))
        super().__init__(cfg, data_preset)
        assert self.action_dim == 7, f"{self.h5_path}: one-arm 7-d actions expected, got {self.action_dim}"
        self.n_grip = 1
        self.pose_dim = 6
        for key, want in (("USE_WRONG_LANG", True), ("EXPERT_CHUNK_REDUCE", "mean"),
                          ("EXPERT_GRIP_REDUCE", "first"), ("EMIT_NEXT_FRAME", True)):
            assert cfg.get(key, want) == want, f"DATASET.{key} must be {want!r}"

        self.gap_set = sorted({int(g) for g in cfg.get("GAP_SET", [1, 2, 4, 8, 16, 30])})
        self.anchor_stride = int(cfg.get("ANCHOR_STRIDE", 2))
        self.anchor_phase = str(cfg.get("ANCHOR_PHASE", "grid"))
        assert self.anchor_phase in ("grid", "random"), (
            f"ANCHOR_PHASE must be 'grid' or 'random', got {self.anchor_phase!r}")
        self.min_lang_dist = float(cfg.get("CROSS_EP_MIN_LANG_DIST", 0.3))
        self.seed = int(cfg.get("SEED", 3072))
        self.grip_events = bool(cfg.get("GRIP_EVENTS", False))

        with h5py.File(self.h5_path, "r") as f:
            self.task_lang_embed = f["task_lang_embed"][:].astype(np.float32)      # (n_ep, 384)
            self.ep_task_idx = f["ep_task_idx"][:].astype(np.int64)
            self.lang_by_task = f["task_lang_embed_by_task"][:].astype(np.float32)  # (n_task, 384)

        self._build_pair_index()
        self._assign_wrong_lang()

    def _build_pair_index(self):
        """(anchor grid position, later grid position, episode slot) triples + progress labels."""
        t0 = time()
        a_chunks, j_chunks, pi_chunks, pj_chunks, slot_chunks = [], [], [], [], []
        ti_chunks, tj_chunks = [], []
        gap_arr = np.asarray(self.gap_set, dtype=np.int64)
        n_short = 0
        for slot, ei in enumerate(self.ep_indices):
            n_grid = int(self.ep_len[ei]) // self.frameskip
            lo, hi = self.history_size - 1, n_grid - 2          # inclusive anchor range
            if hi < lo:
                n_short += 1
                continue
            a_arr = np.arange(lo, hi + 1, self.anchor_stride, dtype=np.int64)
            j_arr = a_arr[:, None] + gap_arr[None, :]
            valid = j_arr <= (n_grid - 1)
            if not valid.any():
                n_short += 1
                continue
            a_valid = np.broadcast_to(a_arr[:, None], j_arr.shape)[valid]
            j_valid = j_arr[valid]
            denom = float(max(n_grid - 1, 1))
            a_chunks.append(a_valid)
            j_chunks.append(j_valid)
            pi_chunks.append(a_valid.astype(np.float32) / denom)
            pj_chunks.append(j_valid.astype(np.float32) / denom)
            ti_chunks.append((denom - a_valid).astype(np.float32))
            tj_chunks.append((denom - j_valid).astype(np.float32))
            slot_chunks.append(np.full(a_valid.shape[0], slot, dtype=np.int64))

        assert a_chunks, (
            f"LiberoLeWMGGFieldDataset[{self.split}]: no valid pairs -- every episode is shorter "
            f"than history_size={self.history_size} + min(GAP_SET)={self.gap_set[0]} grid steps "
            f"at frameskip={self.frameskip}")
        anchor = np.concatenate(a_chunks)
        later = np.concatenate(j_chunks)
        prog_t = np.concatenate(pi_chunks)
        prog_tp1 = np.concatenate(pj_chunks)
        ttg_t = np.concatenate(ti_chunks)
        ttg_tp1 = np.concatenate(tj_chunks)
        slots = np.concatenate(slot_chunks)

        perm = np.random.RandomState(self.seed).permutation(len(anchor))
        self._anchor, self._later = anchor[perm], later[perm]
        self._prog_t, self._prog_tp1 = prog_t[perm], prog_tp1[perm]
        self._ttg_t, self._ttg_tp1 = ttg_t[perm], ttg_tp1[perm]
        self._slot = slots[perm]
        self._n_pairs = int(len(anchor))
        logger.info(f"LiberoLeWMGGFieldDataset[{self.split}]: {self._n_pairs} pairs from "
                    f"{len(self.ep_indices) - n_short} episodes ({n_short} too short), gaps "
                    f"{self.gap_set} grid steps x frameskip {self.frameskip}, anchor_stride "
                    f"{self.anchor_stride}, history_size {self.history_size}, anchor_phase "
                    f"{self.anchor_phase!r} in {time() - t0:.1f}s")

    def _assign_wrong_lang(self):
        """Fixed per-TASK wrong-language partner, broadcast to all of its episodes."""
        rng = np.random.RandomState(self.seed + 1)
        emb = self.lang_by_task / (np.linalg.norm(self.lang_by_task, axis=1, keepdims=True) + 1e-8)
        cos = emb @ emb.T                                   # (n_task, n_task)
        dist = 1.0 - cos
        n_fallback = 0
        wrong_task_of_task = np.zeros(len(emb), dtype=np.int64)
        for t in range(len(emb)):
            cand = np.where(dist[t] >= self.min_lang_dist)[0]
            if len(cand) == 0:
                cand = np.array([int(np.argmax(dist[t]))])
                n_fallback += 1
            wrong_task_of_task[t] = int(cand[rng.randint(len(cand))])
        if n_fallback:
            logger.warning(f"LiberoLeWMGGFieldDataset[{self.split}]: {n_fallback} task(s) had no "
                           f"partner at cosine distance >= {self.min_lang_dist}; used the most "
                           f"distant task instead")
        self._wrong_lang_of_ep = self.lang_by_task[wrong_task_of_task[self.ep_task_idx]]

    def __len__(self) -> int:
        return self._n_pairs

    def _read_views(self, frame_idx: np.ndarray) -> np.ndarray:
        """(n,) storage frame indices -> (n,V,H,W,C) uint8, or (n,H,W,C) for a single camera."""
        views = [buf[frame_idx] for buf in self._ensure_pixels()]
        return views[0] if len(views) == 1 else np.stack(views, axis=1)

    @staticmethod
    def _to_chw(stacked: np.ndarray) -> np.ndarray:
        """(...,H,W,C) uint8 -> (...,C,H,W) float32 in [0,1], the contract `JEPA.encode` expects."""
        return np.moveaxis(stacked, -1, -3).astype(np.float32) / 255.0

    def _draw_phase(self, ep_idx: int, a_grid: int, j_grid: int) -> int:
        """ANCHOR_PHASE random: a raw-frame shift for one sample, uniform over its valid range."""
        k = self.frameskip
        m0 = max(j_grid * k, a_grid * k + self.action_offset + k - 1)
        m0 = max(m0, (a_grid + 1) * k)
        hi = min(self.anchor_stride * k - 1, int(self.ep_len[ep_idx]) - 1 - m0)
        assert hi >= 0, (ep_idx, a_grid, j_grid, hi)
        return int(np.random.randint(0, hi + 1))

    def __getitem__(self, idx: int) -> dict:
        k = self.frameskip
        slot = int(self._slot[idx])
        ep_idx = int(self.ep_indices[slot])
        a_grid, j_grid = int(self._anchor[idx]), int(self._later[idx])
        phase = self._draw_phase(ep_idx, a_grid, j_grid) if self.anchor_phase == "random" else 0
        base = int(self._local_base[slot])

        ctx_grid = a_grid - np.arange(self.history_size - 1, -1, -1, dtype=np.int64)
        ctx_frames = base + ctx_grid * k + phase
        win_frames = np.append(ctx_frames, base + (a_grid + 1) * k + phase)
        win_px = self._to_chw(self._read_views(win_frames))             # (H+1,[V,]C,128,128)
        ctx_px, next_px = win_px[:self.history_size], win_px[self.history_size]

        act_idx = ctx_frames[:, None] + self.action_offset + np.arange(k, dtype=np.int64)[None, :]
        ctx_act = self._actions[act_idx].astype(np.float32)            # (H,k,7) RAW
        anchor_chunk_raw = ctx_act[-1]                                  # (k,7) the expert's own

        later_px = self._to_chw(self._read_views(np.array([base + j_grid * k + phase],
                                                          dtype=np.int64))[0])

        grip_hold_raw = float(self._actions[base + a_grid * k + phase + self.action_offset - 1, 6])
        ctx_act = (ctx_act - self.action_loc) / self.action_scale
        expert_raw = (anchor_chunk_raw - self.action_loc) / self.action_scale
        grip_hold = (grip_hold_raw - float(self.action_loc[6])) / float(self.action_scale[6])
        expert_a = expert_raw.mean(axis=0)
        grip_sign = 1.0 if float(anchor_chunk_raw[0, 6]) > 0.0 else -1.0
        prev_grip_sign = 1.0 if grip_hold_raw > 0.0 else -1.0
        grip_switch = 1.0 if grip_sign != prev_grip_sign else 0.0

        prog_t, prog_tp1 = float(self._prog_t[idx]), float(self._prog_tp1[idx])
        ttg_t, ttg_tp1 = float(self._ttg_t[idx]), float(self._ttg_tp1[idx])
        if phase:
            denom = float(max(int(self.ep_len[ep_idx]) // k - 1, 1))
            prog_t = min(prog_t + phase / (k * denom), 1.0)
            prog_tp1 = min(prog_tp1 + phase / (k * denom), 1.0)
            ttg_t = max(ttg_t - phase / k, 0.0)
            ttg_tp1 = max(ttg_tp1 - phase / k, 0.0)

        sample = {
            "ctx_pixels": torch.from_numpy(np.ascontiguousarray(ctx_px)),
            "ctx_action": torch.from_numpy(np.ascontiguousarray(
                ctx_act.reshape(self.history_size, -1))),
            "pixels_tp1": torch.from_numpy(np.ascontiguousarray(later_px)),
            "grip_hold": torch.tensor(float(grip_hold), dtype=torch.float32),
            "expert_a_norm7": torch.from_numpy(np.ascontiguousarray(expert_a.astype(np.float32))),
            "expert_grip_sign": torch.tensor(grip_sign, dtype=torch.float32),
            "expert_grip_switch": torch.tensor(grip_switch, dtype=torch.float32),
            "lang_embed": torch.from_numpy(self.task_lang_embed[ep_idx]),
            "prog_t": torch.tensor(prog_t, dtype=torch.float32),
            "prog_tp1": torch.tensor(prog_tp1, dtype=torch.float32),
            "ttg_t": torch.tensor(ttg_t, dtype=torch.float32),
            "ttg_tp1": torch.tensor(ttg_tp1, dtype=torch.float32),
            "gap": torch.tensor(float(j_grid - a_grid), dtype=torch.float32),
            "task_idx": torch.tensor(int(self.ep_task_idx[ep_idx]), dtype=torch.long),
            "episode_idx": torch.tensor(ep_idx, dtype=torch.long),
            "frame_idx": torch.tensor(int(self.ep_offset[ep_idx]) + a_grid * k + phase,
                                      dtype=torch.long),
            "ep_len": torch.tensor(int(self.ep_len[ep_idx]), dtype=torch.long),
            "pixels_next": torch.from_numpy(np.ascontiguousarray(next_px)),
            "lang_embed_wrong": torch.from_numpy(self._wrong_lang_of_ep[ep_idx]),
        }
        if self.anchor_phase == "random":
            sample["anchor_phase"] = torch.tensor(phase, dtype=torch.long)
        if self.grip_events:
            g_loc, g_scale = float(self.action_loc[6]), float(self.action_scale[6])
            sample["grip_open_norm"] = torch.tensor((-1.0 - g_loc) / g_scale, dtype=torch.float32)
            sample["grip_close_norm"] = torch.tensor((1.0 - g_loc) / g_scale, dtype=torch.float32)
            r0 = base + a_grid * k + phase + self.action_offset         # this anchor's first command
            nxt = np.nonzero((self._actions[r0: base + int(self.ep_len[ep_idx]), 6] > 0)
                             != (grip_hold_raw > 0))[0]
            sample["grip_event_offset"] = torch.tensor(int(nxt[0]) if len(nxt) else -1,
                                                       dtype=torch.long)
        return sample
