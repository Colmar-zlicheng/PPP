from time import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils.builder import DATASET
from lib.utils.logger import logger

from .lewm_data import _episode_split, ensure_shm_episode_export
from .libero_timing import CORRECTED, resolve_timing


@DATASET.register_module()
class LiberoSeerDataset(Dataset):
    def __init__(self, cfg, data_preset=None):
        self.h5_path = cfg.DATA_ROOT
        self.seq_len = int(cfg.get("SEQ_LEN", 7))
        self.future = int(cfg.get("FUTURE_STEPS", 3))
        self.chunk = int(cfg.get("ACTION_PRED_STEPS", 3))
        assert self.future >= 1 and self.chunk >= 1
        self.window = self.seq_len + self.future
        self.stride = int(cfg.get("WINDOW_STRIDE", 8))
        self.phase_mode = str(cfg.get("WINDOW_PHASE", "random"))
        assert self.phase_mode in ("random", "grid")
        self.gap_set = [int(g) for g in cfg.get("GAP_SET", [4, 8, 16, 32, 64, 120])]
        self.cameras = list(cfg.get("CAMERAS", ["pixels", "pixels_wrist"]))
        assert len(self.cameras) == 2, "Seer reads exactly two views [front, wrist]"
        self.split = cfg.get("SPLIT", "all")
        self.seed = int(cfg.get("SEED", 3072))
        self.min_lang_dist = float(cfg.get("CROSS_EP_MIN_LANG_DIST", 0.3))
        self.later_window = bool(cfg.get("LATER_WINDOW", False))
        assert not self.later_window or self.chunk == self.future, "LATER_WINDOW needs ACTION_PRED_STEPS == FUTURE_STEPS"
        self.later_end = str(cfg.get("LATER_WINDOW_END", "forecast")).lower()
        assert self.later_end in ("forecast", "frame"), f"LATER_WINDOW_END must be forecast or frame, got {self.later_end}"
        assert self.later_end == "forecast" or self.later_window, "LATER_WINDOW_END frame needs LATER_WINDOW: true"
        self.future_states = bool(cfg.get("FUTURE_STATES", False))

        t0 = time()
        with h5py.File(self.h5_path, "r") as f:
            self.ep_offset = f["ep_offset"][:].astype(np.int64)
            self.ep_len = f["ep_len"][:].astype(np.int64)
            self.ep_task_idx = f["ep_task_idx"][:].astype(np.int64)
            self.timing = resolve_timing(cfg, f)
            assert self.timing.contract == CORRECTED and self.timing.action_offset("osc") == 1, (
                "LiberoSeerDataset implements the corrected post-action contract only "
                f"(got {self.timing.as_dict()})")
            self.action_loc = f["action_loc"][:].astype(np.float32)
            self.action_scale = f["action_scale"][:].astype(np.float32)
            self.lang_by_task = f["task_lang_embed_by_task"][:].astype(np.float32)
            ep_indices = _episode_split(self.ep_offset, float(cfg.get("TRAIN_RATIO", 0.9)), self.seed,
                                        self.split, ep_task_idx=self.ep_task_idx,
                                        stratify=bool(cfg.get("STRATIFY", True)))
        self.ep_indices = np.sort(np.asarray(ep_indices, dtype=np.int64))
        assert len(np.unique(self.ep_indices)) == len(self.ep_indices)
        assert not cfg.get("FILTER_SUCCESS", False), "DATASET.FILTER_SUCCESS is not supported"
        lens = self.ep_len[self.ep_indices]
        self._local_base = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        n_frames = int(lens.sum())

        need = max(self.window, self.seq_len + self.chunk)
        slots, starts, n_short = [], [], 0
        for slot, ei in enumerate(self.ep_indices):
            last = int(self.ep_len[ei]) - need
            if last < 0:
                n_short += 1
                continue
            s = np.arange(0, last + 1, self.stride, dtype=np.int64)
            slots.append(np.full(len(s), slot, dtype=np.int64))
            starts.append(s)
        self._slot = np.concatenate(slots)
        self._start = np.concatenate(starts)
        self._need = need

        episodes = np.stack([self.ep_offset[self.ep_indices], lens], axis=1).astype(np.int64)
        self._pixel_paths = [ensure_shm_episode_export(self.h5_path, cam, episodes)
                             for cam in self.cameras]
        self._pixels = None
        with h5py.File(self.h5_path, "r") as f:
            self._actions = np.empty((n_frames, 7), dtype=np.float32)
            self._state = np.empty((n_frames, 8), dtype=np.float32)
            for slot, ei in enumerate(self.ep_indices):
                o, n = int(self.ep_offset[ei]), int(self.ep_len[ei])
                w = int(self._local_base[slot])
                self._actions[w:w + n] = f["actions"][o:o + n]
                self._state[w:w + n, 0:3] = f["ee_pos"][o:o + n]
                self._state[w:w + n, 3:6] = f["ee_ori"][o:o + n]
                self._state[w:w + n, 6:8] = f["gripper_states"][o:o + n]
        self._assign_wrong_lang()
        logger.info(f"LiberoSeerDataset[{self.split}]: {len(self._start)} windows "
                    f"(stride {self.stride}, phase {self.phase_mode}) from "
                    f"{len(self.ep_indices) - n_short} episodes ({n_short} too short), "
                    f"seq_len {self.seq_len} + future {self.future}, chunk {self.chunk}, "
                    f"gaps {self.gap_set} raw ticks, {n_frames} frames in {time() - t0:.1f}s")

    def _assign_wrong_lang(self):
        rng = np.random.RandomState(self.seed + 1)
        emb = self.lang_by_task / (np.linalg.norm(self.lang_by_task, axis=1, keepdims=True) + 1e-8)
        dist = 1.0 - emb @ emb.T
        self.wrong_task_of_task = np.zeros(len(emb), dtype=np.int64)
        for t in range(len(emb)):
            cand = np.where(dist[t] >= self.min_lang_dist)[0]
            if len(cand) == 0:
                cand = np.array([int(np.argmax(dist[t]))])
            self.wrong_task_of_task[t] = int(cand[rng.randint(len(cand))])

    def _ensure_pixels(self):
        if self._pixels is None:
            self._pixels = [np.load(p, mmap_mode="r") for p in self._pixel_paths]
        return self._pixels

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pixels"] = None
        return state

    def __len__(self) -> int:
        return len(self._start)

    def window_at(self, slot: int, s: int, gap: int = None) -> dict:
        """The sample for episode slot `slot`, window start `s` (raw frame within the episode)."""
        ei = int(self.ep_indices[slot])
        T = int(self.ep_len[ei])
        base = int(self._local_base[slot])
        S, K, Fs = self.seq_len, self.chunk, self.future
        assert 0 <= s <= T - self._need, (slot, s, T)
        anchor = s + S - 1 + (Fs if self.later_window else 0)
        if gap is None:
            fits = [g for g in self.gap_set if anchor + g <= T - 1]
            gap = int(fits[np.random.randint(len(fits))]) if fits else T - 1 - anchor
        later = min(anchor + int(gap), T - 1)

        frames = np.append(np.arange(s, s + self.window, dtype=np.int64), later) + base
        px = self._ensure_pixels()
        views = np.stack([px[0][frames], px[1][frames]], axis=1)          # (W+1, 2, h, w, 3)

        pos = np.arange(S, dtype=np.int64)
        act_idx = base + s + pos[:, None] + 1 + np.arange(K, dtype=np.int64)[None, :]   # (S,K)
        actions_raw = self._actions[act_idx]                                         # (S,K,7)
        actions_norm = (actions_raw - self.action_loc) / self.action_scale
        hold = self._actions[base + s + pos, 6]                                      # (S,)
        hold_sign = np.where(hold > 0, 1.0, -1.0).astype(np.float32)
        first_sign = np.where(actions_raw[:, 0, 6] > 0, 1.0, -1.0).astype(np.float32)
        ev = np.full(S, -1, dtype=np.int64)
        ep_close = self._actions[base:base + T, 6] > 0
        for i in range(S):
            r0 = s + i + 1
            nxt = np.nonzero(ep_close[r0:] != (hold_sign[i] > 0))[0]
            if len(nxt):
                ev[i] = int(nxt[0])
        task = int(self.ep_task_idx[ei])
        denom = float(max(T - 1, 1))
        extra = {}
        if self.later_window:
            last = later - (Fs if self.later_end == "forecast" else 0)
            first = last - S + 1
            assert first >= 0, (slot, s, later)
            lf = np.arange(first, last + 1, dtype=np.int64) + base
            extra = {
                "later_frames": torch.from_numpy(np.ascontiguousarray(np.stack([px[0][lf], px[1][lf]], axis=1))),
                "later_states": torch.from_numpy(np.ascontiguousarray(self._state[lf])),
                "later_window_end": torch.tensor(later - last, dtype=torch.long),
            }
            if self.later_end == "forecast":
                la = self._actions[base + last + 1 + np.arange(K, dtype=np.int64)]          # (K,7)
                extra.update({
                    "later_actions_raw": torch.from_numpy(np.ascontiguousarray(la)),
                    "later_actions_norm": torch.from_numpy(np.ascontiguousarray(
                        ((la - self.action_loc) / self.action_scale).astype(np.float32)))})
        if self.future_states:
            extra["future_states"] = torch.from_numpy(np.ascontiguousarray(
                self._state[base + s + S + np.arange(Fs, dtype=np.int64)]))
        return {**extra,
            "frames": torch.from_numpy(np.ascontiguousarray(views)),
            "states": torch.from_numpy(np.ascontiguousarray(self._state[base + s + pos])),
            "actions_raw": torch.from_numpy(np.ascontiguousarray(actions_raw)),
            "actions_norm": torch.from_numpy(np.ascontiguousarray(actions_norm.astype(np.float32))),
            "grip_hold": torch.from_numpy(hold_sign),
            "expert_grip_sign": torch.from_numpy(first_sign),
            "expert_grip_switch": torch.from_numpy((first_sign != hold_sign).astype(np.float32)),
            "grip_event_offset": torch.from_numpy(ev),
            "prog_t": torch.tensor(anchor / denom, dtype=torch.float32),
            "prog_tp1": torch.tensor(later / denom, dtype=torch.float32),
            "gap": torch.tensor(float(later - anchor), dtype=torch.float32),
            "task_idx": torch.tensor(task, dtype=torch.long),
            "wrong_task_idx": torch.tensor(int(self.wrong_task_of_task[task]), dtype=torch.long),
            "episode_idx": torch.tensor(ei, dtype=torch.long),
            "frame_idx": torch.tensor(int(self.ep_offset[ei]) + anchor, dtype=torch.long),
            "ep_len": torch.tensor(T, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> dict:
        slot, s0 = int(self._slot[idx]), int(self._start[idx])
        if self.phase_mode == "random":
            T = int(self.ep_len[self.ep_indices[slot]])
            hi = min(self.stride - 1, T - self._need - s0)
            s = s0 + int(np.random.randint(0, hi + 1))
            return self.window_at(slot, s)
        gap = self.gap_set[idx % len(self.gap_set)]
        return self.window_at(slot, s0, gap=gap)
