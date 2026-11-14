"""Closed-loop controller for Seer and Seer-PPP (one command per 20 Hz control step)."""

from collections import deque

import numpy as np
import torch


def obs_to_seer_state(obs) -> np.ndarray:
    import robosuite.utils.transform_utils as T
    return np.concatenate([np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                           T.quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
                           np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)]).astype(np.float32)


def obs_to_seer_frames(obs) -> np.ndarray:
    """-> (2, h, w, 3) uint8 [front, wrist], unflipped (the stored convention)."""
    return np.stack([np.asarray(obs["agentview_image"], dtype=np.uint8),
                     np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)], axis=0)


class SeerPlanner:
    """Seer: the action head's per-tick chunk, temporally ensembled (Seer's deployment), gripper from
    its ensembled probability. Seer-PPP: the per-tick chunk `a_k = m dp(F(s, u), l)/du_k` at u = 0,
    ensembled the same way, gripper from the field's flip advantage against `grip_field_eps`."""

    def __init__(self, model, action_bounds, device, ensembling_temp: float = 0.01,
                 grip_field_eps: float = 0.0, max_steps: int = 600):
        self.model, self.net, self.device = model, model.net, device
        self.mode = model.policy_mode
        assert self.mode in ("seer", "seer_ppp"), self.mode
        self.S, self.K = model.seq_len, model.chunk
        self.lo = np.asarray(action_bounds[0], dtype=np.float64)
        self.hi = np.asarray(action_bounds[1], dtype=np.float64)
        self.loc = model.action_loc.detach().cpu().numpy().astype(np.float64)
        self.scale = model.action_scale.detach().cpu().numpy().astype(np.float64)
        self.ens_temp = float(ensembling_temp)
        self.grip_field_eps = float(grip_field_eps)
        self.max_steps = int(max_steps)
        self.reset()

    def reset(self):
        self._feat = deque(maxlen=self.S)       # per-frame MAE features (2,197,768), on device
        self._state = deque(maxlen=self.S)
        self._grip_sign = -1.0                  # starts open (the settle action opens the hand)
        self._t = 0
        self._ens = torch.zeros(self.max_steps + self.K + 1, self.max_steps + self.K + 1, 7,
                                device=self.device)
        self._ens_pose = torch.zeros(self.max_steps + self.K + 1, self.max_steps + self.K + 1, 6,
                                     device=self.device)
        self._ens_set = torch.zeros(self.max_steps + self.K + 1, self.max_steps + self.K + 1,
                                    dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def observe(self, obs) -> None:
        fr = torch.from_numpy(obs_to_seer_frames(obs)).to(self.device)
        x = self.net.preprocess(fr.unsqueeze(0), augment=False)            # (1,2,3,224,224)
        self._feat.append(self.net.encode_images(x)[0])
        self._state.append(torch.from_numpy(obs_to_seer_state(obs)).to(self.device))

    def _base(self, text):
        feat = torch.stack(list(self._feat)).unsqueeze(0)                  # (1,n,2,197,768)
        img, cls = self.net.frame_tokens(feat)
        st = torch.stack(list(self._state)).unsqueeze(0).float()
        return self.net.base_tokens(text, st, img, cls)

    def _to_raw(self, a_norm6: np.ndarray, grip_sign: float) -> np.ndarray:
        raw = np.concatenate([self.loc[:6] + self.scale[:6] * a_norm6, [grip_sign]])
        return np.clip(raw, self.lo, self.hi)

    def act(self, task_idx: int) -> dict:
        assert len(self._feat) > 0, "observe() before act()"
        text = self.model.text_table[int(task_idx)].unsqueeze(0)
        diag = {}
        a_raw = self._act_ppp(text, diag) if self.mode == "seer_ppp" else self._act_bc(text, diag)
        self._t += 1
        return {"action_raw": a_raw, "diag": diag}

    @torch.no_grad()
    def _act_bc(self, text, diag) -> np.ndarray:
        base = self._base(text)
        out = self.net.run(base, [{"u6": None}], queries=True, read_group=0)
        arm, glogit = self.net.bc_decode(out["query"])
        arm, gprob = arm[0, -1], torch.sigmoid(glogit[0, -1])                # (K,6), (K,)
        t = self._t
        self._ens[t, t:t + self.K] = torch.cat([arm, gprob.unsqueeze(-1)], dim=-1)
        cur = self._ens[:, t]
        cur = cur[torch.all(cur != 0, dim=1)]
        w = torch.exp(-self.ens_temp * torch.arange(len(cur), device=self.device, dtype=cur.dtype))
        a = (cur * (w / w.sum()).unsqueeze(1)).sum(dim=0)
        grip = 1.0 if float(a[6]) > 0.5 else -1.0
        self._grip_sign = grip
        diag.update({"grip_prob": float(a[6]), "pose_norm6": float(self.model.normalize(a[:6]).norm())})
        raw = np.concatenate([a[:6].double().cpu().numpy(), [grip]])
        return np.clip(raw, self.lo, self.hi)

    def _act_ppp(self, text, diag) -> np.ndarray:
        dev = self.device
        with torch.no_grad():
            base = self._base(text)
        hold = self._grip_sign
        K = self.K
        with torch.enable_grad():
            u0 = torch.zeros(1, 1, K, 6, device=dev, requires_grad=True)
            Z = {"u6": u0, "grip": torch.full((1, 1, K), hold, device=dev), "last": True}
            z0 = self.model.group_states(base, [Z])[0][:, 0]                            # (1,D)
            q = self.model.head(z0, text.expand(1, -1)).squeeze(-1)                       # (1,)
            (g,) = torch.autograd.grad(q.sum(), u0)
        with torch.no_grad():
            chunk, M, _ = self.model.geom(z0.detach().repeat_interleave(K, 0), g.reshape(K, 6))
            chunk = chunk.view(1, K, 6).mean(dim=0)
            a6 = self._ensemble_pose(chunk).view(1, 6)
            g = g.reshape(1, K, 6).mean(dim=0)[:1]
            q = q.mean()
            z = torch.zeros(1, 1, K, 6, device=dev)
            one = torch.ones(1, 1, K, device=dev)
            groups = [{"u6": z, "grip": -one, "last": True}, {"u6": z, "grip": one, "last": True}]
            zz = torch.cat([s[:, 0] for s in self.model.group_states(base, groups)])
            qq = self.model.head(zz, text.expand(len(groups), -1)).squeeze(-1)
            d = float(qq[1] - qq[0])
        adv = -d if hold > 0 else d
        thr = self.grip_field_eps
        grip = -hold if adv > thr else hold
        self._grip_sign = grip
        a6 = a6[0].double().cpu().numpy()
        diag.update({"grip_adv": adv, "grip_d": d, "grip_thr": thr, "q": float(q.detach()),
                     "g_norm": float(g.norm()), "m": float(M[0, 0, 0]),
                     "pose_norm6": float(np.linalg.norm(a6)),
                     "pred_progress_rate": float((g[0].double().cpu().numpy() * a6).sum())})
        return self._to_raw(a6, grip)

    def _ensemble_pose(self, chunk: torch.Tensor) -> torch.Tensor:
        """Seer's temporal ensembling for a per-tick PPP chunk (K,6): every chunk that covers this
        tick, oldest first, weighted exp(-ens_temp * order). -> (6,)."""
        t = self._t
        self._ens_pose[t, t:t + self.K] = chunk
        self._ens_set[t, t:t + self.K] = True
        cur = self._ens_pose[:, t][self._ens_set[:, t]]
        w = torch.exp(-self.ens_temp * torch.arange(len(cur), device=self.device, dtype=cur.dtype))
        return (cur * (w / w.sum()).unsqueeze(1)).sum(dim=0)

    @torch.no_grad()
    def progress_now(self, task_idx: int) -> float:
        if not self.model.is_ppp:
            return float("nan")
        text = self.model.text_table[int(task_idx)].unsqueeze(0)
        K = self.K
        Z = {"u6": torch.zeros(1, 1, K, 6, device=self.device),
             "grip": torch.full((1, 1, K), self._grip_sign, device=self.device), "last": True}
        z = self.model.group_states(self._base(text), [Z])[0][:, 0]
        return float(self.model.head(z, text))
