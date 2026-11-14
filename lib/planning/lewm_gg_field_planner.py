"""Closed-loop controller for the LeWM-based PPP field and the LeWM-based BC policy."""

from typing import Optional

import numpy as np
import torch

from lib.models.lewm.lewm_field_adapter import predict_at_base, progress_covector


class LeWMGGFieldPlanner:
    """One replanning call = one 7-dim normalized OSC_POSE command.

    Args:
        model: a built `LeWMGGFieldCotrainModel` (PPP: pose from the field's action gradient,
            gripper from the field's finite difference) or `LeWMBCModel` (BC: pose head and
            gripper switching head), weights loaded.
        action_loc/action_scale: (7,) raw-action normalization from the consolidated H5 -- the
            same arrays the training actions were normalized with.
        action_bounds: `(in_min, in_max, out_min, out_max)` from
            `lib.envs.libero_env.get_action_bounds`; the input bounds clip the raw command.
        ctx_decim: decisions between two context frames (context frames are `FRAMESKIP` sim
            steps apart, as in training, while the policy replans every step).
        grip_field_eps: the field gripper switches when the flip advantage exceeds this
            (potential units).
        pose_grip_settle: the pose query keeps seeing the gripper as open after a close command
            until the fingers have stopped closing (finger-width feedback).
    """
    SETTLE_TOL = 0.002
    SETTLE_MIN_STEPS = 3
    SETTLE_MAX_STEPS = 15

    def __init__(self, model, action_loc, action_scale, action_bounds, device,
                 ctx_decim: int = 1, grip_field_eps: float = 0.0, pose_grip_settle: bool = False):
        self.policy_mode = str(model.policy_mode)
        assert self.policy_mode in ("field", "bc_wm"), self.policy_mode
        self.has_potential = self.policy_mode == "field"
        self.field_gripper = getattr(model, "field_gripper", None)
        assert self.has_potential == (self.field_gripper is not None)
        self.grip_field_eps = float(grip_field_eps)
        self.pose_grip_settle = bool(pose_grip_settle)
        self.model = model
        self.wm = model.wm
        assert self.wm.n_grip == 1 and self.wm.action_dim == 7, "one-arm 7-DoF actions only"
        self.device = device
        self.action_loc = torch.as_tensor(action_loc, dtype=torch.float32, device=device)
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=device)
        self.action_bounds = action_bounds

        self.history_size = self.wm.history_size
        self.frameskip = self.wm.frameskip
        self.cameras = self.wm.cameras
        self._grip_sign, self._grip_hold_norm = self._open_grip_state()
        self._ctx_emb = None      # (1,H,D) rolling window of frame embeddings
        self._ctx_act = None
        self.ctx_decim = int(ctx_decim)
        assert self.ctx_decim >= 1, f"ctx_decim must be >= 1, got {ctx_decim}"
        assert self.frameskip % self.ctx_decim == 0, (
            f"ctx_decim={self.ctx_decim} must divide FRAMESKIP={self.frameskip}: one context "
            f"action slot holds the commands executed between two context frames")
        self._emb_buf = []        # per-decision frame embeddings, oldest first; [0] is the reset
        self._cmd_buf = []        # `_cmd_buf[k]` carries `_emb_buf[k]` -> `_emb_buf[k+1]`
        self._z_next_pred = None
        self.last_latent_drift = float("nan")
        self._drift_hist = []

    def _grip_raw_to_norm(self, raw: float) -> float:
        c = 6
        return (float(raw) - float(self.action_loc[c].item())) / float(self.action_scale[c].item())

    def _open_grip_state(self):
        return -1.0, self._grip_raw_to_norm(-1.0)

    def _pad_cmd7(self) -> torch.Tensor:
        """(7,) normalized command for action history the episode does not have yet: the pose
        channels at the normalized zero (the corpus median, `action_loc`) and the gripper OPEN."""
        pad = torch.zeros(7, device=self.device)
        pad[6] = self._grip_raw_to_norm(-1.0)
        return pad

    def _to_raw(self, a_norm7: torch.Tensor) -> np.ndarray:
        a_raw = (a_norm7 * self.action_scale + self.action_loc).detach().cpu().numpy()
        return np.clip(a_raw, self.action_bounds[0], self.action_bounds[1])

    def obs_to_frame(self, obs: dict) -> torch.Tensor:
        """Live env obs -> (1,1,V,C,H,W) float32 in [0,1], the world model's `pixels` contract."""
        key = {"pixels": "agentview_image", "pixels_wrist": "robot0_eye_in_hand_image"}
        views = []
        for cam in self.cameras:
            if cam in obs:
                views.append(np.asarray(obs[cam], dtype=np.float32) / 255.0)
            else:
                assert cam in key, f"no live-env observation key known for camera {cam!r}"
                views.append(np.asarray(obs[key[cam]], dtype=np.float32) / 255.0)
        px = np.stack(views, axis=0)                      # (V,H,W,C)
        px = np.moveaxis(px, -1, -3)                      # (V,C,H,W)
        return torch.from_numpy(np.ascontiguousarray(px)).unsqueeze(0).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def bootstrap_from_obs(self, obs: dict) -> None:
        """Start an episode: repeat the initial frame `HISTORY_SIZE` times and fill the past
        action chunks with `_pad_cmd7` (pose at the corpus median, gripper open)."""
        frame = self.obs_to_frame(obs)                                   # (1,1,V,C,H,W)
        emb = self.wm.encode(frame)                                      # (1,1,D)
        self._ctx_emb = emb.expand(1, self.history_size, emb.shape[-1]).contiguous()
        self._ctx_act = (self._pad_cmd7().to(self._ctx_emb.dtype).repeat(self.frameskip)
                         .reshape(1, 1, self.frameskip * 7).repeat(1, self.history_size, 1))
        self._emb_buf, self._cmd_buf = [emb], []
        self._grip_sign, self._grip_hold_norm = self._open_grip_state()
        if self.pose_grip_settle:
            self._ctx_act_pose = self._ctx_act.clone()
            self._cmd_buf_pose = []
            self._pose_grip_closed = False       # the pose has been told the close happened
            self._settle_age = 0                 # steps since the close command, while not told
            self._last_width = self._finger_width(obs)
        self._z_next_pred, self.last_latent_drift, self._drift_hist = None, float("nan"), []

    @torch.no_grad()
    def push_observation(self, obs: dict, executed_a_norm7: torch.Tensor) -> None:
        ex = executed_a_norm7.to(self.device)
        assert ex.numel() == 7, f"push_observation expects a (7,) command, got {tuple(ex.shape)}"
        chunk = ex.reshape(1, 1, 7).repeat(1, 1, self.frameskip)
        self._ctx_act[:, -1:] = chunk.to(self._ctx_act.dtype)
        if self.pose_grip_settle:
            pose_chunk = self._pose_view_chunk(chunk, obs)
            self._ctx_act_pose[:, -1:] = pose_chunk.to(self._ctx_act_pose.dtype)
            self._cmd_buf_pose.append(pose_chunk)
        emb = self.wm.encode(self.obs_to_frame(obs))                     # (1,1,D)
        self._cmd_buf.append(chunk)
        self._emb_buf.append(emb)
        if self._z_next_pred is not None:
            den = float(emb[:, 0].norm().item()) or 1.0
            self.last_latent_drift = float((emb[:, 0] - self._z_next_pred).norm().item()) / den
            self._drift_hist.append(self.last_latent_drift)
        if self.ctx_decim == 1:
            self._ctx_emb = torch.cat([self._ctx_emb[:, 1:], emb], dim=1)
            self._ctx_act = torch.cat([self._ctx_act[:, 1:],
                                       torch.zeros_like(self._ctx_act[:, -1:])], dim=1)
            if self.pose_grip_settle:
                self._ctx_act_pose = torch.cat([self._ctx_act_pose[:, 1:],
                                                torch.zeros_like(self._ctx_act_pose[:, -1:])], dim=1)
        else:
            self._rebuild_decimated_context()

    def _rebuild_decimated_context(self) -> None:
        """Rebuild `_ctx_emb` / `_ctx_act` by SAMPLING the per-decision history every
        `ctx_decim` decisions, newest slot = the current observation."""
        H, S, n = self.history_size, self.ctx_decim, len(self._emb_buf)
        idx = [max(0, n - 1 - (H - 1 - h) * S) for h in range(H)]
        self._ctx_emb = torch.cat([self._emb_buf[i] for i in idx], dim=1)
        template = self._ctx_act[:, -1:]
        self._ctx_act = self._decimated_action_slots(self._cmd_buf, idx, template)
        if self.pose_grip_settle:
            self._ctx_act_pose = self._decimated_action_slots(self._cmd_buf_pose, idx, template)

    def _decimated_action_slots(self, cmd_buf, idx, template) -> torch.Tensor:
        """`_rebuild_decimated_context`'s action slots from one per-decision command buffer."""
        H, S = self.history_size, self.ctx_decim
        rep = self.frameskip // S
        slots = []
        for h, i in enumerate(idx):
            if h == H - 1:
                slots.append(torch.zeros_like(template))
                continue
            cmds = cmd_buf[i:idx[h + 1]]
            first = [c.reshape(1, 1, self.frameskip, 7)[:, :, 0] for c in cmds]
            while len(first) < S:      # younger than the window: pad (pose median, gripper open)
                first.insert(0, self._pad_cmd7().reshape(1, 1, 7).to(template.dtype))
            row = torch.cat([f.repeat(1, 1, rep) for f in first], dim=-1)   # (1,1,frameskip*7)
            slots.append(row.to(template.dtype))
        return torch.cat(slots, dim=1)

    @staticmethod
    def _finger_width(obs: dict) -> Optional[float]:
        q = obs.get("robot0_gripper_qpos") if hasattr(obs, "get") else None
        return None if q is None else float(np.abs(np.asarray(q, dtype=np.float64)).sum())

    def _pose_view_chunk(self, chunk: torch.Tensor, obs: dict) -> torch.Tensor:
        rows = chunk.reshape(1, 1, self.frameskip, 7)
        g_raw = float(self.action_loc[6].item() + self.action_scale[6].item() * rows[0, 0, 0, 6].item())
        closed = g_raw > 0.0
        told_closed = self._pose_grip_closed
        w = self._finger_width(obs)
        if not closed:
            self._pose_grip_closed, self._settle_age = False, 0
        elif not self._pose_grip_closed:
            self._settle_age += 1
            settled = (w is not None and self._last_width is not None
                       and self._settle_age >= self.SETTLE_MIN_STEPS
                       and (w - self._last_width) > -self.SETTLE_TOL)
            if settled or self._settle_age >= self.SETTLE_MAX_STEPS:
                self._pose_grip_closed = True
        self._last_width = w
        if closed and not told_closed:
            rows = rows.clone()
            rows[..., 6] = self._grip_raw_to_norm(-1.0)
        return rows.reshape(1, 1, self.frameskip * 7)

    def _pose_inputs(self):
        """(ctx_act, grip_hold) the POSE query is evaluated at."""
        hold = self._grip_hold_norm
        ctx_act = self._ctx_act
        if self.pose_grip_settle:
            ctx_act = self._ctx_act_pose
            if self._grip_sign > 0 and not self._pose_grip_closed:
                hold = self._grip_raw_to_norm(-1.0)
        return ctx_act, torch.full((1, 1), hold, device=self.device, dtype=self._ctx_emb.dtype)

    @torch.no_grad()
    def progress_at_context(self, lang: torch.Tensor) -> float:
        """`p_theta(z_t, l)` at the last observed frame (NaN for BC, which has no potential)."""
        assert self._ctx_emb is not None, "call bootstrap_from_obs() before progress_at_context()"
        lang = lang if lang.dim() == 2 else lang.unsqueeze(0)
        if not self.has_potential:
            return float("nan")
        return float(self.model.head(self._ctx_emb[:, -1], lang).squeeze().item())

    def plan_action(self, lang: torch.Tensor) -> dict:
        """-> {"action_raw", "action_norm7", "diag": {...}}."""
        assert self._ctx_emb is not None, "call bootstrap_from_obs() before plan_action()"
        lang = lang if lang.dim() == 2 else lang.unsqueeze(0)
        diag = {"nan_detected": False}

        pose_ctx_act, grip_hold = self._pose_inputs()
        if not self.has_potential:
            with torch.no_grad():
                z_next = predict_at_base(self.wm, self._ctx_emb, pose_ctx_act, grip_hold)
                a6 = self.model.policy_head(z_next, lang).squeeze(0)         # (6,)
                logit = self.model.grip_head(self._ctx_emb[:, -1], lang).squeeze(0)
                grip_sign = 1.0 if float(logit.item()) > 0.0 else -1.0
                diag["grip_logit"] = float(logit.item())
            if torch.isnan(a6).any():
                diag["nan_detected"] = True
                a6 = torch.zeros_like(a6)
            diag.update({"predicted_progress_rate": float("nan"), "g_norm": 0.0,
                         "m_trace_over_da": float("nan"), "w_fro": float("nan"),
                         "potential_next": float("nan")})
            self._z_next_pred = z_next.detach()
            return self._finish_action(a6, grip_sign, diag)

        g, z_next, q = progress_covector(self.wm, self.model.head, self._ctx_emb, pose_ctx_act,
                                         lang, grip_hold, create_graph=False)
        if torch.isnan(g).any():
            diag["nan_detected"] = True
            g = torch.zeros_like(g)

        with torch.no_grad():
            M, W = self.model.geom.operators(z_next.detach(), 1, z_next.dtype, z_next.device)
            a6 = torch.einsum("...ij,...j->...i", M + W, g.detach()).squeeze(0)
            grip_sign = self.field_gripper.decide(self, lang, diag)

        diag["predicted_progress_rate"] = float((g.squeeze(0) * a6).sum().item())
        diag["g_norm"] = float(g.norm().item())
        diag["m_trace_over_da"] = float(M.diagonal(dim1=-2, dim2=-1).mean().item())
        diag["w_fro"] = float(W.pow(2).sum().sqrt().item())
        diag["potential_next"] = float(q.detach().item())
        self._z_next_pred = z_next.detach()
        return self._finish_action(a6, grip_sign, diag)

    def _finish_action(self, a6, grip_sign: float, diag: dict) -> dict:
        """Everything between "the policy chose a pose command" and "the environment gets a row"."""
        diag["a6"] = a6.detach().cpu().numpy().tolist()
        grip_norm = self._grip_raw_to_norm(grip_sign)
        gcol = torch.full(a6.shape[:-1] + (1,), grip_norm, device=self.device, dtype=a6.dtype)
        a_norm7 = torch.cat([a6.detach(), gcol], dim=-1)             # (7,)
        self._grip_sign, self._grip_hold_norm = grip_sign, grip_norm

        a_raw = self._to_raw(a_norm7)
        diag.update({
            "pose_norm6": float(a6.norm(dim=-1).mean().item()),
            "action_norm_normalized": float(a_norm7.norm(dim=-1).mean().item()),
            "gripper_snapped_to": grip_sign,
            "action_raw": a_raw.tolist(),
        })
        return {"action_raw": a_raw, "action_norm7": a_norm7, "diag": diag}
