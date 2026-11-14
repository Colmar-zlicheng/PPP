from typing import Optional

import torch

from lib.models.lewm.lewm_field_adapter import base_point_context


def progress_contrast(adapter, head, ctx_emb: torch.Tensor, ctx_act: torch.Tensor,
                      lang: torch.Tensor, grip_open: torch.Tensor, grip_close: torch.Tensor,
                      u6_base: Optional[torch.Tensor] = None,
                      detach_dynamics: Optional[bool] = None):
    """`d = p(F(z, [u0, close]), l) - p(F(z, [u0, open]), l)` and its parts."""
    B, dev, dtype = ctx_emb.shape[0], ctx_emb.device, ctx_emb.dtype
    u6 = torch.zeros(B, 6, device=dev, dtype=dtype) if u6_base is None else u6_base.to(dev, dtype)
    grip = torch.cat([grip_open.to(dev, dtype).reshape(B, 1),
                      grip_close.to(dev, dtype).reshape(B, 1)], dim=0)            # (2B,1)
    field_ctx, act = base_point_context(adapter, ctx_emb.repeat(2, 1, 1), ctx_act.repeat(2, 1, 1),
                                        grip, u6.repeat(2, 1))
    z = adapter.predict_next(field_ctx, act, update_bn_stats=False,
                             detach_dynamics=detach_dynamics)                     # (2B,D)
    q = head(z, lang.repeat(2, 1)).squeeze(-1)                                    # (2B,)
    q_open, q_close = q[:B], q[B:]
    dz = (z[B:] - z[:B]).detach().norm(dim=-1)
    return q_close - q_open, q_open, q_close, dz


def dwell_context(ctx_emb: torch.Tensor, ctx_act: torch.Tensor, grip_hold: torch.Tensor):
    """The CONSTRUCTED DWELL: the current frame's embedding in all H slots, and an action history
    of hold-position commands carrying the command in force."""
    B, H = ctx_emb.shape[:2]
    emb = ctx_emb[:, -1:].expand_as(ctx_emb).contiguous()
    act = ctx_act.clone().view(B, H, -1, 7)
    act[..., :6] = 0.0
    act[..., 6] = grip_hold.to(act.dtype).view(B, 1, 1)
    return emb, act.view_as(ctx_act)


class FieldGripper:
    """`MODEL.GRIPPER_FIELD` -- the gripper decision read off `p_theta o F`."""

    def __init__(self, gcfg, ctx_action_base: str, frameskip: int):
        self.tau_steps = float(gcfg.get("TAU_STEPS", 0.5))
        tol = gcfg.get("TOL_EARLY", {})
        self.tol_close = int(tol.get("CLOSE", 2))
        self.tol_open = int(tol.get("OPEN", 4))
        self.isolate = bool(gcfg.get("ISOLATE", True))
        assert bool(gcfg.get("DWELL", True)) and bool(gcfg.get("BALANCE", True)), (
            "GRIPPER_FIELD.DWELL and GRIPPER_FIELD.BALANCE must be true")
        assert gcfg.get("MARGIN_STEPS", None) is None and float(gcfg.get("BOUNDARY_STEPS", 0.0)) == 0.0, (
            "GRIPPER_FIELD.MARGIN_STEPS / BOUNDARY_STEPS are not supported")
        self.frameskip = int(frameskip)
        assert self.tau_steps > 0, f"GRIPPER_FIELD.TAU_STEPS must be > 0, got {self.tau_steps}"
        assert self.tol_close >= 0 and self.tol_open >= 0
        assert ctx_action_base == "median", (
            "GRIPPER_FIELD builds hold-position histories with the pose channels at the corpus "
            "median, which needs ACT_SUPERVISION.CTX_ACTION_BASE: median")
        self._stash = None
        self.val = {k: [0.0, 0] for k in ("recall", "false", "recall_dwell", "false_dwell")}

    def describe(self) -> str:
        return (f"gripper_field=ON (flip advantage of p o F at u0, switching-region target, early "
                f"tolerance close {self.tol_close} / open {self.tol_open} steps, dwell=True, "
                f"balance=True, isolate={self.isolate}, tau={self.tau_steps} grid steps)")

    def progress_step(self, batch: dict, dev) -> torch.Tensor:
        """(B,) one grid step of progress in `p_theta`'s units: the per-episode clock's
        `1/(n_grid - 1)`."""
        n_grid = torch.div(batch["ep_len"].to(dev), self.frameskip, rounding_mode="floor")
        return 1.0 / (n_grid - 1).clamp_min(1).float()

    def forward(self, model, batch: dict, ctx_emb: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """-> the command logit `d / tau` on the real context ((B,), > 0 = close). Stashes what
        `loss` and the meters need."""
        dev = ctx_emb.device
        ctx_act = batch["ctx_action"].to(dev)
        hold = batch["grip_hold"].to(dev)
        g_open, g_close = batch["grip_open_norm"].to(dev), batch["grip_close_norm"].to(dev)
        if self.isolate:
            ctx_emb = ctx_emb.detach()
        e_d, a_d = dwell_context(ctx_emb, ctx_act, hold)
        embs, acts = [ctx_emb, e_d], [ctx_act, a_d]
        n = len(embs)
        d, _, _, dz = progress_contrast(
            model.wm, model.head, torch.cat(embs), torch.cat(acts), lang.repeat(n, 1),
            g_open.repeat(n), g_close.repeat(n), detach_dynamics=True if self.isolate else None)
        B = ctx_emb.shape[0]
        step = self.progress_step(batch, dev)
        tau = self.tau_steps * step
        closed = hold > 0.5 * (g_open + g_close)
        flip = torch.where(closed, -1.0, 1.0)
        self._stash = {"adv": [flip * d[i * B:(i + 1) * B] for i in range(n)], "tau": tau,
                       "closed": closed, "dz": dz[:B].mean(), "step": step}
        return d[:B] / tau

    def region(self, batch: dict, closed: torch.Tensor) -> torch.Tensor:
        """(B,) bool: the anchor lies in the switching region (`GRIP_EVENTS` labels)."""
        off = batch["grip_event_offset"].to(closed.device)
        tol = torch.where(closed, self.tol_open, self.tol_close)   # hold closed -> next is an open
        return (off >= 0) & (off <= tol)

    def loss(self, batch: dict, w: torch.Tensor, cmd_logit: torch.Tensor):
        """-> (l_grip, stats). `cmd_logit` is `forward`'s output."""
        s = self._stash
        y = self.region(batch, s["closed"]).float()
        w = w.to(y.device, y.dtype).reshape(-1)
        terms, pos_w, neg_w = [], [], []
        for adv in s["adv"]:
            z = adv / s["tau"]
            terms.append(s["tau"] * torch.nn.functional.softplus(torch.where(y > 0, -z, z)))
            pos_w.append(w * y)
            neg_w.append(w * (1 - y))
        per, pw, nw = torch.cat(terms), torch.cat(pos_w), torch.cat(neg_w)
        l_grip = (0.5 * (per * pw).sum() / pw.sum().clamp_min(1.0)
                  + 0.5 * (per * nw).sum() / nw.sum().clamp_min(1.0))
        with torch.no_grad():
            tgt = (batch["expert_grip_sign"].to(y.device) > 0).float()
            correct = ((cmd_logit.detach() > 0).float() == tgt).float()
            stats = {"grip_acc": correct.mean(), "grip_dz": s["dz"],
                     "grip_d_steps": (s["adv"][0] / s["step"]).abs().median(),
                     "grip_region_frac": (y * w).sum() / w.sum().clamp_min(1.0)}
            for name, adv in zip(("", "_dwell"), s["adv"]):
                fire = (adv > 0).float()
                if bool((y * w > 0).any()):
                    stats[f"grip_recall{name}"] = (fire * y * w).sum() / (y * w).sum()
                stats[f"grip_false{name}"] = (fire * (1 - y) * w).sum() / ((1 - y) * w).sum().clamp_min(1.0)
            if "expert_grip_switch" in batch:
                m = batch["expert_grip_switch"].to(y.device) > 0
                if bool(m.any()):
                    stats["grip_acc_switch"] = correct[m].mean()
        return l_grip, stats

    def update_val(self, batch: dict) -> None:
        s = self._stash
        y = self.region(batch, s["closed"])
        for name, adv in zip(("", "_dwell"), s["adv"]):
            fire = adv > 0
            for key, mask in ((f"recall{name}", y), (f"false{name}", ~y)):
                m = self.val[key]
                m[0] += float(fire[mask].float().sum())
                m[1] += int(mask.sum())

    def log_val(self, summary, logger, name: str, epoch_idx: int) -> None:
        parts = []
        for key, (num, cnt) in self.val.items():
            if cnt == 0:
                continue
            v = num / cnt
            parts.append(f"grip_{key}={v:.4f} (n={cnt})")
            if summary is not None:
                summary.add_scalar(f"val/epoch_grip_{key}", v, epoch_idx)
        if parts:
            logger.info(f"{name} val epoch {epoch_idx}: " + " ".join(parts))
        self.val = {k: [0.0, 0] for k in self.val}

    @torch.no_grad()
    def decide(self, planner, lang: torch.Tensor, diag: dict) -> float:
        """The deployed rule on the planner's own (real) context: switch iff `A > eps`."""
        one = torch.ones(1, device=planner.device, dtype=planner._ctx_emb.dtype)
        d, q_open, q_close, dz = progress_contrast(
            planner.wm, planner.model.head, planner._ctx_emb, planner._ctx_act, lang,
            one * planner._grip_raw_to_norm(-1.0), one * planner._grip_raw_to_norm(1.0))
        d = float(d.item())
        adv = -d if planner._grip_sign > 0 else d
        diag.update({"grip_logit": d, "grip_adv": adv, "grip_q_open": float(q_open.item()),
                     "grip_q_close": float(q_close.item()), "grip_dz": float(dz.item())})
        if adv > planner.grip_field_eps:
            return -planner._grip_sign
        return planner._grip_sign
