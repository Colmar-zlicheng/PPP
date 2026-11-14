"""Closed-loop LIBERO evaluation (official protocol) of the LeWM-based PPP and BC checkpoints.

One task per call: the task's official initial states (`.pruned_init`), the official step budget,
`OFFICIAL_SETTLE_STEPS` wait steps before the first command, one command per 20 Hz control step,
stop on success. Results go to `<out_dir>/task<N>.json`.
"""
import argparse
import contextlib
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, ".")

import h5py
import imageio
import numpy as np
import torch
import torch.backends.cuda as _cuda_backends

import lib.models  # noqa: F401  (registers the models)
from lib.datasets.libero_timing import resolve_timing
from lib.envs.libero_env import (OFFICIAL_SETTLE_ACTION, OFFICIAL_SETTLE_STEPS,
                                 OFFICIAL_TRIALS_PER_TASK, build_env, deterministic_reset,
                                 get_action_bounds, hard_reset_to_state, load_official_init_states,
                                 official_max_steps, resolve_init_states_path, settle)
from lib.planning.lewm_gg_field_planner import LeWMGGFieldPlanner
from lib.utils.builder import build_model
from lib.utils.config import get_config
from lib.utils.logger import logger

FIELD_MODEL_TYPES = ("LeWMGGFieldCotrainModel", "LeWMBCModel")
DATA_DEFAULT = "assets/latent_extract/libero/libero_full_consolidated.h5"
ARM = "gg"

_real_sdp_kernel = _cuda_backends.sdp_kernel


@contextlib.contextmanager
def force_math_sdp():
    """Route every `sdp_kernel` request to the math kernel, which supports double backward."""
    def _forced(*a, **k):
        return _real_sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    _cuda_backends.sdp_kernel = _forced
    try:
        yield
    finally:
        _cuda_backends.sdp_kernel = _real_sdp_kernel


def default_out_dir(ckpt: str) -> str:
    ckpt_dir = os.path.normpath(ckpt)
    run_dir = os.path.dirname(os.path.dirname(ckpt_dir))
    return os.path.join(run_dir, "closed_loop", os.path.basename(ckpt_dir))


def default_cfg(ckpt: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(ckpt))), "dump_cfg.yaml")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="dir holding one <ModelName>.pth.tar")
    p.add_argument("--cfg", default=None, help="default: <run_dir>/dump_cfg.yaml")
    p.add_argument("--data", default=DATA_DEFAULT)
    p.add_argument("--task_idx", type=int, default=0,
                   help="task index in the consolidated file: 0-9 LIBERO-Goal, 10-19 Object, "
                        "20-29 Spatial, 30-39 LIBERO-10 (Long)")
    p.add_argument("--n_episodes", type=int, default=None,
                   help=f"rollouts for this task (default: all {OFFICIAL_TRIALS_PER_TASK} official "
                        "initial states)")
    p.add_argument("--grip_field_eps_steps", type=float, default=None,
                   help="PPP only: the field gripper switches when its flip advantage exceeds this "
                        "many grid steps of progress (converted with the task's median "
                        "demonstration length); the paper uses 1.0")
    p.add_argument("--pose_grip_settle", action="store_true",
                   help="PPP only: after a close command the pose query keeps seeing the gripper as "
                        "open until the fingers have stopped closing (gripper settling)")
    p.add_argument("--episode_seed_base", type=int, default=None,
                   help="per-episode seeded scene reset: episode k resets with seed BASE + k "
                        "(the paper uses 0); omitted, one env.reset() is shared by all episodes")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--video_arms", default="gg",
                   help="with --save_video: comma-separated 'gg' (the rollout) and/or 'gg_p' "
                        "(rollout + the field's progress p_theta; PPP only)")
    p.add_argument("--fps", type=int, default=None, help="video frame rate (default: control rate)")
    return p.parse_args()


def load_field_model(cfg_path, ckpt_dir, device):
    cfg = get_config(config_file=cfg_path, merge=False, freeze=False)
    assert cfg.MODEL.TYPE in FIELD_MODEL_TYPES, (
        f"{cfg_path} builds MODEL.TYPE={cfg.MODEL.TYPE!r}, not one of {FIELD_MODEL_TYPES}")
    model = build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN).to(device)
    model.setup(summary_writer=None)
    cands = [f for f in os.listdir(ckpt_dir)
             if f.endswith(".pth.tar") and f != "train_param.pth.tar"]
    assert len(cands) == 1, f"expected one model-state *.pth.tar in {ckpt_dir}, found {cands}"
    sd = torch.load(os.path.join(ckpt_dir, cands[0]), map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"load mismatch: missing={missing} unexpected={unexpected}"
    model.eval()
    model.wm.assert_data_contract(cfg.DATASET.TRAIN)
    logger.info(f"loaded {cands[0]} | policy={model.policy_mode} | world model: "
                f"{model.wm.cfg_path} @ co-trained weights from this checkpoint")
    return model, cfg


def render_progress_video(frames, p_traj, path, fps, title, y_label, readout_fn, scale=3):
    """Side-by-side mp4: LEFT the simulator's own view, RIGHT `p_theta` revealed step by step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(frames), len(p_traj))
    assert n > 1, f"need >= 2 aligned (frame, p) pairs, got {len(frames)} frames / {len(p_traj)} p"
    p = np.asarray(p_traj[:n], dtype=np.float64)
    side = int(frames[0].shape[0]) * scale
    dpi = 100
    lo, hi = float(p.min()), float(p.max())
    pad = max(0.05 * (hi - lo), 1e-3)

    fig = plt.figure(figsize=(side / dpi, side / dpi), dpi=dpi)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.21, right=0.95, top=0.84, bottom=0.14)
    (line,) = ax.plot([], [], lw=2.0, color="#1f77b4")
    (dot,) = ax.plot([], [], "o", ms=6, color="#d62728")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("replan step")
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)
    txt = ax.text(0.03, 0.96, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=11, family="monospace")

    out = []
    for i in range(n):
        line.set_data(np.arange(i + 1), p[: i + 1])
        dot.set_data([i], [p[i]])
        txt.set_text(readout_fn(float(p[i])))
        fig.canvas.draw()
        panel = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
        sim = np.repeat(np.repeat(frames[i], scale, axis=0), scale, axis=1)
        assert panel.shape[0] == sim.shape[0], (
            f"panel {panel.shape} does not match the upscaled sim frame {sim.shape}")
        out.append(np.concatenate([sim, panel], axis=1))
    plt.close(fig)
    imageio.mimsave(path, out, fps=fps)


def commanded_delta_xyz(a_raw, bounds):
    in_min, in_max, out_min, out_max = bounds
    scale = (out_max[:3] - out_min[:3]) / (in_max[:3] - in_min[:3])
    return (np.asarray(a_raw, dtype=np.float64)[:3] - in_min[:3]) * scale + out_min[:3]


def rollout(env, planner, state_i, lang, n_steps, frame_stride, bounds, obs_callback=None,
            record_progress=False):
    obs = hard_reset_to_state(env, state_i)
    obs = settle(env, OFFICIAL_SETTLE_STEPS, action=OFFICIAL_SETTLE_ACTION) or obs
    planner.bootstrap_from_obs(obs)
    p_traj = []

    def _observe(step, obs):
        if obs_callback is not None:
            obs_callback(step, obs)
        if record_progress:
            p_traj.append(planner.progress_at_context(lang))

    _observe(0, obs)

    real_success = [bool(env.check_success())]
    stopped_on_success = real_success[0]
    a_traj = []
    ee_drifts, ee_path, ee_cmd_norms = [], 0.0, []
    has_ee = "robot0_eef_pos" in obs
    pose_norms, full_norms, grip_signs, prog_rates = [], [], [], []

    for step in range(0 if stopped_on_success else n_steps):
        with force_math_sdp():
            out = planner.plan_action(lang)
        a_raw = out["action_raw"]
        a_norm7 = out["action_norm7"].detach().cpu().numpy()
        pose_norms.append(out["diag"]["pose_norm6"])
        full_norms.append(out["diag"]["action_norm_normalized"])
        grip_signs.append(out["diag"]["gripper_snapped_to"])
        prog_rates.append(out["diag"]["predicted_progress_rate"])
        a_traj.append(out["diag"]["a6"])

        a_seq = np.broadcast_to(np.asarray(a_raw, dtype=np.float64), (frame_stride, 7))
        ee0 = np.asarray(obs["robot0_eef_pos"], dtype=np.float64) if has_ee else None
        cmd_xyz = commanded_delta_xyz(a_seq[0], bounds)
        obs, _, _, _ = env.step(a_seq[0])
        if has_ee:
            realized = np.asarray(obs["robot0_eef_pos"], dtype=np.float64) - ee0
            ee_drifts.append(float(np.linalg.norm(realized - cmd_xyz)))
            ee_cmd_norms.append(float(np.linalg.norm(cmd_xyz)))
            ee_path += float(np.linalg.norm(realized))
        real_success.append(bool(env.check_success()))
        planner.push_observation(obs, torch.as_tensor(a_norm7, dtype=torch.float32))
        _observe(step + 1, obs)
        if real_success[-1]:
            stopped_on_success = True
            break

    out = {
        "n_steps_run": len(real_success) - 1,
        "stop_reason": "success" if stopped_on_success else "budget",
        "stopped_on_success": stopped_on_success,
        "n_steps_budget": int(n_steps),
        "success_real": bool(any(real_success)),      # the BDDL success predicate
        "success_real_final": bool(real_success[-1]),  # still satisfied at the end
        "action_norm_mean": float(np.mean(full_norms)) if full_norms else None,
        "pose_norm_mean": float(np.mean(pose_norms)) if pose_norms else None,
        "n_gripper_flips": (int(np.sum(np.diff(np.sign(grip_signs)) != 0))
                            if len(grip_signs) > 1 else None),
        "mean_predicted_progress_rate": float(np.mean(prog_rates)) if prog_rates else None,
        "first_success_step": (int(np.argmax(real_success[1:])) if any(real_success[1:]) else None),
        "a6_traj": a_traj or None,
        "ee_drift_mean": float(np.mean(ee_drifts)) if ee_drifts else None,
        "ee_drift_sum": float(np.sum(ee_drifts)) if ee_drifts else None,
        "ee_cmd_norm_mean": float(np.mean(ee_cmd_norms)) if ee_cmd_norms else None,
        "ee_path_len": float(ee_path) if ee_drifts else None,
        "latent_drift_mean": float(np.mean(planner._drift_hist)) if planner._drift_hist else None,
        "latent_drift_last": float(planner._drift_hist[-1]) if planner._drift_hist else None,
    }
    if record_progress:
        out["p_traj"] = p_traj
    return out


def main():
    args = parse_args()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    args.cfg = args.cfg or default_cfg(args.ckpt)
    args.out_dir = args.out_dir or default_out_dir(args.ckpt)
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info(f"cfg={args.cfg}  out_dir={args.out_dir}")
    t0 = time.time()

    model, cfg = load_field_model(args.cfg, args.ckpt, device)
    frame_stride = model.wm.frameskip
    exec_stride, ctx_stride = 1, frame_stride
    ctx_decim = ctx_stride // exec_stride

    f = h5py.File(args.data, "r")
    ep_len, ep_task_idx = f["ep_len"][:], f["ep_task_idx"][:]
    action_loc = f["action_loc"][:].astype(np.float64)
    action_scale = f["action_scale"][:].astype(np.float64)
    bddl_file = f["bddl_file"][args.task_idx].decode()
    env_args_json = f["env_args_json"][args.task_idx].decode()
    bounds = get_action_bounds(env_args_json)
    task_text = f["task_text_by_task"][args.task_idx].decode()
    task_suite = f["task_suite"][args.task_idx].decode()

    timing = resolve_timing(cfg.DATASET.TRAIN, f)
    logger.info(f"dataset timing={timing.contract}, stored observations={timing.observation_timing}")

    official_states = load_official_init_states(bddl_file)
    n_want = args.n_episodes or OFFICIAL_TRIALS_PER_TASK
    eval_eps = list(range(min(n_want, len(official_states))))
    logger.info(f"task {args.task_idx} ({task_text!r}, {task_suite}): {len(eval_eps)}/"
                f"{len(official_states)} official initial states from "
                f"{resolve_init_states_path(bddl_file)}")
    if len(eval_eps) < OFFICIAL_TRIALS_PER_TASK:
        logger.warning(f"--n_episodes {len(eval_eps)} < {OFFICIAL_TRIALS_PER_TASK}: a subset of the "
                       f"official protocol")

    lang = torch.from_numpy(f["task_lang_embed_by_task"][args.task_idx]).float().to(device).unsqueeze(0)

    env = build_env(bddl_file, env_args_json)
    control_freq = float(getattr(env.env, "control_freq", 0) or
                         json.loads(env_args_json).get("env_kwargs", {}).get("control_freq", 0))
    assert control_freq > 0, f"no control_freq on the env or in env_args_json for task {args.task_idx}"
    if args.fps is None:
        args.fps = max(1, int(round(control_freq / exec_stride)))
    env.reset()

    grip_field_eps = 0.0
    if args.grip_field_eps_steps is not None:
        n_grid = ep_len[ep_task_idx == args.task_idx] // model.wm.frameskip
        grip_field_eps = args.grip_field_eps_steps * float(np.median(1.0 / np.maximum(n_grid - 1, 1)))
        logger.info(f"field gripper threshold {args.grip_field_eps_steps} grid steps = "
                    f"{grip_field_eps:.4f} potential units on task {args.task_idx}")

    planner = LeWMGGFieldPlanner(model, action_loc, action_scale, bounds, device,
                                 ctx_decim=ctx_decim, grip_field_eps=grip_field_eps,
                                 pose_grip_settle=args.pose_grip_settle)

    video_arms = set(a.strip() for a in args.video_arms.split(",") if a.strip())
    assert video_arms <= {ARM, f"{ARM}_p"}, f"--video_arms accepts 'gg' and 'gg_p', got {video_arms}"
    wants_plain = args.save_video and ARM in video_arms
    wants_panel = args.save_video and f"{ARM}_p" in video_arms
    assert not (wants_panel and not planner.has_potential), (
        f"--video_arms gg_p shows the field's progress, but {cfg.MODEL.TYPE} has no potential")
    y_label, readout_fn = ("p_theta  (= progress - 1)",
                           lambda v: f"p    = {v:+.3f}\nprog = {v + 1.0:.3f}")
    if args.save_video:
        video_dir = os.path.join(args.out_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)

    n_steps = int(np.ceil(official_max_steps(task_suite) / exec_stride))
    logger.info(f"official protocol: {n_steps} control steps for {task_suite}; "
                f"{OFFICIAL_SETTLE_STEPS} settle steps before the first command")

    results = []
    for ep_i in eval_eps:
        if args.episode_seed_base is not None:
            deterministic_reset(env, args.episode_seed_base + int(ep_i))
        state_i = np.asarray(official_states[ep_i], dtype=np.float64)
        frames, obs_cb = None, None
        if wants_plain or wants_panel:
            frames = []

            def obs_cb(step, obs, frames=frames):
                frames.append(np.asarray(obs["agentview_image"], dtype=np.uint8)[::-1].copy())

        out = rollout(env, planner, state_i, lang, n_steps, frame_stride, bounds,
                      obs_callback=obs_cb, record_progress=wants_panel)
        if frames:
            tag = "success" if out["success_real"] else "fail"
            stem = os.path.join(video_dir, f"task{args.task_idx}_ep{ep_i:04d}")
            if wants_plain:
                try:
                    imageio.mimsave(f"{stem}_{tag}.mp4", frames, fps=args.fps)
                    out["video"] = f"{stem}_{tag}.mp4"
                except Exception as e:
                    logger.warning(f"video save failed for ep {ep_i}: {e}")
            if wants_panel:
                try:
                    render_progress_video(
                        frames, out["p_traj"], f"{stem}_p_{tag}.mp4", args.fps,
                        title=f"ep {ep_i}\n{task_text[:44]}", y_label=y_label, readout_fn=readout_fn)
                    out["video_p"] = f"{stem}_p_{tag}.mp4"
                except Exception as e:
                    logger.warning(f"progress video failed for ep {ep_i}: {e}")
        results.append(out)
        logger.info(f"init {ep_i:3d} success={out['success_real']} steps={out['n_steps_run']} "
                    f"|a|={out['action_norm_mean']}")

    def _mean_of(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None]
        return float(np.mean(v)) if v else None

    s = {"n": len(results),
         "success_real_rate": float(np.mean([r["success_real"] for r in results])),
         "success_real_final_rate": float(np.mean([r["success_real_final"] for r in results])),
         "mean_action_norm": _mean_of(results, "action_norm_mean")}
    for k in ("ee_drift_mean", "ee_cmd_norm_mean", "ee_path_len", "latent_drift_mean"):
        s[k] = _mean_of(results, k)
    s["ee_drift_frac"] = (s["ee_drift_mean"] / s["ee_cmd_norm_mean"]
                          if s["ee_drift_mean"] is not None and s["ee_cmd_norm_mean"] else None)
    summary = {ARM: s}

    out_path = os.path.join(args.out_dir, f"task{args.task_idx}.json")
    with open(out_path, "w") as fh:
        json.dump({"args": vars(args),
                   "timing": {"dataset_contract": timing.contract,
                              "stored_observations": timing.observation_timing},
                   "task_text": task_text, "episodes": eval_eps,
                   "protocol": "official",
                   "task_suite": task_suite,
                   "official": {"init_states": resolve_init_states_path(bddl_file),
                                "n_init_states": int(len(official_states)),
                                "max_env_steps": official_max_steps(task_suite),
                                "settle_steps": OFFICIAL_SETTLE_STEPS,
                                "settle_action": OFFICIAL_SETTLE_ACTION,
                                "scene_reset": (f"per_episode_seed{args.episode_seed_base}"
                                                if args.episode_seed_base is not None else "per_job"),
                                "trials_per_task_full": OFFICIAL_TRIALS_PER_TASK},
                   "world_model": {"cfg": model.wm.cfg_path, "ckpt": model.wm.ckpt_path,
                                   "cotrained": True},
                   "frame_stride": frame_stride,
                   "exec_stride": exec_stride,
                   "ctx_stride": ctx_stride, "ctx_decim": ctx_decim,
                   "summary": summary,
                   "episode_results": {ARM: results}}, fh, indent=2, default=str)

    print(f"\ntask {args.task_idx} {task_text!r} ({task_suite}): success "
          f"{s['success_real_rate']:.3f} over {s['n']} episodes "
          f"(still satisfied at the end: {s['success_real_final_rate']:.3f})")
    print(f"wrote {out_path}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
