"""Closed-loop LIBERO evaluation (official protocol) of the Seer and Seer-PPP checkpoints.

One task per call: the task's official initial states (`.pruned_init`), the official step budget,
`OFFICIAL_SETTLE_STEPS` wait steps before the first command, one command per 20 Hz control step,
stop on success. Results go to `<out_dir>/task<N>.json`.
"""
import argparse
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

import lib.models  # noqa: F401  (registers SeerPolicyModel)
from lib.envs.libero_env import (OFFICIAL_SETTLE_ACTION, OFFICIAL_SETTLE_STEPS,
                                 OFFICIAL_TRIALS_PER_TASK, build_env, deterministic_reset,
                                 get_action_bounds, hard_reset_to_state, load_official_init_states,
                                 official_max_steps, resolve_init_states_path, settle)
from lib.planning.seer_planner import SeerPlanner
from lib.utils.builder import build_model
from lib.utils.config import get_config
from lib.utils.logger import logger

DATA_DEFAULT = "assets/latent_extract/libero/libero_full_consolidated.h5"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="dir holding SeerPolicyModel.pth.tar")
    p.add_argument("--cfg", default=None, help="default: <run_dir>/dump_cfg.yaml")
    p.add_argument("--data", default=DATA_DEFAULT)
    p.add_argument("--task_idx", type=int, required=True,
                   help="task index in the consolidated file: 0-9 LIBERO-Goal, 10-19 Object, "
                        "20-29 Spatial, 30-39 LIBERO-10 (Long)")
    p.add_argument("--n_episodes", type=int, default=None,
                   help=f"rollouts for this task (default: all {OFFICIAL_TRIALS_PER_TASK} official "
                        "initial states)")
    p.add_argument("--episode_seed_base", type=int, default=None,
                   help="per-episode seeded scene reset: episode k resets with seed BASE + k "
                        "(the paper uses 0); omitted, one env.reset() is shared by all episodes")
    p.add_argument("--grip_field_eps_steps", type=float, default=1.0,
                   help="Seer-PPP: the field gripper switches when its flip advantage exceeds this "
                        "many control steps of progress")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--gpu_id", type=int, default=0)
    return p.parse_args()


def load_seer_model(cfg_path, ckpt_dir, device):
    cfg = get_config(config_file=cfg_path, merge=False, freeze=False)
    assert cfg.MODEL.TYPE == "SeerPolicyModel", f"{cfg_path} builds {cfg.MODEL.TYPE}"
    model = build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN).to(device)
    model.setup(summary_writer=None)
    path = os.path.join(ckpt_dir, "SeerPolicyModel.pth.tar")
    sd = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, f"load mismatch: missing={missing} unexpected={unexpected}"
    model.eval()
    ds = cfg.DATASET.TRAIN
    assert int(ds.get("SEQ_LEN", 7)) == model.seq_len and int(ds.get("FUTURE_STEPS", 3)) == model.future
    logger.info(f"loaded {path}: {model.policy_mode}")
    return model, cfg


def rollout(env, planner, state_i, task_idx, n_steps, frames=None, p_traj=None):
    obs = hard_reset_to_state(env, state_i)
    obs = settle(env, OFFICIAL_SETTLE_STEPS, action=OFFICIAL_SETTLE_ACTION) or obs
    planner.reset()
    planner.observe(obs)

    def record(o):
        if frames is not None:
            frames.append(np.asarray(o["agentview_image"], dtype=np.uint8)[::-1].copy())
        if p_traj is not None:
            p_traj.append(planner.progress_now(task_idx))
    record(obs)
    succ = [bool(env.check_success())]
    pose_norms, grips, advs, rates = [], [], [], []
    ee_path = 0.0
    for _ in range(0 if succ[0] else n_steps):
        out = planner.act(task_idx)
        a = out["action_raw"]
        ee0 = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        obs, _, _, _ = env.step(a)
        ee_path += float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - ee0))
        planner.observe(obs)
        record(obs)
        d = out["diag"]
        pose_norms.append(d.get("pose_norm6"))
        grips.append(float(a[6]))
        if "grip_adv" in d:
            advs.append(d["grip_adv"])
            rates.append(d["pred_progress_rate"])
        succ.append(bool(env.check_success()))
        if succ[-1]:
            break
    res = {
        "n_steps_run": len(succ) - 1,
        "stop_reason": "success" if succ[-1] else "budget",
        "stopped_on_success": bool(succ[-1]),
        "n_steps_budget": int(n_steps),
        "success_real": bool(any(succ)),
        "success_real_final": bool(succ[-1]),
        "first_success_step": int(np.argmax(succ[1:])) if any(succ[1:]) else None,
        "pose_norm_mean": float(np.mean(pose_norms)) if pose_norms else None,
        "n_gripper_flips": int(np.sum(np.diff(np.sign(grips)) != 0)) if len(grips) > 1 else 0,
        "flip_ticks": (np.flatnonzero(np.diff(np.sign(grips)) != 0) + 1).tolist() if len(grips) > 1 else [],
        "ee_path_len": ee_path,
        "action_norm_mean": float(np.mean(pose_norms)) if pose_norms else None,
    }
    if advs:
        res.update({"grip_adv_max": float(np.max(advs)),
                    "mean_predicted_progress_rate": float(np.mean(rates))})
    if p_traj is not None:
        res["p_traj"] = p_traj
    return res


def main():
    args = parse_args()
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
    args.cfg = args.cfg or os.path.join(run_dir, "dump_cfg.yaml")
    args.out_dir = args.out_dir or os.path.join(run_dir, "closed_loop", os.path.basename(args.ckpt))
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    model, cfg = load_seer_model(args.cfg, args.ckpt, device)

    f = h5py.File(args.data, "r")
    bddl_file = f["bddl_file"][args.task_idx].decode()
    env_args_json = f["env_args_json"][args.task_idx].decode()
    task_text = f["task_text_by_task"][args.task_idx].decode()
    task_suite = f["task_suite"][args.task_idx].decode()
    ep_len, ep_task = f["ep_len"][:], f["ep_task_idx"][:]
    bounds = get_action_bounds(env_args_json)
    states = load_official_init_states(bddl_file)
    eps = list(range(min(args.n_episodes or OFFICIAL_TRIALS_PER_TASK, len(states))))
    budget = official_max_steps(task_suite)
    T = ep_len[ep_task == args.task_idx].astype(np.float64)

    grip_eps = 0.0
    if model.policy_mode == "seer_ppp":
        field_step = float(np.median(model.future / np.maximum(T - 1, 1)))   # one control step of progress
        grip_eps = args.grip_field_eps_steps * field_step
        logger.info(f"field gripper threshold {args.grip_field_eps_steps} control steps = {grip_eps:.4f}")
    planner = SeerPlanner(model, bounds, device, grip_field_eps=grip_eps, max_steps=budget)
    env = build_env(bddl_file, env_args_json)
    env.reset()
    logger.info(f"official protocol: task {args.task_idx} {task_text!r} ({task_suite}), {len(eps)} init "
                f"states, {budget} env steps, {model.policy_mode}")

    video_dir = os.path.join(args.out_dir, "videos")
    results = []
    for ep_i in eps:
        if args.episode_seed_base is not None:
            deterministic_reset(env, args.episode_seed_base + int(ep_i))
        frames = [] if args.save_video else None
        p_traj = [] if (args.save_video and model.is_ppp) else None
        res = rollout(env, planner, np.asarray(states[ep_i], dtype=np.float64), args.task_idx,
                      budget, frames, p_traj)
        if frames:
            os.makedirs(video_dir, exist_ok=True)
            tag = "success" if res["success_real"] else "fail"
            path = os.path.join(video_dir, f"task{args.task_idx}_ep{ep_i:04d}_{tag}.mp4")
            try:
                imageio.mimsave(path, frames, fps=args.fps)
                res["video"] = path
            except Exception as e:
                logger.warning(f"video save failed: {e}")
        results.append(res)
        logger.info(f"init {ep_i:3d} success={res['success_real']} steps={res['n_steps_run']}")

    rate = float(np.mean([r["success_real"] for r in results]))
    summary = {"gg": {"n": len(results), "success_real_rate": rate,
                      "success_real_final_rate": float(np.mean([r["success_real_final"] for r in results])),
                      "mean_action_norm": float(np.mean([r["pose_norm_mean"] for r in results
                                                         if r["pose_norm_mean"] is not None] or [np.nan])),
                      "ee_path_len": float(np.mean([r["ee_path_len"] for r in results]))}}
    scene = (f"per_episode_seed{args.episode_seed_base}" if args.episode_seed_base is not None
             else "per_job")
    out_path = os.path.join(args.out_dir, f"task{args.task_idx}.json")
    with open(out_path, "w") as fh:
        json.dump({"args": vars(args), "protocol": "official", "task_suite": task_suite,
                   "task_text": task_text, "episodes": eps,
                   "official": {"init_states": resolve_init_states_path(bddl_file),
                                "n_init_states": int(len(states)), "max_env_steps": budget,
                                "settle_steps": OFFICIAL_SETTLE_STEPS,
                                "settle_action": OFFICIAL_SETTLE_ACTION, "scene_reset": scene,
                                "trials_per_task_full": OFFICIAL_TRIALS_PER_TASK},
                   "model": {"type": "SeerPolicyModel", "policy_mode": model.policy_mode,
                             "cfg": args.cfg, "ckpt": args.ckpt, "ensembling": True,
                             "grip_field_eps": grip_eps},
                   "summary": summary, "episode_results": {"gg": results}}, fh, indent=2, default=str)
    print(f"task {args.task_idx} {task_text!r} ({task_suite}), {model.policy_mode}: "
          f"success {rate:.3f} over {len(results)} episodes ({time.time() - t0:.0f}s)\nwrote {out_path}")


if __name__ == "__main__":
    main()
