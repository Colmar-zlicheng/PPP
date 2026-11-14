import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

EXTERNAL_LIBERO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "external", "LIBERO")


def resolve_bddl_path(bddl_file: str) -> str:
    path = os.path.normpath(os.path.join(EXTERNAL_LIBERO_ROOT, bddl_file))
    assert os.path.exists(path), f"bddl file not found: {path} (from {bddl_file})"
    return path


def build_env(bddl_file: str, env_args_json: str, camera_heights: int = 128, camera_widths: int = 128,
               control_freq: int = None, horizon: int = 1000):
    """Construct a LIBERO OffScreenRenderEnv matching one task's recorded
    demonstration convention exactly."""
    from libero.libero.envs import OffScreenRenderEnv

    env_args = json.loads(env_args_json)
    kwargs = dict(env_args.get("env_kwargs", {}))
    controller_cfg = kwargs.pop("controller_configs", None)
    controller_name = (controller_cfg or {}).get("type", "OSC_POSE")
    kwargs.pop("bddl_file_name", None)  # ControlEnv takes this positionally below
    if control_freq is not None:
        kwargs["control_freq"] = control_freq
    kwargs.setdefault("camera_heights", camera_heights)
    kwargs.setdefault("camera_widths", camera_widths)
    kwargs.setdefault("horizon", horizon)

    path = resolve_bddl_path(bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=path, controller=controller_name, **kwargs)
    return env


def get_action_bounds(env_args_json: str):
    """Return (input_min, input_max, output_min, output_max) as (7,) float
    arrays from the stored controller config -- the exact OSC_POSE
    normalized-command bounds and their physical (m, rad) scale."""
    import numpy as np

    env_args = json.loads(env_args_json)
    cc = env_args["env_kwargs"]["controller_configs"]
    in_max, in_min = cc["input_max"], cc["input_min"]
    out_max, out_min = cc["output_max"], cc["output_min"]

    def _bcast(x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0 or x.shape == ():
            x = np.full(6, float(x))
        if x.shape[0] == 6:
            x = np.concatenate([x, [1.0 if x.mean() > 0 else -1.0]])
        return x

    in_max, in_min = _bcast(in_max), _bcast(in_min)
    out_max, out_min = _bcast(out_max), _bcast(out_min)
    out_max[-1], out_min[-1] = 1.0, -1.0
    return in_min, in_max, out_min, out_max


def deterministic_reset(env, placement_seed):
    """`env.reset()` with the object placement made REPRODUCIBLE."""
    import random

    import numpy as np

    np.random.seed(int(placement_seed))
    random.seed(int(placement_seed))
    return env.reset()


def reset_to_state(env, mujoco_state):
    return env.set_init_state(mujoco_state)


def hard_reset_to_state(env, mujoco_state):
    for robot in env.robots:
        robot.reset(deterministic=True)
    return reset_to_state(env, mujoco_state)


OFFICIAL_MAX_STEPS = {
    "libero_spatial": 220,   # longest training demo has 193 steps
    "libero_object":  280,   # longest training demo has 254 steps
    "libero_goal":    300,   # longest training demo has 270 steps
    "libero_10":      520,   # longest training demo has 505 steps
    "libero_90":      400,   # longest training demo has 373 steps
}

OFFICIAL_SETTLE_STEPS = 10

SETTLE_ACTIONS = {
    "open":  [0.0] * 6 + [-1.0],   # OpenVLA / openpi; ends at gripper qpos +-0.038723
    "zeros": [0.0] * 7,            # LIBERO lifelong/metric.py; ends at +-0.020542
}
OFFICIAL_SETTLE_ACTION = "open"

OFFICIAL_TRIALS_PER_TASK = 50


def resolve_init_states_path(bddl_file: str, pruned: bool = True) -> str:
    """Path to a task's official eval init-state file, from its stored `bddl_file`."""
    rel = bddl_file.replace("bddl_files", "init_files")
    rel = os.path.splitext(rel)[0] + (".pruned_init" if pruned else ".init")
    path = os.path.normpath(os.path.join(EXTERNAL_LIBERO_ROOT, rel))
    assert os.path.exists(path), (
        f"init-state file not found: {path} (from {bddl_file}). The official "
        f"protocol needs external/LIBERO checked out with its init_files tree.")
    return path


def load_official_init_states(bddl_file: str, pruned: bool = True):
    """(n, state_dim) float64 array of a task's official eval initial states."""
    import numpy as np
    import torch
    states = np.asarray(torch.load(resolve_init_states_path(bddl_file, pruned),
                                   weights_only=False), dtype=np.float64)
    assert states.ndim == 2 and len(states) > 0, f"bad init-state array {states.shape}"
    return states


def official_max_steps(suite: str) -> int:
    """OpenVLA's step budget for a suite. Unknown suite -> the libero_goal value."""
    return OFFICIAL_MAX_STEPS.get(suite, OFFICIAL_MAX_STEPS["libero_goal"])


def settle(env, n_steps: int = OFFICIAL_SETTLE_STEPS,
           action: str = OFFICIAL_SETTLE_ACTION):
    """Step `n_steps` wait actions so the scene settles before the policy acts."""
    import numpy as np
    assert action in SETTLE_ACTIONS, (
        f"unknown settle action {action!r}, expected one of {sorted(SETTLE_ACTIONS)}")
    a = np.asarray(SETTLE_ACTIONS[action], dtype=np.float64)
    obs = None
    for _ in range(n_steps):
        obs, _, _, _ = env.step(a)
    return obs
