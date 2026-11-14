"""Per-suite and average LIBERO success of one closed-loop cell (official protocol).

Reads `<run_dir>/closed_loop/<cell>/task*.json`, as written by `scripts/close_loop_lewm_libero.py`
or `scripts/close_loop_seer_libero.py`, and prints the success rate of each suite (10 tasks x 50
initial states) and their mean.

    python scripts/summarize_official_suites.py \\
        --run_dir checkpoints/lewm_ppp_libero --cell epseed0_settle_checkpoint_120
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.envs.libero_env import OFFICIAL_TRIALS_PER_TASK, official_max_steps  # noqa: E402

EVAL_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
TASKS_PER_SUITE = 10


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--cell", required=True,
                   help="the folder under <run_dir>/closed_loop, e.g. epseed0_checkpoint_120")
    p.add_argument("--json_out", default=None, help="also write the aggregate here")
    return p.parse_args()


def main():
    args = parse_args()
    cell_dir = os.path.join(args.run_dir, "closed_loop", args.cell)
    paths = sorted(glob.glob(os.path.join(cell_dir, "task*.json")))
    if not paths:
        raise SystemExit(f"no task*.json in {cell_dir}")

    by_suite, settles, resets = {}, set(), set()
    for path in paths:
        d = json.load(open(path))
        if d.get("protocol") != "official":
            raise SystemExit(f"{path}: protocol={d.get('protocol')!r}, expected 'official'")
        settles.add(str(d["official"]["settle_action"]))
        resets.add(str(d["official"]["scene_reset"]))
        by_suite.setdefault(d["task_suite"], []).append(d)
    if len(settles) > 1 or len(resets) > 1:
        raise SystemExit(f"tasks in {cell_dir} were run under different settings: settle action "
                         f"{sorted(settles)}, scene reset {sorted(resets)}")

    print(f"cell       : {cell_dir}")
    print(f"protocol   : official, settle_action={settles.pop()}, scene_reset={resets.pop()}")
    width = 84
    print("=" * width)
    print(f"{'suite':16} | {'tasks':>5} | {'rollouts':>8} | {'success':>8} | {'rate':>6} | "
          f"{'env steps':>9} | status")
    print("-" * width)

    out, rates, complete_all = {}, [], True
    expected = TASKS_PER_SUITE * OFFICIAL_TRIALS_PER_TASK
    for suite in EVAL_SUITES:
        cells = by_suite.get(suite, [])
        if not cells:
            print(f"{suite:16} | {'-':>5} | {'-':>8} | {'-':>8} | {'-':>6} | "
                  f"{official_max_steps(suite):>9} | missing")
            complete_all = False
            continue
        flat = [bool(r["success_real"]) for d in cells for r in d["episode_results"]["gg"]]
        n, k = len(flat), int(np.sum(flat))
        ok = len(cells) == TASKS_PER_SUITE and n == expected
        complete_all &= ok
        rate = k / n
        rates.append(rate)
        status = "complete" if ok else f"partial ({len(cells)}/{TASKS_PER_SUITE} tasks, {n}/{expected} rollouts)"
        print(f"{suite:16} | {len(cells):>5} | {n:>8} | {k:>8} | {rate:>6.3f} | "
              f"{official_max_steps(suite):>9} | {status}")
        out[suite] = {"tasks": len(cells), "rollouts": n, "successes": k, "rate": rate,
                      "complete": ok, "max_env_steps": official_max_steps(suite)}

    print("-" * width)
    if len(rates) == len(EVAL_SUITES):
        out["average"] = float(np.mean(rates))
        print(f"{'average':16} | {'':>5} | {'':>8} | {'':>8} | {out['average']:>6.3f} | {'':>9} | "
              f"{'complete' if complete_all else 'partial'}")
    else:
        print(f"{'average':16} | not defined: {len(rates)}/{len(EVAL_SUITES)} suites present")
    print("=" * width)

    if args.json_out:
        out["meta"] = {"run_dir": args.run_dir, "cell": args.cell, "complete": complete_all}
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
