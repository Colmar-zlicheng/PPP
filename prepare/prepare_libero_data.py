"""Consolidate the LIBERO demonstrations into one HDF5 file.

Reads `<libero_root>/<suite>/*.hdf5` (the official LIBERO release) and writes, frame by frame over all
demonstrations, the 128x128 agent-view and wrist images (`pixels`, `pixels_wrist`), the 7-d OSC
actions, the simulator states and proprioception; per episode its offset, length and task index; per
task its instruction, `all-MiniLM-L6-v2` embedding, BDDL file and environment arguments; and the
action normalization (`action_loc` = median, `action_scale` = std over all non-terminal actions).

    python prepare/prepare_libero_data.py --suite all --libero_root data/LIBERO
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("USE_TF", "0")

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ALL_SUITES = ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"]
    p.add_argument("--libero_root", default="data/LIBERO")
    p.add_argument("--suite", default="all", choices=ALL_SUITES + ["all"],
                   help="'all' (the released file) consolidates every suite into one file")
    p.add_argument("--out", default=None,
                   help="default: assets/latent_extract/libero/libero_full_consolidated.h5 for --suite all, "
                        "else assets/latent_extract/libero/<suite>_consolidated.h5")
    p.add_argument("--lang_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--chunk_frames", type=int, default=32)
    return p.parse_args()


def _sorted_demo_keys(f):
    keys = [k for k in f["data"].keys() if k.startswith("demo_")]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def main():
    args = parse_args()
    suites = ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"] \
        if args.suite == "all" else [args.suite]
    out_path = args.out or ("assets/latent_extract/libero/libero_full_consolidated.h5" if args.suite == "all"
                             else f"assets/latent_extract/libero/{args.suite}_consolidated.h5")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    task_files = []
    task_suites = []
    for suite in suites:
        suite_files = sorted(glob.glob(os.path.join(args.libero_root, suite, "*.hdf5")))
        assert len(suite_files) > 0, f"no *.hdf5 found under {args.libero_root}/{suite}"
        task_files.extend(suite_files)
        task_suites.extend([suite] * len(suite_files))
    print(f"Suites {suites}: {len(task_files)} task files total")

    out_f = h5py.File(out_path, "w")
    out_f.attrs["observation_timing"] = "post_action"
    out_f.attrs["action_timing"] = "pre_action"
    out_f.attrs["state_timing"] = "pre_action"
    out_f.attrs["source_timing_contract"] = "libero_native_post_action_v1"
    out_f.attrs["timing_description"] = (
        "Rows preserve the source exporter: states[i] is pre-action s_i; "
        "actions[i] is a_i; pixels/proprio[i] are recorded after applying a_i. "
        "For an interior observation row i, outgoing actions start at i+1 and "
        "the aligned stored simulator state is states[i+1]. The final post-action "
        "observation has no aligned stored successor state or outgoing action.")
    datasets = {
        "pixels": out_f.create_dataset("pixels", shape=(0, 128, 128, 3), maxshape=(None, 128, 128, 3),
                                        dtype="u1", chunks=(args.chunk_frames, 128, 128, 3)),
        "pixels_wrist": out_f.create_dataset("pixels_wrist", shape=(0, 128, 128, 3), maxshape=(None, 128, 128, 3),
                                              dtype="u1", chunks=(args.chunk_frames, 128, 128, 3)),
        "actions": out_f.create_dataset("actions", shape=(0, 7), maxshape=(None, 7), dtype="f4",
                                         chunks=(1024, 7)),
        "states": out_f.create_dataset("states", shape=(0,), maxshape=(None,),
                                        dtype=h5py.special_dtype(vlen=np.dtype("float64")), chunks=(256,)),
        "ee_pos": out_f.create_dataset("ee_pos", shape=(0, 3), maxshape=(None, 3), dtype="f4", chunks=(1024, 3)),
        "ee_ori": out_f.create_dataset("ee_ori", shape=(0, 3), maxshape=(None, 3), dtype="f4", chunks=(1024, 3)),
        "gripper_states": out_f.create_dataset("gripper_states", shape=(0, 2), maxshape=(None, 2), dtype="f4",
                                                chunks=(1024, 2)),
        "joint_states": out_f.create_dataset("joint_states", shape=(0, 7), maxshape=(None, 7), dtype="f4",
                                              chunks=(1024, 7)),
        "ep_offset": out_f.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype="i8", chunks=(256,)),
        "ep_len": out_f.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype="i4", chunks=(256,)),
        "ep_task_idx": out_f.create_dataset("ep_task_idx", shape=(0,), maxshape=(None,), dtype="i4", chunks=(256,)),
        "ep_success": out_f.create_dataset("ep_success", shape=(0,), maxshape=(None,), dtype="i1", chunks=(256,)),
    }

    def append(ds, arr):
        arr = np.asarray(arr)
        n0 = ds.shape[0]
        ds.resize(n0 + arr.shape[0], axis=0)
        ds[n0:] = arr

    def append_vlen(ds, rows):
        n0 = ds.shape[0]
        ds.resize(n0 + len(rows), axis=0)
        for k, row in enumerate(rows):
            ds[n0 + k] = np.asarray(row, dtype=np.float64)

    task_texts, bddl_files, env_args_jsons = [], [], []
    n_frames_total = 0
    n_action_dim_mismatch = 0

    for task_idx, task_path in enumerate(task_files):
        f = h5py.File(task_path, "r")
        data_attrs = f["data"].attrs
        problem_info = json.loads(data_attrs["problem_info"])
        task_text = problem_info["language_instruction"]
        bddl_file = str(data_attrs["bddl_file_name"])
        env_args_json = str(data_attrs["env_args"])
        task_texts.append(task_text)
        bddl_files.append(bddl_file)
        env_args_jsons.append(env_args_json)

        demo_keys = _sorted_demo_keys(f)
        print(f"[{task_idx+1}/{len(task_files)}] ({task_suites[task_idx]}) {os.path.basename(task_path)}: "
              f"'{task_text}' ({len(demo_keys)} demos)")

        for dk in demo_keys:
            g = f["data"][dk]
            actions = g["actions"][:].astype(np.float32)
            states = g["states"][:].astype(np.float64)
            T = actions.shape[0]
            if actions.shape[1] != 7:
                n_action_dim_mismatch += 1
                continue
            append(datasets["pixels"], g["obs"]["agentview_rgb"][:])
            append(datasets["pixels_wrist"], g["obs"]["eye_in_hand_rgb"][:])
            append(datasets["actions"], actions)
            append_vlen(datasets["states"], states)
            append(datasets["ee_pos"], g["obs"]["ee_pos"][:].astype(np.float32))
            append(datasets["ee_ori"], g["obs"]["ee_ori"][:].astype(np.float32))
            append(datasets["gripper_states"], g["obs"]["gripper_states"][:].astype(np.float32))
            append(datasets["joint_states"], g["obs"]["joint_states"][:].astype(np.float32))
            append(datasets["ep_offset"], [n_frames_total])
            append(datasets["ep_len"], [T])
            append(datasets["ep_task_idx"], [task_idx])
            append(datasets["ep_success"], [1])
            n_frames_total += T
        f.close()

    if n_action_dim_mismatch:
        print(f"WARNING: skipped {n_action_dim_mismatch} demos with actions.shape[1] != 7")

    actions = datasets["actions"][:]
    keep = np.ones(len(actions), dtype=bool)
    keep[datasets["ep_offset"][:] + datasets["ep_len"][:] - 1] = False      # each episode's last action
    out_f.create_dataset("action_loc", data=np.median(actions[keep], axis=0).astype(np.float32))
    out_f.create_dataset("action_scale", data=actions[keep].std(axis=0).astype(np.float32))

    print(f"Embedding {len(task_texts)} task instructions with {args.lang_model} ...")
    from sentence_transformers import SentenceTransformer
    lang_model = SentenceTransformer(args.lang_model)
    task_lang_embed_by_task = lang_model.encode(task_texts, show_progress_bar=True).astype(np.float32)

    ep_task_idx_arr = np.array(datasets["ep_task_idx"], dtype=np.int64)  # h5py dataset -> array, already flushed
    out_f.create_dataset("task_lang_embed", data=task_lang_embed_by_task[ep_task_idx_arr])
    out_f.create_dataset("task_text", data=np.array(task_texts, dtype=object)[ep_task_idx_arr],
                          dtype=h5py.string_dtype())
    out_f.create_dataset("task_lang_embed_by_task", data=task_lang_embed_by_task)
    out_f.create_dataset("task_text_by_task", data=np.array(task_texts, dtype=object), dtype=h5py.string_dtype())
    out_f.create_dataset("bddl_file", data=np.array(bddl_files, dtype=object), dtype=h5py.string_dtype())
    out_f.create_dataset("env_args_json", data=np.array(env_args_jsons, dtype=object), dtype=h5py.string_dtype())
    out_f.create_dataset("task_suite", data=np.array(task_suites, dtype=object), dtype=h5py.string_dtype())
    out_f.create_dataset("suite", data=args.suite)
    n_episodes = datasets["ep_offset"].shape[0]

    out_f.close()
    print(f"Saved {out_path}: {n_frames_total} frames, {n_episodes} episodes, {len(task_texts)} tasks "
          f"across {len(suites)} suite(s) {suites}")


if __name__ == "__main__":
    main()
