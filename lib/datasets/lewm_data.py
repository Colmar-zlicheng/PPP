import os

import h5py
import hdf5plugin  # noqa: F401  registers HDF5 compression codecs
import numpy as np

from lib.utils.logger import logger


def ensure_shm_episode_export(h5_path: str, dataset: str, episodes: np.ndarray,
                              cache_dir: str = "/dev/shm/ppp_cache") -> str:
    """Materialize a SELECTED SUBSET of an HDF5 frame dataset as one tmpfs-backed .npy
    and return its path, so concurrent DDP ranks share ONE physical copy."""
    import fcntl
    import hashlib
    import shutil

    src = os.path.abspath(h5_path)
    episodes = np.asarray(episodes, dtype=np.int64).reshape(-1, 2)
    n_frames = int(episodes[:, 1].sum())

    with h5py.File(src, "r") as f:
        ds = f[dataset]
        frame_shape, dtype = ds.shape[1:], ds.dtype
    row_bytes = int(np.prod(frame_shape)) * dtype.itemsize
    expected = n_frames * row_bytes

    key = hashlib.sha1(b"|".join([
        src.encode(), dataset.encode(), str(os.path.getsize(src)).encode(),
        episodes.tobytes(),
    ])).hexdigest()[:16]
    dst = os.path.join(cache_dir, f"{os.path.basename(src)}.{dataset}.{key}.npy")

    def _ready() -> bool:
        return os.path.exists(dst) and os.path.getsize(dst) >= expected

    if _ready():
        logger.info(f"ensure_shm_episode_export: reusing {dst} ({expected / 1e9:.1f}GB)")
        return dst

    os.makedirs(cache_dir, exist_ok=True)
    lock_path = dst + ".lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if not _ready():
                free = shutil.disk_usage(cache_dir).free
                assert free > expected * 1.05, (
                    f"ensure_shm_episode_export: need {expected / 1e9:.1f}GB in {cache_dir} "
                    f"but only {free / 1e9:.1f}GB free. tmpfs is shared and finite -- "
                    f"clear stale exports (rm -rf {cache_dir}) or set SHM_CACHE: false.")
                logger.info(f"ensure_shm_episode_export: building {dst} "
                            f"({n_frames} frames of '{dataset}', {expected / 1e9:.1f}GB) ...")
                tmp = f"{dst}.tmp.{os.getpid()}"
                out = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype,
                                                shape=(n_frames,) + frame_shape)
                with h5py.File(src, "r") as f:
                    d = f[dataset]
                    w = 0
                    for base, n in episodes:      # contiguous per-episode reads
                        d.read_direct(out, np.s_[base:base + n], np.s_[w:w + n])
                        w += int(n)
                    assert w == n_frames
                out.flush()
                del out
                os.rename(tmp, dst)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return dst


def _episode_split(
    ep_offset: np.ndarray,
    train_ratio: float,
    seed: int,
    split: str,
    ep_task_idx: np.ndarray = None,
    stratify: bool = False,
    val_ratio: float = None,
) -> np.ndarray:
    """Return episode indices for the requested split using a deterministic permutation."""
    num_eps = len(ep_offset)
    if not stratify:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(num_eps)
        n_train = int(num_eps * train_ratio)
        return perm[:n_train] if split == "train" else perm[n_train:]

    assert ep_task_idx is not None, "_episode_split(stratify=True) requires ep_task_idx"
    ep_task_idx = np.asarray(ep_task_idx)
    rng = np.random.RandomState(seed)
    train_eps, val_eps, test_eps = [], [], []
    for task_id in np.unique(ep_task_idx):
        task_eps = np.where(ep_task_idx == task_id)[0]
        perm = rng.permutation(task_eps)
        n = len(perm)
        n_train = int(round(n * train_ratio))
        if val_ratio is None:
            train_eps.append(perm[:n_train])
            val_eps.append(perm[n_train:])
            test_eps.append(perm[n_train:])
        else:
            n_val = int(round(n * val_ratio))
            if n >= 3:
                n_train = min(n_train, n - 2)
                n_val = max(n_val, 1)
                n_val = min(n_val, n - n_train - 1)
            train_eps.append(perm[:n_train])
            val_eps.append(perm[n_train:n_train + n_val])
            test_eps.append(perm[n_train + n_val:])
    train_eps = np.concatenate(train_eps) if train_eps else np.empty(0, dtype=np.int64)
    val_eps = np.concatenate(val_eps) if val_eps else np.empty(0, dtype=np.int64)
    test_eps = np.concatenate(test_eps) if test_eps else np.empty(0, dtype=np.int64)
    if split == "train":
        return train_eps
    if split == "val":
        return val_eps
    if split == "test":
        return test_eps
    if split == "trainval":
        return np.concatenate([train_eps, val_eps])
    if split == "all":
        if val_ratio is None:
            return np.concatenate([train_eps, val_eps])
        return np.concatenate([train_eps, val_eps, test_eps])
    raise ValueError(f"_episode_split(stratify=True): unknown split {split!r}")


