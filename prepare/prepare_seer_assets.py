"""Assets of the Seer-based models: the MAE ViT-B/16 weights (downloaded from FAIR) and the CLIP
ViT-B/32 text features of the LIBERO instructions in the consolidated file.

    python prepare/prepare_seer_assets.py
    # -> checkpoints/vit_mae/mae_pretrain_vit_base.pth, assets/seer/clip_text_libero.npz
"""
import argparse
import hashlib
import os
import urllib.request

import h5py
import numpy as np


MAE_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
MAE_PATH = "checkpoints/vit_mae/mae_pretrain_vit_base.pth"
MAE_BYTES = 343249461
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_OUT = "assets/seer/clip_text_libero.npz"
DATA_DEFAULT = "assets/latent_extract/libero/libero_full_consolidated.h5"


def fetch_mae(path: str = MAE_PATH) -> str:
    if os.path.exists(path) and os.path.getsize(path) == MAE_BYTES:
        print(f"MAE weights present: {path}")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    print(f"downloading {MAE_URL} -> {path}")
    urllib.request.urlretrieve(MAE_URL, tmp)
    assert os.path.getsize(tmp) == MAE_BYTES, f"unexpected size {os.path.getsize(tmp)}"
    os.replace(tmp, path)
    with open(path, "rb") as fh:
        print(f"sha256 {hashlib.sha256(fh.read()).hexdigest()}")
    return path


def build_clip_text(data: str, out: str = CLIP_OUT) -> str:
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    with h5py.File(data, "r") as f:
        texts = [t.decode() if isinstance(t, bytes) else str(t) for t in f["task_text_by_task"][:]]
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    with torch.no_grad():
        enc = tok(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        feat = model.get_text_features(input_ids=enc["input_ids"],
                                       attention_mask=enc["attention_mask"]).float().numpy()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, feat=feat.astype(np.float32), text=np.asarray(texts), model=CLIP_MODEL)
    cos = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    cos = cos @ cos.T
    off = cos[~np.eye(len(cos), dtype=bool)]
    print(f"wrote {out}: {feat.shape}, pairwise cosine min {off.min():.3f} median "
          f"{np.median(off):.3f} max {off.max():.3f}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DATA_DEFAULT)
    p.add_argument("--skip_mae", action="store_true")
    p.add_argument("--skip_clip", action="store_true")
    a = p.parse_args()
    if not a.skip_mae:
        fetch_mae()
    if not a.skip_clip:
        build_clip_text(a.data)


if __name__ == "__main__":
    main()
