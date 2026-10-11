"""Classify a house photo with the trained model.

Usage:
    python -m hc_mlx.inference path/to/photo.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from hc_mlx.config import BEST_CKPT, LAST_CKPT, MEAN, STD, TARGET_CLASSES
from hc_mlx.models.cnn import HouseCNN


def predict(
    path: str | Path,
    ckpt: str | Path | None = None,
    size: int = 128,
    repo: str | None = None,
) -> list[tuple[str, float]]:
    if ckpt is None and repo is not None:
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(repo_id=repo, filename="house_cnn_best.safetensors")
    if ckpt is None:
        ckpt = str(BEST_CKPT if BEST_CKPT.exists() else LAST_CKPT)
    if ckpt is None:
        ckpt = str(BEST_CKPT if BEST_CKPT.exists() else LAST_CKPT)
    model = HouseCNN(num_classes=len(TARGET_CLASSES))    model.load_weights(str(ckpt))
    model.eval()

    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(MEAN)) / np.array(STD)
    probs = mx.softmax(model(mx.array(arr[None])), axis=-1)[0]
    mx.eval(probs)
    probs = np.array(probs.tolist(), dtype=np.float64)
    return [(TARGET_CLASSES[i], float(probs[i])) for i in np.argsort(-probs)]


def main() -> None:
    p = argparse.ArgumentParser(description="Classify a house photo")
    p.add_argument("image", help="path to photo")
    p.add_argument("--ckpt", default=None, help="checkpoint path (default: best if present)")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--top", type=int, default=3)
    a = p.parse_args()
    for label, prob in predict(a.image, a.ckpt, a.size)[: a.top]:
        print(f"{label:10s} {prob:.3f}")


if __name__ == "__main__":
    main()
