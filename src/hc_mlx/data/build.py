"""Local dataset building: raw Hub downloads -> train-ready arrays."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np

from hc_mlx.config import CACHE_DIR, DATA_DIR, IMG_EXTS, MEAN, STD, TARGET_CLASSES


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_combined(
    raw_places: str | Path = "data/raw/places365-val",
    raw_extra: str | Path | None = "data/raw/places365-256px",
    dest: str | Path | None = None,
    indian_pool: str | Path | None = None,
    indian_assign: str = "haveli",
) -> Path:
    """Flatten raw per-class folders into data/combined/<class>/*.jpg.

    Merges every raw source passed in. Re-runnable: existing files are kept.
    """
    dest = Path(dest or DATA_DIR)
    for target in TARGET_CLASSES:
        (dest / target).mkdir(parents=True, exist_ok=True)

    # Hash everything already in dest so re-runs and overlapping raw
    # sources never create duplicate images (which would leak train to test).
    seen = set()
    for target in TARGET_CLASSES:
        for f in (dest / target).glob("*"):
            if f.is_file():
                seen.add(_md5(f))
    counts = {t: sum(1 for _ in (dest / t).glob("*")) for t in TARGET_CLASSES}
    dupes = 0
    for raw in [Path(raw_places), *([Path(raw_extra)] if raw_extra and Path(raw_extra).exists() else [])]:
        for target in TARGET_CLASSES:
            src_dir = raw / target
            if not src_dir.exists():
                continue
            for f in sorted(src_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    digest = _md5(f)
                    if digest in seen:
                        dupes += 1
                        continue
                    seen.add(digest)
                    out = dest / target / f"{target}_{counts[target]:05d}{f.suffix.lower()}"
                    if not out.exists():
                        shutil.copyfile(f, out)
                    counts[target] += 1

    if indian_pool is not None and Path(indian_pool).exists() and indian_assign in TARGET_CLASSES:
        for f in sorted(Path(indian_pool).rglob("*")):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                digest = _md5(f)
                if digest in seen:
                    dupes += 1
                    continue
                seen.add(digest)
                out = dest / indian_assign / f"indian_{counts[indian_assign]:05d}{f.suffix.lower()}"
                if not out.exists():
                    shutil.copyfile(f, out)
                counts[indian_assign] += 1

    print(f"[build] combined class counts: {counts} (dupes skipped: {dupes}) -> {dest}")
    return dest


def summarize_combined(data_dir: str | Path | None = None) -> dict[str, int]:
    data_dir = Path(data_dir or DATA_DIR)
    counts = {}
    for target in TARGET_CLASSES:
        d = data_dir / target
        counts[target] = (
            sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
            if d.exists()
            else 0
        )
    print(f"[summary] {counts} (total={sum(counts.values())})")
    return counts


def _manifest_count(data_dir: Path) -> int:
    return sum(
        1
        for t in TARGET_CLASSES
        for p in (data_dir / t).rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def load_images(
    size: int = 128, data_dir: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load data/combined into RAM as normalized float32 arrays.

    Results are cached under data/processed. The cache name includes the
    image count, so growing the dataset never reuses a stale cache.
    """
    from datasets import load_dataset
    from PIL import Image

    data_dir = Path(data_dir or DATA_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"arrays_{size}_n{_manifest_count(data_dir)}.npz"
    if cache.exists():
        print(f"[data] loading cache {cache}")
        z = np.load(cache)
        return z["X"], z["y"]

    ds = load_dataset("imagefolder", data_dir=str(data_dir), drop_labels=False)["train"]
    label_names: list[str] = ds.features["label"].names
    print(f"[data] {len(ds)} images, labels: {label_names}")
    remap = [TARGET_CLASSES.index(n) for n in label_names]

    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)
    Xs, ys, skipped = [], [], 0
    for i in range(len(ds)):
        try:
            ex = ds[i]
            img = ex["image"].convert("RGB").resize((size, size), Image.BILINEAR)
        except Exception as e:
            skipped += 1
            print(f"[data] WARNING: skipping unreadable image idx={i} ({e})")
            continue
        Xs.append((np.asarray(img, dtype=np.float32) / 255.0 - mean) / std)
        ys.append(remap[int(ex["label"])])
    print(f"[data] kept={len(Xs)} skipped={skipped}")

    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int32)
    np.savez_compressed(cache, X=X, y=y)
    print(f"[data] cached {X.shape} {y.shape} -> {cache}")
    return X, y


def stratified_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for c in range(len(TARGET_CLASSES)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_te = max(1, int(len(idx) * test_size))
        te_idx += idx[:n_te].tolist()
        tr_idx += idx[n_te:].tolist()
    tr_idx = np.array(rng.permutation(tr_idx))
    te_idx = np.array(te_idx)
    return (X[tr_idx], y[tr_idx]), (X[te_idx], y[te_idx])
