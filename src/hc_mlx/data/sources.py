"""Fetchers: pull raw house images from the Hugging Face Hub.

Sources:
  A. rumenguin/places365-val      (primary, labeled, 100 imgs/class)
  B. ljnlonoljpiljm/places365-256px (scale-up, 1.8M imgs, DuckDB fast path)
  C. scikit-harry Indian sets     (unlabeled, Indian context)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from hc_mlx.config import (
    INDIAN_REPOS,
    PLACES_256_LABEL_IDS,
    PLACES_256_N_SHARDS,
    PLACES_256_REPO,
    PLACES_256_URL,
    PLACES_VAL_MAP,
    PLACES_VAL_REPO,
    TARGET_CLASSES,
)


def download_places_val(
    dest: str | Path = "data/raw/places365-val",
    classes: list[str] | None = None,
) -> Path:
    """Download only the class folders we need (skips the other 359)."""
    from huggingface_hub import HfFileSystem

    classes = classes or TARGET_CLASSES
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem()
    total = 0
    for target in classes:
        for folder in PLACES_VAL_MAP[target]:
            out_dir = dest / target / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            files = fs.glob(f"datasets/{PLACES_VAL_REPO}/images/{folder}/*")
            for src in files:
                out_file = out_dir / src.rsplit("/", 1)[-1]
                if out_file.exists():
                    continue
                with fs.open(src, "rb") as fsrc, open(out_file, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst)
                total += 1
            print(f"[places-val] {target}/{folder}: {len(files)} files")
    print(f"[places-val] downloaded {total} new files -> {dest}")
    return dest


def stream_places256(
    per_class: int = 500,
    classes: list[str] | None = None,
    dest: str | Path = "data/raw/places365-256px",
) -> Path:
    """Stream the big parquet set and keep matching labels.

    Slow on class-grouped shards. Prefer fetch_places256_duckdb below.
    """
    from datasets import load_dataset

    classes = classes or TARGET_CLASSES
    dest = Path(dest)
    wanted: dict[int, str] = {lid: t for t in classes for lid in PLACES_256_LABEL_IDS[t]}
    counts = {t: 0 for t in classes}
    for t in classes:
        (dest / t).mkdir(parents=True, exist_ok=True)

    ds = load_dataset(PLACES_256_REPO, split="train", streaming=True)
    saved = 0
    for ex in ds:
        label = int(ex["label"])
        if label not in wanted:
            continue
        target = wanted[label]
        if counts[target] >= per_class:
            if all(counts[t] >= per_class for t in classes):
                break
            continue
        if ex["image"] is None:
            continue
        ex["image"].convert("RGB").save(dest / target / f"places256_{label}_{counts[target]:05d}.jpg", "JPEG", quality=90)
        counts[target] += 1
        saved += 1
        if saved % 100 == 0:
            print(f"[places256] progress {counts}")
    print(f"[places256] done {counts} -> {dest}")
    return dest


def fetch_places256_duckdb(
    per_class: int = 1000,
    classes: list[str] | None = None,
    dest: str | Path = "data/raw/places365-256px",
) -> Path:
    """Fast scale-up fetch using DuckDB over the Hub parquet URLs.

    The shards are roughly grouped by class, so we first scan only the
    cheap label column of each shard, then pull image bytes solely from
    shards that contain labels we want.
    """
    import io

    import duckdb
    from PIL import Image

    classes = classes or TARGET_CLASSES
    dest = Path(dest)
    for t in classes:
        (dest / t).mkdir(parents=True, exist_ok=True)

    wanted: dict[int, str] = {lid: t for t in classes for lid in PLACES_256_LABEL_IDS[t]}
    counts = {t: sum(1 for _ in (dest / t).glob("*.jpg")) for t in classes}
    print(f"[p256] existing counts: {counts}")

    con = duckdb.connect()
    hits: dict[int, list[int]] = {}
    import time

    def _distinct_labels(url: str) -> list[int] | None:
        for attempt in range(4):
            try:
                return [r[0] for r in con.execute(f"SELECT DISTINCT label FROM read_parquet('{url}')").fetchall()]
            except Exception as e:
                wait = 15 * (attempt + 1)
                print(f"[p256] scan retry in {wait}s ({e})".split("(")[0])
                time.sleep(wait)
        return None

    for i in range(PLACES_256_N_SHARDS):
        url = PLACES_256_URL.format(i=i)
        labels = _distinct_labels(url)
        time.sleep(3)
        if labels is None:
            print(f"[p256] shard {i}: scan failed after retries, skipping")
            continue
        got = sorted(set(labels) & set(wanted))
        if got:
            hits[i] = got
            print(f"[p256] shard {i}: labels {got}")
    for i, lids in hits.items():
        if all(counts[t] >= per_class for t in classes):
            break
        need = sorted({lid for lid in lids if counts[wanted[lid]] < per_class})
        if not need:
            continue
        url = PLACES_256_URL.format(i=i)
        for attempt in range(4):
            try:
                rows = con.execute(
                    f"SELECT label, image.bytes FROM read_parquet('{url}') WHERE label IN ({','.join(map(str, need))})"
                ).fetchall()
                break
            except Exception:
                print(f"[p256] shard {i}: fetch retry in 30s")
                time.sleep(30)
        else:
            print(f"[p256] shard {i}: fetch failed after retries, skipping")
            continue
        for label, blob in rows:
            target = wanted[int(label)]
            if counts[target] >= per_class:
                continue
            try:
                img = Image.open(io.BytesIO(bytes(blob))).convert("RGB")
            except Exception:
                continue
            img.save(dest / target / f"p256_{label}_{counts[target]:05d}.jpg", "JPEG", quality=90)
            counts[target] += 1
        print(f"[p256] shard {i} -> {counts}")
    print(f"[p256] done {counts} -> {dest}")
    return dest


def download_indian_pool(dest: str | Path = "data/raw/indian-pool") -> Path:
    """Snapshot the small Indian residential repos (unlabeled extras)."""
    from huggingface_hub import snapshot_download

    from hc_mlx.config import IMG_EXTS

    dest = Path(dest)
    for repo in INDIAN_REPOS:
        out = dest / repo.split("/")[-1]
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(out))
        n = sum(1 for p in out.rglob("*") if p.suffix.lower() in IMG_EXTS)
        print(f"[indian-pool] {repo}: {n} images -> {out}")
    return dest
