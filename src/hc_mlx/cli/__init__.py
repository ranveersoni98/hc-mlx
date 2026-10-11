"""The `hc-mlx` command: one entry point for data, training, and prediction."""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="hc-mlx", description="Useless house-type classifier (MLX)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show classes and dataset sources")

    dl = sub.add_parser("download", help="fetch HF sources and build data/combined")
    dl.add_argument("--per-class", type=int, default=0,
                    help="also scale up via DuckDB to this many images per class (0 = skip)")

    tr = sub.add_parser("train", help="train the classifier")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch", type=int, default=32)
    tr.add_argument("--size", type=int, default=128)

    pr = sub.add_parser("predict", help="classify one photo")
    pr.add_argument("image", help="path to photo")
    pr.add_argument("--top", type=int, default=3)
    pr.add_argument("--repo", default=None, help="Hub model repo to download weights from")

    a = p.parse_args()
    if a.cmd == "info":
        from hc_mlx.config import INDIAN_REPOS, PLACES_256_LABEL_IDS, PLACES_VAL_MAP, TARGET_CLASSES

        print("classes:", TARGET_CLASSES)
        print("places-val map:", PLACES_VAL_MAP)
        print("places-256 labels:", PLACES_256_LABEL_IDS)
        print("indian pools:", INDIAN_REPOS)
    elif a.cmd == "download":
        from hc_mlx.data import build_combined, download_places_val, fetch_places256_duckdb, summarize_combined

        download_places_val()
        if a.per_class > 0:
            fetch_places256_duckdb(per_class=a.per_class)
        build_combined()
        summarize_combined()
    elif a.cmd == "train":
        from hc_mlx.training import train

        train(epochs=a.epochs, batch=a.batch, size=a.size)
    elif a.cmd == "predict":
        from hc_mlx.inference import predict

        for label, prob in predict(a.image, repo=a.repo)[: a.top]:
            print(f"{label:10s} {prob:.3f}")


if __name__ == "__main__":
    main()
