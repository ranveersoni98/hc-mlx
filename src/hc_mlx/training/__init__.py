"""Train the house-type classifier.

Usage:
    python -m hc_mlx.training [--epochs 30] [--batch 32] [--size 128]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from hc_mlx.config import BEST_CKPT, CKPT_DIR, LAST_CKPT, TARGET_CLASSES
from hc_mlx.data.build import load_images, stratified_split
from hc_mlx.models.cnn import HouseCNN, count_params


def accuracy(logits: mx.array, y: mx.array) -> float:
    return float(mx.mean(mx.argmax(logits, axis=-1) == y).item())


def train(
    epochs: int = 20,
    batch: int = 32,
    size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    weight_decay: float = 5e-4,
    flip_aug: bool = True,
) -> Path:
    np.random.seed(seed)
    mx.random.seed(seed)

    X, y = load_images(size)
    (Xtr, ytr), (Xte, yte) = stratified_split(X, y)
    print(f"[split] train={len(Xtr)} test={len(Xte)}")
    if flip_aug:
        Xtr = np.concatenate([Xtr, Xtr[:, :, ::-1, :]], axis=0)
        ytr = np.concatenate([ytr, ytr], axis=0)
        print(f"[aug] +horizontal flip -> train={len(Xtr)}")

    counts = np.bincount(ytr, minlength=len(TARGET_CLASSES)).astype(np.float32)
    cw = counts.sum() / (len(TARGET_CLASSES) * np.maximum(counts, 1))
    print(f"[data] train class counts={counts.astype(int).tolist()} weights={cw.round(3).tolist()}")

    model = HouseCNN(num_classes=len(TARGET_CLASSES))
    print(f"[model] HouseCNN params={count_params(model):,}")
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(model: HouseCNN, Xb: mx.array, yb: mx.array, wb: mx.array) -> mx.array:
        logits = model(Xb)
        ce = nn.losses.cross_entropy(logits, yb, reduction="none", label_smoothing=0.05)
        return mx.mean(ce * wb)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    Xtr_m, ytr_m = mx.array(Xtr), mx.array(ytr)
    Xte_m, yte_m = mx.array(Xte), mx.array(yte)
    wtr = mx.array(cw[ytr])

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_acc, best_ep = 0.0, 0
    n = len(Xtr)
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n)
        tot_loss, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = mx.array(perm[i : i + batch])
            loss, grads = loss_and_grad(model, Xtr_m[idx], ytr_m[idx], wtr[idx])
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            tot_loss += float(loss.item()) * len(idx)
            nb += len(idx)
        model.eval()
        tr_acc = accuracy(model(Xtr_m), ytr_m)
        te_acc = accuracy(model(Xte_m), yte_m)
        model.train()
        if te_acc > best_acc:
            best_acc, best_ep = te_acc, epoch
            model.save_weights(str(BEST_CKPT))
        print(
            f"[ep {epoch:02d}/{epochs}] loss={tot_loss / nb:.4f} "
            f"train_acc={tr_acc:.3f} val_acc={te_acc:.3f} best={best_acc:.3f}@{best_ep}"
        )

    model.eval()
    print(f"[done] final val_acc={accuracy(model(Xte_m), yte_m):.3f} best={best_acc:.3f}@ep{best_ep}")
    model.save_weights(str(LAST_CKPT))
    print(f"[ckpt] last -> {LAST_CKPT} | best -> {BEST_CKPT}")
    return BEST_CKPT


def main() -> None:
    p = argparse.ArgumentParser(description="Train the house-type classifier (MLX)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wd", type=float, default=5e-4, help="AdamW weight decay")
    p.add_argument("--no-aug", action="store_true", help="disable horizontal-flip augmentation")
    a = p.parse_args()
    train(
        epochs=a.epochs,
        batch=a.batch,
        size=a.size,
        lr=a.lr,
        seed=a.seed,
        weight_decay=a.wd,
        flip_aug=not a.no_aug,
    )


if __name__ == "__main__":
    main()
