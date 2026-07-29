#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.2",
#     "torchvision>=0.17",
#     "numpy>=1.26",
# ]
# ///
"""
AlexNet adapted to CIFAR-10 — a proper, reference-quality training script.

Why an *adapted* AlexNet (not the 2012 original)?
  The original AlexNet was built for 224x224 ImageNet: an 11x11 stride-4 conv1
  plus aggressive pooling. Fed a 32x32 CIFAR image, the feature map collapses to
  0x0 after a couple of stages and the hard-coded 9216-wide FC input no longer
  matches. So we keep AlexNet's *spirit* (5 conv blocks with the classic
  64->192->384->256->256 channel ladder + a 3-layer MLP head with dropout) but
  swap to 3x3 / stride-1 convs and gentle 2x pooling so 32x32 survives to the head.
  BatchNorm is added (the original used Local Response Norm) for stable, modern
  training — this is the pragmatic "gold standard" for CIFAR.

Best practices included (this is what made the repo's old SmallAlexNet NOT robust):
  - Correct preprocessing: per-channel Normalize with CIFAR-10 mean/std.
  - Data augmentation: RandomCrop(pad=4) + horizontal flip (train only).
  - A real train/val split (45k/5k) AND a held-out test evaluation.
  - Accuracy is actually measured and reported every epoch.
  - SGD + momentum + nesterov + weight decay, cosine LR schedule, label smoothing.
  - Mixed precision (AMP) on CUDA, reproducible seeding, best-checkpoint saving.

Run (uv handles the deps automatically):
    uv run scripts/data_prep/alexnet_cifar10.py                # train ~35 epochs
    uv run scripts/data_prep/alexnet_cifar10.py --epochs 5     # quick smoke test
    uv run scripts/data_prep/alexnet_cifar10.py --export-npy   # also dump per-layer .npy

Expected: ~88-90% top-1 test accuracy after ~35 epochs on a single modern GPU.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

# CIFAR-10 per-channel statistics (computed over the 50k training images).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class AlexNetCIFAR(nn.Module):
    """AlexNet's structure, re-scaled so a 32x32 input survives to the head.

    Spatial trace for a 32x32 input:
        conv1 -> 32 ; pool -> 16
        conv2 -> 16 ; pool ->  8
        conv3 ->  8b
        conv4 ->  8
        conv5 ->  8 ; pool ->  4
        flatten = 256 * 4 * 4 = 4096
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir: str, batch_size: int, workers: int, seed: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )

    # Two views of the same 50k images so train/val use different transforms.
    full_train = CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    full_val = CIFAR10(data_dir, train=True, download=True, transform=eval_tf)
    test_set = CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    n_val = 5000
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(full_train), generator=gen)
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    train_set = torch.utils.data.Subset(full_train, train_idx.tolist())
    val_set = torch.utils.data.Subset(full_val, val_idx.tolist())

    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        preds = model(imgs).argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total


def export_npy(model: nn.Module, out_dir: str) -> None:
    """Dump each weight/bias tensor to a .npy (float32) for inspection / C++ use."""
    os.makedirs(out_dir, exist_ok=True)
    for name, tensor in model.state_dict().items():
        if tensor.dtype.is_floating_point:
            path = os.path.join(out_dir, name.replace(".", "_") + ".npy")
            np.save(path, tensor.cpu().numpy().astype(np.float32))
    print(f"[export] wrote {len(os.listdir(out_dir))} .npy files to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train AlexNet adapted to CIFAR-10.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="saved_params")
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-npy", action="store_true", help="dump per-layer .npy after training")
    args = p.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple-silicon GPU (fp32; AMP stays CUDA-only)
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[setup] device={device} epochs={args.epochs} batch={args.batch_size} lr={args.lr}")

    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, args.batch_size, args.workers, args.seed
    )

    model = AlexNetCIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0
    best_path = os.path.join(args.out_dir, "alexnet_cifar10_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(imgs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * imgs.size(0)
        scheduler.step()

        train_loss = running / (len(train_loader) * args.batch_size)
        val_acc = evaluate(model, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]
        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            flag = "  <- best"
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  "
            f"val_acc {val_acc:5.2f}%  lr {lr_now:.4f}{flag}"
        )

    # Final test accuracy with the best checkpoint.
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc = evaluate(model, test_loader, device)
    print(f"\n[done] best val_acc {best_acc:.2f}%  |  test_acc {test_acc:.2f}%")
    print(f"[done] checkpoint: {best_path}")

    if args.export_npy:
        export_npy(model, os.path.join(args.out_dir, "npy"))


if __name__ == "__main__":
    main()
