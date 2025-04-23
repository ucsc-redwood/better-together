#!/usr/bin/env python3
"""
Train a small AlexNet‑style model on CIFAR‑10 and save parameters to .npy files.

Architecture:
  Conv1(3→16, 3×3, pad=1) → ReLU → MaxPool(2)
  Conv2(16→32, 3×3, pad=1) → ReLU → MaxPool(2)
  Conv3(32→64, 3×3, pad=1) → ReLU
  Conv4(64→64, 3×3, pad=1) → ReLU
  Conv5(64→64, 3×3, pad=1) → ReLU → MaxPool(2)
  ↓ flatten (64×4×4=1024)
  Linear(1024→10)
Saves:
  u_conv1_w, u_conv1_b, …, u_linear_w, u_linear_b
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class SmallAlexNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # → (16,32,32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → (16,16,16)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → (32,16,16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → (32,8,8)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (64,8,8)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → (64,8,8)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # → (64,8,8)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → (64,4,4)
        )
        self.classifier = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def main(
    data_dir: str = "./data",
    batch_size: int = 128,
    epochs: int = 5,
    out_dir: str = "saved_params",
):
    # prepare directories
    os.makedirs(out_dir, exist_ok=True)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset & loader
    train_ds = CIFAR10(root=data_dir, train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # model, loss, opt
    model = SmallAlexNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # training loop
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch:2d}/{epochs} — Loss: {avg_loss:.4f}")

    # save each parameter tensor to .npy
    state = model.state_dict()
    mapping = {
        "features.0.weight": "u_conv1_w",
        "features.0.bias": "u_conv1_b",
        "features.3.weight": "u_conv2_w",
        "features.3.bias": "u_conv2_b",
        "features.6.weight": "u_conv3_w",
        "features.6.bias": "u_conv3_b",
        "features.8.weight": "u_conv4_w",
        "features.8.bias": "u_conv4_b",
        "features.10.weight": "u_conv5_w",
        "features.10.bias": "u_conv5_b",
        "classifier.weight": "u_linear_w",
        "classifier.bias": "u_linear_b",
    }

    for torch_name, np_name in mapping.items():
        tensor = state[torch_name].cpu().numpy()
        path = os.path.join(out_dir, f"{np_name}.npy")
        np.save(path, tensor)
        print(f"Saved {path} (shape={tensor.shape})")


if __name__ == "__main__":
    main()
