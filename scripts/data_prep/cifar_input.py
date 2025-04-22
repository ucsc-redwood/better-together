#!/usr/bin/env python3
"""
Split CIFAR-10 train set into 10 .npy files, each containing 128 images
in channel-first (C, H, W) format.
"""

import os
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms


def main(output_dir: str = "cifar_batches"):
    # 1) Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2) Download/load CIFAR-10 train set
    # By default, data is loaded as H×W×C (32×32×3) uint8 images.
    cifar_train = CIFAR10(root="./data", train=True, download=True)

    # Extract raw numpy array of shape (50000, 32, 32, 3)
    images = cifar_train.data  # numpy array, dtype=uint8

    # 3) We only need the first 10 batches of 128 images (i.e. 1280 images total)
    num_batches = 10
    batch_size = 128

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        stop = start + batch_size

        # slice out (128, 32, 32, 3)
        batch = images[start:stop]

        # 4) Convert to channel-first: (128, 3, 32, 32)
        batch_chw = batch.transpose(0, 3, 1, 2)

        # 5) Save to .npy
        out_path = os.path.join(output_dir, f"batch_{batch_idx}.npy")
        np.save(out_path, batch_chw)
        print(f"Saved {out_path} with shape {batch_chw.shape}")


if __name__ == "__main__":
    main()
