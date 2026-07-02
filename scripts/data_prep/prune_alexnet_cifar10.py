#!/usr/bin/env python3
"""Prune AlexNetCIFAR's convs, fine-tune, fold BatchNorm, export deployment weights.

Produces the two REAL weight sets the C++ apps load (04-alexnet-cifar-spec.md §4),
both BN-folded (the kernels are Conv+ReLU with no BN):

  saved_params/export/dense/   conv{1..5}_{w,b}.npy  fc{1..3}_{w,b}.npy
  saved_params/export/sparse/  conv{i}_csr_{values,col_idx,row_ptr}.npy  conv{i}_b.npy
                               fc{1..3}_{w,b}.npy               (FC head stays dense)
  saved_params/export/         test_batch.npy (128,3,32,32 normalized)  test_labels.npy

The sparse variant mirrors the paper: magnitude-prune the CONV layers only (CSR over
the (out_ch, in_ch*kh*kw) flattened matrix -- exactly cifar_sparse::CSRMatrix's
layout), fine-tune with the masks pinned so accuracy recovers, then fold BN (zeros
stay zeros under per-channel scaling). The dense export folds the pristine
checkpoint. Both exports print their test accuracy; the folded model is re-verified
against the BN model before writing anything.

Usage:
  uv run --with torch --with torchvision python3 scripts/data_prep/prune_alexnet_cifar10.py \
      [--sparsity 0.75] [--finetune-epochs 6] [--checkpoint saved_params/alexnet_cifar10_best.pt]
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alexnet_cifar10 import AlexNetCIFAR, build_loaders, evaluate, set_seed  # noqa: E402

# (conv, bn) feature indices and the export names, in stage order.
CONV_BN = [(0, 1), (4, 5), (8, 9), (11, 12), (14, 15)]
CONV_NAMES = ["conv1", "conv2", "conv3", "conv4", "conv5"]
FC_IDX = [1, 4, 6]
FC_NAMES = ["fc1", "fc2", "fc3"]


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Return folded (weight, bias): w' = w*g/s, b' = (b-mean)*g/s + beta, s=sqrt(var+eps)."""
    g, beta = bn.weight.detach(), bn.bias.detach()
    mean, var, eps = bn.running_mean.detach(), bn.running_var.detach(), bn.eps
    scale = g / torch.sqrt(var + eps)
    w = conv.weight.detach() * scale[:, None, None, None]
    b = (conv.bias.detach() - mean) * scale + beta
    return w, b


def folded_copy(model: AlexNetCIFAR) -> AlexNetCIFAR:
    """A BN-free copy computing the same function: BN folded into convs, BN -> Identity."""
    m = copy.deepcopy(model).eval()
    for ci, bi in CONV_BN:
        w, b = fold_bn(m.features[ci], m.features[bi])
        m.features[ci].weight.data.copy_(w)
        m.features[ci].bias.data.copy_(b)
        m.features[bi] = nn.Identity()
    return m


def prune_masks(model: AlexNetCIFAR, sparsity: float):
    """Per-conv-layer magnitude masks keeping the largest (1-sparsity) fraction."""
    masks = {}
    for ci, _ in CONV_BN:
        w = model.features[ci].weight.detach()
        k = int(round(w.numel() * (1.0 - sparsity)))
        thresh = w.abs().flatten().kthvalue(w.numel() - k + 1).values
        masks[ci] = (w.abs() >= thresh).float()
    return masks


def apply_masks(model: AlexNetCIFAR, masks) -> None:
    for ci, m in masks.items():
        model.features[ci].weight.data.mul_(m.to(model.features[ci].weight.device))


def finetune(model, masks, train_loader, val_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            apply_masks(model, masks)  # pin the pruned zeros
        sched.step()
        acc = evaluate(model, val_loader, device)
        print(f"[finetune] epoch {epoch}/{epochs}  val_acc {acc:5.2f}%")


def save(path, arr, dtype):
    np.save(path, np.ascontiguousarray(arr.cpu().numpy().astype(dtype)))


def export_dense(model_folded, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, (ci, _) in zip(CONV_NAMES, CONV_BN):
        conv = model_folded.features[ci]
        save(os.path.join(out_dir, f"{name}_w.npy"), conv.weight.detach(), np.float32)
        save(os.path.join(out_dir, f"{name}_b.npy"), conv.bias.detach(), np.float32)
    for name, li in zip(FC_NAMES, FC_IDX):
        fc = model_folded.classifier[li]
        save(os.path.join(out_dir, f"{name}_w.npy"), fc.weight.detach(), np.float32)
        save(os.path.join(out_dir, f"{name}_b.npy"), fc.bias.detach(), np.float32)


def export_sparse(model_folded, out_dir):
    """CSR per conv over the (out_ch, in_ch*kh*kw) flattened matrix (CSRMatrix layout)."""
    os.makedirs(out_dir, exist_ok=True)
    for name, (ci, _) in zip(CONV_NAMES, CONV_BN):
        conv = model_folded.features[ci]
        w2d = conv.weight.detach().cpu().numpy().reshape(conv.weight.shape[0], -1)
        values, col_idx, row_ptr = [], [], [0]
        for row in w2d:
            (nz,) = row.nonzero()
            values.extend(row[nz])
            col_idx.extend(nz)
            row_ptr.append(len(col_idx))
        np.save(
            os.path.join(out_dir, f"{name}_csr_values.npy"), np.asarray(values, dtype=np.float32)
        )
        np.save(
            os.path.join(out_dir, f"{name}_csr_col_idx.npy"), np.asarray(col_idx, dtype=np.int32)
        )
        np.save(
            os.path.join(out_dir, f"{name}_csr_row_ptr.npy"), np.asarray(row_ptr, dtype=np.int32)
        )
        save(os.path.join(out_dir, f"{name}_b.npy"), conv.bias.detach(), np.float32)
        density = len(values) / w2d.size
        print(f"[sparse] {name}: nnz={len(values)} density={density:.3f}")
    for name, li in zip(FC_NAMES, FC_IDX):
        fc = model_folded.classifier[li]
        save(os.path.join(out_dir, f"{name}_w.npy"), fc.weight.detach(), np.float32)
        save(os.path.join(out_dir, f"{name}_b.npy"), fc.bias.detach(), np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="saved_params/alexnet_cifar10_best.pt")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="saved_params/export")
    p.add_argument("--sparsity", type=float, default=0.75, help="conv zero fraction")
    p.add_argument("--finetune-epochs", type=int, default=6)
    p.add_argument("--finetune-lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir, args.batch_size, workers=4, seed=args.seed
    )

    model = AlexNetCIFAR().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    base_acc = evaluate(model, test_loader, device)
    print(f"[dense] checkpoint test_acc {base_acc:5.2f}%")

    # Folding sanity gate: the BN-free copy must match the BN model.
    dense_folded = folded_copy(model)
    folded_acc = evaluate(dense_folded, test_loader, device)
    print(f"[dense] BN-folded test_acc {folded_acc:5.2f}%")
    assert abs(folded_acc - base_acc) < 0.3, "BN folding changed the model"

    # Prune -> fine-tune (masks pinned; BN keeps adapting) -> fold -> verify zeros.
    sparse_model = copy.deepcopy(model)
    masks = prune_masks(sparse_model, args.sparsity)
    apply_masks(sparse_model, masks)
    pruned_acc = evaluate(sparse_model, test_loader, device)
    print(f"[sparse] pruned@{args.sparsity:.0%} (no finetune) test_acc {pruned_acc:5.2f}%")
    finetune(
        sparse_model,
        masks,
        train_loader,
        val_loader,
        device,
        args.finetune_epochs,
        args.finetune_lr,
    )
    sparse_acc = evaluate(sparse_model, test_loader, device)
    print(f"[sparse] fine-tuned test_acc {sparse_acc:5.2f}%")
    sparse_folded = folded_copy(sparse_model)
    for ci, _ in CONV_BN:
        w = sparse_folded.features[ci].weight.detach()
        assert (w == 0).float().mean() >= args.sparsity - 0.01, "folding disturbed zeros"

    export_dense(dense_folded, os.path.join(args.out_dir, "dense"))
    export_sparse(sparse_folded, os.path.join(args.out_dir, "sparse"))

    # A normalized eval batch + labels for on-device end-task accuracy checks.
    x, y = next(iter(test_loader))
    np.save(os.path.join(args.out_dir, "test_batch.npy"), x.numpy().astype(np.float32))
    np.save(os.path.join(args.out_dir, "test_labels.npy"), y.numpy().astype(np.int32))
    print(f"[export] done -> {args.out_dir} (dense {base_acc:.2f}%, sparse {sparse_acc:.2f}%)")


if __name__ == "__main__":
    main()
