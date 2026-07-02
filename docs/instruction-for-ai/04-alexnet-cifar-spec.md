# AlexNet (CIFAR-10) — Canonical Model Spec

> **Status: canonical.** This is the AlexNet variant the project standardizes on
> going forward. Reference implementation:
> [`scripts/data_prep/alexnet_cifar10.py`](../../scripts/data_prep/alexnet_cifar10.py)
> (class `AlexNetCIFAR`). When hand-writing OMP/CUDA/Vulkan kernels or exporting
> weights, match the shapes and semantics below exactly.

---

## 1. Why this shape (not the 2012 original, not the old `SmallAlexNet`)

The 2012 AlexNet was built for 224×224 ImageNet: an 11×11 stride-4 conv1 plus
aggressive pooling. Fed a 32×32 CIFAR image the feature map collapses to 0×0 and
the hard-coded 9216-wide FC input no longer matches — it cannot run unmodified.

`AlexNetCIFAR` keeps AlexNet's **spirit** — the classic
`64 → 192 → 384 → 256 → 256` channel ladder and a **3-layer MLP head with
dropout** — but swaps to 3×3 / stride-1 convs with gentle 2× pooling so a 32×32
input survives to the classifier. BatchNorm replaces the original's Local
Response Norm for stable, modern training.

This is **not** the repo's older `cifar_alex.py` `SmallAlexNet` (channels
`16/32/64/64/64`, a single FC `1024→10`, 9 pipeline stages). That one matched the
existing C++ kernels but was not a faithful AlexNet. The two are **not weight- or
shape-compatible**.

---

## 2. Architecture — per-layer trace (input 32×32 RGB)

Conv: 3×3 kernel, stride 1, padding 1 (spatial-preserving). Pool: 2×2 max,
stride 2. Each conv block is `Conv → BatchNorm → ReLU`. ReLU/BN fuse into the
conv at inference time.

| # | Layer            | In ch | Out ch | Op (k/s/p)      | Spatial out |
|---|------------------|-------|--------|-----------------|-------------|
| 1 | Conv1+BN+ReLU    | 3     | 64     | conv 3/1/1      | 32×32       |
| 2 | MaxPool1         | 64    | 64     | pool 2/2        | 16×16       |
| 3 | Conv2+BN+ReLU    | 64    | 192    | conv 3/1/1      | 16×16       |
| 4 | MaxPool2         | 192   | 192    | pool 2/2        | 8×8         |
| 5 | Conv3+BN+ReLU    | 192   | 384    | conv 3/1/1      | 8×8         |
| 6 | Conv4+BN+ReLU    | 384   | 256    | conv 3/1/1      | 8×8         |
| 7 | Conv5+BN+ReLU    | 256   | 256    | conv 3/1/1      | 8×8         |
| 8 | MaxPool3         | 256   | 256    | pool 2/2        | 4×4         |
|   | *flatten*        |       |        | 256×4×4 = 4096  | —           |
| 9 | FC1+ReLU (+Drop) | 4096  | 4096   | linear          | —           |
| 10| FC2+ReLU (+Drop) | 4096  | 4096   | linear          | —           |
| 11| FC3 (logits)     | 4096  | 10     | linear          | —           |

Spatial trace: `32 →(pool) 16 →(pool) 8 →(pool) 4`.

---

## 3. Pipeline stages (paper's definition)

Under the framework's "each independently schedulable conv/pool/linear = one
stage" rule (ReLU/BN fused into conv; dropout is a train-only no-op), this model
decomposes into **11 stages**:

```
1 Conv1   2 Pool1   3 Conv2   4 Pool2   5 Conv3   6 Conv4   7 Conv5   8 Pool3
9 FC1    10 FC2    11 FC3
```

Versus the old `SmallAlexNet`'s **9 stages** — the +2 comes entirely from the
3-FC head (old model had a single FC). If BatchNorm is **not** folded into the
preceding conv it becomes 5 extra stages (→ 16); the canonical assumption is
**BN folded**, matching how the OMP kernels fuse ReLU.

---

## 4. Weight tensors (for `.npy` / kernel export)

Layout is PyTorch-native, row-major (C-order), **float32**:
conv weight `(out, in, kh, kw)` (OIHW), linear weight `(out, in)`, bias `(out,)`.

| Tensor       | Shape              | Params      |
|--------------|--------------------|-------------|
| conv1_w / _b | (64, 3, 3, 3)      | 1,728 / 64  |
| conv2_w / _b | (192, 64, 3, 3)    | 110,592 / 192 |
| conv3_w / _b | (384, 192, 3, 3)   | 663,552 / 384 |
| conv4_w / _b | (256, 384, 3, 3)   | 884,736 / 256 |
| conv5_w / _b | (256, 256, 3, 3)   | 589,824 / 256 |
| fc1_w / _b   | (4096, 4096)       | 16,777,216 / 4096 |
| fc2_w / _b   | (4096, 4096)       | 16,777,216 / 4096 |
| fc3_w / _b   | (10, 4096)         | 40,960 / 10 |

Plus per-conv BatchNorm params `weight, bias, running_mean, running_var` each of
shape `(out_ch,)` — fold these into the conv weight/bias for an inference kernel
(`w' = γ·w/√(var+ε)`, `b' = γ·(b−mean)/√(var+ε) + β`), or load them separately.

Total ≈ **52.6M** parameters (the two 4096×4096 FC layers dominate — this is why
the checkpoint is ~137 MB and is a deliberate AlexNet trait, not a bug).

> Migrated 2026-07-02: both cifar apps' `AppData` and `run_stage_*` kernels now
> implement exactly the table above (`u_conv1_w` … `u_fc3_w`), verified on all
> three backends on real hardware. `scripts/data_prep/prune_alexnet_cifar10.py`
> exports the BN-folded dense weights and the magnitude-pruned (25%-density)
> sparse CSR variant the sparse app mirrors.

---

## 5. Preprocessing contract (must match train & inference)

Per-channel normalization over CIFAR-10:

```
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```

Train augmentation: `RandomCrop(32, padding=4)` + horizontal flip. Eval/export:
`ToTensor` + `Normalize(mean, std)` only. Any exported input batch fed to a
kernel must already be normalized **float32 `(N, 3, 32, 32)`** — raw uint8
`[0,255]` is wrong and will not match the trained weights.

---

## 6. Training reference

```bash
uv run scripts/data_prep/alexnet_cifar10.py --epochs 35          # ~88–90% test acc
uv run scripts/data_prep/alexnet_cifar10.py --epochs 5           # quick smoke test
uv run scripts/data_prep/alexnet_cifar10.py --export-npy         # dump per-layer .npy
```

SGD (momentum 0.9, nesterov, weight-decay 5e-4), cosine LR from 0.1, label
smoothing 0.1, AMP on CUDA. 45k/5k train/val split + held-out 10k test.
Best-val checkpoint saved to `saved_params/alexnet_cifar10_best.pt`.

---

## 7. Deploying the real weights

`scripts/data_prep/prune_alexnet_cifar10.py` leaves the trained export in
`saved_params/export/`: `dense/conv{1..5}_{w,b}.npy` + `fc{1..3}_{w,b}.npy`
(BN-folded, f32, OIHW / `(out,in)` row-major), `sparse/conv{i}_csr_{values,col_idx,row_ptr}.npy`
(CSR over `(out_ch, in_ch*3*3)`, plus biases and the dense FC head), and a
normalized `test_batch.npy` `(128,3,32,32)` + `test_labels.npy` `(128,)`. Both
cifar `AppData`s honor **`BT_WEIGHTS_DIR`**: unset → the synthetic seeded init
(hermetic tests, byte-identical to before); set → the real weights and real
test batch are loaded and **any missing file or shape mismatch throws** —
asking for real weights never silently falls back. Deploy with

```bash
scripts/deploy-weights.sh jetson              # -> doremy@duck-stable:/tmp/bt/weights
scripts/deploy-weights.sh rocky               # -> rocky-ryzen:/tmp/bt/weights
scripts/deploy-weights.sh android R5CY21Y3VEV # -> /data/local/tmp/bt/weights
```

after which the `run-on-{jetson,rocky,android}.sh` scripts export
`BT_WEIGHTS_DIR` automatically when the deployed dir exists. The end-to-end
check is `RealWeights_EndTaskAccuracy` in the two OMP test binaries (skips
without `BT_WEIGHTS_DIR`; asserts test-batch accuracy ≥ 0.85 — dense measures
~0.90, sparse ~0.90):

```bash
BT_WEIGHTS_DIR=$PWD/saved_params/export \
    ./build/pc/test-cifar-dense-omp --gtest_filter='*RealWeights*'
```
