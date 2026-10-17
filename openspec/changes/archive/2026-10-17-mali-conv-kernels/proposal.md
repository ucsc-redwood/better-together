# Proposal

## Why

On the Pixel 7a (Mali-G710), the cifar convolutions are the largest cost left in the project, and the
kernel work done so far did not reach them. Both recent conv speed-ups get their gain from loading the
weights as scalar (subgroup-uniform) data, which Mali has no scalar unit to exploit.

Measured with `bm-prof-cifar-{sparse,dense}-vk` on the phones, isolated scenario, medians over 3
interleaved rounds (2026-10-15/16; `docs/reports-for-human/2026-10-15-vk-shader-profile.md` sections 8-9):

| stage (3x3 conv) | 1 | 3 | 5 | 6 | 7 | app sum |
|---|---|---|---|---|---|---|
| cifar-sparse, Pixel 7a (ms) | 22.1 | 166.1 | 247.0 | 326.7 | 219.9 | ~1046 |
| cifar-sparse, Samsung (ms) | 5.3 | 34.9 | 81.7 | 98.2 | 59.7 | ~280 |
| cifar-dense, Pixel 7a (ms) | 7.1 | 24.5 | 36.3 | 48.8 | 32.6 | ~167 |
| cifar-dense, Samsung (ms) | 0.9 | 2.3 | 2.7 | 3.4 | 2.1 | ~19 |

* The five convs are ~94 % of the Pixel's cifar-sparse and ~89 % of its cifar-dense time.
* The sparse CSR fast path (subgroup-uniform row bounds) is -42 % on the Samsung and **0 %** on the
  Pixel.
* The dense conv's branchless-load + uniform-weight path is -45 % on the Samsung but **+12 %** on
  the Pixel. The shader therefore keeps the previous kernel when `SUBGROUP_SIZE < 32`, so nothing on
  Mali got faster.

## What Changes

- Add a Mali code path (selected when the subgroup size is below 32, the same switch the dense conv
  already uses) to the two 3x3/stride-1/pad-1 conv kernels:
  - `new_cifar_sparse_conv2d_k3s1p1.comp`;
  - `new_cifar_dense_conv2d_k3s1p1.comp`.

  The aim is to reuse each loaded value across several outputs without relying on scalar loads, for
  example by staging the input window and the weights in shared memory, blocking several output
  pixels per invocation, and using vectorized loads. design.md picks among these by measurement.
- Keep the AMD-style path (subgroup >= 32) bit-for-bit unchanged, so the Samsung and the 780M cannot
  regress.
- Keep FP32 everywhere. Reduced precision is a non-goal here; it belongs to `cifar-gpu-kernels`.
- Add a Mali profiling step to the kernel workflow:
  - static analysis with `malioc` (Arm Performance Studio), which is not installed today;
  - per-stage Pixel A/B with the existing `bm-prof` method.
- Relationship to the unstarted `cifar-gpu-kernels` change: its dense-conv step 6 ("a Mali-tuned
  variant") is owned by this change. Its other steps are untouched; they target subgroup >= 32 GPUs
  and FP16/INT8.

## Capabilities

### New Capabilities
- `gpu-kernels/mali-conv`: the contract of the cifar 3x3 convolution kernels on GPUs with a subgroup
  size below 32 and no scalar unit (Mali). It covers which path such a GPU runs, its correctness
  against the double-precision reference at any subgroup size and map size, the guarantee that the
  subgroup >= 32 path does not change, and the measured-speed-up gate that decides whether a Mali
  path ships.

### Modified Capabilities
<!-- none: no main spec covers kernel paths yet (gpu-kernels/* exists only in the unarchived
     cifar-gpu-kernels change). -->

## Impact

- **Code:**
  - `platform/engine/vulkan/shaders/comp/new_cifar_sparse_conv2d_k3s1p1.comp`;
  - `platform/engine/vulkan/shaders/comp/new_cifar_dense_conv2d_k3s1p1.comp`;
  - possibly the two apps' conv dispatch geometry (`apps/cifar-{sparse,dense}/vulkan/dispatchers.cpp`)
    if a Mali path covers more than one output per invocation;
  - `apps/cifar-{sparse,dense}/vulkan/test_main.cpp` (the existing `SparseConvK3S1P1` and
    `DenseConvK3S1P1` tests already cover subgroup-dependent paths at 6x6 and 8x8).
- **Tools:** `malioc` from Arm Performance Studio on the build box (a manual install). Without it the
  work proceeds on device measurements only.
- **Gates:**
  - local `ctest -L omp`;
  - rocky-ryzen `ctest -L vulkan` (correctness only, its path is unchanged);
  - the Pixel 7a and the Samsung running the four VK suites;
  - per-stage A/B on both phones.
  - Profiling runs only on the phones, never on rocky.
- **Out of scope:**
  - FP16/INT8;
  - cooperative matrix;
  - conv+pool fusion (the 11 stages are the scheduler's units);
  - the FC layers;
  - Adreno or other non-Mali GPUs with subgroup < 32 (none in the fleet).
