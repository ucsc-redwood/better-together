# Proposal

## Why

The cifar apps' Vulkan kernels leave a lot of performance unused, and that distorts which hybrid
schedules the paper finds, since the scheduler only offloads balanced work.

Measured per-stage GPU time on rocky-ryzen (Radeon 780M, RADV), 2026-10-15
(`bm-cifar-{dense,sparse}-vk`, the `real_time` column):

| | conv1 | conv2 | conv3 | conv4 | conv5 | fc1 | fc2 | app total |
|---|---|---|---|---|---|---|---|---|
| cifar-dense (N=16) | 0.116 | 0.883 | 1.989 | 2.426 | 1.576 | 0.869 | 0.876 | ≈9 ms |
| cifar-sparse (N=128) | 1.73 | 17.54 | 27.43 | 38.20 | 23.72 | 6.41 | 6.39 | ≈143 ms |

That's three separate problems, one per topic:

- **Sparse conv** (`new_cifar_sparse_conv2d_k3s1p1.comp`): conv4 achieves ≈0.095 TFLOP/s of useful
  work, about 1% of the 780M's FP32 peak. The dense kernel on zero-filled weights would be ≈2×
  faster than the CSR kernel.
- **Dense conv** (`new_cifar_dense_conv2d_k3s1p1.comp`): 0.68–1.03 TFLOP/s, about 10–20% of FP32
  peak (≈4 TFLOP/s without dual-issue, ≈8 with; to verify at measured clocks). It's a direct
  convolution with no LDS tiling, a 4-channel × 1-pixel register tile, and weight indices that
  aren't workgroup-uniform.
- **FC** (`new_cifar_dense_linear_bt.comp`, used by both apps): already DRAM-bound (fc1/fc2 stream
  64 MB of FP32 weights each in 0.87 ms ≈ 77 GB/s at N=16). cifar-sparse at N=128 re-streams the
  weights once per 16-image batch tile (`grid.y = ceil(N/16) = 8`), so 512 MB per layer in
  ≈6.4 ms.

## What Changes

### Sparse conv rewrite
- Rewrite the 3×3/s1/p1 CSR conv so each workgroup owns one output channel
  (`out_c = gl_WorkGroupID.y`). The CSR row bounds and metadata become workgroup-uniform: scalar
  loads and uniform loop bounds.
- Decode each nonzero's `(in_c, ky, kx)` into a packed input offset plus tap id on the host at
  weight-load time (synthetic `build_from_dense()` and real `load_real_weights()`), removing the
  in-shader decode.
- Stage the CSR row in LDS in bounded chunks.
- Keep unstructured CSR. Structured sparsity is recorded as a research decision (design.md).

### Dense conv optimization
Incremental, each step gated:
1. workgroup-uniform weights (`oct = gl_WorkGroupID.y`);
2. LDS halo tiling of the input;
3. an 8 oc × 2–4 px register tile;
4. opt-in FP16 packed math;
5. an optional RDNA3 `VK_KHR_cooperative_matrix` implicit-GEMM path;
6. a Mali-tuned variant.

**Non-goal:** conv+pool fusion, because the 11 stages are the scheduler's units.

### FC weight bandwidth
- Restructure `linear_bt` so weights are streamed **once per batch** regardless of N. Today every
  16-image tile re-reads them.
- Opt-in weight-only quantization: FP16 (2× fewer bytes) or INT8 with per-output-row scales (4×),
  dequantized to FP32 in-shader. Activations stay FP32, so this needs only 16/8-bit storage access,
  not `VK_KHR_shader_integer_dot_product` (that pays off only when compute-bound, and these layers
  are bandwidth-bound). Bounded by an end-task accuracy check on the real trained weights.
- Replace the lane-0 serial shared-memory reduction with `subgroupClusteredAdd` (cluster 16). Since
  the Vulkan campaign's Phase B, subgroup size is pinned and ≥ 16 on every fleet GPU. This drops
  `sh_red` and its barriers.
- Cleanup candidate: the generic `new_cifar_{dense,sparse}_linear.comp` are only taken when
  `in_features % 4 != 0`, which no layer hits.

### Shared engine work
- Optional (enabled-if-present, never required) device features:
  - `shaderFloat16` and `storageBuffer16BitAccess`;
  - `cooperativeMatrix`.
- An opt-in precision mode for the reduced-precision paths. FP32 stays the default everywhere.

## Capabilities

### New Capabilities
- `gpu-kernels/cifar-sparse-conv`: Correctness and performance contract of cifar-sparse's pruned
  3×3 convolution kernels.
- `gpu-kernels/cifar-dense-conv`: Correctness, precision and performance contract of cifar-dense's
  3×3 convolution kernels.
- `gpu-kernels/fully-connected`: Correctness, weight precision, accuracy and weight-traffic contract
  of the cifar apps' fully-connected kernels.

### Modified Capabilities
<!-- none -->

## Impact

- **Shaders:** `new_cifar_sparse_conv2d_k3s1p1.comp`, `new_cifar_dense_conv2d_k3s1p1.comp` and
  `new_cifar_dense_linear_bt.comp`, plus new variants (tiled, FP16, cooperative-matrix, Mali, INT8
  FC), all under `platform/engine/vulkan/shaders/comp/`, with their baked `*_spv.h`.
- **Dispatchers and appdata:**
  - `apps/cifar-{dense,sparse}/vulkan/dispatchers.{hpp,cpp}` (grid shapes, variant selection);
  - a VK-side packed-offset array for the sparse CSR (`apps/cifar-sparse/appdata.hpp`);
  - reduced-precision weight copies made at load time.
- **Engine:** `platform/engine/vulkan/base_engine.cpp` (optional features, precision mode).
- **Tests:**
  - The FP32 paths keep the existing oracle tolerance.
  - Reduced-precision paths get documented tolerances.
  - New VK end-task accuracy tests on the real weights, mirroring the OMP
    `RealWeights_EndTaskAccuracy` tests.
- **Profiling:** cifar × VK stage times change, so profiling stores and schedules must be
  regenerated before reporting new fleet results.
