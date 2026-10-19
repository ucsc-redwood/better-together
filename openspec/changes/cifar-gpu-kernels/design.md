# Design

Sections are per topic: **S** = sparse conv, **C** = dense conv, **F** = FC. Decision ids carry the
topic letter. Motivation and measurements are in proposal.md (Why).

## Context

### Sparse conv

- **`new_cifar_sparse_conv2d_k3s1p1.comp`:**
  - One invocation per output element `(b, out_c, oh, ow)`, flat over `gl_GlobalInvocationID.x`,
    grid `ceil(total/256)` (`apps/cifar-sparse/vulkan/dispatchers.cpp`, e.g. `record_stage_1`).
  - Each invocation walks CSR row `out_c` (`weight_row_ptr[out_c] .. [out_c+1]`). Per nonzero it
    does three global loads (`weight_col_idx`, `weight_vals`, `input_data`), decodes `/ 9` and `/ 3`
    (strength-reduced, literal), checks bounds, and accumulates into two interleaved accumulators.
  - `out_c` comes from the flat id, so the compiler can't prove the CSR metadata loads uniform:
    they're vector loads of identical addresses across the subgroup (to verify in ACO output with
    `RADV_DEBUG=shaders`).
- **`CSRMatrix`** (`apps/cifar-sparse/appdata.hpp`) is shared by OMP, CUDA and VK. It has
  `rows = K`, `cols = C·9`, `values`, `row_ptr`, and a flat-kernel-index `col_idx`.
  - Shapes: conv1 64×27, conv2 192×576, conv3 384×1728, conv4 256×3456, conv5 256×2304, about 25%
    density.
  - It's built by `build_from_dense()` (synthetic) or `load_real_weights()` (real export).
- **Batch:** N = 128 (`AppData::BATCH_SIZE`). Feature maps are 32², 16², 8², 8² and 8².

### Dense conv

- **`new_cifar_dense_conv2d_k3s1p1.comp`:**
  - `local_size_x = 256`. One invocation per `(n, oct, oh, ow)`, flat over `gl_GlobalInvocationID.x`
    (`ow` fastest, then `oh`, `oct`, `n`).
  - `OC_TILE = 4`, with fully unrolled 3×3 taps and hoisted edge masks.
  - Per input channel: 9 input loads and 36 weight loads from global memory.
  - The dispatcher (`apps/cifar-dense/vulkan/dispatchers.cpp`) uses it for every conv when
    `K % 4 == 0` and the shape is 3×3/s1/p1, with grid `ceil(total/256)`. The generic
    `new_cifar_dense_conv2d.comp` is the fallback.
- **Shapes (N=16):** conv1 3→64 at 32², conv2 64→192 at 16², conv3 192→384 at 8², conv4 384→256 at
  8², conv5 256→256 at 8². Weights are `[K, C, 3, 3]` FP32.
- **Device support:**
  - The 780M (RADV 25.2.7, from rocky's vulkaninfo profile) reports `shaderFloat16`,
    `storageBuffer16BitAccess`, `VK_KHR_cooperative_matrix` (`cooperativeMatrix: true`),
    `VK_KHR_shader_integer_dot_product`, and subgroup ops including CLUSTERED.
  - Pixel 7a (Mali-G710) and Samsung (Xclipse 940): to verify on device (`cmd gpu vkjson` or a
    small probe in `test-kiss-vk`).
- **Engine:** the required-feature table enables only required features (Vulkan campaign Phase A).
  Optional features follow the push-descriptor pattern: enabled if present, never required.
### FC

- **`new_cifar_dense_linear_bt.comp`** (used for fc1–fc3 by both apps; the generic
  `new_cifar_{dense,sparse}_linear.comp` are taken only when `in_features % 4 != 0`, which no layer
  hits):
  - `local_size_x = 256`, with `ROW_W = 16` lanes per output feature and `ROWS = 16` features per
    workgroup.
  - Inputs are staged in shared memory in `CHUNK4 = 32` vec4 chunks for `MAX_N = 16` images (8 KB);
    each lane holds 16 FP32 accumulators.
  - Grid `(ceil(out/16), ceil(N/16), 1)`, so at N=128 (cifar-sparse) there are 8 batch tiles, and
    each re-reads the full FP32 weight matrix.
  - The final reduction writes `vec4` partials to `sh_red` (4 KB). Lane 0 of each row then sums 16
    entries serially, 4 passes, with a barrier on each side of every pass.
- **Weights:** `[out, in]` FP32. fc1 and fc2 are 4096×4096 (64 MB each); fc3 is 10×4096. The
  `RealWeights_EndTaskAccuracy` OMP tests assert ≥ 0.75 (dense) and ≥ 0.85 (sparse). FP32 accuracy
  is 90.48% / 90.58%.
- **780M support:** 16-bit storage, 8-bit storage (already a required engine feature) and CLUSTERED
  subgroup ops (from rocky's vulkaninfo profile). Phones: to verify.

## Goals / Non-Goals

### Sparse conv

**Goals (S):**
- Make conv2–conv5 faster beyond noise on the Samsung (where this path runs) without regressing either phone.
- Keep the change inside the VK kernel, its dispatcher, and an additive CSR-side array.

**Non-Goals (S):**
- Changing the OMP or CUDA sparse kernels.
- Changing the pruning, the density, or the 11-stage model.
- Fusing conv with pool.
### Dense conv

**Goals (C):**
- Make the FP32 conv2–conv5 sum faster beyond noise on the Samsung.
- Offer FP16 and cooperative-matrix modes as opt-in, accuracy-bounded paths.
- Don't regress the phones.

**Non-Goals (C):**
- Conv+pool fusion, since stages are scheduling units.
- Changing conv1. It's 0.116 ms with C=3, so it's left on the current kernel unless a step helps it
  for free.
- OMP and CUDA kernels.
### FC

**Goals (F):**
- Weights streamed once per batch for any N.
- Opt-in FP16 and INT8 weight-only paths that meet the spec's time ratios and the accuracy bound.
- A barrier-free subgroup reduction.

**Non-Goals (F):**
- Activation quantization or integer-dot-product math. The layers are bandwidth-bound, so
  weight-only quantization captures the byte savings.
- fc3 (10 outputs, ≈0.1 ms).

## Decisions

### Sparse conv (S)

#### S1. One output channel per workgroup row: `out_c = gl_WorkGroupID.y`
- **Grid:** `(ceil(N·P·Q / 256), K, 1)`. Each workgroup computes 256 output pixels (spanning
  images) for one `out_c`.
- **Why:** `row_ptr[out_c]`, `row_ptr[out_c+1]` and every `(value, offset)` of the row become
  dynamically uniform. RADV/ACO can then use scalar loads, and the nonzero loop has uniform bounds:
  no divergence from varying row lengths, and the loop-carried dependency is the only cost.
- **Alternative:** keep the flat mapping and rely on L1 broadcast. Rejected: it's what we measure
  today.

#### S2. Host-side tap decode into an offset + tap id
- **What's stored:** per nonzero, a packed `int`: the input-plane offset
  `in_c·H·W + (ky−1)·W + (kx−1)` in the high bits, and the tap id `ky·3+kx` (4 bits) in the low
  bits. It's computed per layer with that layer's H and W, which are fixed per CSR matrix.
- **In the shader:** each invocation builds a 9-bit "tap valid at this (oh, ow)" mask once. Each
  nonzero then costs one value load, one offset load, a mask test and one input load. There's no
  division and no 4-way bounds compare.
- **Where it lives:** a VK-only `std::pmr::vector<int>` beside the CSR, filled after
  `build_from_dense()` / `load_real_weights()`.
  - *Alternative:* add it to `CSRMatrix` for all backends. Rejected: it costs memory on OMP and
    CUDA, which don't read it.
  - *As built (task 1.3):* `AppData::conv{1..5}_taps`, filled by `pack_conv_taps()` and read only by
    the Vulkan kernel. `AppData` is shared by every backend and does not know which one it serves,
    so OMP and CUDA allocate the arrays too. Each is sized like its CSR's `col_idx` (rows·cols),
    because a buffer is placed when it allocates, before nnz is known: 9.0 MB per `AppData`, about
    3 % of its ≈263 MB.
- **Risk:** offset width. conv4's largest offset is `383·64 + …`, far below 2^27, so 27 bits of
  offset plus 4 bits of tap fit a signed 32-bit int (to verify per layer in a unit test).

#### S3. Stage the CSR row in LDS
- **How:** the workgroup cooperatively copies `(value, packed offset)` for its row into shared
  memory in chunks of ≤ 1024 nonzeros (8 KB), loops, and re-stages.
- **Why chunk:** it bounds shared memory under the 16 KB Vulkan minimum, whatever the row length.
- **Mali:** not here. The Mali sparse conv path is owned by `mali-conv-kernels` (main spec
  `gpu-kernels/mali-conv`); S3 changes only the subgroup-32-and-wider path.

#### S4. Sparsity format stays unstructured CSR in this change (decision recorded)
**Alternatives considered:**
- **Structured 2:4 (N:M) sparsity:** maps to vector loads and is easy to speed up. But it changes
  the pruning (retraining), the accuracy (dense 90.48% / sparse 90.58% today), and the paper's
  premise that cifar-sparse is the *irregular-memory* workload.
- **Channel-blocked sparsity:** prune whole `(out_c, in_c)` 3×3 blocks. Regular access, but the
  same retraining and premise issues apply.
- **Dense kernel on zero-filled weights:** about 2× faster today by estimate, but it erases the
  sparse workload.

**Decision:** keep unstructured CSR, since the irregular-memory character is part of the research
claim. Revisit with the paper authors if S1–S3 don't reach the spec's 2× bound.
### Dense conv (C)

#### C1. Uniform weights: `oct = gl_WorkGroupID.y`
- **Grid:** `(ceil(N·H·W / 256), K / OC_TILE, 1)`.
- **Why:** all weight addresses of a workgroup depend only on workgroup id plus the loop counter, so
  ACO can emit scalar (SMEM) loads and the 36 weight loads per channel leave the vector memory
  pipe. To verify with `RADV_DEBUG=shaders` before and after.
- **Caveat:** at 8×8 maps, `N·H·W = 1024`, so 4 workgroups per channel tile. That's still
  `K/OC_TILE × 4 ≥ 256` workgroups at conv3–conv5.

#### C2. LDS halo tiling
- **How:** each workgroup stages an input tile with a 1-pixel halo, for a block of `C_TILE` input
  channels, into shared memory. It's reused by all 3×3 taps and by all output channels the
  workgroup computes.
- **Size:** chosen so the tile fits ≤ 16 KB (the Vulkan minimum `maxComputeSharedMemorySize`), with
  a larger tile where the device reports more.
- **Alternative:** rely on L1 for the 9× tap reuse. That's today's behavior, and the measurements
  say it isn't enough.

#### C3. Register tile 8 oc × 2–4 px
- **Why:** each thread accumulates 8 output channels for 2–4 adjacent pixels, which raises FMA per
  load from 4:1 toward 16–32:1.
- **Shape:** via spec constants (`OC_TILE`, `PX_TILE`), so per-device tuning doesn't fork the source.

#### C4. FP16 packed math, opt-in
- **How:** FP16 storage and arithmetic (`f16vec2`), giving RDNA3 `v_pk_fma_f16`/`v_dot2` and the
  Mali FP16 rate. Accumulation in FP32 is the default, with FP16 accumulation as a separate measured
  sub-option.
- **Selection:** an engine-level precision mode (e.g. `BT_VK_PRECISION=fp16`), FP32 by default.
- **Spec:** separate tolerance plus end-task accuracy within 0.5 points on the real weights (dense
  test accuracy is 90.48% in FP32).

#### C5. Cooperative-matrix implicit GEMM (RDNA3), optional
- **How:** the conv is an implicit GEMM `[K] × [C·9] · [C·9] × [N·P·Q]`, using
  `VK_KHR_cooperative_matrix` with FP16 inputs and FP32 accumulate in the 16×16×16 shapes RADV
  reports (to verify via `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`).
- **When:** only when the device lists a usable shape. Otherwise C1–C4.
- **Alternative:** hand-written WMMA via vendor intrinsics. Rejected: not portable Vulkan.

#### C6. Mali variant (removed 2026-10-18)
Owned by `mali-conv-kernels`, which shipped a Mali conv path selected by `SUBGROUP_SIZE` < 32 (main spec
`gpu-kernels/mali-conv`). C1-C5 change only the subgroup-32-and-wider path, and must keep the Mali path's
selection and outputs unchanged.
### FC (F)

#### F1. Stream weights once per batch
- **Today:** each `grid.y` batch tile re-reads the weights.
- **Change:** the workgroup loops over all batch tiles itself. For each weight chunk it holds its
  rows' chunk in registers (or LDS), then re-stages each 16-image input tile and accumulates.
  Accumulators for N up to 128 don't fit in registers (16 per tile per lane), so outputs are
  accumulated per tile across the K loop. Two layouts are candidates, chosen by measurement:
  - **(a) K-outer, tiles-inner:** keep a weight chunk resident, iterate the 8 input tiles, keep
    per-tile partials in LDS (16 rows × 128 images × 4 B = 8 KB).
  - **(b) Batch-split:** lanes partition images instead of K. Each lane owns images `j, j+16, …`
    and reads broadcast weights from an LDS-staged chunk.
- **Traffic:** in both layouts the weight DRAM traffic is 1× per batch. Input re-reads
  (128×4096×4 B = 2 MB per layer) should come from L2 (to verify against the 780M's L2 size with a
  counter or timing).
- **Spec bound:** sparse fc1 ≤ 2× dense fc1.

#### F2. Weight-only quantization, opt-in
- **FP16:** weights stored as `float16_t`/`f16vec4` (16-bit storage), converted to FP32 in
  registers. 2× fewer bytes.
- **INT8:** per-output-row symmetric scale (`w ≈ scale[row] · q`), stored as `int8_t` with packed
  loads, dequantized in registers. 4× fewer bytes, plus 4 KB of scales per layer.
- **Quantization:** done once at weight-load time on the host, for synthetic and real weights.
- **Selection:** a precision mode, e.g. `BT_VK_FC_WEIGHTS=fp16|int8`, FP32 by default. Unsupported
  devices fall back to FP32 with a warning.
- **Alternative:** INT8 activations with `VK_KHR_shader_integer_dot_product`. Rejected for now: it
  needs per-batch activation quantization and gains compute that isn't the bottleneck.

#### F3. Clustered subgroup reduction
- **Change:** replace the `sh_red` and lane-0 serial sum with `subgroupClusteredAdd(acc, 16)` per
  image (`GL_KHR_shader_subgroup_clustered`). Valid because the subgroup size is pinned and ≥ 16 on
  every fleet GPU since the Vulkan campaign's Phase B.
- **Effect:** removes 4 KB of shared memory and 8 barriers per workgroup.
- **Fallback:** keep the shared-memory reduction behind a spec constant for a device that lacks
  CLUSTERED support (phones: to verify).

#### F4. Generic linear fallbacks
`new_cifar_dense_linear.comp` and `new_cifar_sparse_linear.comp` are dead for every current shape.
- **Decision:** keep them as the fallback contract in this change, and list removal as an optional
  cleanup task gated on a test that forces `in_features % 4 != 0`.

## Risks / Trade-offs

### Sparse conv

- **[Risk]** Uniform `out_c` per workgroup means 8×8 feature maps × N=128 give 8,192 outputs per
  channel = 32 workgroups per channel. That's plenty of parallelism at K ≥ 192, but conv1
  (K=64, 32×32) changes shape the most.
  → **Mitigation:** measure conv1 too, and keep the old kernel selectable by a dispatcher flag until
  the new one wins everywhere.
- **[Risk]** Phones are register- and occupancy-sensitive (Mali-G710).
  → **Mitigation:** the spec's no-regression scenario on both phones; a per-device spec constant for
  `STAGE_ROW`.
- **[Risk]** The scalar-load assumption is unverified.
  → **Mitigation:** task 1.2 inspects the ACO disassembly before investing in S3.
### Dense conv

- **[Risk]** Each step must be measured separately, or gains get misattributed.
  → **Mitigation:** a gate plus timing after each step (C1–C5), in the proposal's order (task group 2).
- **[Risk]** The FP16 tolerance hides real accuracy loss.
  → **Mitigation:** the end-task accuracy requirement on real weights, not just per-stage tolerance.
- **[Risk]** The cooperative-matrix path adds a third precision/format and doubles test time.
  → **Mitigation:** it's opt-in, tested only where supported, and could be dropped if C1–C4 already
  meet the spec.
- **[Trade-off]** More variants mean more shaders to bake and keep in sync.
  → **Mitigation:** spec constants first, separate sources only for genuinely different algorithms
  (cooperative matrix).
### FC

- **[Risk]** Layout F1(a)'s LDS partials or F1(b)'s broadcast weights hurt occupancy, especially on
  Mali.
  → **Mitigation:** measure both layouts on all three devices; the spec bound decides.
- **[Risk]** INT8 per-row quantization loses more than 0.5 points on the real weights.
  → **Mitigation:** the end-task accuracy test gates it. If it fails, ship FP16 only and record the
  INT8 accuracy.
- **[Risk]** Input re-reads in F1 spill out of L2 on smaller-cache GPUs.
  → **Mitigation:** time sparse fc1 on both phones.

## Migration Plan

- **Order:** sparse conv (task group 1), then dense conv (group 2), then FC (group 3), each as one
  or more PRs into `dev`.
- **Gate:** every PR passes local `ctest -L omp`; `ctest -L vulkan` plus the app's `test-*-vk` on
  rocky-ryzen; and the app's `test-*-vk` on the Pixel 7a and the Samsung. It also records
  before/after per-stage timing, on the phones only.
- **Defaults:** reduced-precision modes (C4, C5, F2) are opt-in, so default (FP32) behavior changes
  only through the FP32 kernels' speed.
- **Rollback:** revert the PR. Old kernels stay registered and selectable by a dispatcher flag until
  each new kernel has passed the fleet gate.
- **Profiling data:** cifar × VK profiling stores and fitted schedules are stale after each group
  merges, and must be regenerated before reporting fleet results.

## Open Questions

- Does the paper team want structured sparsity (S4) as a later experiment?
- Are FP16, INT8 or cooperative-matrix results acceptable in the paper's headline numbers, or only as
  extra data points?

Neither affects the tasks: the reduced-precision paths are opt-in.
