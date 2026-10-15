# Vulkan compute-shader profile and optimization plan (Radeon 780M / RADV)

Date: 2026-10-15. Branch: `campaign2/vk-shader-cleanup` (analysis only; no shader or engine
code in this repo was changed by this report). Base commit `1694bee`.

**Device / driver:** rocky-ryzen, AMD Radeon 780M (RADV PHOENIX, RDNA3, `--device minipc`),
Mesa 25.2.7, Vulkan 1.4.318, subgroup size 64, kernel 6.12 (Rocky 10). Governor
`powersave`, DPM `auto`, clocks not touched.

**Contents:** headline, how it was measured, per-shader tables, prototypes, ranked plan,
what was measured vs inferred, caveats, commands.

---

## 1. Headline

All numbers below are GPU-timestamp medians from `bm-prof-*-vk`, isolated scenario. They are
standalone kernel times, not pipelined wall time.

* The three apps are dominated by a few kernels, and the biggest wins need no new
  algorithm, only code the compiler can schedule:
  * `new_cifar_dense_conv2d_k3s1p1` and `new_cifar_sparse_conv2d_k3s1p1` are **latency-bound
    by divergent guarded loads and per-lane loads of wave-uniform data**. The RADV ISA shows
    it (section 3). Removing that was measured in the real apps: **dense pipeline sum
    8.64 -> 4.18 ms (-52%)**, **sparse 121.9 -> 58.8 ms (-52%)**. The edited kernels pass
    the repo's own differential suites (`test-cifar-dense-vk` 12/12, `test-cifar-sparse-vk` 12/12).
  * The tree radix sort (45% of the tree pipeline) re-reads a 256x256-entry histogram table in
    every scatter workgroup. Changing one constant (`kRadixNumWorkgroups` 256 -> 64) was
    measured **0.773 -> 0.404 ms** on stage 2 (tree sum 1.705 -> 1.338 ms, -22%);
    `test-tree-vk` 14/14.
  * Dense FC (stages 9, 10) is **at the DRAM roof already** (79.5 GB/s vs the 86.7 GB/s
    `global_read` roof); only fewer bytes can help there.
  * Maxpool kernels are at or near the memory roof and tiny (0.3-1.6% of dense, <0.6% of sparse).
* The prototypes live only in scratch trees (`/tmp/vkproto` on this host, `~/bt-src-vkprof`
  and `~/vkprof-run` on rocky-ryzen). Nothing was pushed or committed except this report.

## 2. Method and measurement hygiene

Follows the shape of `openspec/changes/vk-engine-runtime/tasks.md` task 4.2 / design D1
(note: the `openspec/` tree is not in this worktree; I read it from the main checkout
`/home/doremy/Desktop/ucsc/better-together/openspec/changes/vk-engine-runtime/`).
The GPU DVFS slowdown after idle dominates short kernels (D1: tree 1-3 1.09 ms hot ->
2.98 ms after a 20 ms gap). The numbers here are therefore **hot-clock** kernel times; in a
pipelined run with idle gaps every kernel is slower and short kernels are slowed most.

Per-run gate (script `~/vkprof-run/run.sh` on rocky-ryzen): 5 s settle, then 5 consecutive
`gpu_busy_percent` readings that must all be 0 (abort otherwise), load average logged, then
the binary. Load averages at run start: 0.29-1.38 (the tree run r3 started at 1.38, above
the 0.05-0.49 band of task 4.2; its stage times agree with r1/r2 to <1%). I did not
re-check `/dev/dri` holders before every run (one check at the start: none; `fuser`
cannot see other users' fds). `bm-prof` itself warms the GPU back-to-back for 150 ms before
sampling each cell (`BT_PROF_GPU_WARM_MS`), which is why stage numbers are hot-clock.
The runs were interleaved (base, variant, base, variant ...), 3 rounds each, median of the
per-run `p50`.

The 780M was used by someone else at the start (`gpu status` showed `locked`, 57-97% busy,
a `llama_main` process). I did not use `--force`; I waited until `gpu status` said idle.
During the later gpu-lab prototype runs it went briefly busy again (exit 75 from `./gpu`);
those runs were retried with `--busy-threshold 1` and a retry loop, and one reading taken at
busy=7.4% was discarded. After each bm-prof run `gpu_busy_percent` read 55-69% for a moment
(a stale counter right after exit; the next run's pre-check always read 0).

Not done on purpose: no RGP/SQTT capture (forbidden on an AMD iGPU per the skill's safety
rules), no clock changes, no `rga`. Not measured: the GPU shader clock *during* a run. The
`gpu_clock_mhz` field in the JSONL provenance is read before the warm-up (800 MHz on the
cifar cells, 1046-1067 MHz on tree cells) and is not the running clock; a sysfs sampler I
tried did not overlap the run and read 800 MHz throughout, so I make no claim about the
running clock.

Tools: `bm-prof-{cifar-dense,cifar-sparse,tree}-vk` (built from this worktree's sources on
rocky-ryzen with `scripts/build-bench-on-rocky.sh vulkan`, `BT_BENCH_SRC=bt-src-vkprof`),
`RADV_DEBUG=shaders` and `RADV_DEBUG=shaderstats,nocache` for ISA and compile statistics,
and `~/Desktop/gpu-lab/gpu run 780m` (GPU timestamps) for isolated-shader prototypes.
The `bm-phases-*` tools do not exist in this branch (they live on
`campaign2/vk-engine-runtime`); `bm-prof` already gives per-stage GPU-timestamp time.

Cross-check of the two timers on the same shader (dense conv stage 6, repo shader, no
changes): `bm-prof` 2.288 ms vs `gpu-lab` 2.167 ms median over 300 iterations (5% apart;
the two use different data and `gpu-lab` runs one dispatch at a time).

Roofs (780M, from `~/Desktop/igpu-roofline/results/fleet-fast-20260926/rocky-ryzen/780m/report/REPORT.md`,
Mesa 25.2.7, confirmed medians): `alu_fp32` 5.756 TFLOP/s, `global_read` 86.67 GB/s,
`global_write` 77.4 GB/s, `global_copy` 71.5 GB/s, `cache_read_effective` 3.34 TB/s,
`shared_fp32_read` 2.28 TB/s. Ridge point for FP32 FMA vs DRAM read = 66 FLOP/byte. The
roof run is 3 weeks old; the clock/thermal conditions of this session were not re-verified.

Workloads (real ones): cifar-dense N=16, cifar-sparse N=128 with 25%-density CSR convs and
real weights from `/tmp/bt/weights`, tree n=307,200 points (`kDefaultInputSize` 640x480,
synthetic random cloud). Pipeline "share" below is the share of the sum of isolated stage
times, not of pipelined wall time.

## 3. Per-shader results

Occupancy/ISA data are RADV `shaderstats` (the "Subgroups per SIMD" figure is RADV's
maximum waves per SIMD for that shader). No shader spills (`Spilled VGPRs 0`, `Scratch 0`
everywhere). Pipeline order = `make_algo` order.

### 3.1 cifar-dense (sum of stage medians 8.72 ms, 3 runs; stage numbers are medians of 3)

| stage | kernel | ms (runs) | share | FLOP / bytes | rate vs roof |
|---|---|---|---|---|---|
| 1 conv1 3->64, 32x32 | `new_cifar_dense_conv2d_k3s1p1` | 0.1834 (0.171-0.184) | 2.1% | 57 MF / 4.4 MB | 0.31 TF (5%) |
| 2 pool1 | `new_cifar_dense_maxpool` | 0.0929 | 1.1% | 5.2 MB | 56 GB/s |
| 3 conv2 64->192, 16x16 | conv k3s1p1 | 0.8458 (0.809-1.217, DVFS outlier in run 1) | 9.7% | 0.91 GF / 4.6 MB | 1.07 TF (19%) |
| 4 pool2 | maxpool | 0.0704 | 0.8% | | |
| 5 conv3 192->384, 8x8 | conv k3s1p1 | 1.8559 (1.829-1.869) | 21.3% | 1.36 GF / 5.0 MB | 0.73 TF (12.7%) |
| 6 conv4 384->256 | conv k3s1p1 | 2.2876 (2.280-2.288) | 26.2% | 1.81 GF / 5.3 MB | 0.79 TF (13.7%) |
| 7 conv5 256->256 | conv k3s1p1 | 1.5252 (1.495-1.529) | 17.5% | 1.21 GF / 3.6 MB | 0.79 TF (13.7%) |
| 8 pool3 | maxpool | 0.0256 | 0.3% | | |
| 9 FC1 4096x4096 | `new_cifar_dense_linear_bt` | 0.8456 (0.843-0.848) | 9.7% | 0.54 GF / 67.1 MB | 79.4 GB/s = 91.6% of `global_read` |
| 10 FC2 | linear_bt | 0.8500 | 9.7% | same | 79.0 GB/s |
| 11 FC3 4096x10 | linear_bt | 0.1392 | 1.6% | 164 KB weights | latency-bound (1 workgroup) |

Conv arithmetic intensity is 250-270 FLOP/B, well right of the 66 FLOP/B ridge, so these
are compute-side kernels sitting at 13-19% of the FP32 roof. FLOP counts are 2*MAC.

ISA / stats (dense, pipeline order 0..4 = conv2d, conv2d_k3s1p1, maxpool, linear, linear_bt):

| shader | VGPR | LDS | waves/SIMD | instr | VALU / VMEM / SMEM | note |
|---|---|---|---|---|---|---|
| conv2d (generic, unused by the model) | 32 | 0 | 32 | 306 | 129 / 4 / 3 | fallback only |
| conv2d_k3s1p1 | 40 | 0 | 24 | 434 | 204 / 26 / 7 | see below |
| maxpool | 32 | 0 | 32 | 272 | 122 / 2 / 2 | |
| linear (generic) | 16 | 0 | 32 | 92 | 42 / 4 / 3 | not used for the FC head |
| linear_bt | 48 | 12288 | 20 | 847 | 434 / 22 / 6 | |

**What the dense conv ISA shows** (`RADV_DEBUG=shaders`, dump kept at
`~/vkprof-run/dumps/conv_k3.txt` on rocky-ryzen, loop lines ~1100-1500):

1. Each of the 9 taps is a separate guarded load: `s_and_saveexec_b64 / s_cbranch_execz`,
   then a re-load of the 128-bit buffer descriptor (`s_load_b128 ...; s_waitcnt lgkmcnt(0)`),
   `buffer_load_b32`, and an else-path `s_waitcnt vmcnt(0); v_mov_b32 vN, 0`. The `vmcnt(0)`
   in the else path drains all in-flight loads before each tap. On an 8x8 map a 64-lane wave
   is a whole image, so *every* wave has edge lanes and executes both paths for every tap.
   The taps therefore issue one at a time.
2. The 36 weights per input channel are 12 per-lane `buffer_load_b128/b32` (VGPR-addressed),
   even though the weight row index is the same for every lane of the wave (the output
   channel group is constant across a wave whenever `H*W % 64 == 0`). 21 VMEM
   instructions per channel step for 36 FMAs.
3. `VOPD 0`: no dual-issue.

**Tests of the hypothesis, measured** (section 4): branchless taps alone -36%, plus
wave-uniform weight loads (scalar `s_load`) -59%, plus `OC_TILE` 8 -67% (conv stage 6,
`gpu-lab`).

### 3.2 cifar-sparse (sum 121.77 ms)

| stage | kernel | ms (3 runs) | share |
|---|---|---|---|
| 1 conv1 | `new_cifar_sparse_conv2d_k3s1p1` (CSR) | 1.5632 | 1.3% |
| 2 pool1 | `new_cifar_sparse_maxpool` | 0.5767 | 0.5% |
| 3 conv2 | CSR conv | 16.978 | 13.9% |
| 4 pool2 | maxpool | 0.4425 | 0.4% |
| 5 conv3 | CSR conv | 27.284 | 22.4% |
| 6 conv4 | CSR conv | 38.080 | 31.3% |
| 7 conv5 | CSR conv | 23.719 | 19.5% |
| 8 pool3 | maxpool | 0.1815 | 0.1% |
| 9 FC1 | `new_cifar_dense_linear_bt`, grid (256, 8) | 6.3973 | 5.3% |
| 10 FC2 | linear_bt | 6.3961 | 5.3% |
| 11 FC3 | linear_bt | 0.1488 | 0.1% |

Run-to-run spread is below 0.5% on stages 3, 5, 6, 7, 9, 10.

Roofline position, from the real weights (density 25%): conv stage 5 executes
~1.36 G MACs = 2.72 GFLOP in 27.3 ms = 0.10 TFLOP/s (1.7% of the FP32 roof). The same
layer at full density in dense form (8 x dense conv3 for N=128) would be 10.9 GFLOP; the
dense conv at `OC_TILE=8` does 8 images of it in ~8 x 0.62 = 4.95 ms, so even after the CSR fix (12.85 ms)
the pruned conv is ~2.6x *slower* than the unpruned dense one. Sparsity is currently a
slowdown.

FC (N=128, 8 batch tiles of 16): the weight matrix is re-streamed 8 times,
8 x 67.1 MB = 537 MB in 6.40 ms = 84 GB/s, i.e. at the DRAM roof of 86.7 GB/s. (Inference
below in section 5.)

ISA / stats (pipeline order = conv2d, conv2d_k3s1p1, maxpool, linear, linear_bt):
CSR conv k3s1p1: VGPR 32, 0 LDS, 32 waves/SIMD, 376 instr, VMEM 12 / SMEM 11. The ISA
loop (`dumps/cifar-sparse.txt`, shader 2) is the same pattern as the dense one, worse: per
nonzero the `col_idx` load, then input and weight loads that depend on it, each followed by
`s_waitcnt vmcnt(0)` and a descriptor `s_load_b128` reload. Two dependent memory round-trips per
nonzero, one nonzero at a time.

### 3.3 tree (sum 1.7013 ms at n=307,200)

| stage | kernels | ms (3 runs) | share | notes |
|---|---|---|---|---|
| 1 morton | `tree_morton` | 0.0765 | 4.5% | 6.1 MB / 0.0765 ms = 80 GB/s = 92% of read roof |
| 2 radix sort | 4 x (`multi_radixsort_histograms` + `multi_radixsort_warp64`) = 8 dispatches, 256 workgroups each | 0.7738 | 45.5% | see below |
| 3 dedup | `tree_find_dups`, 3-pass scan (`tree_scan_local/_block_sums/_add`), `tree_move_dups` = 5 dispatches | 0.2337 | 13.7% | ~0.047 ms per dispatch |
| 4 radix tree | `tree_build_radix_tree` | 0.2034 | 12.0% | |
| 5 edge count | `tree_edge_count` | 0.0338 | 2.0% | |
| 6 scan | 3-pass scan | 0.1525 | 9.0% | |
| 7 octree | `tree_build_octree` | 0.2277 | 13.4% | |

ISA / stats (11 pipelines, order = `make_algo` order: morton, hist, scatter, find_dups,
scan_local, scan_block_sums, scan_add, move_dups, radix_tree, edge_count, octree; full
list in `~/vkprof-run/dumps/stats-tree.txt`): VGPR 8-40, no spills. Scatter
(`multi_radixsort_warp64`): VGPR 40, LDS 9728 B, 24 waves/SIMD. `scan_local`/`scan_block_sums`:
LDS 8192 B, VGPR 16/8. `tree_build_octree`: VGPR 32, 81 static VMEM instructions.

Radix sort diagnosis (**inferred from code, then confirmed by experiment**):
`multi_radixsort_warp64.comp` lines 40-48: each of the 256 scatter workgroups has 256 threads
(one per bin) that loop `j = 0 .. g_num_workgroups-1` reading `g_histograms[256*j + bin]`.
That is 256 x 256 x 256 = 16.8 M loads (67 MB of L2 traffic) per pass, 268 MB per sort,
to sort 1.2 MB of keys. `kRadixNumWorkgroups = 256` is a host constant in
`apps/tree/vulkan/vk_appdata.hpp:14`. Measured with 128/64/32 workgroups (section 4.4).

Kernels `octree_*`, `tmp_single_radixsort_*`, `tree_merge_sort`, `tree_naive_prefix_sum`
are compiled but not referenced by any dispatcher (`grep` of `apps/` and `platform/engine`
finds them only in `all_shaders.hpp`); they were not profiled.

## 4. Prototypes (measured)

All prototypes were built and run in scratch trees; the repo shaders are unchanged. Every
prototype was validated by the repo's own differential suites (OMP oracle) run on rocky-ryzen:
`test-cifar-dense-vk`, `test-cifar-sparse-vk`, `test-tree-vk` with `--device minipc`.
The `ctest -L vulkan` wrapper was not used; the binaries were run directly.

### 4.1 Dense conv k3s1p1 (isolated, `gpu-lab`, bit-exact output checksums)

Variants (sources in `/tmp/vkproto/conv_v1.comp`, `conv_v2.comp`, `conv_v3_oc8.comp`):
* v1: all 9 taps loaded unconditionally from a clamped index, then selected to 0.0 (no
  branches, no `vmcnt(0)` between taps).
* v2: v1 plus `w0 = subgroupBroadcastFirst(oc) * wrow` (`#extension GL_KHR_shader_subgroup_ballot`),
  so the weight index is provably wave-uniform and RADV emits scalar loads.
* v3: v2 with `OC_TILE` 8 (8 accumulators, same expression order, so results are bit-identical).

Medians of 3 interleaved rounds of 300 iterations, `gpu-lab` GPU timestamps, microseconds
(identical `fnv1a64` of the output buffer for all variants and all four shapes):

| layer | base | v1 | v2 | v3 (OC 8) |
|---|---|---|---|---|
| conv1 (N16 C3 K64 32x32) | 182.8 | - | 131.1 | **85.5** |
| conv2 (C64 K192 16x16) | 799.0 | - | 346.1 | **329.4** |
| conv3 (C192 K384 8x8) | 1783.8 | - | 669.5 | **613.7** |
| conv4 (C384 K256 8x8) | 2167 (2166.8/2163.6/2168.3) | 1378 | 889 | **721** |
| conv5 (C256 K256 8x8) | 1453.8 | - | 530.5 | **466.2** |

`OC_TILE=16` was slower than 8 on conv4 (947 us). Stats of the v3 shader: VGPR 32, 0 LDS,
32 waves/SIMD, SMEM 18 (weights now scalar loads), VMEM 19.

**In the real app** (`bm-prof-cifar-dense-vk`, 3 interleaved rounds, ms; the dispatcher
grid divisor `out_channels / 4` -> `/ 8` changed in the scratch copy at 5 sites in
`apps/cifar-dense/vulkan/dispatchers.cpp`):

| stage | base | OC8 variant | change |
|---|---|---|---|
| 1 | 0.1842 | 0.0855 | -54% |
| 3 | 0.8104 | 0.2688 | -67% |
| 5 | 1.8435 | 0.6193 | -66% |
| 6 | 2.2786 | 0.7306 | -68% |
| 7 | 1.4998 | 0.4442 | -70% |
| sum of 11 stages | 8.637 | 4.179 | **-51.6%** |

`test-cifar-dense-vk`: 12/12 pass. After the change the convs are at 2.2-3.4 TFLOP/s =
38-58% of the FP32 roof, and the FC head (1.7 ms) is 41% of the dense pipeline sum.
(That binary also contained the linear_bt tile swizzle of 4.3; with grid.y = 1 in the dense
app it is a no-op, FC stage times are unchanged within noise: 0.845/0.850 -> 0.853/0.848.)

### 4.2 Sparse conv k3s1p1 (real app, `bm-prof-cifar-sparse-vk`, ms)

Variant `/tmp/vkproto/sparse_k3_v1.comp`: row bounds from `subgroupBroadcastFirst(weight_row_ptr[...])`
(the output channel is constant per wave when `out_h*out_w % 64 == 0`), so `nz`, the CSR decode
and the `col_idx`/`weight_vals` loads are scalar; each tap does one input load on a clamped
index (no branch) and selects 0 for padding; 4 independent accumulators, unroll 4. No engine
change (same grid and bindings).

| stage | base | variant | change |
|---|---|---|---|
| 1 | 1.5640 | 0.9626 | -38% |
| 3 | 16.995 | 8.2045 | -52% |
| 5 | 27.293 | 12.850 | -53% |
| 6 | 38.178 | 18.239 | -52% |
| 7 | 23.723 | 11.561 | -51% |

3 interleaved rounds each; run spread < 1%. `test-cifar-sparse-vk` 12/12 pass. Variant stats:
VGPR 32, SMEM 22, VMEM 8. Output differs only in summation order (NearEqual oracle).

### 4.3 Linear bt (FC head) for N=128 (real app, sparse, ms)

Two edits to `new_cifar_dense_linear_bt.comp`, on top of 4.2:
* **swizzle**: remap `(gl_WorkGroupID.x, y)` through the linear workgroup id so the 8 batch
  tiles of one weight row-group are adjacent in launch order and share the weight fetch
  through L2. Shader-only, no engine change. Stage 9/10: 6.40 -> 5.16 / 5.12 (-19%).
* **2 output features per row-group** (`FPR = 2`, one LDS read of the input vec4 feeds two
  weight rows; 32 accumulators; grid.x divisor 16 -> 32 at 3 sites in
  `apps/cifar-sparse/vulkan/dispatchers.cpp`). Stage 9/10: -> **2.79 / 2.82** (-56% vs base).
  Stage 11 (FC3, N_out=10) regresses 0.149 -> 0.192 ms (one workgroup, more work in it).

`test-cifar-sparse-vk` 12/12 pass with both. Whole sparse pipeline sum (conv fix + swizzle +
FPR 2): **121.90 -> 58.81 ms (-51.8%)**.

Why the FC change works (this part is inferred, the effect is measured): the base kernel does
per weight vec4 per lane 16 `ds_read_b128` (one per image) for 64 FMAs. Total LDS reads =
33.5 M weight-vec4 x 16 x 16 B = 8.6 GB, which at the 2.28 TB/s `shared_fp32_read` roof is
3.7 ms of the 5.1 ms after the swizzle. Reading each input vec4 once for two weight rows halves
that traffic. I did not take LDS counters; this is arithmetic, not a measured stall.

### 4.4 Tree radix sort workgroup count (real app, `bm-prof-tree-vk`, ms)

`kRadixNumWorkgroups` in `apps/tree/vulkan/vk_appdata.hpp` (scratch copy, restored after):

| NWG | stage 2 (3 runs) | other tree stages |
|---|---|---|
| 256 (base) | 0.7733 (0.7748 / 0.7726 / 0.7733) | unchanged |
| 128 | 0.4214 | |
| 64 | **0.4042** | stage 3 0.2374, 4 0.2095, 7 0.2231 (noise) |
| 32 | 0.4078 | |

Tree sum 1.705 -> 1.338 ms at NWG=64 (-21.6%). `test-tree-vk.nwg64` and `.nwg32`: 14/14 pass.
The remaining 0.40 ms for 8 dispatches is ~0.05 ms per dispatch, which looks like launch/barrier
latency (inference: stage 5 `tree_edge_count` is one 150k-thread dispatch and takes 0.034 ms).

## 5. Ranked plan

Ranking = expected saving in a standalone full-model pass x confidence. "Measured" means the
change was built into the real app on rocky-ryzen and the saving is the observed one;
"inferred" means I have not run it.

| # | Change | Shader / files | Expected saving | Confidence | Status | Risk |
|---|---|---|---|---|---|---|
| 1 | **Sparse CSR conv**: uniform row bounds via `subgroupBroadcastFirst`, branchless clamped tap, 4 accumulators | `new_cifar_sparse_conv2d_k3s1p1.comp` only | **-55.9 ms** of 121.9 (convs 107.6 -> 51.7) | 0.9 | measured, tests 12/12 | `subgroupBroadcastFirst` is only correct if `out_h*out_w % subgroupSize == 0` (true for 32x32/16x16/8x8 with sub 16/32/64). Guard it in the dispatcher or fall back to the generic shader. Not run on Mali / Jetson / Samsung. |
| 2 | **Dense conv k3s1p1**: branchless taps, uniform weight loads, `OC_TILE` 8 | `new_cifar_dense_conv2d_k3s1p1.comp` + 5 grid-divisor edits (4 -> 8) in `apps/cifar-dense/vulkan/dispatchers.cpp`; contract `K % 8 == 0` (all convs here: 64/192/384/256) | **-4.56 ms** of 8.64 (convs 6.70 -> 2.15) | 0.9 | measured, tests 12/12, bit-exact in isolation | Same uniformity guard as #1. `OC_TILE=16` is slower. Mali with 16-lane subgroups was not tried. |
| 3 | **linear_bt**: batch-tile swizzle + 2 output features per row-group | `new_cifar_dense_linear_bt.comp` + grid.x divisor 16 -> 32 (sparse; the dense dispatcher needs the same edit if the shader is shared) | **-7.2 ms** sparse (FC 12.79 -> 5.61); dense unaffected (already at DRAM roof) | 0.85 | measured on sparse, tests 12/12 | The shader is shared by both apps, so the dense dispatcher must change its divisor too (I did not build dense with the FPR-2 shader). FC3 (N_out=10) regresses by 0.04 ms. |
| 4 | **Tree radix sort**: `kRadixNumWorkgroups` 256 -> 64 | `apps/tree/vulkan/vk_appdata.hpp:14` (one constant) | **-0.37 ms** of 1.705 (-22%) | 0.85 | measured, `test-tree-vk` 14/14 | Fewer, longer workgroups; was only tested at n=307,200 on the 780M. On Mali/Jetson the optimum may differ; n=3M on real data not tried. |
| 5 | Sparse conv, second step: stage the input image in LDS or give each thread several images so one decoded nonzero feeds several loads/FMAs; target 1 VMEM per several FMAs | sparse conv shader | est. 15-25 ms of the remaining 51.7 ms (it is still ~2.6x slower than unpruned dense conv) | 0.3 | **inferred** | Moderate: LDS use vs occupancy; I have not profiled the post-fix kernel's stalls. |
| 6 | linear_bt `FPR = 4` (4 weight rows per input read) | `new_cifar_dense_linear_bt.comp` | ~1-2 ms for sparse FC (floor is 0.77 ms per layer for one weight pass) | 0.4 | **inferred** from the LDS arithmetic above | VGPR growth (acc 64 + 64) lowers occupancy. |
| 7 | Dense FC weights in fp16/bf16 (or int8) | linear_bt + weight export | up to ~0.85 ms of dense (2 x 0.42) | 0.3 | **inferred** (DRAM-bound, 91.6% of read roof) | Changes numerics; the NearEqual oracle tolerance and the "real weights" contract must be agreed. Highest implementation cost. |
| 8 | Fuse tree stage-3 `find_dups` into `tree_scan_local` and `move_dups` into `tree_scan_add` (5 -> 3 dispatches); same for any 3-pass scan | tree scan shaders + dispatcher | ~0.05-0.09 ms | 0.5 | **inferred** from ~0.047 ms/dispatch | Touches correctness-critical stream compaction. |
| 9 | FC3 (4096x10) split-K over several workgroups | linear_bt + dispatcher | ~0.10 ms dense, ~0.10 ms sparse | 0.5 | **inferred** (1 workgroup, 0.139 ms for 164 KB) | Needs a second reduction pass or atomics on the output. |
| 10 | Fuse maxpool into the conv epilogue (stages 1+2, 3+4, 7+8) | conv shaders | up to 0.19 ms dense, ~1.2 ms sparse (the pool kernels plus the intermediate round trip) | 0.3 | **inferred**; maxpools are at 56-73 GB/s, near the 71-87 GB/s roofs | Breaks the stage = one dispatch contract of the scheduler; the pipeline partitioner expects pooling as its own stage. |

After the top 4, residual standalone totals (measured with the prototypes combined):
dense 4.18 ms (FC 41%), sparse 58.8 ms (convs 88%), tree 1.34 ms (stage 2 30%).

Not recommended: the generic `new_cifar_*_conv2d`, `new_cifar_*_linear` shaders (no stage
in the 11-stage model uses them when the specializations are available; the generic linear
is only the fallback); the dead shaders listed in 3.3.

## 6. Measured vs inferred, and what could be wrong

* Measured in the real apps: sections 3 tables (baseline) and 4.1 (dense stages), 4.2, 4.3
  (swizzle + FPR 2), 4.4. The per-stage deltas are far outside the run-to-run spreads
  (< 1% on the stages that matter).
* Measured in isolation only (`gpu-lab`): the v1/v2 conv breakdown (4.1 first table). The
  "branchless alone = -36%, uniform weights = another -23%" split is from conv4 only.
* Inferred: all roofline percentages derive from my FLOP counts (2*MAC) and the three-week-old
  roof file; the LDS-bandwidth explanation of the FC kernel; the launch-latency explanation of
  the tree stages; items 5-10 of the ranking. The radix-sort cost model (O(NWG^2) histogram
  reads) is derived from the source and confirmed only by the end-to-end effect, not by a
  counter.
* Baseline timing caveat: one dense stage 3 run came out at 1.2165 ms against 0.809/0.846 for
  the other two (an idle-clock outlier of the kind described in design D1); I report medians.
* "Share of pipeline" is a share of the *sum* of isolated stage times. In the pipelined
  schedule the kernels run after idle gaps (D1), so absolute times are larger and the
  shortest kernels (tree stages, pools) are slowed most, so their relative share is understated
  here.
* The 780M was shared with other users during the session (see section 2); every number kept
  was taken with `gpu_busy_percent` 0 before the run, but I cannot exclude a short external
  burst inside a run. The A/B pairs were interleaved to limit that.
* Not tested at all: any of the four changes on Mali (Pixel 7a subgroup 16, Samsung subgroup 32),
  Jetson Vulkan (subgroup 32), or with `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1`. The
  `subgroupBroadcastFirst` uniformity assumption is the main portability point; changes 1-3
  should go through the phone gate (`scripts/run-mali-oracle.sh`) before merge.
* Not run: `ctest -L vulkan`, the hybrid e2e suites, `scripts/refactor_gate.sh`; `just fmt`
  is not applicable (no repo code changed).

## 7. Commands run (all on rocky-ryzen via `ssh rocky-ryzen bash -s`, since its login shell is fish)

Build the benchmark binaries from this worktree (rsync + build on rocky, binaries land in
`~/bt-src-vkprof/build/vulkan/`):

```bash
cd <worktree>
BT_BENCH_SRC=bt-src-vkprof scripts/build-bench-on-rocky.sh vulkan
```

Per-stage GPU time (the gate script, 3 rounds interleaved over `cifar-dense cifar-sparse tree`;
`BIN=` selects a variant binary):

```bash
~/vkprof-run/run.sh <app> <tag>   # sleep 5; 5x gpu_busy_percent==0 check; then:
BT_WEIGHTS_DIR=/tmp/bt/weights BT_PROF_SCENARIO=isolated BT_PROF_RUN=1 \
  bm-prof-<app>-vk --device minipc --benchmark_filter='vulkan/' > <app>-<tag>.jsonl
python3 ~/vkprof-run/summ.py        # median of p50 per stage across the 3 runs
```

ISA and statistics:

```bash
RADV_DEBUG=shaders,nocache      bm-prof-<app>-vk --device minipc --benchmark_filter='vulkan/stage1$' 2> dumps/<app>.txt
RADV_DEBUG=shaderstats,nocache  bm-prof-<app>-vk --device minipc --benchmark_filter='vulkan/stage1$' 2> stats-<app>.txt
```

(`shaders` gives NIR/ACO/disassembly, no statistics in Mesa 25.2.7; `shaderstats` gives
VGPR/SGPR/LDS/waves/instruction counts.)

Isolated shader prototypes (from this machine; shaders compiled by `gpu-lab` with `glslc -O`,
same flags as `cmake/bt_shaders.cmake`), shown for conv4:

```bash
~/Desktop/gpu-lab/gpu run 780m <shader>.comp --threads 65536 \
  --buf 1536K:randf --buf 3456K:randf --buf 4K:randf --buf 1M:zero \
  --push 16,384,8,8,256,3,3,1,1,1 --iters 300 --warmup 50 --busy-threshold 1 --json
```

Prototype shaders into the scratch tree (not the repo):

```bash
glslc --target-env=vulkan1.3 -O -fshader-stage=compute X.comp -o name.spv
{ echo '#pragma once'; xxd -i name.spv; } > name_spv.h
scp name_spv.h rocky-ryzen:bt-src-vkprof/platform/engine/vulkan/shaders/h/
# on rocky: cmake --build --preset vulkan --target bm-prof-<app>-vk test-<app>-vk
```

Correctness: `BT_WEIGHTS_DIR=/tmp/bt/weights ./test-cifar-{dense,sparse}-vk --device minipc`
and `./test-tree-vk.nwg{64,32} --device minipc` (12/12, 12/12, 14/14 passed).

Scratch locations (not in git): `/tmp/vkproto` (this host: all prototype `.comp`, `gpu-lab`
outputs `out_*.json`, parse scripts), rocky-ryzen `~/bt-src-vkprof` (scratch source copy with
edited headers, `dispatchers.cpp`, `vk_appdata.hpp` restored to original for the tree constant;
the dense `/ 8` and sparse `/ 32` divisor edits and the prototype shader headers are still in
that scratch tree), `~/vkprof-run` (JSONL, ISA dumps `dumps/`, scripts). Variant binaries:
`bm-prof-cifar-dense-vk.{base,dv2,dv3}`, `bm-prof-cifar-sparse-vk.{base,sv1,sv2,sv3}`,
`bm-prof-tree-vk.{base,nwg128,nwg64,nwg32}` in `~/bt-src-vkprof/build/vulkan/`.

## 8. Applied: sparse CSR conv (item 1 of the ranking)

`new_cifar_sparse_conv2d_k3s1p1.comp` now has the fast path of the prototype in 4.2 (row
bounds from `subgroupBroadcastFirst`, branchless clamped taps, 4 accumulators), but guarded by
construction: the fast path runs only in a subgroup whose active lanes all share one output
channel (`subgroupAllEqual(out_c)`); otherwise the previous per-lane walk runs. So correctness
does not depend on `out_h*out_w` being a multiple of the subgroup size, and no dispatcher guard is
needed. Without the guard the broadcast silently returns another channel's row; the new test
`SparseConvK3S1P1.FallbackWhenSubgroupSpansChannels` (6x6 maps, hand-made CSR with an empty row,
double-precision reference) fails on an unguarded shader and passes on this one. A second test
covers the fast path (8x8).

Per-stage GPU time on rocky-ryzen (`bm-prof-cifar-sparse-vk`, isolated scenario, 3 rounds
interleaved base/new, `gpu_busy_percent` 0 before each run, run spread < 1 %):

| stage | base ms | new ms | change |
|---|---|---|---|
| 1 conv | 1.573-1.599 | 0.987-0.994 | -37 % |
| 3 conv | 16.98-17.00 | 8.29-8.32 | -51 % |
| 5 conv | 27.26-27.29 | 12.96-12.97 | -52 % |
| 6 conv | 38.01-38.12 | 18.24-18.48 | -52 % |
| 7 conv | 23.72-23.74 | 11.68 | -51 % |
| other stages | unchanged | unchanged | within 1 % |
| **sum of stages** | **121.71** | **66.44** | **-45 %** |

The FC stages (9, 10: 6.4 ms each) are untouched; they are item 3 of the ranking.
