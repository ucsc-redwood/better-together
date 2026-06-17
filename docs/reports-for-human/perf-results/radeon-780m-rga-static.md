# Radeon 780M — RGA static shader analysis

**Date:** 2026-06-17 · **Target:** `rocky-ryzen` AMD Radeon 780M Graphics (RADV
PHOENIX, **gfx1103**, RDNA3, subgroup 64) · **Tool:** Radeon GPU Analyzer (RGA)
2.14.2.8, offline SPIR-V mode · **Scope:** all 29 `.comp` kernels.

> **One-line finding:** no kernel spills registers and none is register/LDS-bound
> on the 780M — RGA *rules out* "a shader compiles badly" as a bottleneck. The
> cost is elsewhere (dispatch / fence-sync / memory), which matches the
> runtime-overhead thesis in [`../../instruction-for-ai/05-profiling.md`](../../instruction-for-ai/05-profiling.md).

## What RGA gives us (and what it doesn't)

RGA's `vk-spv-offline` mode compiles each SPIR-V binary with the bundled AMD
compiler for a chosen ASIC, *without a GPU or a live driver*. It is the AMD analog
of `malioc` for Mali — a **static**, offline, CI-friendly check. It reports:

- **register pressure** — `USED_VGPRs` / `USED_SGPRs` against the RDNA3 budget
  (256 VGPR, 106 SGPR per SIMD32);
- **register spills** — `VGPR_SPILLS` / `SGPR_SPILLS` and `SCRATCH_MEM` (any
  non-zero = the compiler ran out of registers and is round-tripping to scratch —
  the first thing to fix);
- **LDS usage** — `USED_LDS_BYTES` against 65536;
- **ISA size** and the full disassembly (`.isa`).

It does **not** give achieved occupancy / wavefront counts: in offline mode
`WAVEFRONT_SIZE` and `THREADS_PER_WORKGROUP` come back `0`. Real occupancy needs a
**live** run (`rga -s vulkan` against a real pipeline) or an **RGP** capture. So
RGA answers *"is the shader itself fat?"*, not *"which stage is slow end-to-end?"*.

## Method (reproducible)

RGA was not installed on rocky; it ships as a self-contained portable tarball
(bundles its own compilers, no root). Rocky's login shell is **fish** → wrap in
`bash -lc` / `bash -s` (same gotcha as [`run-on-rocky.sh`](../../../scripts/run-on-rocky.sh)).

```bash
# 1. fetch RGA on rocky (has internet; github reachable)
ssh rocky-ryzen bash -s <<'EOF'
cd /tmp && rm -rf rga && mkdir rga && cd rga
url=$(curl -s https://api.github.com/repos/GPUOpen-Tools/radeon_gpu_analyzer/releases/latest \
      | grep -oE '"browser_download_url": *"[^"]+rga-[0-9.]+\.tgz"' | grep -oE 'https[^"]+')
curl -sL -o rga.tgz "$url" && tar xzf rga.tgz
EOF

# 2. stage the SPIR-V (already built into shaders/spv/)
scp builtin-apps/common/kiss-vk/shaders/spv/*.spv rocky-ryzen:/tmp/bt-spv/

# 3. batch-analyse every kernel for gfx1103
ssh rocky-ryzen bash -s <<'EOF'
RGA=/tmp/rga/rga-*/rga ; cd /tmp/bt-spv ; mkdir -p /tmp/bt-rga-out
for s in *.spv; do n=${s%.spv}
  $RGA -s vk-spv-offline -c gfx1103 --comp "$s" \
       --isa "/tmp/bt-rga-out/$n.isa" --analysis "/tmp/bt-rga-out/$n.csv"
done
EOF
```

Each run emits `gfx1103_<name>_comp.csv` (one stats row) + `..._comp.isa`
(disassembly). Confirm the ASIC is supported with `rga -s vk-spv-offline -l | grep gfx1103`.

## Results — all 29 kernels, gfx1103 (sorted by VGPR)

Budget: **256 VGPR · 106 SGPR · 65536 B LDS**. `SPILL`/`SCRATCH` are the
load-bearing columns — **all zero**.

| Shader | VGPR | SGPR | LDS B | V-spill | S-spill | Scratch | ISA B |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tmp_single_radixsort_warp64` | 43 | 65 | 11776 | 0 | 0 | 0 | 1852 |
| `tmp_single_radixsort_warp16` | 43 | 65 | 11776 | 0 | 0 | 0 | 1844 |
| `tmp_single_radixsort_warp32` | 37 | 65 | 11776 | 0 | 0 | 0 | 2296 |
| `stressor` | 29 | 19 | 0 | 0 | 0 | 0 | 2936 |
| `tree_build_octree` | 22 | 106 | 0 | 0 | 0 | 0 | 3524 |
| `tree_scan_local` | 21 | 58 | 8192 | 0 | 0 | 0 | 2316 |
| `tree_scan_block_sums` | 20 | 46 | 8192 | 0 | 0 | 0 | 2152 |
| `new_cifar_sparse_conv2d` | 20 | 46 | 0 | 0 | 0 | 0 | 1528 |
| `multi_radixsort_warp64` | 20 | 52 | 9728 | 0 | 0 | 0 | 1856 |
| `multi_radixsort_warp32` | 20 | 52 | 9728 | 0 | 0 | 0 | 1844 |
| `multi_radixsort_warp16` | 20 | 52 | 9728 | 0 | 0 | 0 | 1856 |
| `octree_build_octree_nodes` | 17 | 30 | 0 | 0 | 0 | 0 | 916 |
| `tree_morton` | 13 | 19 | 0 | 0 | 0 | 0 | 512 |
| `tree_build_radix_tree` | 13 | 42 | 0 | 0 | 0 | 0 | 1528 |
| `tree_merge_sort` | 11 | 25 | 0 | 0 | 0 | 0 | 576 |
| `octree_build_radix_tree` | 11 | 40 | 0 | 0 | 0 | 0 | 1180 |
| `new_cifar_sparse_maxpool` | 10 | 26 | 0 | 0 | 0 | 0 | 1092 |
| `new_cifar_sparse_linear` | 10 | 26 | 0 | 0 | 0 | 0 | 492 |
| `new_cifar_dense_maxpool` | 10 | 38 | 0 | 0 | 0 | 0 | 912 |
| `new_cifar_dense_conv2d` | 10 | 45 | 0 | 0 | 0 | 0 | 1016 |
| `octree_edge_count` | 7 | 14 | 0 | 0 | 0 | 0 | 216 |
| `new_cifar_dense_linear` | 7 | 16 | 0 | 0 | 0 | 0 | 412 |
| `multi_radixsort_histograms` | 6 | 21 | 1024 | 0 | 0 | 0 | 656 |
| `tree_edge_count` | 5 | 16 | 0 | 0 | 0 | 0 | 264 |
| `octree_morton` | 5 | 16 | 0 | 0 | 0 | 0 | 436 |
| `tree_scan_add` | 3 | 12 | 0 | 0 | 0 | 0 | 520 |
| `tree_find_dups` | 3 | 14 | 0 | 0 | 0 | 0 | 156 |
| `tree_naive_prefix_sum` | 2 | 14 | 0 | 0 | 0 | 0 | 160 |
| `tree_move_dups` | 2 | 14 | 0 | 0 | 0 | 0 | 152 |

Artifacts on rocky: `/tmp/bt-rga-out/*.csv` (stats) + `*.isa` (disassembly);
RGA itself at `/tmp/rga/`.

## Findings

1. **Zero spills, zero scratch, across all 29 kernels.** Nothing is register-
   starved on RDNA3. This is the headline pass/fail and it passes.
2. **No kernel is register-bound.** The heaviest user is 43 VGPR (`single_radixsort`);
   against a 256-VGPR file that leaves ample room for many resident waves. VGPR is
   not what caps occupancy here.
3. **Where occupancy *is* mildly capped, it's LDS, not registers.** The
   `single_radixsort` (11776 B) and `multi_radixsort` / `tree_scan_*` (8–9.7 KB)
   kernels limit residency to ~5–8 workgroups/CU by LDS (64 KB ÷ ~11 KB). Still
   comfortable; only relevant if these ever become the measured bottleneck.
4. **The cifar kernels (conv2d / linear / maxpool, dense & sparse) are tiny** —
   7–20 VGPR, mostly **zero LDS**, sub-1.5 KB ISA. They are compute-trivial and
   will be **memory-/dispatch-bound**, not ALU-bound. Tuning their SPIR-V would
   buy nothing; the win is in feeding them (overlap, fewer round-trips).
5. **`tmp_single_radixsort_warp{16,32,64}` look experimental/unused in the hot
   path.** They are registered in
   [`all_shaders.hpp`](../../../builtin-apps/common/kiss-vk/shaders/all_shaders.hpp)
   but the sort pipeline uses the multi-pass `multi_radixsort_*` variants; the
   single-pass `tmp_*` ones (the only kernels above 37 VGPR) appear to be dead
   weight. Worth confirming and pruning — not a perf issue, just clarity.

## Insights

- RGA's value on this project is **negative-result confirmation**: it removes
  "bad shader codegen" from the suspect list, redirecting effort to the runtime
  overhead (submit/fence/UMA-maintenance/SPSC), which is exactly the first-order
  KPI the profiling doc tells us to chase.
- The subgroup-16/32/64 radixsort variants compile to **near-identical register
  footprints** (43/43/37 VGPR), so picking a variant per device is a correctness/
  occupancy-by-subgroup decision, **not** a register-pressure trade-off.
- Static analysis is cheap enough to gate in CI (no device, runs on the build
  box for any `gfx*` target). A spill regression — e.g. someone unrolls a loop or
  bloats a struct — would show up as a non-zero `*_SPILLS`/`SCRATCH` before it ever
  reaches hardware.

## Suggestions / next steps

1. **CI gate (high ROI, no device):** wrap the batch above as
   `scripts/rga-analyze.sh` and fail the build if any kernel reports
   `VGPR_SPILLS>0 || SGPR_SPILLS>0 || SCRATCH_MEM>0`. Pair with `malioc --format
   json` for the Mali targets — same idea, both static, both CI-friendly.
2. **For actual occupancy / "which stage is slow":** RGA static has done its job;
   move to a **live** signal — the in-app per-stage Vulkan timestamps (already in
   `sequence.cpp`, see [`05-profiling.md`](../../instruction-for-ai/05-profiling.md)
   Tier S), then **RGP** for wavefront-level timeline on the 780M. RGP on RADV +
   headless compute needs the `RADV_THREAD_TRACE_TRIGGER` file mechanism (there is
   no swapchain/present to trigger on).
3. **Prune or rename `tmp_single_radixsort_*`** if confirmed unused, so the shader
   set reflects what actually runs.
4. **Don't micro-optimize the cifar kernels' ISA.** Finding 4 says they're not
   ALU-bound; spend the effort on dispatch/sync overlap instead.
