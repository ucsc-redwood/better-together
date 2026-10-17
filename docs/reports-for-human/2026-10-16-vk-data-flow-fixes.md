# Vulkan data-flow fixes: what was done and what it bought (2026-10-16)

Follow-up to [`2026-10-16-vk-data-flow-profile.md`](2026-10-16-vk-data-flow-profile.md), which ranked
three changes. All three were implemented, each measured on the two phones (Pixel 7a, Samsung) with
the same interleaved method; **nothing was run on rocky except correctness tests**. The short version:
two of the three are worth keeping on by default; the third is implemented but **off by default**,
because the data does not support turning it on.

| # | change | default | measured effect |
|---|---|---|---|
| 3 | directional cache maintenance + `readonly`/`writeonly` on 18 shader bindings + a lint | on | correct, small: bytes invalidated for tree stages 1-3 11.78 -> 7.09 MiB; whole chunk -2..-3 % (noise) |
| 1 | tree app uses buffer classes; GPU-only buffers in coherent memory on the Pixel 7a | on | Pixel tree chunk **-7 %**, cifar -3 %; Samsung unchanged |
| 2 | place the thread that drives a GPU chunk | **off** (`BT_GPU_THREAD_CORES=auto` to try) | Pixel +3..+13 % slower pinned, Samsung -2..-43 % but runs 2x apart |

## 3. Directional maintenance, shader qualifiers, lint

* `Sequence` flushes only buffers some dispatch accesses and invalidates only buffers some dispatch
  writes (it did both for the whole allocation of every bound buffer). A binding the SPIR-V declares no
  variable for counts as unused (`Algorithm`: `kNone`, was read-write).
* 18 bindings in 5 shaders got `readonly`/`writeonly`. They come from the optimized SPIR-V's loads and
  stores, **not** from trial compiles: glslang rejects a write to a `readonly` buffer but accepts a read
  of a `writeonly` one, so trial compiles would have marked `tree_scan_add` binding 0 and
  `tree_build_octree` binding 3 `writeonly`. `scripts/check_shader_access.py` (`just check-shaders`, in CI)
  compares every declaration with the SPIR-V and was checked against two deliberately wrong shaders.
* Tests: `CacheMaintenance.IsDirectional` (new shader `engine_test_copy`: read-only input, write-only
  output, an unused third binding; mutation-checked), `SpirvReflection.BufferAccess` updated.
* Effect is small, as predicted: most buffers of a tree chunk are intermediates, written by one stage and
  read by the next, so they are read-write to the engine. Pixel 7a, tree, unpinned:

  | chunk | MiB invalidated | whole chunk ms |
  |---|---|---|
  | 1-3 | 11.78 -> 7.09 | 27.6 -> 26.2 (batch 1-3 only) |
  | 4-7 | 13.77 of 15.53 | 12.2 -> 13.7 (noise; same bytes) |
  | 1-7 | | 42.6 -> 41.3 (-3 %); on the big cores 29.9 -> 29.2 (-2 %) |

## 1. Buffer classes for the tree app

* `tree::AppData` and `VkAppData` have a `PlaceBuffer` constructor like the cifar apps; the old
  constructor delegates to it with `place_all_in(mr)`. Each buffer declares the stages that touch it on any
  backend (0 = host code: the final octree and the two counts the dispatcher reads back from `u_out_idx`
  and `u_edge_offset_s6`). The input points are declared as **weights** (written once at construction,
  then only read by stage 1).
* `gpu_local` had the same placement as the host-cached boundary class on every device. It now follows the
  weights rule: a table of devices where `uncached` measured faster, and `BT_VK_GPU_LOCAL_PLACEMENT` to
  override. Pixel 7a is in the table.
* Tests: `Placement.CpuChunkBuffersStayBoundary` for tree (hardware-free; every 1-7 GPU range) and
  `Placement.DeclaredStagesMatchBindings` for tree (the declared stages equal the stages that bind each
  buffer; mutation-checked).
* Measured, whole tree chunk 1-7 on the big cores, 3 rounds:

  | step | Pixel 7a | Samsung |
  |---|---|---|
  | `gpu_local` only (intermediates), default -> uncached | 29.39 -> 28.20 ms (-4 %) | +2 % (noise) |
  | + input points as weights, new defaults vs both classes forced host-cached | 29.42 -> **27.51 ms (-7 %)** | 9.07 -> 9.24 ms (+2 %, noise) |
  | cifar-dense / cifar-sparse GPU chunk, `gpu_local` uncached | -3 % / -3 % | +1 % / 0 % |

* **Why only -7 % and not the -31 % of the profile report's experiment.** That experiment put *every*
  boundary buffer in coherent memory. What stays boundary here, by contract, is what host code reads:
  the final octree outputs (8.2 MiB, invalidated every item) and two counts. A pipeline that never reads
  the octree (benchmarks, the usual run) still pays that invalidate. Closing the gap needs a way to say
  "the host reads this occasionally, not per item" (see Follow-ups).

## 2. The thread that drives a GPU chunk

`runtime/gpu_host_thread.hpp` implements the policy proposed in the profile report (a free medium or big
tier, medium first; never little, never the super core). The measurements did **not** support enabling it:

Tree pipeline `Big[1] | GPU[2-4] | Big[5-7]`, interval per item (logger ticks), 4 rounds interleaved:

| | unpinned | core 4 | policy set | another core |
|---|---|---|---|---|
| Pixel 7a | 240 361 | 259 665 (+8.0 %) | 271 381 (cores 4,5: **+12.9 %**) | 248 336 (core 6: +3.3 %) |
| Samsung | 246 720 | 241 520 (-2.1 %) | 140 867 (cores 4,5,6: -42.9 %) | 193 706 (core 7: -21.5 %) |

The Samsung runs behind the pinned medians are 245k / 170k / 108k / 112k and 192k / 248k / 118k / 196k
(unpinned: 258k / 246k / 247k / 235k): the pinned configurations are sometimes much faster, sometimes not
faster at all. The Pixel is slower in every pinned variant. Earlier single-experiment readings (Pixel
"core 4 = unpinned", Samsung "-26 %") did not reproduce, which is why the default stays unpinned.
Control (cifar-dense, GPU-dominated): no effect on either phone, as expected.

So `BT_GPU_THREAD_CORES` is **opt-in**: unset = the OS places the thread (behaviour unchanged);
`none`, `<id>[,<id>...]` and `auto` (the policy) are available. Wired into `run_pipeline` and
`bm-gen-logs`. Tests: `GpuHostCores.*` (8, hardware-free).

## What the gates said

| gate | result |
|---|---|
| local | kiss 30, tree 18, cifar-dense 15, cifar-sparse 16, e2e 5/3/2, chained 1, gpu-host-thread 8, schedule 26 |
| rocky (native build, correctness only) | the same suites, all passed |
| Pixel 7a and Samsung | kiss 29, tree 18, cifar-dense 15, cifar-sparse 16; exit 0 |
| CI | runs on the PR |

## Found on the way (not fixed here)

* **Vulkan stage 7 never writes the octree's leaf masks.** `tree_build_octree.comp` declares
  `ChildLeafMasks` (binding 4) and three more bindings (10-12) it never uses, so `u_oct_child_leaf_mask_s7`
  keeps its initial value on the Vulkan path, while the OMP path fills it. The Vulkan differential oracle
  does not compare that buffer, so the tests are green. (The engine now skips cache maintenance for those
  four unused bindings, which is a side benefit.)
* The tree GPU chunk is **two submissions** (stage 3|4: the host reads `n_unique`), so
  `bm-phases` shows only the second (corrected in the profile report).

## Follow-ups worth doing

1. A way to declare "the host reads this buffer occasionally" so the final octree can sit in coherent
   memory and stop being invalidated per item (the remaining gap to the -31 % of the profile experiment).
2. Remove the stage 3|4 round trip (`vkCmdDispatchIndirect` sized by a GPU-written `n_unique`).
3. Write the octree leaf masks in `tree_build_octree.comp` and add them to the Vulkan oracle.
4. Per-device GPU-thread placement only if a stable effect is found (e.g. record the Samsung's
   bimodal runs against the actual core and frequency, not just the interval).
