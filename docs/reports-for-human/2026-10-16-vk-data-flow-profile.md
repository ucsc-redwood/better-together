# Vulkan data flow: where the time goes (2026-10-16)

**Question.** The Vulkan runtime looks under-optimized in how data moves: is a data transfer the
bottleneck, and where? **Short answer.** There is no explicit transfer to optimize (no staging
buffer, no `vkCmdCopy`; every buffer is host-visible, persistently mapped memory). The data cost is
**host cache maintenance** (`vkFlushMappedMemoryRanges` before each submit, `vkInvalidate...` after
each wait) and **where the thread that drives the GPU runs**. Both matter a lot on the Pixel 7a
(Mali) with the tree app: the GPU chunk spends 51 % of its wall time on cache maintenance and the
GPU itself only 35 %. On the Samsung (Xclipse) and for both cifar apps they do not matter: the cifar
chunks are 92-99 % GPU compute, so there the bottleneck is the shaders.

Everything was measured on the two phones, **not on rocky** (instruction: no profiler on rocky). The
local box's integrated GPU (RADV, coherent memory) only served as a sanity check of the tools:
flush/invalidate cost about 0 there.

## 1. How data moves today (from the code)

* All buffers come from `VulkanMemoryResource` classes; every placement is host-visible and mapped
  (`platform/engine/vulkan/vma_pmr.hpp`). There is no `memcpy`/`std::copy` between stages in
  `runtime/` or the apps' Vulkan code.
* `Sequence::submit()` calls `vmaFlushAllocation(a, 0, VK_WHOLE_SIZE)` on **every** buffer a chunk
  binds, and `Sequence::wait()` calls `vmaInvalidateAllocation(a, 0, VK_WHOLE_SIZE)` on every one
  (`sequence.cpp`). It does not look at whether the buffer is read or written by the GPU, whether the
  host wrote it since, or whether the host will read it afterwards. The cost of cache maintenance is
  proportional to the range, not to the dirty data. On coherent memory both calls are no-ops.
* Buffer classes (`boundary`, `weights`, `gpu_local`) let a schedule put buffers that only GPU stages
  touch into other memory (`runtime/placement.hpp`). Two gaps: the **tree app does not use them** (its
  AppData has no `PlaceBuffer` constructor, so all its buffers are in the host-cached `boundary`
  class), and `gpu_local` has the same placement as `boundary` on every device, because no
  measurement had promoted it.
* Hand-off between pipeline chunks is a busy-yield SPSC queue (`runtime/pipeline.hpp`); OMP chunks are
  pinned to their core tier, **the thread that drives a GPU chunk is not pinned**.

## 2. Phases of a GPU chunk (`bm-phases-*-vk`, 100 submissions, medians, host thread unpinned)

`maint` = flush + invalidate (the invalidate time also holds the timestamp read-back). `touched` =
MiB the two walk. On the Pixel the cifar weights must be in coherent memory (flushing 129 MiB took
0.09 ms), so the cifar rows' maintenance cost is not proportional to `touched`.

| device | app | stages | wall ms | flush ms | invalidate ms | maint | GPU ms | GPU | touched MiB |
|---|---|---|---|---|---|---|---|---|---|
| Pixel 7a | tree | 1-7 | 18.17 | 4.60 | 4.66 | 51 % | 6.36 | 35 % | 15.5 |
| Pixel 7a | tree | 2-2 | 14.39 | 0.98 | 1.04 | 14 % | 9.17 | 64 % | 3.6 |
| Pixel 7a | tree | 5-7 | 7.75 | 1.46 | 1.53 | 39 % | 3.09 | 40 % | 15.5 |
| Pixel 7a | cifar-dense | 1-6 | 123.83 | 3.59 | 3.52 | 6 % | 114.32 | 92 % | 17.8 |
| Pixel 7a | cifar-dense | 1-11 | 165.51 | 4.10 | 4.06 | 5 % | 154.99 | 94 % | 150.0 |
| Pixel 7a | cifar-dense | 7-11 | 43.39 | 0.27 | 0.34 | 1 % | 40.62 | 94 % | 133.2 |
| Pixel 7a | cifar-dense | 9-11 | 17.91 | 0.09 | 0.13 | 1 % | 15.48 | 86 % | 128.9 |
| Pixel 7a | cifar-sparse | 1-6 | 802.27 | 14.81 | 14.78 | 4 % | 760.75 | 95 % | 104.2 |
| Pixel 7a | cifar-sparse | 1-11 | 1069.08 | 16.56 | 16.37 | 3 % | 1033.90 | 97 % | 250.9 |
| Pixel 7a | cifar-sparse | 7-11 | 281.33 | 1.98 | 2.19 | 1 % | 275.28 | 98 % | 154.7 |
| Pixel 7a | cifar-sparse | 9-11 | 60.83 | 0.54 | 0.63 | 2 % | 57.58 | 95 % | 134.2 |
| Samsung | tree | 1-7 | 3.95 | 0.02 | 0.43 | 11 % | 2.11 | 54 % | 15.5 |
| Samsung | tree | 2-2 | 1.53 | 0.00 | 0.02 | 1 % | 1.21 | 79 % | 3.6 |
| Samsung | tree | 5-7 | 1.15 | 0.00 | 0.02 | 2 % | 0.93 | 81 % | 15.5 |
| Samsung | cifar-dense | 1-6 | 8.32 | 0.03 | 0.09 | 1 % | 6.95 | 84 % | 17.8 |
| Samsung | cifar-dense | 1-11 | 12.37 | 0.05 | 0.09 | 1 % | 10.76 | 87 % | 150.0 |
| Samsung | cifar-dense | 7-11 | 7.55 | 0.01 | 0.07 | 1 % | 5.94 | 79 % | 133.2 |
| Samsung | cifar-dense | 9-11 | 5.50 | 0.01 | 0.07 | 1 % | 4.14 | 75 % | 128.9 |
| Samsung | cifar-sparse | 1-6 | 158.69 | 0.04 | 0.12 | 0 % | 155.76 | 98 % | 104.2 |
| Samsung | cifar-sparse | 1-11 | 223.57 | 0.07 | 0.11 | 0 % | 220.81 | 99 % | 250.9 |
| Samsung | cifar-sparse | 7-11 | 67.85 | 0.01 | 0.06 | 0 % | 65.71 | 97 % | 154.7 |
| Samsung | cifar-sparse | 9-11 | 21.78 | 0.01 | 0.07 | 0 % | 20.22 | 93 % | 134.2 |

Reading it:

* **Pixel 7a x tree is the one case where cache maintenance dominates**: 9.3 ms of 18.2 ms for stages
  1-7 (51 %), 39 % for 5-7. 15.5 MiB at about 3.3 GiB/s.
* The Samsung's maintenance is 0-11 % (its memory is coherent; the 0.4 ms is the timestamp read-back).
* For both cifar apps the GPU is 75-99 % of the chunk wall time: **data flow is not their bottleneck**.
* `wait - GPU` (start latency plus wake-up) is about 20 % of a short tree chunk (Pixel 8.0 vs 6.4 ms,
  Samsung 3.0 vs 2.1 ms). Not much to gain there.

## 3. The core the host thread runs on changes the maintenance cost 3x

Pixel 7a, tree stages 1-7, 100 submissions, process pinned with `taskset` (cores 0-3 little, 4-5
medium, 6-7 big):

| host thread on | flush ms | invalidate ms | chunk wall ms | GPU ms |
|---|---|---|---|---|
| unpinned (the OS put it on a little core) | 4.45 | 4.54 | 17.58 | 5.98 |
| little 0-3 | 4.39 | 4.47 | 16.84 | 5.97 |
| medium 4-5 | 1.55 | 1.60 | 10.93 | 5.77 |
| big 6-7 | 1.48 | 1.68 | 10.66 | 5.76 |

`bm-phases` blocks while the GPU runs; the likely reason is that the scheduler then parks its thread
on a little core (the unpinned row is as slow as the little-core row). In the
real pipeline the GPU thread is not pinned either (section 1), so the same can happen there.

## 4. Pipeline: where the GPU-driving thread runs (`bm-gen-logs-*-vk`, `BT_GPU_THREAD_CORES`)

Schedules: tree `Big[1] | GPU[2-4] | Big[5-7]`, cifar-dense `Big[1-4] | GPU[5-11]`. Time per item at
steady state (ticks of the logger's counter, tasks 20-99; 3 rounds interleaved, 20 s apart):

| device | app | GPU thread unpinned | little core 0 | medium core 4 |
|---|---|---|---|---|
| Pixel 7a | tree | 282 779 | 482 553 (**+70.6 %**) | 283 301 (+0.2 %) |
| Pixel 7a | cifar-dense | 3 181 825 | 3 174 333 (-0.2 %) | 3 170 652 (-0.4 %) |
| Samsung | tree | 237 293 | 249 733 (+5.2 %, noisy) | 176 059 (**-25.8 %**) |
| Samsung | cifar-dense | 444 797 | 446 375 (+0.4 %) | 445 266 (+0.1 %) |

* Pixel tree: by default the GPU thread does not end up on a little core in the pipeline (it probably
  counts as a busy thread because it spins on the queue), but on a little core the pipeline is 71 % slower.
  The risk is real, the default is not currently hitting it.
* Samsung tree: the bottleneck chunk is the CPU one (`Big[5-7]`); moving the GPU thread to a medium
  core makes the pipeline 26 % faster. Inferred cause: the unpinned, spinning GPU thread takes time
  from the OMP workers on the big cores. Not proven.
* cifar-dense (GPU- or CPU-chunk-dominated) does not react: the control behaves as predicted.

## 5. What the maintenance is worth: boundary buffers in coherent memory

`BT_VK_BOUNDARY_PLACEMENT` (new experiment knob) moves the boundary class to a coherent memory type.
Tree stages 1-7 as one GPU chunk, host thread on big cores, 100 submissions, 3 rounds interleaved:

| device | placement | wall ms | flush ms | invalidate ms | GPU ms | wall vs default |
|---|---|---|---|---|---|---|
| Pixel 7a | default (host-cached) | 10.55 | 1.48 | 1.64 | 5.76 | |
| Pixel 7a | uncached | **6.50** | 0.00 | 0.04 | 4.74 | **-38 %** |
| Pixel 7a | device_local | **6.44** | 0.00 | 0.04 | 4.66 | **-39 %** |
| Samsung | default | 3.75 | 0.00 | 0.41 | 2.11 | |
| Samsung | uncached | 3.64 | 0.00 | 0.40 | 2.11 | -3 % |
| Samsung | device_local | 3.64 | 0.00 | 0.40 | 2.11 | -3 % |

On the Pixel the GPU time itself drops 18 % as well. The Samsung does not change (already coherent).
Against the unpinned 17.6 ms of section 3, the chunk is 63 % shorter.

Caveat: this moves **all** boundary buffers, including any a CPU chunk reads, so it is a measurement,
not a setting for schedules with CPU chunks (CPU reads of uncached memory are slow). The right
mechanism is the existing class-based placement, which needs the tree app to use it.

## 6. Shader qualifiers (needed for any direction-aware maintenance)

`Algorithm` reads each binding's `readonly`/`writeonly` from the SPIR-V; a plain `buffer` counts as
read-write. 29 of 84 bindings in `shaders/comp` have no qualifier (26 in the tree shaders, 3 in the
test and stress shaders): `tree_build_octree` 13 of 13,
`tree_build_radix_tree` 3 of 6, both radix-sort shaders 5 of 5, `tree_merge_sort` 2 of 2,
`tree_naive_prefix_sum`/`tree_scan_add`/`tree_scan_block_sums` 1 each. The cifar shaders are all
qualified. For the tree chunk 1-7 the engine therefore sees 14 buffers, 15.53 MiB, all read-write,
all non-coherent on the Pixel. (Even with qualifiers, buffers written then read inside one chunk are
read-write; the engine cannot tell they never reach the host. That knowledge is in the app, via the
stages each buffer declares.)

## 7. Ranking

| # | change | evidence | expected gain | risk |
|---|---|---|---|---|
| 1 | Give the tree app buffer classes (`PlaceBuffer`) and put `gpu_local` in coherent memory on devices whose default is non-coherent | section 5: Pixel tree GPU chunk -38 % (GPU time -18 % too); Samsung neutral | tree on Pixel: about 4 ms per GPU chunk (10.55 -> 6.50 ms with the thread on a big core) | must measure per device; CPU never touches these buffers by construction of the class rules; uncached CPU access is slow if a rule is wrong |
| 2 | Pin, or at least steer, the thread that drives a GPU chunk (a medium core), and say so in the schedule | sections 3-4: +71 % if it lands on a little core (Pixel tree); -26 % when moved off the big cores (Samsung tree) | pipeline-level, up to 26 % on Samsung tree | which core is device-specific; needs a field in `devices/*.json` or the schedule |
| 3 | Direction-aware maintenance: no invalidate for buffers no shader writes, no flush for write-only buffers, none for buffers that never leave the chunk | section 6: today everything is walked whole | recovers part of #1 on devices where #1 is not enough; little on top of #1 | needs qualifiers on the 29 tree bindings and per-buffer liveness from the app |
| 4 | Nothing to do: submission overhead (record/submit <= 0.7 ms), wake-up latency, cifar data flow | section 2 | - | - |

Largest numbers outside data flow: the Pixel's cifar-sparse takes 1069 ms per item (Samsung 224 ms);
its conv shaders (about 165-330 ms each) are untouched by the sparse-conv fix, which relies on scalar
loads Mali lacks. That is a shader job for Mali.

## 8. Limits of this study

* Two phones plus a coherent local GPU; no Jetson (a different memory model) and no rocky.
* One schedule per app in section 4; phone clocks and temperature are not controlled (runs are
  interleaved, 15-20 s apart, and spreads are in the raw numbers above).
* Logger ticks in section 4 are only compared as ratios.
* `bm-phases` runs the chunk alone; the pipeline numbers (section 4) include the CPU chunks.
* The maintenance rate (GiB/s) is meaningful only for tree: cifar weights are in coherent memory on
  the Pixel and the byte count includes them.
* The busy-yield contention cause in section 4 is inferred.

## 9. How to reproduce

```bash
cmake --preset android -DBT_BUILD_BENCHMARKS=ON && cmake --build --preset android \
  --target bm-phases-tree-vk bm-phases-cifar-dense-vk bm-phases-cifar-sparse-vk bm-gen-logs-tree-vk
# per chunk shape (phone serial as --device); SPDLOG_LEVEL=debug lists the bound buffers
adb -s <serial> shell 'cd /data/local/tmp/btflow && LD_LIBRARY_PATH=. ./bm-phases-tree-vk --device <serial> -s 1 -e 7 -n 100'
# core placement: taskset -c 6,7 (Pixel), taskset 180 (Samsung's taskset has no -c)
# placement: BT_VK_BOUNDARY_PLACEMENT=uncached;  GPU thread: BT_GPU_THREAD_CORES=4
```
