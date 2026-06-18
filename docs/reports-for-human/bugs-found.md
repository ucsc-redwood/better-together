# Bugs surfaced by the differential-oracle test suite

> Record of real defects found while replacing the smoke-level gtest suite with
> OMP-as-oracle differential correctness tests (method: [`../instruction-for-ai/03-unit-testing.md`](../instruction-for-ai/03-unit-testing.md); status: [`testing-status.md`](testing-status.md)).
> Each was invisible to the old "did it run / did the buffer change" checks.
> Branch: `refactor/framework-device-axis`.

---

## 1. CUDA managed buffers never stream-attached for the GPU — **RESOLVED (commit 4161664)**

> **Resolved (2026-06-16, commit `4161664`):** instead of the stream-attach surgery,
> the CUDA dispatchers' `CudaManager` was switched from `CudaManagedResource` to
> **`CudaPinnedResource`** (zero-copy mapped pinned: `cudaHostAlloc(cudaHostAllocMapped)`
> + `cudaHostGetDevicePointer`). On the Jetson Orin UMA, pinned memory is physically
> shared/coherent and stays host-accessible *concurrently* with GPU kernels, fixing
> both the sequential visibility race and the CPU+GPU hybrid pipeline — with no
> per-buffer stream-attach juggling and the kernel launches unchanged. The
> `CudaManagedResource` below is left intact (no caller uses it on Tegra now); if it
> is ever reused there, the stream-attach analysis still applies.
>
> **Re-verified on Jetson 2026-06-17** (current `dev`, incl. the Phase 1/2 robustness
> changes): `ctest -L cuda` GREEN — tree-cu 7/7, cifar-dense-cu 10/10,
> cifar-sparse-cu 10/10, deterministic, no hangs. The analysis below is kept for the
> record (and for any future reuse of the managed resource).

**File:** `builtin-apps/common/cuda/cu_mem_resource.cu`, `CudaManagedResource::do_allocate()`

```cpp
cudaError_t err = cudaMallocManaged(&ptr, bytes, cudaMemAttachHost);
```

**Root cause.** `cudaMemAttachHost` attaches the allocation to the host. To let a
GPU kernel touch it you must `cudaStreamAttachMemAsync(stream, ptr, 0,
cudaMemAttachSingle)` onto the stream the kernel runs on, and launch the kernel on
that stream. The code does **neither** — `cudaStreamAttachMemAsync` appears
nowhere and all 13 kernel launches use the **default stream**. On devices with
`concurrentManagedAccess = 0` (Tegra / **Jetson Orin** — verified via
`cudaDevAttrConcurrentManagedAccess`), GPU access to host-attached managed memory
that was never stream-attached is **undefined**: the kernel's writes are only
partially visible to the host and **vary run-to-run**.

**Symptom.** Every CUDA stage of every app returned mostly-zero output on the
Jetson. Measured: tree stage-1 morton had only ~184k / 307200 elements non-zero,
the count changing each run (181k / 192k / 195k). The values that *did* land were
correct — a visibility/coherence race, not wrong math.

**Why it was hidden.** (a) The old tests asserted only `EXPECT_TRUE(is_different)`
("the buffer changed") — partial output still counts as changed, so they passed
(tree CUDA 13/13). (b) CUDA only ever *ran* on the Jetson; the discrete-GPU build
box can't build CUDA (CUDA-13 removed `cub::DivideAndRoundUp`), and discrete GPUs
have `concurrentManagedAccess = 1` with looser rules that mask it.

**Why the obvious one-liner is WRONG.** Switching to default
`cudaMallocManaged(&ptr, bytes)` (= `cudaMemAttachGlobal`) makes *sequential* GPU
output correct (verified: cifar-dense/sparse CUDA 0/9 → 9/9, tree 5/7). **But it
breaks the CPU+GPU hybrid that is the point of the paper.** The hybrid executor
(`pipe/*-cu/main.cu`) runs CPU (OMP) and GPU (CUDA) pipeline stages on separate
`std::thread`s concurrently, ping-ponging AppData items through SPSC queues — so a
CPU thread does **host** access to one item's managed buffers while a kernel runs
on another item's buffers. On `concurrentManagedAccess = 0`, the host may not
access **any** *global*-attached managed memory while a kernel is running → the
CPU thread faults. `cudaMemAttachHost` was the deliberate hack that keeps that
concurrent host access legal; the authors just never added the GPU half.

**The real fix (preserves both correctness and concurrency).** Keep
`cudaMemAttachHost`, and in the dispatcher: before a kernel uses a buffer,
`cudaStreamAttachMemAsync(mgr_stream, ptr, 0, cudaMemAttachSingle)`; launch the
kernel on `mgr_stream` (not the default stream); `cudaStreamSynchronize(mgr_stream)`
before the host reads that output. Buffers belonging to *other* pipeline items
stay host-attached and remain CPU-accessible concurrently. (Alternative worth
evaluating for Tegra UMA: zero-copy pinned memory — the existing
`CudaPinnedResource` via `cudaHostAlloc` + `cudaHostGetDevicePointer` — which is
physically shared and coherent without attach juggling.) Either needs validation
of **both** the sequential differential oracle **and** the hybrid pipeline on the
Jetson.

**Status.** Left as `cudaMemAttachHost` with an explanatory warning comment. The
CUDA differential tests (`*-cu`) correctly **fail** until the real fix lands —
they are revealing a genuine, latent correctness defect in the published CUDA
backend (GPU pipeline stages on Jetson have never produced fully-correct output;
the paper measures latency, so it went unnoticed).

---

## 2. `octree/appdata.hpp` missing `<algorithm>` — **FIXED**

**File:** `builtin-apps/octree/appdata.hpp`

Used `std::ranges::generate` without including `<algorithm>`. It compiled under
the xmake/clang build via transitive includes, but **failed to compile** in the
CMake/gcc path the moment the octree tests were registered. Added the include.

---

## 3. cifar test batch-size constant drift — **FIXED**

**Files:** `cifar-{dense,sparse}/**/test_main.{cpp,cu}`

Tests hard-coded `constexpr int kTestBatchSize = 128;` duplicated from the
AppData. cifar-sparse's AppData uses `BATCH_SIZE = 512`, so the CUDA test's
dimension assertions checked the wrong number (a latent failure that only bites
on the CUDA target). Replaced every copy with `AppData::BATCH_SIZE` so the
constant cannot diverge.

---

## 4. tree morton keys read as `float` — **FIXED**

**Files:** `tree/{omp,cuda,vulkan}/test_main.*`

Morton keys are `uint32_t`, but the tests read them into `std::vector<float>`
(`u_morton_keys_s1_out.begin()..end()` → `vector<float>`), converting each 32-bit
key to float and corrupting any future value comparison. Changed to `uint32_t`
(and, for the OMP path, replaced with the real multiset oracle).

---

## 5. cifar-sparse computes all zeros as shipped (empty CSR) — **DOCUMENTED, worked around in tests**

**File:** `builtin-apps/cifar-sparse/appdata.hpp` (`CSRMatrix`)

`CSRMatrix` zero-initializes `row_ptr` / `col_idx` and the AppData fills only
`values`; **no step ever builds the CSR indices**. Every sparse conv therefore
iterates an empty row range (`row_ptr[oc] == row_ptr[oc+1] == 0`) → output =
bias = 0. Verified at runtime: conv1 output is 0 non-zero of 8.4M elements.

`nnz` is `const int = 0`, so it cannot be corrected in place. The differential
tests build a deterministic valid CSR **in-test** (so the kernel exercises real
sparse conv) and a `GTEST_SKIP` guard test
`CifarSparseAppData.ShippedCsrIndicesAreEmpty_KnownIssue` surfaces the defect.
**Not yet fixed in shipped code** — the AppData should build the CSR (or load real
sparse `.npy` weights, RFC C1; both cifar AppDatas have commented-out
`load_npy_to_ndarray`).

---

## 6. tree radix-tree (stage 4) / octree (stage 7) are order-sensitive — **CHARACTERIZED**

Not a single bug but a real property: `process_radix_tree`/`process_link_leaf`
do cross-node writes under `#pragma omp for`, so `u_brt_parents_s4` and the
octree `children`/`child_leaf_mask` are **not bitwise-stable run-to-run** under
parallel OMP, and CUDA produces a different *valid* representation. The OMP
oracle is pinned single-threaded (`omp_set_num_threads(1)`) to get a canonical
reference; cross-backend these stages need an **invariant / canonical**
comparison (e.g. compare the set of `(parent, child)` edges) rather than exact
buffer equality. Tracked as the remaining tree gap (CUDA tree 5/7).

---

## 7. Vulkan cross-backend findings (first run on rocky-ryzen AMD 780M) — **TRIAGE / DEFERRED**

Vulkan Runners wired for all three apps (same harness). First differential run on
the rocky-ryzen iGPU (`--device minipc`, subgroup 64):

- **cifar-sparse-vk: 9/9 PASS** — the sparse Vulkan path is numerically correct.
- **cifar-dense-vk: FIXED — 9/9.** The dense conv2d and maxpool shaders expect a
  **3D dispatch grid** (`gl_WorkGroupID.z`=batch n, `.y`=channel k/c,
  `.x`=spatial), but `cifar-dense/vulkan/dispatchers.cpp` launched them flat 1D
  (`{div_ceil(total_output,256), 1, 1}`), so `WorkGroupID.z`/`.y` were always 0 →
  only batch-0 / channel-0 was computed; every other output stayed at its
  zero-init (hence `out=0`). Fixed the 5 conv + 3 pool dispatches to
  `{div_ceil(P*Q,256), channels, batch}`; verified 9/9 vs the double-precision
  reference on rocky-ryzen, stable. (cifar-sparse-vk was already correct — its
  dispatcher launches the right grid; only the dense dispatcher was wrong.)
  Unambiguous fix, no tradeoff — committed.
- **tree-vk: stage-2 sort — FIXED on AMD/Xclipse, replaced with a device-wide
  sort.** The original failure (`out[0]=855638110`, unsorted) was a placeholder
  single-workgroup LSD radix sort (`tmp_single_radixsort_*`, Mirco Werner's
  VkRadixSort, 4×8-bit) whose 4 (even) ping-pong passes landed the result back
  in binding-0 while the test read binding-1. Per **TODO(better-sort)**, rather
  than just patch the parity, stage 2 now uses Mirco Werner's **multi-workgroup
  (device-wide) LSD radix sort**: a histogram kernel (`multi_radixsort_histograms`)
  + a scatter kernel (`multi_radixsort_warp{16,32,64}`, selected by
  `get_vulkan_warp_size()`), 256 workgroups, 4 passes, ping-pong
  `s1→tmp→out→tmp→out` so the sorted keys end in `u_morton_keys_sorted_s2_out`.
  - **Verified on 3 subgroup sizes:** AMD Radeon 780M (`minipc`, sg64) PASS;
    Galaxy S24 / Xclipse (`R5CY21Y3VEV`, sg32) PASS; both keep stages 1/3/5/6
    green (4/7 are the §6 structural stages). The warp16 variant is logically
    correct (see Mali note below) but blocked by a Mali engine-layer defect.
  - **Sub-finding (real sync bug the device-wide sort surfaced):** the first
    cut dispatched each of the 8 passes as its own `submit()`+`wait_for_fence()`.
    A fence wait orders execution but does **not** make an SSBO's
    shader-writes *available/visible* to the next submit's shader-reads — that
    needs a pipeline memory barrier. AMD/Xclipse happen to be coherent and
    tolerated it; **Mali-G710 did not** — the scatter pass read a *partially
    written* histogram (probed: non-deterministic ~65/256 workgroups visible,
    run-to-run) → garbage. **Fix:** record all 8 dispatches into **one command
    buffer / one submit** with an explicit `shaderWrite→shaderRead`
    `vk::MemoryBarrier` between each (each pass gets its own descriptor set via
    `num_sets(4)`). This expresses the device-side dependencies correctly on all
    backends instead of relying on incidental driver coherence.
  - stages 4/7 fail = the §6 structural stages (need invariant compare).

- **Mali-G710 host-coherency defect (located, NOT fixed — engine layer).**
  After the device-side barriers were correct, the warp16 (Mali) variant still
  failed with `out[0]=0`, but **the sort is actually correct on Mali**: a probe
  inside `run_stage_2` (right after `wait_for_fence`) reads `out[0]=4441` ==
  reference. gtest, reading the same buffer *after* `run_stage_2` returns, sees
  `0`. Root cause: Mali's host-visible memory is **non-coherent**, and kiss-vk's
  VMA allocator (`VMA_MEMORY_USAGE_AUTO` + `HOST_ACCESS_RANDOM`, mapped) never
  calls `vkInvalidateMappedMemoryRanges` after GPU work, so the host CPU reads
  stale/partially-visible data. This is an **engine-wide** defect (like §1 for
  CUDA), not specific to the sort: the single-dispatch stage-5 (`edge_count`,
  unrelated to sorting) **also fails on Mali**, matching the known
  `cifar-dense-vk 5/10 on Mali-G710` note. The real fix touches the memory
  model (invalidate mapped ranges on GPU→host, or force `HOST_COHERENT`) and
  must be validated across all backends; deferred. AMD/Intel/Xclipse are
  coherent and unaffected.

These are deferred triage items (like §1), not fully fixed here — the point is the
oracle now *runs cross-backend on Vulkan* and pinpoints them per stage. The old
smoke suite reported 35/35 on this box.

## 8. OMP reference (the differential-oracle golden) tree bugs — **FIXED**

The OMP path is the golden every backend is diff-tested against, so its own
correctness is load-bearing. A line-by-line review against the CUDA kernels
found several real defects. Fixed this session (each verified against
`test-tree-omp`, then re-validated `test-tree-vk` 7/7 on rocky-ryzen):

- **Stage-4 over-iteration / OOB read (FIXED).** Both `safe_tree_appdata.cpp`
  (`initialize`) and `omp/dispatchers.cpp` (`run_stage_4`) looped `i` to
  `n_unique`, but brt has only `n_brt_nodes = n_unique - 1` internal nodes. The
  extra last node reads `codes[i+1] = codes[n_unique]` (out of the valid unique
  range) and writes a non-existent brt slot. Bound changed to `n_brt_nodes`
  (matches CUDA `k_BuildRadixTree`). The compared range `[0, n_brt_nodes)` is
  unchanged, so no oracle regression.

- **Stage-7 wrong loop bound (FIXED).** `i` is a *brt* node index but the loop
  ran to `n_octree_nodes` (an octree-node count). Measured: this left **81238 of
  150716** octree nodes with zero writers (a half-empty octree). Bound changed to
  `n_brt_nodes` in both `initialize` and `run_stage_7` (matches CUDA
  `k_MakeOctNodes`). The Vulkan shader + dispatcher were updated to the same
  bound; the now-unused `n_octree_nodes` push constant was removed.

- **Golden never ran `process_link_leaf` (FIXED).** `initialize` (which builds
  the golden buffers) called only `process_oct_node`, so the golden's
  `children` / `child_leaf_mask` were incomplete and the oracle comment claiming
  link-leaf was "completed" was untrue. Added the `process_link_leaf` call to the
  golden loop, matching `run_stage_7`.

- **`process_link_leaf` unguarded parent-walk (FIXED).** Its `while
  (edge_counts[rt_node]==0) rt_node = rt_parents[rt_node];` had no escape, and the
  root brt node has `rt_parents[0]==0` (a never-assigned self-loop) → potential
  infinite loop. Added the same `counter>30` guard `process_oct_node` already
  uses, on both the left and right branches. (CUDA has it on the left branch
  only; this is stricter.)

- **Non-atomic mask OR (FIXED).** `set_child`'s `u_child_node_mask[node_idx] |=
  1<<which_child` is an OR-reduction where several brt nodes target the same
  octree node under `#pragma omp parallel for` — a racy read-modify-write. Made
  it `#pragma omp atomic`. OR is commutative, so the mask is now order-independent
  and a stable oracle target (the GPU shaders use `atomicOr` to match).

- **Octree geometry has no single owner per node (FIXED, all backends).** Even
  with the corrected `n_brt_nodes` bound, octree nodes were written by **more than
  one** brt node, sometimes with *different* levels, and the root (node 0) had no
  writer at all. Root cause: stage 6 uses an **inclusive** `std::partial_sum`, so
  `edge_offset[i]` already includes `edge_count[i]`, yet `process_oct_node` used
  `edge_offset[i]` as brt `i`'s range *start*, treating the range as
  `[edge_offset[i], edge_offset[i]+edge_count[i])`. That shifted every range
  forward by `edge_count[i]`, leaving node 0 a hole and overlapping the next
  non-empty brt node's range, so `oct_cell_size`/`oct_corner` were
  last-writer-wins (nondeterministic and wrong where the two writers disagreed on
  level).

  **Fix (option A — localized, keeps stage 6 inclusive so `CheckStage6` is
  unaffected):** brt `i`'s range *start* is the **exclusive** prefix sum
  `edge_offset[i] - edge_count[i]`. Applied to every "start of a range" read —
  `oct_idx` (own start), `oct_parent` (rt_parent's start), and
  `bottom_oct_idx` (rt_node's start, both link-leaf branches) — in all three
  copies of `process_oct_node` / `process_link_leaf`:
  `omp/func_octree.hpp`, `cuda/07_octree.cu`, and the Vulkan shader
  `kiss-vk/shaders/comp/tree_build_octree.comp`. **Node 0 (root):** with the
  exclusive start, node 0 is owned and written exactly once by the first
  non-empty brt node's chain (brt node 1), which assigns it `cell_size = 16`
  (level 6 relative to `root_level=0`). CUDA's old `threadIdx.x==0` root init
  (`cell_size[0]=range`) was always *overwritten* by that chain write (it runs
  before the main loop, separated by `__syncthreads()`) and disagreed with
  OMP/Vulkan, so it was **removed** — all three now rely solely on the chain
  writer and agree byte-for-byte.

  **Diagnostic proof (temporary, since removed)** over the seeded 150716-node
  octree: INCLUSIVE → **9111** zero-writer holes (incl. node 0), **9107**
  multi-writer collisions; EXCLUSIVE → **0** holes, **0** collisions, node 0
  written exactly once. With one writer per node the geometry is order-independent
  and a valid oracle target, so `tree_diff_oracle.hpp::CheckStage7` now compares
  `oct_cell_size` and `oct_corner` (`NearEqual`) in addition to `child_node_mask`.
  `children`/`child_leaf_mask` remain excluded (their per-slot writes are still
  order-sensitive). Verified: `test-tree-omp` 7/7 (with geometry) locally;
  `test-tree-vk` 7/7 (with geometry) on rocky-ryzen. CUDA: source fix applied and
  cross-compiles clean (`test-tree-cu`, aarch64).

  **CUDA numerical verification is BLOCKED by §1, not by this fix.** Ran the
  cross-built `test-tree-cu` on the Jetson Orin (`duck-naughty`): 4/7 fail, but
  the failure is the §1 managed-memory race, NOT a geometry regression — Stage1
  morton (untouched by this work) already fails with `out=0` from index 0, and
  the first-mismatch index is non-deterministic run-to-run (252032, then 262144),
  the classic partial-visibility signature of §1. The values that DO land are
  correct math. So the CUDA geometry fix cannot be validated on the Jetson until
  §1 (`cudaStreamAttachMemAsync`) is addressed; the source change is correct by
  construction (identical edit to the OMP/Vulkan paths that pass with geometry).

## 9. `bm-baseline-cifar-dense-vk` segfaults on exit (Jetson) — **RESOLVED (2026-06-17)**

> **Resolved:** `~BaseEngine` (`kiss-vk/base_engine.cpp`) only destroyed the VMA
> allocator — the `vk::Device` and `vk::Instance` were leaked, so the Vulkan loader's
> own static/atexit teardown raced and segfaulted on Tegra. The dtor now drains
> (`device_.waitIdle()`, guarded — it can throw) and destroys, in order, the VMA
> allocator → device → instance. Safe because the dispatcher declares `engine` first
> (destroyed last), so the sequences/algorithms (pools) and the engine's memory
> resource (buffers) are already gone when the base dtor runs.
>
> **Verified on Jetson 2026-06-17:** `bm-baseline-cifar-dense-vk`, `test-tree-vk`, and
> `bm-gen-logs-tree-vk` all **exit 0** (were 139/SIGSEGV); `ctest -L vulkan` on rocky
> stays GREEN (tree 7/7, cifar-dense/sparse 10/10). The `03_run_schedule.py`
> `check=False` workaround can stay (harmless) but is no longer required for Jetson VK.

**Symptom (original).** The single-PU baseline on the Jetson
(`./bm-baseline-cifar-dense-vk --device jetson --benchmark_min_time=2s`) prints
**all** results correctly, then **segfaults during process teardown**:

```
OMP/CIFAR-dense/Baseline/LittleCores        196 ms          191 ms           15
VK/Baseline                                59.0 ms         1.84 ms         1000
...Segmentation fault (core dumped)
```

**Impact.** None on the numbers — the crash is *after* the last result is flushed,
on exit, so the measurements are valid (these are the no-framework baselines used
for the schedule-vs-single-PU speedup: pure-CPU 196 ms, pure-GPU 59 ms/task). Only
the process exit code is non-zero, which could trip a CI gate that checks it. The
schedule-execution path (`bm-gen-logs`) and the profiler (`bm-prof`) were **not**
observed to crash.

**Suspected cause (unverified).** Static-destruction order of the kiss-vk Vulkan
engine (`DispatcherT` / VMA pools) vs. google-benchmark global state, or a
double-free in the engine dtor after 1000 GPU iterations on Tegra. The VK baseline
holds the engine for the whole run; OMP-only baselines have not been seen to crash.
May interact with the managed-mem path (§1). Not yet reproduced under a debugger.

**Repro.** Cross-build the `jetson` preset → `scp build/jetson/bm-baseline-cifar-dense-vk`
→ run on `duck-naughty` as above. First seen 2026-06-17 while collecting the
no-framework baseline.

**Next step (deferred by request).** Run under `cuda-gdb`/`gdb` on the Jetson for
the teardown backtrace; likely a one-line dtor-ordering fix in the kiss-vk engine,
or an explicit `disp.reset()` before `benchmark::Shutdown()`.

**Update (2026-06-17): generalizes to `bm-gen-logs-*-vk` on the Jetson.** The same
teardown segfault hits the schedule executor — it prints all `### Python Begin ###`
records, then crashes on exit (non-zero code). This silently broke
`03_run_schedule.py`, which used `subprocess.run(check=True)` and so *discarded the
captured stdout* when the executor exited non-zero (the `jetson_sparse_vk` cell came
back with an empty log). **Worked around:** `03` now runs the executor with
`check=False`, keeping the records regardless of exit code (a teardown crash after
the data is flushed is harmless to the measurement). The underlying engine-dtor
crash is still the real fix; until then, Jetson VK schedule/baseline runs exit
non-zero but produce valid logs.

## 10. Concurrent GPU pipeline chunks share one command buffer → device loss — **GUARDED (2026-06-17)**

**Symptom:** the `PipelineE2EVk.DISABLED_AlternatingBoundary` schedule
`{VK 1-3, OMP 4, VK 5, OMP 6, VK 7}` reproducibly SIGSEGVs on rocky/RADV; the
backtrace lands in `run_stage_1 → cmd_end` after `radv/amdgpu: The CS has been
cancelled because the context is lost`.

**Earlier triage was WRONG.** The 2026-06-17 triage blamed "GPU re-entry into the
data-dependent octree (stale `n_brt_nodes`)". That cannot be it: the test uses
`VkAppData_Safe`, whose `n_unique`/`n_brt_nodes`/`n_octree_nodes` are **`const`,
fixed at construction from the OMP golden** (`safe_tree_appdata.hpp`), so the
stage-7 count `run_stage_7` reads is always correct. The octree was a red herring.

**Real root cause (GPU-assisted validation, rocky):** a concurrent command-buffer
race. Validation reports `VUID-vkBeginCommandBuffer-commandBuffer-00049` —
*"vkBeginCommandBuffer on an active command buffer still in the recording state."*
Each schedule chunk runs on its own worker thread, but **all Vulkan chunks share
one `VulkanDispatcher` → one `Sequence`/command buffer/fence**
(`pipeline_test_runner.hpp` captures `&disp` into every GPU chunk's lambda). With
≥2 Vulkan chunks, two threads record into that one buffer at once → corruption →
`VK_ERROR_DEVICE_LOST` → the next item's `cmd_begin` SIGSEGVs on the lost context.
Crash frequency tracks the **number of concurrent Vulkan chunks**, not octree
re-entry: a single contiguous GPU chunk (`{OMP 1-3, VK 4-7}`, `{VK 1-7}`) never
races; the 3-Vulkan-chunk schedule reliably crashes; the 2-chunk one
(`{VK 1-3, OMP 4-6, VK 7}`) is flaky (passed in our run).

**Why production is unaffected:** the z3 solver assigns exactly one contiguous
chunk per PU, so it never emits a multi-GPU-chunk schedule.

**Fix (guard, not a dispatcher rewrite):** `first_concurrent_gpu_chunk()`
(`pipeline/schedule.hpp`) rejects any schedule that puts a GPU backend in >1 chunk;
`run_pipeline()` calls it up front and fails cleanly instead of racing the GPU.
Making the dispatcher concurrency-safe would only enable a capability nothing uses
(one GPU engine serializes anyway). Unit-tested on pc (`ScheduleGpuReuse.*` in
`test-schedule-omp`); the ex-DISABLED case is now
`PipelineE2EVk.RejectsMultiGpuChunkSchedule`, asserting the rejection (no GPU
needed). Diagnosis path: GPU-AV via
`VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT` on rocky
(validation layers installed 2026-06-17).

## Latent issue noticed (not fixed)

`SETUP_DEFAULT_LAUNCH_PARAMS` (`builtin-apps/common/cuda/helpers.cuh`) declares
`static const auto grid_dim = div_up(TOTAL_ITER, ...)`. Being `static`, the grid
is computed once and **cached across calls** to the same stage function. Harmless
when the per-stage iteration count is fixed (the seeded tests), but wrong if a
stage is dispatched on inputs of different sizes within one process. Left as-is;
noted for the registry/refactor work.

---

## Non-standard `worker<QueueT>` in the `pipe/*-cu/main.cu` cell runners — **PORTABILITY NIT, nvcc-only-accepted (surfaced 2026-06-17; resolved as a side effect of P3)**

**Files:** `pipe/{tree,cifar-dense,cifar-sparse}-cu/main.cu` (the `run-pipe-*-cu`
targets, `CMakeLists.txt:191-193`).

Each calls the pipeline worker as a **template-id**: `std::thread t0(worker<QueueT>, …)`
(`pipe/tree-cu/main.cu:15,23`). But `worker` in `pipe/pipeline_common.hpp:54` is a
**non-template** `static inline void worker(QueueT&, …)` — it has been a plain function
since commit `a72c259` ("lift the duplicated const.hpp plumbing into
pipeline_common.hpp"), which replaced the per-cell templated workers with one
magic-typedef function but did **not** update these `main.cu` call sites.

**Correction to the first triage:** a template-argument-list on a non-template name is
ill-formed in standard C++ — a standalone **g++ -std=c++20** repro rejects it
(`error: expected '(' after template-argument-list`). I initially inferred the cu
targets must be broken. They are **not**: **nvcc accepts** the construct (it tolerates
the bogus `<QueueT>` and binds to the plain `worker`). Verified by a clean docker
cross-rebuild (`CCACHE_DISABLE=1`, source touched, no ccache configured anywhere) —
`run-pipe-tree-cu` builds green, rc=0. So this is a **latent portability nit** (compiles
only because the CUDA toolchain is lenient and is the only toolchain these cells use),
**not** a break. No test inventory is affected (they are `run-` drivers, not ctest tests).

**Resolution:** P3 templatizes `worker` into a real `template<class Queue, class AppData>`
(`runtime/pipeline.hpp`) and updates the call sites to the explicit, standard-conforming
`worker<QueueT, AppDataT>` — which compiles on **both** g++ and nvcc. So P3 removes the
nit mechanically; not a deliberate behavior change.
