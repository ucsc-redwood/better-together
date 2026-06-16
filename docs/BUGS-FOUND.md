# Bugs surfaced by the differential-oracle test suite

> Record of real defects found while replacing the smoke-level gtest suite with
> OMP-as-oracle differential correctness tests (see [`TESTING.md`](TESTING.md)).
> Each was invisible to the old "did it run / did the buffer change" checks.
> Branch: `refactor/framework-device-axis`.

---

## 1. CUDA managed buffers never stream-attached for the GPU — **CONFIRMED · DEFERRED (TODO, address later)**

> **TODO (deferred by decision):** do not apply an aggressive fix now. The correct
> fix touches the production CUDA memory model and must be validated on the Jetson
> against *both* the sequential differential oracle and the concurrent hybrid
> pipeline. Tracked here and as `TODO(cuda-managed-mem)` in
> `builtin-apps/common/cuda/cu_mem_resource.cu`. The `*-cu` differential tests
> stay red on the Jetson until then — by design.

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
- **tree-vk:** stage-2 **sort** returns garbage (`out=855638110`) — most likely
  the hardcoded `--device`→subgroup map (`minipc`=64) selecting the wrong
  `radixsort_warp{16,32,64}` shader variant for this GPU; this is the
  "warp-size from device string" trap the audit flagged (fix = runtime subgroup
  query, plan T3). stages 4/7 fail = the §6 structural stages (expected; need
  invariant compare).

These are deferred triage items (like §1), not fixed here — the point is the
oracle now *runs cross-backend on Vulkan* and pinpoints them per stage. The old
smoke suite reported 35/35 on this box.

## Latent issue noticed (not fixed)

`SETUP_DEFAULT_LAUNCH_PARAMS` (`builtin-apps/common/cuda/helpers.cuh`) declares
`static const auto grid_dim = div_up(TOTAL_ITER, ...)`. Being `static`, the grid
is computed once and **cached across calls** to the same stage function. Harmless
when the per-stage iteration count is fixed (the seeded tests), but wrong if a
stage is dispatched on inputs of different sizes within one process. Left as-is;
noted for the registry/refactor work.
