# Unit testing — how to run & how to write

> This is the **actionable** testing doc: the commands to run tests on each target
> and the correctness method every new test must follow. For the audit of the
> current suite, the coverage matrix, and the T0–T4 roadmap, see
> [`../reports-for-human/testing-status.md`](../reports-for-human/testing-status.md).
> Device hosts / serials / access shells are in [`01-hardware.md`](01-hardware.md).

## The method — OMP-as-oracle, in-process differential

Each **application** (`tree`, `cifar-dense`, `cifar-sparse`) is a sequence
of **stages**; each stage has a kernel in up to **3 backends** — OMP (CPU), CUDA,
Vulkan (GLSL compute). A correct test proves the three backends compute the *same
correct answer*, not just "it ran without throwing".

The technique: **OMP is the reference oracle, computed in-process on the same
target**. For each `(app, stage S, backend B≠omp)`, from the **same fixed seed**
(`mt19937 gen(114514)`), run stages `1..S` on OMP → `ref`, run `1..S` on B → `out`,
and compare the stage-S output buffer **element-wise**:

- **Integer / structural stages** (tree: morton, sort, unique, radix-tree,
  edge-count, prefix-sum, octree-build) → **exact** (`bt::testing::ExactEqual`; for
  sort, `is_sorted` + permutation oracle).
- **Float stages** (cifar conv / linear) → `bt::testing::NearEqual` with rel+abs
  tolerance (≈ `rtol=1e-4`, `atol=1e-5` for fp32; looser for fp16/tensor-core paths).
  Pool/ReLU are near-exact. On failure, report max-abs-diff and the worst index.

Computing the OMP reference in-process on each target sidesteps x86-vs-aarch64 FP
drift — the binary is self-validating wherever it runs, with no shipped goldens.
Oracle helpers: `platform/util/testing/oracle.hpp`. Canonical model shapes for
cifar: [`04-alexnet-cifar-spec.md`](04-alexnet-cifar-spec.md).

**Hardware gating, not hardware-required:** probe for the CUDA/Vulkan device at
startup and `GTEST_SKIP()` when absent (gtest exits 0 on skip → a skip is a pass).
`--device` parsing is non-fatal: an unknown/missing one self-skips core-pinning
tests instead of aborting the binary.

## Full coverage — the definition of done

"All stages correct" means running the per-stage differential test for **every
(application × backend × hardware) cell that the hardware actually supports**:

```
for app A in {tree, cifar-dense, cifar-sparse}:
  for backend B in {OMP, CUDA, Vulkan}:
    for hardware HW in {Samsung phone, Jetson, MiniPC}:
      if HW supports B:
        run the unit test for (A, B) on HW   # proves every stage of A runs correctly there
```

A cell is **green** only when every stage of the app matches the OMP oracle within
tolerance on that hardware. The `if HW supports B` gate is what keeps the matrix
honest — a missing backend is a **skip, not a pass** (see hardware gating above):

| Hardware | OMP | CUDA | Vulkan |
|---|---|---|---|
| **Samsung phone** (`R5CY21Y3VEV`, subgroup 64; 32..64) | ✅ | — no NVIDIA GPU | ✅ |
| **Jetson Orin** (`duck-stable`, sm_87; twin `duck-naughty` is benchmark-only) | ✅ | ✅ | ✅ |
| **Rocky MiniPC** (`minipc`, Radeon iGPU) | ✅ | — no NVIDIA GPU | ✅ |

Notes that make the gate precise on this fleet:
- **OMP runs everywhere** — it's the in-process oracle, so every HW row is always exercised.
- **CUDA = Jetson only.** No phone/MiniPC has an NVIDIA GPU; the build box's RTX is
  discrete and breaks the CUDA-13 build, so it's **build-only** (compile-checks, no run).
- **Vulkan needs an integrated GPU** (`kiss-vk` hard-selects `eIntegratedGpu`): Jetson,
  the MiniPC iGPU, and the phones run it; the discrete-GPU build box does not. The
  subgroup size is queried from the device at runtime (one radix-sort shader for every
  width): the Pixel is the only subgroup-16 target, rocky-ryzen and the Samsung run 64 and,
  with `BT_VK_SUBGROUP_SIZE=32`, 32 — the Jetson alone can't cover those.

The richer matrix (per-stage counts, the Pixel 7a row, build-only cells, the T0–T4
rollout) lives in [`../reports-for-human/testing-status.md`](../reports-for-human/testing-status.md).

## Running the tests

The per-stage differential tests use the `bt::testing` oracle (one harness per app,
`*_diff_oracle.hpp`; one Runner per backend). Tests are tagged with a backend
`LABELS` so CTest can select them: `-L omp` / `-L cuda` / `-L vulkan`.

Build with the matching preset, then **deploy+run with the `scripts/run-on-*.sh`
helper** — the scripts handle the tmp staging, rocky's fish login shell,
and the adb-stdin gotcha. Full deploy/exec details + error lookup: [`01-hardware.md`](01-hardware.md).

**Local — OMP (the everyday command, no GPU/devices):**
```bash
cmake --preset pc && cmake --build --preset pc
ctest --test-dir build/pc -L omp --output-on-failure        # success: "100% tests passed, 0 failed"
# single binary / subset:
./build/pc/test-cifar-dense-omp --device pc --gtest_filter='*Conv*'
./build/pc/test-tree-omp --gtest_list_tests
```

**CUDA — Jetson Orin (`doremy@duck-stable`):** cross-build in the container, then run.
```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/workspace/build \
  -v "$PWD:/workspace" -w /workspace bt-cross:7.2-glslc bash -lc \
  'cmake --preset jetson && cmake --build --preset jetson --target \
     test-tree-cu test-cifar-dense-cu test-cifar-sparse-cu'
scripts/run-on-jetson.sh                 # deploy to /tmp/bt + run all *-cu
```
(Green as of 2026-06-16 — the `cuda-managed-mem` visibility defect was fixed with
zero-copy pinned memory; see [`../reports-for-human/bugs-found.md`](../reports-for-human/bugs-found.md) §1.
Reverified 2026-06-18: tree-cu 7 / cifar-dense-cu 10 / cifar-sparse-cu 10.
The Jetsons were **reflashed to JetPack 7.2 / CUDA 13.2 on 2026-07-01** and
re-verified on 2026-07-02: all three `test-*-cu` suites (7/10/10) **pass on both
ducks** using the CUDA 12.6 cross binaries — the 12.6-binary-on-7.2-stack path is
correctness-verified (see [`02-building.md`](02-building.md)).)

**EXPERIMENTAL, on-demand only — `test-schedule-permutation-cu`** (tree, CUDA): sweeps
all 29 valid contiguous CPU/GPU stage-split schedules for the tree pipeline on the
genuinely-chained `tree::AppData` dispatch path, checking both per-schedule
correctness (vs. the OMP oracle) and genuine CPU/GPU overlap (>= 3 of 5 repeated runs
must show measured concurrent execution, not serialization). `LABELS "experimental"` —
deliberately excluded from `ctest -L cuda`; build + run it explicitly:
`cmake --build --preset jetson --target test-schedule-permutation-cu` then
`scripts/run-on-jetson.sh test-schedule-permutation-cu`. See
`specs/002-cpu-gpu-schedule-coverage/` for the full spec/plan/tasks.

**Vulkan — rocky-ryzen iGPU (x86, easiest):** build natively, then run.
```bash
cmake --preset vulkan
cmake --build --preset vulkan --target test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk
scripts/run-on-rocky.sh                   # deploy + run all *-vk on the iGPU box
```
All Vulkan suites green as of the 2026-06-16 all-green milestone (reverified on
rocky-ryzen 2026-06-18: tree-vk 7 / cifar-dense-vk 10 / cifar-sparse-vk 10).

**Vulkan validation is opt-in (`BT_VK_VALIDATION=1`).** `kiss_vk::Engine` requests the
Khronos validation layer only when `BT_VK_VALIDATION=1` is set; by default no layer is
requested. If it is set but the layer is not installed, the engine warns and continues
without it. The fleet scripts (`scripts/run-on-rocky.sh`, `optimizer/orchestrate/transport.py`)
*also* export `VK_LOADER_LAYERS_DISABLE='~all~'`, which hides every layer from the loader —
so for a validation run, call the binary directly and **do not** set that variable. The
layer is installed on rocky-ryzen. From the build dir there:
```bash
BT_VK_VALIDATION=1 ./test-tree-vk --device minipc          # validation errors go to stdout
BT_VK_VALIDATION=1 VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT \
  VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true \
  ./test-tree-vk --device minipc                           # + synchronization validation
SPDLOG_LEVEL=debug BT_VK_VALIDATION=1 ./test-kiss-vk       # logs "Vulkan instance layers enabled: ..."
```
Synchronization validation needs `VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true`. Without it the layer
(1.4.328 on rocky, 1.4.341 on the build box) does not check shader accesses through descriptors **at all**. With the
inter-stage barriers removed, `test-tree-vk` fails 9 tests and still prints 0 `SYNC-HAZARD` lines; with the
heuristic it prints 210 (vk-engine-runtime tasks 1.4).
Validation output is printed as `Validation Error: [ VUID-… ]` lines, so grep for that.
Leaving validation off locally means you can miss errors, so run it before touching the engine.

**Subgroup widths.** The engine specializes the subgroup-dependent radix sort with the
subgroup size the device reports, pinned where it can be (rocky-ryzen and the Samsung: 64,
the Pixel 7a: 16; see [`01-hardware.md`](01-hardware.md)). No fleet GPU defaults to 32, so
cover that width by pinning the device's minimum with `BT_VK_SUBGROUP_SIZE` (it throws if
the device cannot pin that size, e.g. 32 on the Pixel):
```bash
BT_VK_SUBGROUP_SIZE=32 ./test-tree-vk --device minipc     # rocky-ryzen, pinned at 32
```

**Mali conv paths and the conv grid contract (mali-conv-kernels, cifar-gpu-kernels).**
- Below subgroup 32 both cifar 3x3 convs run a Mali path: 4 adjacent output pixels per invocation, 4 channels each
  for the dense one. cifar-dense's shader (`new_cifar_dense_conv2d_k3s1p1.comp`) is subgroup-size dependent and at
  32 and above runs the path the 780M and the Xclipse were tuned on. cifar-sparse has two shaders and its dispatcher
  picks by `engine.subgroup_size()`: `new_cifar_sparse_conv2d_k3s1p1.comp` is the Mali path only, and at 32 and above
  it runs `new_cifar_sparse_conv2d_k3s1p1_oc.comp` (one output channel per workgroup row on the packed CSR taps,
  `AppData::conv{1..5}_taps`). The Pixel 7a is the only fleet GPU that takes the Mali paths, so a change to the
  dispatch on it is only tested by the phone gate; rocky-ryzen, the build box and the Samsung cover the other paths.
- `platform/engine/vulkan/shaders/conv_k3s1p1_contract.hpp` gives the invocation counts (`kDenseConvMaliPixels` /
  `kSparseConvMaliPixels` must equal `PX` in the shader): the dense shader's for a subgroup size, the sparse Mali
  shader's (`sparse_conv_k3s1p1_mali_invocations`). Both apps' dispatchers and `DenseConvK3S1P1` / `SparseConvK3S1P1`
  size `grid.x` from it. A contract that counts more outputs per invocation than the shader computes leaves outputs
  at the tests' `-1` fill, so the tests fail; this was checked by mutation on the build box and on the Pixel. That
  only works while each test shape needs more than one 256-wide workgroup on both paths, because the grid rounds up
  to whole workgroups and a shortfall inside the last one goes unseen. Keep the tests' batch and channel counts large
  enough for that. The sparse Mali shader does not depend on the subgroup size, so `SparseConvK3S1P1` tests it on
  every GPU.
- Those tests use 6x6, 8x8 and 7x5 maps (7x5 for an odd width, not a multiple of 4), and the sparse CSR has an empty
  row and a row whose taps all read the padded border at the right edge. `SparseConvPerChannel` checks the `_oc`
  shader the same way against the OMP kernel (plus a top-edge-only row and a row longer than one `STAGE_ROW` chunk),
  and `SparseConvTaps.DecodeToColIdx` checks the packed taps of the five model convs.
- `BT_VK_SPARSE_STAGE_ROW=1` makes the `_oc` shader stage each CSR row in shared memory, 1024 nonzeros at a time
  (its `STAGE_ROW` spec constant). [Unset: off; on the Samsung it was not faster and was slower on conv4 and conv5,
  cifar-gpu-kernels task 1.8.]
- Static Mali analysis: `malioc -c Mali-G710 --vulkan -S 0=16 <shader>.spv` on the glslc output (Arm Performance
  Studio's Mali Offline Compiler; on the build box under `~/tools/arm-performance-studio/`). Measurements and the
  rejected variants: [`../reports-for-human/2026-10-16-mali-conv-kernels.md`](../reports-for-human/2026-10-16-mali-conv-kernels.md).

**Binding paths.** Stage buffers are bound with push descriptors (`VK_KHR_push_descriptor`,
enabled on every fleet GPU) or, where the extension is missing, from a per-`Sequence`
transient descriptor pool. `BT_VK_FORCE_DESCRIPTOR_FALLBACK=1` forces the pool path even
where the extension exists, so one device covers both; `SPDLOG_LEVEL=debug` logs which path
an engine uses (`Descriptor binding: …`):
```bash
BT_VK_FORCE_DESCRIPTOR_FALLBACK=1 ./test-tree-vk --device minipc
```

**Engine knobs (vk-engine-runtime).** All unset by default; the default behaviour is in brackets.
- `BT_VK_BARRIER_ALL=1` puts an inter-stage barrier before every dispatch but a command buffer's first. [Unset: the
  `Sequence` records one only where a dispatch reads or writes a buffer written since the last barrier, or writes one
  read since it. `SPDLOG_LEVEL=debug` logs `Sequence: command buffer with N dispatches, M inter-stage barriers`.]
- `BT_VK_WEIGHTS_PLACEMENT=default|device_local|uncached` overrides the memory type of the weights buffer class, for
  re-measuring placement. [Unset: `uncached` on the Pixel 7a's Mali-G710, where fc1/fc2 measured 31–35 % faster, and
  today's host-cached placement on every other device. Only weights no CPU chunk reads move, and only through a
  schedule-aware placement (`runtime/placement.hpp`); the differential tests keep everything host-cached.]
- **Shader buffer qualifiers** (`readonly` / `writeonly`) are load-bearing: `Sequence` flushes only buffers
  some dispatch accesses and invalidates only those some dispatch writes, and takes that from the SPIR-V
  decorations (a binding the module declares no variable for counts as unused). `just check-shaders`
  (`scripts/check_shader_access.py`, needs glslc and spirv-dis; run in CI) compares each declaration with what
  the optimized SPIR-V does with the buffer; glslang does not catch a `writeonly` buffer that is read.
- `BT_VK_GPU_LOCAL_PLACEMENT=default|device_local|uncached` overrides the memory type of the GPU-local buffer class,
  which holds the buffers every accessor of which is a GPU stage of the schedule (`runtime/placement.hpp`; the tree
  app now declares its buffers like cifar does). [Unset: `uncached` on the Pixel 7a's Mali-G710 (a whole tree GPU
  chunk -7 %, cifar-dense and -sparse -3 %), host-cached elsewhere. The tree input points are declared as weights:
  written once at construction, then only read by stage 1.]
- `BT_VK_BOUNDARY_PLACEMENT=default|device_local|uncached` overrides the memory type of the boundary class (what
  `Engine::get_mr()` returns: every buffer of an AppData built without buffer classes, the tree app's, and the buffers
  a CPU chunk touches), to measure the host cache maintenance of buffers no CPU stage reads. An experiment knob.
  [Unset: host-cached. On the Pixel 7a the tree GPU chunk is 38 % shorter with `uncached`; the Samsung does not
  change. Do not use it with a schedule that has CPU chunks: they would read uncached memory.]
- **`bt::kHostSparse`** (`platform/util/buffer_placement.hpp`) is not a knob but a declaration: in a buffer's accessor
  list it replaces `0` when host code outside the stages only reads the buffer rarely (checks, final results) or a few
  elements per item (a read-back count). It does not keep the buffer in the boundary class; the stages that touch it
  decide (a buffer only the marker touches stays boundary). Reads stay correct in every class, just slower per element
  on an uncached mapping, so a host reader that scans the buffer every item keeps `0`. The tree app declares its
  final octree and the `u_out_idx` / `u_edge_offset_s6` counts this way: a whole tree GPU chunk (stages 1-7, big
  cores) on the Pixel 7a went 29.54 -> 17.79 ms, the Samsung (host-cached GPU-local class) did not change
  (occasional-host-reads, design D4). Hardware-free tests: `Placement.SparseHostReadsDoNotForceBoundary`
  (`test-tree-vk`), `Placement.UndeclaredHostAccessKeepsPlacement` (cifar OMP suites).
- `BT_VK_QUEUE=universal|compute` picks the queue family; `compute` falls back to the universal family, with a
  warning, where none exists (the Pixel). [`universal`: the first family with compute.]
- `BT_VK_QUEUE_PRIORITY=low|medium|high` requests a global queue priority; a missing extension or
  `VK_ERROR_NOT_PERMITTED` (rocky refuses `high`) warns and keeps the default. [The driver's default priority.]
- `BT_VK_INFLIGHT=<n>` lets each Vulkan GPU chunk of `run_pipeline` / `bm-gen-logs` keep n items in flight, handed on
  in order; `run_pipeline` asserts pool ≥ n × GPU chunks + CPU chunks. [1: submit, wait, hand on.]
- `BT_VK_PHASES=1` times every submission's phases (record, submit, wait, …) and verifies the timestamp calibration
  first; `bm-phases-{tree,cifar-dense,cifar-sparse}-vk` turns it on and prints them per chunk, with the bytes the
  flush and invalidate walk and how the shaders use them (`SPDLOG_LEVEL=debug` lists the bound buffers). [Off.]
- `BT_GPU_THREAD_CORES=none|auto|<id>[,<id>...]` decides where the thread that drives a Vulkan GPU chunk (records,
  flushes, submits, waits, invalidates) runs (`runtime/gpu_host_thread.hpp`; OMP chunks are pinned to their tier,
  this thread is not). `<ids>` pin it to exactly those cores; `auto` picks a medium or big core tier no CPU chunk of
  the schedule uses (medium first; none free: unpinned; never a little core). [Unset: the OS places it. Pinning
  measured slower on the Pixel 7a (+3..+13 %) and unstable on the Samsung (-2..-43 %, runs 2x apart), so it is
  opt-in; see `2026-10-16-vk-data-flow-fixes.md`.]
- The solver's `allow_gpu_multi_chunk` (`02_gen_schedule_merged.py --allow-gpu-multi-chunk`) lets z3 give the GPU
  several chunks, priced as their sum. [Off: one chunk per PU.] `optimizer/tests/test_gpu_multi_chunk.py` (`uv run
  pytest -q optimizer`) checks both modes; `PipelineE2EVk.ThreeVulkanChunksAlternating` (`test-pipeline-e2e-vk`) runs
  such a schedule against the oracle.

**Pipeline cache (`BT_VK_PIPELINE_CACHE_DIR`).** Each `kiss_vk::Engine` loads its compiled
pipelines from `<dir>/bt-vk-pipeline-cache-<vendorID>-<deviceID>-<pipelineCacheUUID>.bin`
and saves them back when it is destroyed. `<dir>` is `$BT_VK_PIPELINE_CACHE_DIR`, by default
the current directory: each test's build subdirectory under `build/vulkan` for `ctest`
(`platform/`, `apps/<app>/`), `/tmp/bt` for the rocky and Jetson scripts,
`/data/local/tmp/bt` on the phones. A file from another device or driver, or a
corrupt one, is ignored and the pipelines compile cold; an unwritable directory just means
nothing is saved. The cache only speeds up dispatcher construction, which is outside every
timed region, and the files are safe to delete at any time. `SPDLOG_LEVEL=debug` logs the
load (`Pipeline cache: loaded N bytes from …`, or why it compiled cold) and, per pipeline,
whether the driver reports a cache hit. The files are git-ignored, and `just bench-clean`
also deletes any left in the repo root.

**OMP on Android phones** (both phones' adb is on the build box; run the script there —
`adb -s <serial>` selects the phone):
```bash
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/29.0.14206865
cmake --preset android
cmake --build --preset android --target test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp
scripts/run-on-android.sh 3A021JEHN02756  # Pixel 7a   (on the build box)
scripts/run-on-android.sh R5CY21Y3VEV     # Samsung    (same host, other serial)
```

**What success looks like:** gtest prints `[  PASSED  ] N tests`; `ctest` prints
`100% tests passed, 0 failed`. A `GTEST_SKIP` (backend/device absent) **exits 0 → counts
as a pass**. Each `run-on-*.sh` echoes `== <target> ==` per binary and **exits non-zero if
any target fails**, so its exit code is your CI signal. Verified green (OMP) on all four
targets: PC, Pixel 7a, Jetson, Samsung.

## CI gate

The labeled binaries are identical everywhere; each target runs what it can:
`ctest -L omp` on every PR (desktop, deterministic, no GPU); `ctest -L cuda` on the
Jetson self-hosted runner; `ctest -L vulkan` on an integrated-GPU box / Android via
adb. Fail on any non-skip failure.
