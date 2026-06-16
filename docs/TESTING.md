# Testing — audit & target architecture

> Status: **audit done, architecture proposed** (2026-06-15). An independent review
> of the current gtest suite, plus the plan for a professional-grade, CI/CD-ready,
> cross-backend correctness suite. Companion to [`CMAKE-MIGRATION-RFC.md`](CMAKE-MIGRATION-RFC.md)
> (which covers the build) and [`BUILD.md`](BUILD.md) (how to build/run).

## Running the tests

The per-stage differential tests (`bt::testing` oracle: OMP is the reference,
exact for integer/structural stages, `NearEqual` for float). One harness per app
(`*_diff_oracle.hpp`), one Runner per backend.

**Local — OMP (the everyday command, no GPU/devices):**
```bash
cmake --preset pc && cmake --build --preset pc
ctest --test-dir build/pc -L omp --output-on-failure        # 5/5 green
# single binary / subset:
./build/pc/test-cifar-dense-omp --device pc --gtest_filter='*Conv*'
./build/pc/test-tree-omp --gtest_list_tests
```
`--device` is non-fatal: an unknown/missing one self-skips core-pinning tests.

**CUDA — Jetson Orin (`duck-naughty`):** cross-build in the container, copy, run.
```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/workspace/build \
  -v "$PWD:/workspace" -w /workspace bt-cross:6.1 bash -lc \
  'cmake --preset jetson && cmake --build --preset jetson --target \
     test-tree-cu test-cifar-dense-cu test-cifar-sparse-cu'
scp build/jetson/test-*-cu duck-naughty:~/bt-omp-test/
ssh duck-naughty 'cd ~/bt-omp-test && ./test-cifar-dense-cu --device jetson'
```
(Currently red — blocked by `TODO(cuda-managed-mem)`, see BUGS-FOUND.md §1.)

**Vulkan — rocky-ryzen iGPU (x86, easiest):** build natively, copy x86 binaries,
run there (its login shell is fish → wrap in `bash -lc`).
```bash
cmake --preset vulkan
cmake --build --preset vulkan --target test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk
scp build/vulkan/test-*-vk doremy@rocky-ryzen:~/bt-vk-test/
ssh doremy@rocky-ryzen 'bash -lc "cd ~/bt-vk-test && LD_LIBRARY_PATH=. ./test-cifar-dense-vk --device minipc"'
```
cifar-dense-vk / cifar-sparse-vk = 9/9; tree-vk has the sort + 4/7 TODOs (§7).

**OMP on Android phones** (Pixel 7a on this box; Samsung on rocky-ryzen):
```bash
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/29.0.14206865
cmake --preset android
cmake --build --preset android --target test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp
LIBCXX=$(find "$ANDROID_NDK_HOME" -name libc++_shared.so -path '*aarch64*' | head -1)
adb -s 3A021JEHN02756 push build/android/test-*-omp "$LIBCXX" /data/local/tmp/bt/
adb -s 3A021JEHN02756 shell 'cd /data/local/tmp/bt && chmod 755 test-* && \
  LD_LIBRARY_PATH=. ./test-tree-omp --device 3A021JEHN02756'
```

Host/serial/access details (Jetson `yanwen@duck-naughty`, Samsung via
`doremy@rocky-ryzen`, NDK path, libc++) are also in `BUILD.md` and the device
inventory. Verified green (OMP) on all four targets: PC, Pixel 7a, Jetson, Samsung.

## The goal

Each **application** (`tree`, `cifar-dense`, `cifar-sparse`; more later) is a sequence
of **stages**; each stage has a kernel implemented in **3 backends** — OMP (CPU),
CUDA, Vulkan (GLSL compute). We want unit tests that **guarantee every stage's
kernel is correct in every backend, on every target HW** (Jetson, Android phones,
desktop), and that run as a **CI/CD gate**.

## Audit verdict (independent review): not robust yet

The current suite is almost entirely **smoke + shape + "buffer changed"** checks.
- **0** numerical-correctness assertions (`EXPECT_NEAR`/`EXPECT_FLOAT_EQ` across all 13 files).
- **0** cross-backend / golden-oracle comparisons — each backend tests itself in isolation.
- Consequence: a CUDA/Vulkan kernel that produces **wrong-but-nonzero** output passes everything.
- The only file with real correctness checks (`octree/omp/test_sort.cpp`) is **not registered in CMake**.

So the suite proves "kernels run without throwing and write *something*", not "the
three backends compute the *same correct* answer".

### Inventory

| File | App × Backend | #TEST | Assertion style |
|---|---|---|---|
| `builtin-apps/cifar-dense/omp/test_main.cpp` | cifar-dense × OMP | 12 | dims + no-throw + is_different |
| `builtin-apps/cifar-dense/cuda/test_main.cu` | cifar-dense × CUDA | 17 | dims + no-throw + is_different + mixing |
| `builtin-apps/cifar-dense/vulkan/test_main.cpp` | cifar-dense × Vulkan | 10 | dims + no-throw + is_different |
| `builtin-apps/cifar-sparse/omp/test_main.cpp` | cifar-sparse × OMP | 9 | dims + no-throw |
| `builtin-apps/cifar-sparse/cuda/test_main.cu` | cifar-sparse × CUDA | 17 | dims + no-throw + is_different + mixing |
| `builtin-apps/cifar-sparse/vulkan/test_main.cpp` | cifar-sparse × Vulkan | 9 | **pure no-throw smoke** |
| `builtin-apps/tree/omp/test_main.cpp` | tree × OMP | 7 | no-throw + 1 is_different (stage 1) |
| `builtin-apps/tree/cuda/test_main.cu` | tree × CUDA | 13 | no-throw + is_different + mixing |
| `builtin-apps/tree/vulkan/test_main.cpp` | tree × Vulkan | 14 | no-throw + is_different + 5s loop |
| `builtin-apps/octree/omp/test_main.cpp` | octree × OMP | 8 | **no-throw smoke (real asserts commented out)** |
| `builtin-apps/octree/omp/test_sort.cpp` | radix sort × OMP | 17 | **REAL correctness (sorted + permutation)** |
| `builtin-apps/octree/vulkan/test_main.cpp` | octree × Vulkan | 7 | **pure no-throw smoke** |
| `builtin-apps/common/kiss-vk/test_main.cpp` | Vulkan engine | 2 | no-throw |

Only `test_sort.cpp` has genuine value checks (`is_sorted` + element-wise permutation +
`std::sort` oracle, e.g. `:84`, `:95`, `:652`). It is **xmake-only**, absent from `CMakeLists.txt`.

### #1 finding — the oracle gap

No cross-backend or golden comparison exists anywhere. Worse, the infrastructure is
already there and unused:
- Inputs are **deterministically seeded** (`cifar-dense/appdata.hpp:60` `mt19937 gen(114514)`,
  `tree/tree_appdata.cpp:53` same seed) — an oracle would be cheap and reproducible.
- The **"Mixing" tests** (`cifar-dense/cuda/test_main.cu:302-370`) already run CUDA and OMP
  stages on the *same* `AppData` — the perfect differential setup — yet only assert `EXPECT_NO_THROW`.
- The only output check, `EXPECT_TRUE(is_different)` (`!ranges::equal(before, after)`), passes
  for garbage/NaN/transposed/off-by-one output as long as the buffer is written.

### Concrete bugs / risks (file:line)

| Issue | Location | Nature |
|---|---|---|
| cifar-sparse batch mismatch: `BATCH_SIZE=512` vs test `kTestBatchSize=128` | `cifar-sparse/appdata.hpp:57` vs `cuda/test_main.cu:16,23` | constant duplicated from appdata → silent drift |
| `uint32_t` morton keys read into `std::vector<float>` | `tree/omp/test_main.cpp:16`, `tree/vulkan:18`, `tree/cuda:18` | wrong read; corrupts any future value check |
| `--device` is `->required()` + validated against a hardcoded allow-list; warp size derived from the device string | `app.hpp:69`, `app.cpp:5,27` | CI portability trap — a new board must be added to source or all tests abort |
| 5-second `sleep_for` + unsynchronized `std::queue` (can pop empty → UB) | `tree/vulkan/test_main.cpp:270` | flaky; wastes 5s/run |
| best tests (radix sort) not registered in CMake | `octree/omp/test_sort.cpp` | real correctness never runs in the CMake/CI path |
| octree real assertions commented out; octree CUDA has no test file | `octree/omp/test_main.cpp:28-187` | coverage hole |

### Coverage gaps
- Numerical correctness: **every stage × every backend uncovered**.
- `cifar-sparse/vulkan` weakest (pure no-throw). octree backends are smoke-only; octree CUDA untested.
- GPU sort never checked for sortedness/permutation (only indirect `is_different`).

## Target architecture (professional-grade, cross-backend, CI/CD)

### 1. OMP-as-oracle differential testing (the core)
Make OMP the reference. For each (app, stage S, backend B≠omp): from the **same fixed
seed**, run stages `1..S` on OMP → `ref`, run `1..S` on B → `out`, compare the stage-S
output buffer **element-wise**. Reuse the existing "Mixing" plumbing that already shares
`AppData` across backends.

### 2. Tolerance policy by stage type
- **Integer / structural** (tree: morton, sort, unique, radix-tree, edge-count, prefix-sum,
  octree-build): **exact** `EXPECT_EQ` element-wise; for sort, reuse `is_sorted` + permutation
  oracle from `test_sort.cpp`.
- **Float** (cifar: conv / linear): `EXPECT_NEAR` with rel+abs tolerance (≈ `rtol=1e-4, atol=1e-5`
  for fp32; looser per-backend if fp16/tensor-cores are used). Pool/ReLU ≈ near-exact.
- On failure, report max-abs-diff and the worst index for debuggability.

### 3. Golden data (optional second tier)
Live OMP oracle needs no stored goldens. For a reference *independent* of the OMP
implementation, generate goldens from a committed PyTorch/NumPy script under
`tests/golden/<app>/<stage>` (keyed by the fixed seed, kept tiny), loaded via a
`BT_GOLDEN_DIR` CMake var.

### 4. Hardware gating (not hardware-required)
Replace the hard `->required()` + abort with a runtime probe: detect CUDA/Vulkan device at
startup and `GTEST_SKIP()` if absent (as `test_sort.cpp` already does for cores). Decouple
warp size from the `--device` allow-list — query it from the device at runtime.

### 5. CTest labels + CI gate
- Register every test with a `LABELS` of `omp` / `cuda` / `vulkan` in CMake; set
  `SKIP_RETURN_CODE` so skips aren't failures.
- CI: `ctest -L omp` on every PR (desktop, deterministic, no GPU); `ctest -L cuda` on a
  Jetson self-hosted runner; `ctest -L vulkan` on an integrated-GPU box / Android via adb.
  Fail on any non-skip failure.

## Plan — full (app × stage × backend × target) correctness coverage

> Goal restated concretely: **guarantee every stage of every application is
> numerically correct on every backend, on every target hardware**, as a CI gate.
> The unit of coverage is one *cell* = "stage S of app A on backend B, executed on
> target T, matches the OMP oracle within tolerance".

### T0 — foundation (LANDED, verified `ctest -L omp` 5/5 on the `pc` preset)
- `builtin-apps/common/testing/oracle.hpp` — `bt::testing::ExactEqual` (integer/
  structural, first-mismatch report) and `NearEqual` (float, rtol/atol, worst-
  element + max-abs-diff, NaN/Inf-aware). Two-range templates (`pmr::vector` vs
  `vector`).
- Bug fixes: `kTestBatchSize` → `AppData::BATCH_SIZE` (kills the cifar-sparse
  128-vs-512 drift); `uint32_t` morton reads (was `vector<float>`); missing
  `#include <algorithm>` in `octree/appdata.hpp` (surfaced by the gcc path).
- `test-sort-omp` + `test-octree-omp` registered in CMake (were xmake-only);
  every test tagged with a backend `LABELS` (`omp`/`cuda`/`vulkan`). gtest exits 0
  on `GTEST_SKIP`, so a skip is a pass — no `SKIP_RETURN_CODE` needed.
- `parse_args_test()` — non-fatal device parse: an unknown/missing `--device`
  warns and the relevant tests self-skip instead of `exit(1)` aborting the binary.
- First real cross-stage assertion: tree Stage-1 morton as a **multiset oracle**
  (this is the template the rest adopt).

### A. The coverage matrix (definition of done)

Stages per app and compare mode: **tree** 7 (exact), **octree** 7 (exact),
**cifar-dense** / **cifar-sparse** 9 today → **11** under the canonical model
(see §E), float/`near`.

Targets of record: **PC** (i9 + RTX 4070 Ti S, dGPU, CUDA 13), **Jetson Orin**
(sm_87, CUDA 12.6 + Vulkan), **Android-A = Pixel 7a** (Mali-G710, subgroup **16**),
**Android-B = Samsung `R5CY21Y3VEV`** (subgroup **32**). Optional CI helper: the
**rocky-ryzen iGPU** box (Vulkan, subgroup 64).

| app·backend | PC (dGPU, CUDA13) | Jetson Orin | Pixel 7a (Mali/16) | Samsung (/32) |
|---|---|---|---|---|
| **·OMP** (the oracle) | ✅ run | ✅ | ✅ | ✅ |
| tree·CUDA | build-only¹ | ✅ | — | — |
| tree·Vulkan | ✗ dGPU² | ✅ subgroup 32 | ✅ **subgroup 16** | ✅ subgroup 32 |
| octree·CUDA | ∅ no kernels³ | ∅ | ∅ | ∅ |
| octree·Vulkan | ✗² | ✅ | ✅ 16 | ✅ 32 |
| cifar-d/s·CUDA | build-only¹ | ✅ | — | — |
| cifar-d/s·Vulkan | ✗² | ✅ | ✅ | ✅ |

¹ CUDA 13 on the PC breaks the build (CUB removal) → PC is **build-only** via the
cross container; CUDA *runs* on the Jetson. ² The Vulkan engine hard-selects an
integrated GPU, so the discrete-GPU PC never runs Vulkan — the phones (and the
iGPU box) do. ³ octree has **no CUDA backend** (only OMP + Vulkan kernels exist);
this cell is empty by design, not a coverage gap. **The two phones are what
uniquely exercise the subgroup-16 (Mali) vs subgroup-32 shader variants**
(`tmp_single_radixsort_warp{16,32,64}`) — the Jetson alone cannot.

### B. The harness — one parametrized differential test, not 32×3 hand-written

**OMP-in-process oracle (the key choice).** Every target has a CPU, so each test
binary computes the OMP reference *in-process on that same target* and compares
the backend's stage output against it. Integer stages → exact everywhere; float
stages → `NearEqual` vs the *same-target* CPU, which sidesteps x86-vs-aarch64 FP
drift entirely. No shipped goldens, no PC↔device transfer of expected values — the
binary is self-validating wherever it runs. (`SafeAppData`'s golden buffers
already are this in-process reference for tree; cifar needs the same two-AppData
pattern, reusing the existing "Mixing" plumbing.)

Components (land once, reused across the whole matrix):
- **`oracle.hpp`** — done.
- **Per-app `StageSpec`**: stage i → `{name, ref-accessor, out-accessor,
  valid-length, mode (exact|near), tol}`. This encodes the per-stage gotchas —
  tree Stage-1 is a multiset; unique uses `n_unique`; brt uses `n_brt_nodes`;
  octree uses `n_octree_nodes`; **and a golden buffer must be checked for survival
  of later in-place ops** (`initialize()` sorts `u_morton_keys_s1` in place, so the
  Stage-1 golden is the *sorted* `u_morton_keys_sorted_s2`, not `u_morton_keys_s1`).
  This descriptor is the **seed of the Stage/Kernel registry** (REARCHITECTURE
  Phase 4) — test work and framework work converge here.
- **`TEST_P` over stage index**, instantiated per (app, backend). Four harness
  files (tree / octree / cifar-dense / cifar-sparse) × backend cover the matrix
  from ~4 templates instead of 32×3 copies.
- **Hardware gating**: probe the device and `GTEST_SKIP()` when the backend is
  absent (as `test_sort.cpp` already does for cores); **query the Vulkan subgroup
  size at runtime** (`VkPhysicalDeviceSubgroupProperties`) to replace the
  hardcoded `--device`→warp map — so a new board selects the correct shader
  variant without a source edit (today `get_vulkan_warp_size()` reads the device
  string; this is a real correctness dependency for Android).

### C. Per-target execution — targets are a runner concern, not code

The labeled binaries are identical everywhere; each target runs what it can:
- **PC** — `ctest -L omp` on every PR; build-only CUDA/Vulkan through the cross
  toolchain (catches compile/link regressions without a GPU).
- **Jetson** (self-hosted) — `Dockerfile.cross` build → `scp` → `ctest -L 'cuda|vulkan'`.
- **Android ×2** — `android` (NDK) preset build → `adb -s <serial> push` the
  binary + `libc++_shared.so` to `/data/local/tmp` → run with
  `--gtest_output=xml` → collect. A `scripts/run-on-android.sh <serial>` wrapper
  (the `justfile` already has the adb scaffolding for `3A021JEHN02756` and
  `R5CY21Y3VEV`).
- **iGPU box** (optional) — a Vulkan CI runner that needs no phone for Tier-0.

### D. Phases (each shippable; marked where verifiable)

| Phase | Work | Done when | Verifiable on |
|---|---|---|---|
| **T1 tree** | `StageSpec` + `TEST_P` harness; exact compare, 7 stages; wire OMP(self)/CUDA/Vulkan | tree 7×3 differential green | OMP=PC now · CUDA=Jetson · VK=Jetson+phones |
| **T2 cifar** | Same harness, `near` (rtol 1e-4 / atol 1e-5); dense+sparse × 3 backends | cifar stages within tol | OMP=PC · CUDA=Jetson · VK=Jetson+phones |
| **T2b octree** | Exact, 7 stages, OMP+Vulkan (CUDA ∅); un-comment the real octree asserts | octree 7×2 green | OMP=PC · VK=Jetson+phones |
| **T3 Android + CI** | **Runtime Vulkan subgroup query**; `run-on-android.sh`; redesign the 5 s sleep/queue test; per-runner CI gates | every matrix cell green on its target; CI fails on any non-skip failure | all targets |
| **T4 (opt)** | Independent PyTorch/NumPy goldens (`BT_GOLDEN_DIR`) as a second tier vs a systematically-wrong OMP | golden tier agrees | PC |

T1 is the template that de-risks the rest; T3 is what actually lights up the two
phones.

### E. Interaction with the canonical-model migration (cifar)

[`ALEXNET-CIFAR-SPEC.md`](ALEXNET-CIFAR-SPEC.md) makes **`AlexNetCIFAR` (11 stages)**
canonical, but the C++ kernels still implement the old **`SmallAlexNet` (9 stages,
1024→10 single FC)** — the two are **not** shape/weight-compatible. Adopting the
canonical model re-shapes `AppData` and every cifar `run_stage_*`.

The parametrized harness is **model-agnostic**: it iterates whatever stages the
`StageSpec` declares, so the 9→11 change is a descriptor + tolerance update, not a
test rewrite. Recommended sequencing: build the cifar harness now against the
current 9 stages (proves the float-tolerance path end-to-end and guards the
re-shape itself), then re-point the `StageSpec` when the 11-stage kernels land —
the differential test then *validates the migration* instead of being invalidated
by it. tree and octree are geometry and unaffected.

### F. Open decisions / known gaps

- **octree·CUDA = ∅** — accepted: no kernels exist; not counted as a gap. (Implementing
  them is a *feature*, out of test scope.)
- **Vulkan subgroup**: move from the `--device`→warp map to a runtime query (T3) —
  required for any board not already in the map; also removes a CI portability trap.
- **subgroup-64 (Adreno)** is covered only by the optional iGPU box, not by the two
  chosen phones (Pixel 7a/16 + Samsung/32). Add an Adreno phone later if that path
  needs device coverage.
- **CUDA/Vulkan test `main()`s** still call the fatal `parse_args`; switch them to
  `parse_args_test` in T1/T2 when they compile on-target.
- **Float tolerance** (`rtol`/`atol`) is provisional; confirm per stage, and loosen
  for any fp16/tensor-core path.

Once T1–T3 land, the suite is a real **correctness gate** across backends and
hardware — "the three backends compute the same correct answer on every target" —
instead of a "did it crash" gate.
