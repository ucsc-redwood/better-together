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
| **Samsung phone** (`R5CY21Y3VEV`, subgroup 32) | ✅ | — no NVIDIA GPU | ✅ |
| **Jetson Orin** (`duck-stable`, sm_87; twin `duck-naughty` is benchmark-only) | ✅ | ✅ | ✅ |
| **Rocky MiniPC** (`minipc`, Radeon iGPU) | ✅ | — no NVIDIA GPU | ✅ |

Notes that make the gate precise on this fleet:
- **OMP runs everywhere** — it's the in-process oracle, so every HW row is always exercised.
- **CUDA = Jetson only.** No phone/MiniPC has an NVIDIA GPU; the build box's RTX is
  discrete and breaks the CUDA-13 build, so it's **build-only** (compile-checks, no run).
- **Vulkan needs an integrated GPU** (`kiss-vk` hard-selects `eIntegratedGpu`): Jetson,
  the MiniPC iGPU, and the phones run it; the discrete-GPU build box does not. The two
  phones uniquely exercise the **subgroup-16 (Pixel/Mali) vs subgroup-32 (Samsung)** shader
  variants — the Jetson alone can't cover those.

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
  -v "$PWD:/workspace" -w /workspace bt-cross:7.2 bash -lc \
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

**OMP on Android phones** (both phones' adb is on rocky-ryzen now; copy `build/android`
there and run the script on rocky — `adb -s <serial>` selects the phone):
```bash
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/29.0.14206865
cmake --preset android
cmake --build --preset android --target test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp
scripts/run-on-android.sh 3A021JEHN02756  # Pixel 7a   (on rocky-ryzen)
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
