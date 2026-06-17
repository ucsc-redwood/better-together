# Unit testing — how to run & how to write

> This is the **actionable** testing doc: the commands to run tests on each target
> and the correctness method every new test must follow. For the audit of the
> current suite, the coverage matrix, and the T0–T4 roadmap, see
> [`../reports-for-human/testing-status.md`](../reports-for-human/testing-status.md).
> Device hosts / serials / access shells are in [`01-hardware.md`](01-hardware.md).

## The method — OMP-as-oracle, in-process differential

Each **application** (`tree`, `cifar-dense`, `cifar-sparse`; `octree`) is a sequence
of **stages**; each stage has a kernel in up to **3 backends** — OMP (CPU), CUDA,
Vulkan (GLSL compute). A correct test proves the three backends compute the *same
correct answer*, not just "it ran without throwing".

The technique: **OMP is the reference oracle, computed in-process on the same
target**. For each `(app, stage S, backend B≠omp)`, from the **same fixed seed**
(`mt19937 gen(114514)`), run stages `1..S` on OMP → `ref`, run `1..S` on B → `out`,
and compare the stage-S output buffer **element-wise**:

- **Integer / structural stages** (tree/octree: morton, sort, unique, radix-tree,
  edge-count, prefix-sum, octree-build) → **exact** (`bt::testing::ExactEqual`; for
  sort, `is_sorted` + permutation oracle).
- **Float stages** (cifar conv / linear) → `bt::testing::NearEqual` with rel+abs
  tolerance (≈ `rtol=1e-4`, `atol=1e-5` for fp32; looser for fp16/tensor-core paths).
  Pool/ReLU are near-exact. On failure, report max-abs-diff and the worst index.

Computing the OMP reference in-process on each target sidesteps x86-vs-aarch64 FP
drift — the binary is self-validating wherever it runs, with no shipped goldens.
Oracle helpers: `builtin-apps/common/testing/oracle.hpp`. Canonical model shapes for
cifar: [`04-alexnet-cifar-spec.md`](04-alexnet-cifar-spec.md).

**Hardware gating, not hardware-required:** probe for the CUDA/Vulkan device at
startup and `GTEST_SKIP()` when absent (gtest exits 0 on skip → a skip is a pass).
`--device` parsing is non-fatal: an unknown/missing one self-skips core-pinning
tests instead of aborting the binary.

## Full coverage — the definition of done

"All stages correct" means running the per-stage differential test for **every
(application × backend × hardware) cell that the hardware actually supports**:

```
for app A in {tree, cifar-dense, cifar-sparse}:       # (+ octree, structural, OMP+VK only)
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
| **Jetson Orin** (`jetson`, sm_87) | ✅ | ✅ | ✅ |
| **Rocky MiniPC** (`minipc`, Radeon iGPU) | ✅ | — no NVIDIA GPU | ✅ |

Notes that make the gate precise on this fleet:
- **OMP runs everywhere** — it's the in-process oracle, so every HW row is always exercised.
- **CUDA = Jetson only.** No phone/MiniPC has an NVIDIA GPU; the build box's RTX is
  discrete and breaks the CUDA-13 build, so it's **build-only** (compile-checks, no run).
- **Vulkan needs an integrated GPU** (`kiss-vk` hard-selects `eIntegratedGpu`): Jetson,
  the MiniPC iGPU, and the phones run it; the discrete-GPU build box does not. The two
  phones uniquely exercise the **subgroup-16 (Pixel/Mali) vs subgroup-32 (Samsung)** shader
  variants — the Jetson alone can't cover those.
- **octree** is the structural 4th app (OMP + Vulkan kernels only — no CUDA backend
  exists, so those cells are ∅ by design, not a gap).

The richer matrix (per-stage counts, the Pixel 7a row, build-only cells, the T0–T4
rollout) lives in [`../reports-for-human/testing-status.md`](../reports-for-human/testing-status.md).

## Running the tests

The per-stage differential tests use the `bt::testing` oracle (one harness per app,
`*_diff_oracle.hpp`; one Runner per backend). Tests are tagged with a backend
`LABELS` so CTest can select them: `-L omp` / `-L cuda` / `-L vulkan`.

Build with the matching preset, then **deploy+run with the `scripts/run-on-*.sh`
helper** — the scripts handle the tmp staging, the fish login shells (Jetson + rocky),
and the adb-stdin gotcha. Full deploy/exec details + error lookup: [`01-hardware.md`](01-hardware.md).

**Local — OMP (the everyday command, no GPU/devices):**
```bash
cmake --preset pc && cmake --build --preset pc
ctest --test-dir build/pc -L omp --output-on-failure        # success: "100% tests passed, 0 failed"
# single binary / subset:
./build/pc/test-cifar-dense-omp --device pc --gtest_filter='*Conv*'
./build/pc/test-tree-omp --gtest_list_tests
```

**CUDA — Jetson Orin (`duck-naughty`):** cross-build in the container, then run.
```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/workspace/build \
  -v "$PWD:/workspace" -w /workspace bt-cross:6.1 bash -lc \
  'cmake --preset jetson && cmake --build --preset jetson --target \
     test-tree-cu test-cifar-dense-cu test-cifar-sparse-cu'
scripts/run-on-jetson.sh                 # deploy to /tmp/bt + run all *-cu
```
(Currently red — blocked by `TODO(cuda-managed-mem)`, see
[`../reports-for-human/bugs-found.md`](../reports-for-human/bugs-found.md) §1.)

**Vulkan — rocky-ryzen iGPU (x86, easiest):** build natively, then run.
```bash
cmake --preset vulkan
cmake --build --preset vulkan --target test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk
scripts/run-on-rocky.sh                   # deploy + run all *-vk on the iGPU box
```
cifar-dense-vk / cifar-sparse-vk = 9/9; tree-vk has the sort + 4/7 TODOs.

**OMP on Android phones** (Pixel 7a on the build box; Samsung on rocky-ryzen):
```bash
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/29.0.14206865
cmake --preset android
cmake --build --preset android --target test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp
scripts/run-on-android.sh 3A021JEHN02756  # Pixel; for Samsung run the same script on rocky-ryzen
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
