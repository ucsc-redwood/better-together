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

## Running the tests

The per-stage differential tests use the `bt::testing` oracle (one harness per app,
`*_diff_oracle.hpp`; one Runner per backend). Tests are tagged with a backend
`LABELS` so CTest can select them: `-L omp` / `-L cuda` / `-L vulkan`.

**Local — OMP (the everyday command, no GPU/devices):**
```bash
cmake --preset pc && cmake --build --preset pc
ctest --test-dir build/pc -L omp --output-on-failure        # 5/5 green
# single binary / subset:
./build/pc/test-cifar-dense-omp --device pc --gtest_filter='*Conv*'
./build/pc/test-tree-omp --gtest_list_tests
```

**CUDA — Jetson Orin (`yanwen@duck-naughty`):** cross-build in the container, copy, run.
```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/workspace/build \
  -v "$PWD:/workspace" -w /workspace bt-cross:6.1 bash -lc \
  'cmake --preset jetson && cmake --build --preset jetson --target \
     test-tree-cu test-cifar-dense-cu test-cifar-sparse-cu'
scp build/jetson/test-*-cu duck-naughty:~/bt-omp-test/
ssh duck-naughty 'cd ~/bt-omp-test && ./test-cifar-dense-cu --device jetson'
```
(Currently red — blocked by `TODO(cuda-managed-mem)`, see
[`../reports-for-human/bugs-found.md`](../reports-for-human/bugs-found.md) §1.)

**Vulkan — rocky-ryzen iGPU (x86, easiest):** build natively, copy x86 binaries,
run there (its login shell is fish → wrap in `bash -lc`).
```bash
cmake --preset vulkan
cmake --build --preset vulkan --target test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk
scp build/vulkan/test-*-vk doremy@rocky-ryzen:~/bt-vk-test/
ssh doremy@rocky-ryzen 'bash -lc "cd ~/bt-vk-test && LD_LIBRARY_PATH=. ./test-cifar-dense-vk --device minipc"'
```
cifar-dense-vk / cifar-sparse-vk = 9/9; tree-vk has the sort + 4/7 TODOs.

**OMP on Android phones** (Pixel 7a on the build box; Samsung on rocky-ryzen):
```bash
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/29.0.14206865
cmake --preset android
cmake --build --preset android --target test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp
LIBCXX=$(find "$ANDROID_NDK_HOME" -name libc++_shared.so -path '*aarch64*' | head -1)
adb -s 3A021JEHN02756 push build/android/test-*-omp "$LIBCXX" /data/local/tmp/bt/
adb -s 3A021JEHN02756 shell 'cd /data/local/tmp/bt && chmod 755 test-* && \
  LD_LIBRARY_PATH=. ./test-tree-omp --device 3A021JEHN02756'
```

Verified green (OMP) on all four targets: PC, Pixel 7a, Jetson, Samsung. Host names,
serials, and access shells: [`01-hardware.md`](01-hardware.md).

## CI gate

The labeled binaries are identical everywhere; each target runs what it can:
`ctest -L omp` on every PR (desktop, deterministic, no GPU); `ctest -L cuda` on the
Jetson self-hosted runner; `ctest -L vulkan` on an integrated-GPU box / Android via
adb. Fail on any non-skip failure.
