# Tasks

Gates used below:
- **"local":** `ctest --test-dir build/pc -L omp` and `uv run pytest -q optimizer`.
- **"phone gate":** `scripts/run-on-android.sh <serial> test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk` on
  3A021JEHN02756 (Pixel 7a) and R5CY21Y3VEV (Samsung).
- **"phone profiling":** as in `docs/instruction-for-ai/06-end-to-end-scheduling.md` Step 1, 20 s between runs; on
  the Samsung `taskset` takes a hex mask. **Nothing is profiled on rocky-ryzen.**

## 1. Pinning (runtime/cpu-chunks)

- [x] 1.1 Add the D2 pinning test for tree, cifar-dense and cifar-sparse. Verify: it fails for tree on the current code (record the observed masks and thread counts) and passes for both cifar apps.
  - *Test:* `bt::testing::ExpectChunkPinned` (`platform/util/testing/cpu_pinning.hpp`), run by
    `CpuPinningOmp.TreeAppData` (`test-tree-omp`, stages 1-7), `CpuPinningOmp.CifarDense` and
    `CpuPinningOmp.CifarSparse` (stages 1-11). On a fresh thread whose default team is the whole machine it runs
    the range unpinned, then through the pinned `dispatch_multi_stage(cores, n, …)` on the first half of the
    process's CPUs and then on the second half (n = the half's size), and checks per pinned run: every thread that
    recorded stage work has a mask inside the list, at most n distinct threads did, and every region equals the
    unpinned run's. The observation is a test hook, `BT_PIN_PROBE_HERE()` (`platform/util/pin_probe.hpp`): it
    records thread id and `sched_getaffinity` once per thread and arming, and compiles to nothing unless
    `BT_PIN_PROBE` is defined. Only the three `test-*-omp` targets define it; `test-tree-omp` compiles its own
    copy of `apps/tree/omp/dispatchers.cpp` for that (it defines every symbol of bt_core's copy, so the linker
    never pulls the latter). Probes: inside every loop body of tree stages 1, 4, 5, 7 and at the top of 2, 3, 6;
    before each stage in the cifar apps' pinned dispatch, after the bind.
  - *Current code (a999b1a plus the test, build box, 12 CPUs):* `CpuPinningOmp.TreeAppData` fails. For cores
    {0-5}, n 6 and for cores {6-11}, n 6 alike, 12 threads ran stage work, each with mask {0-11}: 12 of 12 outside
    the list, 12 > 6 threads. Outputs equal the unpinned run's. `CpuPinningOmp.CifarDense` and
    `CpuPinningOmp.CifarSparse` pass.
- [x] 1.2 Pin tree's AppData path (D1; per-region binding if 1.1 shows pool reuse defeats inheritance). Verify: the test passes for all three apps, the tree OMP and VK/CUDA differential suites still pass under local and the phone gate.
  - *D1 as written fails* (tried first: bind the calling thread, `omp_set_num_threads(n)`, restore both after):
    6 threads ran stage work, the caller with mask {0-5} (then {6-11}) and the 5 pool threads with {0-11}, which
    libgomp created during the unpinned run and reuses. So per-region binding (design D1, "Check needed").
  - *Fix:* each `run_stage_N(tree::AppData&, const CpuPin& = {})`; stages 1, 4, 5, 7 open
    `#pragma omp parallel num_threads(n)`, every team thread binds itself to the cores, then `#pragma omp for`.
    The pinned `dispatch_multi_stage(cores, n, AppData&, …)` binds the calling thread for the single-threaded
    stages 2, 3, 6 with a scope guard (`ScopedThreadAffinity`, `platform/registry/affinity.hpp`) that restores its
    previous mask; `dispatch_stage(cores, n, AppData&, s)` forwards to it. Unpinned calls keep the default team.
  - *local:* `ctest --test-dir build/pc -L omp` 11/11 passed (the three `CpuPinningOmp` tests included);
    `uv run pytest -q optimizer` 41 passed.
  - *phone gate* (`BT_ANDROID_DEST=/data/local/tmp/pin scripts/run-on-android.sh <serial> test-tree-vk
    test-cifar-dense-vk test-cifar-sparse-vk test-tree-omp test-cifar-dense-omp test-cifar-sparse-omp`): Pixel 7a
    and Samsung each 22 / 16 / 17 VK tests passed, and the OMP suites 24 / 16 / 17 passed (the real-weights
    accuracy test skipped, no `BT_WEIGHTS_DIR`), so the three `CpuPinningOmp` tests also pass under the NDK's
    libomp. Exit 0. CUDA is not in either gate (no Jetson is used in this run).
- [x] 1.3 Check the tier records on one phone: time tree stage 1 and stage 2 on the Pixel's little tier with `bm-prof-tree-vk`, and again in a process pinned with `taskset` to the little cores. Verify: the medians agree within the spread (spec "Profiling a tree stage on one tier"), numbers recorded in design.md.
  - Pixel 7a, isolated, `--benchmark_filter='^little/stage[12]/'`, 3 rounds alternating the plain process and
    `taskset f` (little cores 0-3), 20 s before every run. p50 medians, ms: stage 1 2.432 [2.424-2.472] vs
    2.446 [2.391-2.468]; stage 2 26.220 [26.194-26.253] vs 26.242 [26.238-26.270]. Each median lies inside the
    other's spread. Details: design.md, "Measurements".
- [x] 1.4 Remove "Tree's CPU tiers are nominal" from `docs/instruction-for-ai/05-profiling.md`. Verify: the doc says tree pins like the cifar apps.
  - The bullet is replaced by "CPU cells run on their tier, in every app": "Tree pins like the cifar apps", how,
    and the `CpuPinningOmp` guard. The two comments that pointed at the removed "Core-pinning intentionally
    unsupported" note (`test_pipeline_chained.cpp`, `test_schedule_permutation_cu.cu`) now just say they run
    unpinned.

## 2. Refresh and gate (optimizer/cost-model)

- [ ] 2.1 Re-profile tree on both phones (isolated and interference, 2 runs each), moving the previous runs to `superseded-<date>/`. Verify: the loaders accept the store; old vs new CPU p50 per tier and stage recorded here.
- [ ] 2.2 Solve top-10 tree schedules per (phone, table) with the per-stage model and execute them as in the 2026-10-18 gate. Verify: every cell has executions in `data/sched_logs`.
- [ ] 2.3 Apply the spec's gate against `docs/reports-for-human/2026-10-18-cpu-chunk-costs-gate.md`. Verify: a decision record in `docs/reports-for-human/` lists every (phone, table) with before / after CPU error, best period and its run-to-run spread, and states whether each bound holds.

## 3. Final gate

- [ ] 3.1 local and phone gate green at the last commit. Verify: results listed in the PR.
