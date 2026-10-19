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

- [x] 2.1 Re-profile tree on both phones (isolated and interference, 2 runs each), moving the previous runs to `superseded-<date>/`. Verify: the loaders accept the store; old vs new CPU p50 per tier and stage recorded here.
  - *Run (2026-10-19 20:54 UTC, both phones at once, each phone's runs in sequence):* the 2026-10-18 runs moved to
    `data/profiling/<serial>/tree/vulkan/<scenario>/superseded-2026-10-18/` first, then `uv run
    optimizer/orchestrate/01_collect_profiling.py --device <serial> --app tree --backend vk --adb-serial <serial>
    --build-dir build/android --dest /data/local/tmp/pin --tools prof --runs 2 --scenarios isolated,interference
    --cooldown-s 20`, the binary of 3d2dc35. 8 runs, 28 records each, none failed. The range records
    (`range-*.jsonl`) were not re-collected (GPU only, unchanged).
  - *Loaders accept it:* `profiling_loader.py --device <serial> --app tree --backend vulkan` loads 56 records from 2
    runs on both phones, exit 0; `smt.data_loader.load_stage_timings` builds all four tables (2 phones x 2
    scenarios) with a value in every (stage, PU) cell.
  - *Old (`superseded-2026-10-18/`, unpinned) vs new (pinned) CPU p50 per tier and stage, run 1 / run 2 (ms).*
    The old tiers are one unpinned measurement taken three times (e.g. Pixel isolated stage 1: 1.14-1.16 on every
    tier); the new ones separate. Little is 2-7x slower than before on both phones (stage 2: 26.1 ms on the Pixel,
    18.0 on the Samsung, was 7.0 and 5.6), the Pixel's big tier is faster (stage 7: 4.15, was 7.8), and the
    Samsung's interference cells lost most of their run-to-run spread (stage 4 medium 3.22 / 3.22, was 8.40 /
    16.39). The Samsung's big tier (2 cores) is slower than its medium tier (3 cores) on the parallel stages
    (stage 4: 4.33 vs 3.22). Its interference little stage 7 still differs between the runs (27.8 / 46.8).

    Pixel 7a, isolated
    | stage | little old | little new | medium old | medium new | big old | big new |
    |---|---|---|---|---|---|---|
    | 1 | 1.141 / 1.161 | 2.456 / 2.467 | 1.148 / 1.156 | 1.161 / 1.162 | 1.142 / 1.143 | 0.746 / 0.751 |
    | 2 | 7.020 / 7.148 | 26.125 / 26.183 | 7.006 / 7.074 | 9.070 / 9.084 | 15.904 / 16.318 | 6.730 / 6.752 |
    | 3 | 0.194 / 0.194 | 1.304 / 1.235 | 0.219 / 0.204 | 0.246 / 0.249 | 0.223 / 0.402 | 0.178 / 0.178 |
    | 4 | 4.321 / 4.331 | 9.406 / 9.414 | 4.330 / 4.352 | 5.207 / 5.195 | 4.354 / 4.350 | 4.081 / 4.101 |
    | 5 | 0.525 / 0.525 | 1.079 / 1.080 | 0.525 / 0.525 | 0.336 / 0.336 | 0.525 / 0.525 | 0.273 / 0.274 |
    | 6 | 0.199 / 0.199 | 0.606 / 0.587 | 0.190 / 0.199 | 0.209 / 0.209 | 0.570 / 0.585 | 0.182 / 0.182 |
    | 7 | 7.676 / 7.955 | 16.611 / 16.605 | 7.656 / 7.976 | 5.384 / 5.376 | 7.846 / 7.800 | 4.153 / 4.144 |

    Pixel 7a, interference
    | stage | little old | little new | medium old | medium new | big old | big new |
    |---|---|---|---|---|---|---|
    | 1 | 1.596 / 1.893 | 2.985 / 2.906 | 2.318 / 2.234 | 1.184 / 1.167 | 1.980 / 2.112 | 1.053 / 1.049 |
    | 2 | 7.933 / 8.405 | 26.160 / 26.135 | 7.888 / 7.881 | 9.027 / 8.999 | 9.038 / 7.876 | 6.952 / 7.165 |
    | 3 | 0.219 / 0.249 | 1.238 / 1.225 | 0.219 / 0.219 | 0.247 / 0.247 | 0.260 / 0.219 | 0.200 / 0.207 |
    | 4 | 6.953 / 7.035 | 10.549 / 10.688 | 6.852 / 7.925 | 5.326 / 5.352 | 6.928 / 7.031 | 5.087 / 5.200 |
    | 5 | 0.750 / 0.641 | 1.614 / 1.518 | 0.564 / 0.589 | 0.337 / 0.337 | 0.689 / 0.761 | 0.355 / 0.395 |
    | 6 | 0.243 / 0.605 | 0.551 / 0.551 | 0.208 / 0.212 | 0.209 / 0.209 | 0.243 / 0.607 | 0.226 / 0.227 |
    | 7 | 10.245 / 12.496 | 22.282 / 23.025 | 8.498 / 13.200 | 6.068 / 5.886 | 9.778 / 14.475 | 5.954 / 5.990 |

    Samsung, isolated
    | stage | little old | little new | medium old | medium new | big old | big new |
    |---|---|---|---|---|---|---|
    | 1 | 1.010 / 1.009 | 2.589 / 2.587 | 1.009 / 1.009 | 0.679 / 0.680 | 1.009 / 1.010 | 0.907 / 0.910 |
    | 2 | 5.654 / 5.616 | 17.901 / 18.030 | 5.440 / 5.459 | 7.898 / 7.895 | 6.465 / 7.925 | 7.080 / 7.082 |
    | 3 | 0.138 / 0.138 | 0.685 / 0.684 | 0.134 / 0.134 | 0.266 / 0.265 | 0.631 / 0.631 | 0.238 / 0.237 |
    | 4 | 2.879 / 3.578 | 6.893 / 6.887 | 2.784 / 3.572 | 3.227 / 3.205 | 2.774 / 3.587 | 4.328 / 4.318 |
    | 5 | 0.353 / 0.354 | 0.916 / 0.914 | 0.353 / 0.354 | 0.232 / 0.232 | 0.353 / 0.353 | 0.307 / 0.307 |
    | 6 | 0.104 / 0.105 | 0.486 / 0.486 | 0.176 / 0.158 | 0.181 / 0.180 | 0.472 / 0.472 | 0.162 / 0.161 |
    | 7 | 3.295 / 3.401 | 7.794 / 7.906 | 3.082 / 3.267 | 2.737 / 2.726 | 2.983 / 3.061 | 3.658 / 3.653 |

    Samsung, interference
    | stage | little old | little new | medium old | medium new | big old | big new |
    |---|---|---|---|---|---|---|
    | 1 | 0.946 / 3.748 | 3.543 / 4.094 | 0.953 / 3.745 | 0.725 / 0.730 | 0.999 / 3.938 | 1.151 / 1.157 |
    | 2 | 6.294 / 6.488 | 18.671 / 17.622 | 6.053 / 6.448 | 7.929 / 7.905 | 7.091 / 8.127 | 7.083 / 7.089 |
    | 3 | 0.171 / 0.247 | 0.685 / 0.685 | 0.213 / 0.235 | 0.266 / 0.266 | 0.212 / 0.164 | 0.238 / 0.238 |
    | 4 | 8.749 / 16.482 | 10.236 / 9.892 | 8.401 / 16.389 | 3.218 / 3.217 | 11.434 / 19.410 | 4.327 / 4.529 |
    | 5 | 2.624 / 2.307 | 1.913 / 1.912 | 2.339 / 1.470 | 0.232 / 0.232 | 1.973 / 2.105 | 0.310 / 0.310 |
    | 6 | 0.124 / 0.183 | 0.487 / 0.487 | 0.124 / 1.271 | 0.181 / 0.181 | 0.506 / 0.199 | 0.162 / 0.162 |
    | 7 | 18.851 / 16.390 | 27.831 / 46.799 | 17.798 / 17.150 | 4.624 / 4.972 | 20.753 / 19.781 | 6.329 / 6.404 |
- [x] 2.2 Solve top-10 tree schedules per (phone, table) with the per-stage model and execute them as in the 2026-10-18 gate. Verify: every cell has executions in `data/sched_logs`.
  - The 2026-10-18 tree executions and schedule files moved to `data/archive/2026-10-18-cpu-chunk-costs/` first
    (`sched_logs/<serial>_tree_vk_<table>/`, `schedules_<table>/<serial>/tree/`), so that their logs (unpinned
    code) do not pool with the new ones by assignment. The cifar logs and schedules stay in place.
  - *Solve:* `02_gen_schedule_merged.py --profiling_root data/profiling --device <serial> --app tree --backend vk
    --table_type <btpm|isolated> --minimize_mode tmax --num_solutions 10 --output_folder data/schedules_<table>`
    (per-stage, no `overhead.json` in the store, so no union sweep), 4 files of 10 candidates.
  - *Execute* (2026-10-19 20:59-21:21 UTC): per cell the 10 by `metrics.max_time`, each as a one-schedule JSON in its
    own `bm-gen-logs-tree-vk` process from `/data/local/tmp/pin`, 3 rounds in the same order (btpm 1-10, isolated
    1-10), 20 s idle before every run, both phones at once. 120 runs (20 per round per phone), every one exit 0 with
    its task lines.
  - Every cell has executions: `data/sched_logs/<serial>_tree_vk_<table>/`, 30 logs each, on both phones.
- [x] 2.3 Apply the spec's gate against `docs/reports-for-human/2026-10-18-cpu-chunk-costs-gate.md`. Verify: a decision record in `docs/reports-for-human/` lists every (phone, table) with before / after CPU error, best period and its run-to-run spread, and states whether each bound holds.
  - Decision record: `docs/reports-for-human/2026-10-19-tree-cpu-pinning-gate.md` (all four cells, before / after
    CPU error, best period with its [min-max] and the best schedule's spread, each bound's verdict). The measure is
    committed as `optimizer/analysis/chunk_accuracy.py`; before use it reproduced the 2026-10-18 report's section 2
    exactly (every cell, both pricings) and the 2026-10-17 report's per-stage tree rows.
  - *CPU error not higher than before + 0.1:* holds in all four. Before (2026-10-18; Samsung as solved / no
    overhead) and after: Pixel btpm 0.218 to 0.084, isolated 0.297 to 0.062; Samsung btpm 0.366 / 0.229 to 0.207,
    isolated 1.165 / 1.783 to 0.268.
  - *Best period not worse beyond noise:* holds in all four. Pixel btpm 11.54 [10.44-21.44] to 8.60 [8.46-8.98]
    (-2.94 vs noise 11.00), isolated 26.55 [24.96-26.70] to 6.87 [6.85-6.87] (-19.68 vs 1.74); Samsung btpm 3.88
    [3.83-4.84] to 3.95 [3.83-4.23] (+0.07 vs 1.01), isolated 7.11 [6.84-9.55] to 3.94 [3.78-4.51] (-3.17 vs 2.71).
  - Not gated: the btpm GPU-chunk error rose (Pixel 0.438 to 0.955, Samsung 0.687 to 1.521), the known btpm GPU
    column artifact (record, section 4).

## 3. Final gate

- [x] 3.1 local and phone gate green at the last commit. Verify: results listed in the PR.
  - At d58af37 (the commit after it changes only this file): `ctest --test-dir build/pc -L omp` 11/11 passed
    (`CpuPinningOmp.*` and `InputPurityOmp.*` included); `uv run pytest -q optimizer` 41 passed; phone gate from the
    `android` build of that commit, `BT_ANDROID_DEST=/data/local/tmp/pin scripts/run-on-android.sh <serial>
    test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk`: Pixel 7a and Samsung each 22 / 16 / 17 passed, 0
    skipped, exit 0. `openspec validate tree-cpu-pinning --strict`: valid.
  - No PR is opened in this run (its instructions forbid it); these results are for the PR description.
