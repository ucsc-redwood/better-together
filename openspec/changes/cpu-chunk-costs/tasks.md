# Tasks

Gates used below:
- **"local":** `ctest --test-dir build/pc -L omp` and `uv run pytest -q optimizer`.
- **"phone gate":** `scripts/run-on-android.sh <serial> test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk` on
  3A021JEHN02756 (Pixel 7a) and R5CY21Y3VEV (Samsung).
- **"phone profiling":** as in `docs/instruction-for-ai/06-end-to-end-scheduling.md` Step 1, 20 s between runs; on
  the Samsung `taskset` takes a hex mask. **Nothing is profiled on rocky-ryzen.**

## 1. Input-pure stages (profiling/cost-table)

- [x] 1.1 Add the D2 test for tree, cifar-dense and cifar-sparse (OMP). Verify: it fails on the current code for tree stage 2 (record which buffers change), and the list of every failing (app, stage) is written under this task.
  - *Test:* `bt::testing::ExpectStageInputPure` (`platform/util/testing/input_purity.hpp`) runs a stage twice on
    one AppData whose earlier stages ran. Every listed region (each buffer and scalar field) the stage does not write
    must equal its contents before the first run, after each run, and the regions it writes must match between the
    two runs. Tests `InputPurityOmp.{TreeAppData,TreeSafeAppData}` (`test-tree-omp`), `InputPurityOmp.CifarDense`
    (`test-cifar-dense-omp`), `InputPurityOmp.CifarSparse` (`test-cifar-sparse-omp`). `TreeAppData` is the chained
    `tree::AppData` that `bm-prof-tree-vk` and the pipeline run; `TreeSafeAppData` is the oracle's path and repeats
    stage 2 with 4 OMP threads on re-run stage-1 keys, since `test-tree-omp` runs single-threaded and the segment
    merge only happens with several threads.
  - *Fails on the current code (4765310 plus the test, build box):* tree stage 2 only.
    - `tree::AppData` stage 2 (`run_stage_2(tree::AppData&)`, `std::ranges::sort` in place): `u_morton_keys_s1`,
      1220389 of 1228800 bytes differ after the first run and after the second.
    - `SafeAppData` stage 2 (`omp::parallel_sort`): `u_morton_keys_s1_out`, 1220258 of 1228800 bytes differ after each
      run, with 1 thread and with 4.
    - Every other stage passes: tree 1 and 3-7 on both paths, cifar-dense 1-11, cifar-sparse 1-11 (inputs, weights,
      CSR arrays and `nnz` unchanged, outputs identical across the two runs).
- [x] 1.2 Make every failing stage keep its input intact (D1). Verify: the test passes; the OMP oracle suites and the VK/CUDA differential suites that compare with OMP still pass under local and the phone gate.
  - *Fix:* `run_stage_2(tree::AppData&)` copies `u_morton_keys_s1` into `u_morton_keys_sorted_s2` and sorts there.
    `omp::parallel_sort` takes its input `const`: each thread copies its segment into the output and sorts it there,
    and the segments are merged pairwise with `std::inplace_merge` in the output (the old ping-pong merged back into
    the input); `merge_segments` became unused and was removed.
  - *local:* `ctest --test-dir build/pc -L omp` 11/11 passed (the four `InputPurityOmp` tests included);
    `uv run pytest -q optimizer` 41 passed.
  - *phone gate* (`BT_ANDROID_DEST=/data/local/tmp/cpucost scripts/run-on-android.sh <serial> test-tree-vk
    test-cifar-dense-vk test-cifar-sparse-vk test-tree-omp`): Pixel 7a and Samsung each 20 / 16 / 17 / 23 tests
    passed, exit 0. CUDA is not in either gate (no Jetson run here).
- [x] 1.3 Validate the timing on one phone (D3): time OMP stage 2 on the Pixel's Medium tier with the fixed code, once by `bm-prof-tree-vk` and once with fresh input before every sample. Verify: the two medians agree within the spread, and both numbers plus the old 0.64 ms are recorded in design.md.
  - Pixel 7a, isolated, `medium/stage2`, process pinned to the Medium cores (`taskset 30`), 3 alternating rounds 20 s
    apart: `bm-prof-tree-vk` p50 9.110 / 9.156 / 9.116 ms (median 9.116, spread 9.110-9.156), fresh input
    9.108 / 9.126 / 9.114 ms (median 9.114, spread 9.108-9.126). Each median lies inside the other's spread. Old
    price 0.64 ms. Details and the harness variant: design.md, "Measurements".

## 2. Refresh the store and fit the overhead (optimizer/cost-model)

- [ ] 2.1 Re-profile tree on both phones (isolated and interference, 2 runs each) into `data/profiling/<serial>/tree/vulkan/`. Verify: the loaders accept the store, and the old vs new stage-2 CPU times per tier are recorded here.
- [ ] 2.2 Fit `overhead.json` for each phone with `fit_overhead.py` from the existing executions (D4). Verify: the fitted constants are recorded here, and pytest still passes.

## 3. Acceptance gate (optimizer/cost-model)

- [ ] 3.1 Solve top-10 (cifar-sparse top-5) per phone, app and table with the per-stage model, the refreshed store and the overhead, and execute them as in the 2026-10-17 gate. Verify: every cell has executions in `data/sched_logs`.
- [ ] 3.2 Compute per-chunk CPU errors on the new executions with and without the overhead, and compare with the 2026-10-17 values (gate report section 3, per-stage rows). Verify: a decision record in `docs/reports-for-human/` lists every (phone, app, table) with before / after CPU error and best measured period, and states whether each spec bound holds. If the overhead fails the held-out check, `overhead.json` is not shipped for that phone and the reason is recorded.
- [ ] 3.3 Update `docs/instruction-for-ai/05-profiling.md` (input-pure stages) and `06-end-to-end-scheduling.md` (the shipped overhead). Verify: the documented commands reproduce one cell of 3.1.

## 4. Final gate

- [ ] 4.1 local and phone gate green at the last commit. Verify: results listed in the PR.
