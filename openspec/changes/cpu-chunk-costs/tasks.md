# Tasks

Gates used below:
- **"local":** `ctest --test-dir build/pc -L omp` and `uv run pytest -q optimizer`.
- **"phone gate":** `scripts/run-on-android.sh <serial> test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk` on
  3A021JEHN02756 (Pixel 7a) and R5CY21Y3VEV (Samsung).
- **"phone profiling":** as in `docs/instruction-for-ai/06-end-to-end-scheduling.md` Step 1, 20 s between runs; on
  the Samsung `taskset` takes a hex mask. **Nothing is profiled on rocky-ryzen.**

## 1. Input-pure stages (profiling/cost-table)

- [ ] 1.1 Add the D2 test for tree, cifar-dense and cifar-sparse (OMP). Verify: it fails on the current code for tree stage 2 (record which buffers change), and the list of every failing (app, stage) is written under this task.
- [ ] 1.2 Make every failing stage keep its input intact (D1). Verify: the test passes; the OMP oracle suites and the VK/CUDA differential suites that compare with OMP still pass under local and the phone gate.
- [ ] 1.3 Validate the timing on one phone (D3): time OMP stage 2 on the Pixel's Medium tier with the fixed code, once by `bm-prof-tree-vk` and once with fresh input before every sample. Verify: the two medians agree within the spread, and both numbers plus the old 0.64 ms are recorded in design.md.

## 2. Refresh the store and fit the overhead (optimizer/cost-model)

- [ ] 2.1 Re-profile tree on both phones (isolated and interference, 2 runs each) into `data/profiling/<serial>/tree/vulkan/`. Verify: the loaders accept the store, and the old vs new stage-2 CPU times per tier are recorded here.
- [ ] 2.2 Fit `overhead.json` for each phone with `fit_overhead.py` from the existing executions (D4). Verify: the fitted constants are recorded here, and pytest still passes.

## 3. Acceptance gate (optimizer/cost-model)

- [ ] 3.1 Solve top-10 (cifar-sparse top-5) per phone, app and table with the per-stage model, the refreshed store and the overhead, and execute them as in the 2026-10-17 gate. Verify: every cell has executions in `data/sched_logs`.
- [ ] 3.2 Compute per-chunk CPU errors on the new executions with and without the overhead, and compare with the 2026-10-17 values (gate report section 3, per-stage rows). Verify: a decision record in `docs/reports-for-human/` lists every (phone, app, table) with before / after CPU error and best measured period, and states whether each spec bound holds. If the overhead fails the held-out check, `overhead.json` is not shipped for that phone and the reason is recorded.
- [ ] 3.3 Update `docs/instruction-for-ai/05-profiling.md` (input-pure stages) and `06-end-to-end-scheduling.md` (the shipped overhead). Verify: the documented commands reproduce one cell of 3.1.

## 4. Final gate

- [ ] 4.1 local and phone gate green at the last commit. Verify: results listed in the PR.
