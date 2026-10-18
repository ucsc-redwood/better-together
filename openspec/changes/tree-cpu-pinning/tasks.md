# Tasks

Gates used below:
- **"local":** `ctest --test-dir build/pc -L omp` and `uv run pytest -q optimizer`.
- **"phone gate":** `scripts/run-on-android.sh <serial> test-tree-vk test-cifar-dense-vk test-cifar-sparse-vk` on
  3A021JEHN02756 (Pixel 7a) and R5CY21Y3VEV (Samsung).
- **"phone profiling":** as in `docs/instruction-for-ai/06-end-to-end-scheduling.md` Step 1, 20 s between runs; on
  the Samsung `taskset` takes a hex mask. **Nothing is profiled on rocky-ryzen.**

## 1. Pinning (runtime/cpu-chunks)

- [ ] 1.1 Add the D2 pinning test for tree, cifar-dense and cifar-sparse. Verify: it fails for tree on the current code (record the observed masks and thread counts) and passes for both cifar apps.
- [ ] 1.2 Pin tree's AppData path (D1; per-region binding if 1.1 shows pool reuse defeats inheritance). Verify: the test passes for all three apps, the tree OMP and VK/CUDA differential suites still pass under local and the phone gate.
- [ ] 1.3 Check the tier records on one phone: time tree stage 1 and stage 2 on the Pixel's little tier with `bm-prof-tree-vk`, and again in a process pinned with `taskset` to the little cores. Verify: the medians agree within the spread (spec "Profiling a tree stage on one tier"), numbers recorded in design.md.
- [ ] 1.4 Remove "Tree's CPU tiers are nominal" from `docs/instruction-for-ai/05-profiling.md`. Verify: the doc says tree pins like the cifar apps.

## 2. Refresh and gate (optimizer/cost-model)

- [ ] 2.1 Re-profile tree on both phones (isolated and interference, 2 runs each), moving the previous runs to `superseded-<date>/`. Verify: the loaders accept the store; old vs new CPU p50 per tier and stage recorded here.
- [ ] 2.2 Solve top-10 tree schedules per (phone, table) with the per-stage model and execute them as in the 2026-10-18 gate. Verify: every cell has executions in `data/sched_logs`.
- [ ] 2.3 Apply the spec's gate against `docs/reports-for-human/2026-10-18-cpu-chunk-costs-gate.md`. Verify: a decision record in `docs/reports-for-human/` lists every (phone, table) with before / after CPU error, best period and its run-to-run spread, and states whether each bound holds.

## 3. Final gate

- [ ] 3.1 local and phone gate green at the last commit. Verify: results listed in the PR.
