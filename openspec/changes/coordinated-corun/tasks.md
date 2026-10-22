# Tasks

Rules for every task:
- **Phones only** for timing: Samsung S24+ `R5CY21Y3VEV` (main), Pixel 7a `3A021JEHN02756` (optional part). Nothing
  is timed on rocky-ryzen or the build box. No clock or governor setting is changed.
- The study's phone directory is `/data/local/tmp/ccr`, removed at the end. Files the study did not create are read,
  never modified or deleted (the owner's `lcpp`, `llama*`, and `llm-phase-affinity`'s leftovers if any).
- Each measured value: median of 3 trials with min-max, masks, GPU clock, thermal status; procedure of design D3.
- Raw outputs go to `data/coordinated-corun/<serial>/`; committed per stage. No model files or binaries committed.

## 1. Setup

- [x] 1.1 Rebase this branch on `dev` once `llm-phase-affinity` is merged, so its drivers, data and analysis module are present. Verify: `data/llm-phase-affinity/drivers/lpa_corun.sh` and `optimizer/analysis/llm_phase_affinity.py` exist on the branch.
- [x] 1.2 Write `data/coordinated-corun/drivers/lpa_pace.cpp` (design D1) and build it for arm64 against llama.cpp `8345f33` with the Samsung build's flags. Verify: it runs on the Samsung in both modes and prints the per-step JSONL and the summary.
- [x] 1.3 Calibrate (spec "The paced harness matches the reference rates"): 1B GPU prefill P = 512 and 2048, GPU decode depth 0, CPU decode on `3f0`, all at u = 1; then GPU prefill at u = 0.5. Verify: rates within 5% of the `llm-phase-affinity` medians (or the difference recorded and the harness's alone rates adopted), and the achieved duty of the u = 0.5 run reported.
- [x] 1.4 Check thread placement for masks `1f0`, `1c0`, `180` with the harness's CPU decode. Verify: one thread per core for each kept mask; any packed mask dropped and recorded in design.md.

## 2. Sweeps (Samsung, llama.cpp)

- [ ] 2.1 Paced-alone rates: GPU prefill (P = 512, 2048) and GPU decode at u = 1.0, 0.8, 0.67, 0.5, 0.33; CPU decode alone on each kept mask. Verify: a table of rate and achieved duty per setting, with clocks.
- [ ] 2.2 Stage A: GPU prefill paced at the five duties next to CPU decode on `3f0`, P = 512 and 2048. Verify: both rates and ratios per duty.
- [ ] 2.3 Stage D: GPU decode paced at the five duties next to CPU decode on `3f0`. Verify: same table for the data-parallel pair.
- [ ] 2.4 Stage B: CPU decode on each kept mask next to GPU prefill at u = 1.0 and the best u of 2.2; repeat 2.3 at the best mask. Verify: same tables.
- [ ] 2.5 Stage C: C = 128 and 256 at the best (u, mask). Verify: same table.
- [ ] 2.6 Stage E: CPU prefill on `3f0` next to GPU decode at u = 1.0 and 0.5. Verify: same table.
- [ ] 2.7 Stage F: 3B at the best specialized and data-parallel settings and at u = 1.0. Verify: same tables for 3B.

## 3. Optional: Pixel ExecuTorch

- [ ] 3.1 Design D5 on the Pixel with the `llm-phase-affinity` ExecuTorch runner and `.pte` files (rebuilt or re-pushed into `/data/local/tmp/ccr` if the earlier study removed them). Verify: the tables, or the skip recorded with its reason.

## 4. Prediction and decision

- [ ] 4.1 Write `optimizer/analysis/coordinated_corun.py` (design D4) over the raw results, with a pytest on a hand-computed case. Verify: `uv run pytest -q optimizer` passes.
- [ ] 4.2 Produce the grid per mix: best coordinated specialized (with setting), best coordinated data-parallel (with setting), the uncoordinated values from `llm-phase-affinity`, the baseline and the gain, and the trade-off curves. Verify: every input traceable to a raw file.
- [ ] 4.3 Write `docs/reports-for-human/<date>-coordinated-corun.md` with the curves, the grid and the go/no-go per the spec; link it from `docs/reports-for-human/README.md`. Verify: `openspec validate coordinated-corun --strict` passes, the record states the decision, and `/data/local/tmp/ccr` is removed from the phones.
