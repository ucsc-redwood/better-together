# Vulkan engine runtime: decisions and measurements (2026-10-15)

**TL;DR** — The OpenSpec change `vk-engine-runtime` (branch `campaign2/vk-engine-runtime`, groups 1–4) changed four
things in the Vulkan engine:
- synchronization2 barriers, recorded only where a stage depends on an earlier one;
- per-device memory placement for weights;
- a selectable queue, and several GPU chunks per schedule;
- phase timing, pre-recorded command buffers and items in flight.

Every default stays as it was unless a measurement beat the noise. The one placement default that changed is the
Pixel 7a's weights (uncached, fc1/fc2 −31 to −35 %). The single-GPU-chunk guard is lifted for Vulkan. Depth 1 stays
the in-flight default. The per-chunk overhead constant the solver uses was refit and did not regress.
Full evidence: `openspec/changes/vk-engine-runtime/{design,tasks}.md`; all measurements below are on rocky-ryzen
(Radeon 780M, RADV) unless a phone is named.

## Knobs and defaults

| knob | default | where it is described |
|---|---|---|
| `BT_VK_BARRIER_ALL=1` | unset: a barrier only where a dispatch touches a buffer written (or, for a write, read) since the last one | 03-unit-testing "Engine knobs" |
| `BT_VK_WEIGHTS_PLACEMENT=default\|device_local\|uncached` | unset: `uncached` on the Pixel 7a (Mali-G710), host-cached elsewhere | same |
| `BT_VK_QUEUE=universal\|compute` | `universal` (first family with compute) | same |
| `BT_VK_QUEUE_PRIORITY=low\|medium\|high` | the driver's default priority | same |
| `BT_VK_INFLIGHT=<n>` | `1` | same; 06-end-to-end-scheduling step 3 |
| `BT_VK_PHASES=1` (`bm-phases-*-vk`) | off | 03-unit-testing |
| solver `allow_gpu_multi_chunk` (`02 … --allow-gpu-multi-chunk`) | off: one chunk per PU | 06-end-to-end-scheduling step 2 |

## A. Synchronization (group 1)

- `synchronization2` is a required 1.3 feature (supported on rocky, the Pixel and the Samsung). Every barrier is now
  `vkCmdPipelineBarrier2`, and both timestamps are `vkCmdWriteTimestamp2` at COMPUTE_SHADER. That move shifted every
  stage by at most ±4.4 %, mostly faster.
- **Conditional barriers kept, but they save nothing on production data.** On the chained production AppData every
  dispatch depends on the one before it, so the barrier counts per chunk are unchanged:

  | app | GPU chunk | dispatches | barriers, `BT_VK_BARRIER_ALL=1` | barriers, tracked |
  |---|---|---|---|---|
  | tree | stages 1–2 | 9 | 8 | 8 |
  | tree | stages 4–7 | 6 | 5 | 5 |
  | cifar-dense | stages 1–11 | 11 | 10 | 10 |
  | cifar-sparse | stages 1–11 | 11 | 10 | 10 |

  They drop on tree's golden-decoupled test data, and the dependency tracking is what memory placement and several
  GPU chunks build on.
- Synchronization validation needs `VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true`. Without it the
  layer does not check descriptor accesses at all: a run with the barriers removed failed 9 tests and printed 0
  hazards; with the heuristic it printed 210.

## B. Memory placement (group 2)

fc1 / fc2 GPU time per arm (`bm-cifar-dense-vk`, ms, means; Δ vs default). Rule: go only if the mean improves by
more than the default arm's spread on both stages.

| device | stage | default | device_local | uncached | default spread | decision |
|---|---|---|---|---|---|---|
| rocky (6 runs) | fc1 | 0.8801 | 0.8548 (−2.9 %) | 0.8661 (−1.6 %) | 0.0373 | no-go |
| | fc2 | 0.8769 | 0.8536 (−2.7 %) | 0.8710 (−0.7 %) | 0.0221 | |
| Pixel 7a (3 runs) | fc1 | 16.6416 | 11.4737 (−31.1 %) | 11.1127 (−33.2 %) | 0.8820 | **go, `uncached`** |
| | fc2 | 16.3291 | 12.1540 (−25.6 %) | 10.5960 (−35.1 %) | 1.0266 | |
| Samsung (3 runs) | fc1 | 3.4437 | 3.2539 (−5.5 %) | 3.2325 (−6.1 %) | 0.5917 | no-go |
| | fc2 | 3.2916 | 3.3225 (+0.9 %) | 3.2489 (−1.3 %) | 0.1163 | |

- rocky's `device_local` carve-out is faster in every run, but by less than the spread; a larger sample could revisit
  it.
- Only weights no CPU chunk reads move. The placement is computed from the schedule (`runtime/placement.hpp`), and
  the differential tests keep every buffer host-cached.
- With the Pixel's weights class forced back to the default, the full model is fc1 −31.1 % / fc2 −26.3 % with the
  measured placement, and no stage is slower beyond noise.

## C. Queues and several GPU chunks (group 3)

- **Queue A/B (tree, cifar-dense; isolated and under interference; 3 interleaved runs).** Every arm is within ±1.5 %
  of `universal` at the default priority, mostly inside its spread. The two larger wall-clock moves (+8.8 %, +19.7 %)
  each come from one outlier run.
  - rocky's driver refuses `high` priority (`VK_ERROR_NOT_PERMITTED`), so the engine warns and keeps the default. The
    priority half of the A/B therefore could not be measured there.
  - **Defaults stay.**
- **Several GPU chunks:** each Vulkan GPU chunk gets its own `Sequence`, push constants travel with the record call,
  and the Engine serializes `vkQueueSubmit`.
  - The guard (`bugs-found.md` §10) is lifted for Vulkan and still rejects repeated CUDA chunks.
  - The once-crashing VK 1–3 / CPU 4 / VK 5 / CPU 6 / VK 7 tree schedule now matches the oracle on all three Vulkan
    devices.
- **Solver:** `allow_gpu_multi_chunk` is opt-in. It prices the GPU's chunks as their sum (one shared device), and
  the default mode never emits two GPU chunks (pytest).

## D. Submission overhead (group 4)

**What the ≈1.5 ms fitted per-chunk overhead is.** Phase timing per submission (`bm-phases`, median ms, back-to-back
chunks):

| shape | dispatch wall | record | submit | wait | GPU | wait − GPU | Σ stage GPU alone |
|---|---|---|---|---|---|---|---|
| tree 1–3 | 1.1862 | 0.0039 | 0.0188 | 1.1578 | 1.0886 | 0.0691 | 1.5705 |
| tree 4–7 | 0.6279 | 0.0027 | 0.0182 | 0.6021 | 0.5522 | 0.0499 | 0.8440 |
| cifar-dense 1–11 | 9.5740 | 0.0324 | 0.0857 | 9.4070 | 8.9758 | 0.4311 | 9.0319 |

GPU time of the chunk after an idle gap before it:

| shape | 0 ms | 1 ms | 5 ms | 20 ms |
|---|---|---|---|---|
| tree 1–3 | 1.0886 | 1.4790 | 2.5427 | 2.9824 |
| tree 4–7 | 0.5522 | 0.8602 | 0.8650 | 1.4022 |
| cifar-dense 1–11 | 8.9758 | 8.9893 | 12.2304 | 24.1065 |

- Host submission costs 0.07–0.6 ms per chunk, and recording only µs. A hot chunk runs *faster* than its stages timed
  alone.
- The constant is mostly the GPU slowing down after idling between pipelined items (DVFS), plus the SPSC hand-off.
  So pre-recording (D2) can only remove µs; keeping work queued (D3) is the lever.
- On the Pixel, cache maintenance of its non-coherent buffers costs 2.3 ms flush + 2.3 ms invalidate per tree chunk.
  That makes a coherent placement for GPU-only buffers a candidate follow-up.
- rocky's calibrated timestamps are wrong: queue timestamps run 2.898e9 ticks ahead of the calibrated device clock.
  The engine now verifies the calibration and treats a failing device as uncorrelated.

**Pre-recorded command buffers and items in flight**:
- An equal recipe reuses its recorded command buffer (up to 32 per Sequence); the descriptor-fallback path re-records.
- Submissions return timeline-semaphore tickets.
- `BT_VK_INFLIGHT` keeps n items in flight per GPU chunk, in FIFO order.

**Refit gate.** Four interleaved rounds of full fleet runs: before = 9236dda, then the group 4 code at depth 1 and
at depth 2.

| metric (ms) | before mean | depth 1 mean | before spread | before mean + spread |
|---|---|---|---|---|
| fitted `gpu_vulkan` per_chunk (all apps) | 1.4645 | 1.6644 | 0.3583 | **1.8228** (met) |
| fitted `gpu_vulkan` per_chunk (tree only) | 0.8711 | 0.8664 | 0.0131 | 0.8842 |
| fitted `cpu` per_chunk (tree only) | 0.2094 | 0.2861 | 0.0178 | 0.2272 |
| tree best makespan (max chunk) | 1.7223 | 1.7324 | 0.0364 | 1.7588 |
| tree GPU chunk 1–3 duration | 1.9287 | 1.9328 | 0.0500 | 1.9786 |
| tree best steady period | 1.5278 | 1.5335 | 0.0310 | 1.5588 |
| tree best per-task makespan | 1.7996 | 1.9677 | 0.0489 | 1.8485 |

- **Accepted.** The GPU constant did not regress beyond noise.
- **Found:** the group 4 builds stall the CPU chunk at the **end** of a run.
  - In 21 of 32 tree schedule runs, one of the last six items takes over 5 ms (up to 24 ms); before group 4, none
    did. That is the per-task makespan and CPU-constant rise above; the steady period is unchanged.
  - An experiment that keeps each GPU chunk's `Sequence` alive past its thread cuts the stalls from 10/15 to 3/15. So
    most of it is the Sequence teardown (32 recorded command buffers) on the GPU thread while the OMP chunk still runs.
- **Depth 2 stays off.**
  - At depth > 1 the logged chunk ticks are submit → complete latency. The fitted GPU constant reads 8.48 ms and
    the max-chunk makespan 2.96 ms: neither measures cost.
  - By throughput, depth 2 beats depth 1 but only matches the pre-group-4 baseline. Its steady-state gain is under
    1 % (cifar-dense −0.7 %, cifar-sparse −0.5 %, tree level).
  - Enabling it would also need constants fitted at depth 2, which the tooling cannot produce yet.

## Open follow-ups

- Destroy each GPU chunk's `Sequence` after the timed region in `bm-gen-logs` (and find the stall's remainder).
- Log a chunk's service time (completion to completion) at depth > 1, so that `fit_overhead` and `speedup_summary`
  can measure in-flight runs; then re-run the depth A/B.
- Measure a coherent `gpu_local` placement on the Pixel (its flush/invalidate cost); revisit rocky's `device_local`
  carve-out with a larger sample.
- Several CUDA chunks per schedule (per-chunk streams) remain out of scope.
