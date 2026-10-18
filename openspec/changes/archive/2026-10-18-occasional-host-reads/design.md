# Design

## Context

See proposal.md (Why). Current state:

- **Declarations.** `bt::PlaceBuffer(kind, stages)` (`platform/util/buffer_placement.hpp`): an AppData
  declares, per buffer, the stages that touch it. 0 stands for host code outside any stage.
- **The placement rule.** `runtime/placement.hpp::schedule_placement` puts a buffer in the
  weights / GPU-local class when every listed accessor is a GPU stage of the schedule, and in the
  boundary class otherwise. 0 is never a GPU stage, so any host accessor forces boundary.
  `bt_placement_test::ExpectCpuBuffersStayBoundary` checks this for every one-GPU-range schedule;
  `Placement.DeclaredStagesMatchBindings` checks the declared stages against the dispatchers'
  bindings.
- **Tree declarations** (`apps/tree/tree_appdata.cpp`, `vulkan/vk_appdata.hpp`):
  - the five octree outputs: `{0, 7}`;
  - `u_edge_offset_s6`: `{0, 6, 7}`;
  - `u_out_idx`: `{0, 3}`.

  The host reads `u_out_idx[n-1]` in `submit_multi_stage` / `complete_multi_stage` and
  `u_edge_offset_s6[n_brt_nodes-1]` in `complete_multi_stage`. Checks read the octree.
- **What the classes are on the phones** (data-flow fixes): GPU-local is `uncached` (coherent) on the
  Pixel 7a and host-cached on the Samsung. Every class is host-mapped.
- **Visibility.** The `vulkan-engine/execution-sync` rule (memory dependency to HOST/HOST_READ, then the
  completion wait) makes GPU writes visible in coherent memory with no invalidate. The engine
  invalidates only non-coherent buffers some dispatch writes (directional maintenance, #28).

## Goals / Non-Goals

**Goals:**
- Buffers the host reads rarely, or a few elements of per item, stop forcing host-cached placement
  and per-item invalidation.
- No change for any declaration that does not opt in.

**Non-Goals:**
- Making the engine skip invalidation of host-cached buffers. That would change the completion-wait
  rule.
- Declaring cifar buffers; their final outputs are a few KiB.

## Decisions

### D1. A marker in the accessor list
Add `bt::kHostSparse` (a negative stage id, so it cannot collide with a stage or with 0) to
`buffer_placement.hpp`. A declaration lists it in place of 0 when the host code outside the stages
only reads the buffer sparsely: rarely, or a few elements per item. `schedule_placement` ignores it
when deciding whether every accessor is a GPU stage.

*Alternatives:*
- A new `BufferKind` (kResult). Kind already means what the data is (activation or weights) and
  selects between the weights and GPU-local classes. Access frequency is about the accessor, so it
  belongs in the accessor list.
- A per-buffer "never invalidate" flag in the engine. Rejected: it changes visibility for host-cached
  memory.

### D2. Which tree buffers
- The five octree outputs: `{kHostSparse, 7}`, read only by checks.
- `u_out_idx`: `{kHostSparse, 3}` and `u_edge_offset_s6`: `{kHostSparse, 6, 7}`, one element per item.

The input points stay weights (`{1}`). Nothing else in tree has a host accessor.

*Re-checked against the current code (2026-10-18, after tree-app-improvements):* the list holds.
- Stage 7 is now two dispatches of `tree_build_octree` (phase 0 nodes, phase 1 leaves). Both bind the same five
  octree buffers plus inputs from stages 3-6, so the octree is still touched by stage 7 only.
- The stage-3 compaction is now two dispatches. It still writes `u_out_idx` (whose last element is `n_unique`) and
  uses `u_scan_block_sums` (`{3, 6}`, no host reader). The host still reads `u_out_idx[n-1]` in `submit_multi_stage` /
  `complete_multi_stage` and `u_edge_offset_s6[n_brt_nodes-1]` in `complete_multi_stage`, nowhere else.
- New since planning: the leaf masks are initialized to 0xFF at construction (tree-app-improvements task 3.2), a
  one-time host write. It is rare host access like the checks, and stays correct in every class: an uncached
  mapping takes the write (as weights do), and on a non-coherent class the submission flushes every buffer a
  dispatch accesses. It does not change the declaration.
- `u_sums`, `u_prefix_sums` (`{0}`, no stage) and `u_num_selected_out` (`{0}`) are untouched: a buffer no stage
  touches is not affected.

### D3. Tests
- `ExpectCpuBuffersStayBoundary` learns the marker: an accessor list made only of GPU stages plus
  `kHostSparse` must land in the GPU class; any CPU stage still forces boundary.
- The tree `Placement.DeclaredStagesMatchBindings` compares stages >= 1 only, so it is unaffected.
- A hardware-free check that the cifar apps' placement is unchanged covers the spec's "existing
  declarations" scenario.
- The pipeline tests (`test-pipeline-e2e-vk`, `test-pipeline-chained-vk`) already read the counts and
  the octree of pipelined items. Run on the Pixel, they are the correctness check for the non-coherent
  device.

### D4. Measurement
The same whole-chunk A/B as the data-flow fixes:
- `bm-phases-tree-vk -s 1 -e 7`, host thread on the big cores (`taskset -c 6,7`, the Samsung
  `taskset 180`);
- 3 rounds alternating the binary with and without the D2 declarations, 20 s apart;
- `dispatch_wall` medians and spreads.

The expected range on the Pixel is between today's 27.51 ms and the all-coherent 20.78 ms. Phones only.

*Measured (task 2.2, 2026-10-18).* A = the tree declarations with `0` (the state before task 2.1), B = with
`bt::kHostSparse`; otherwise the same tree, default engine settings. `bm-phases-tree-vk --device <serial> -s 1 -e 7`
(200 chunks after 20 warm-up, 4 items), host thread on the big cores, rounds A, B, A, B, A, B, 20 s apart. Per run the
median `dispatch_wall` (ms); an arm's median is the median of its three runs, its spread their min..max.

| | A r1 / r2 / r3 | A median (spread) | B r1 / r2 / r3 | B median (spread) | change |
|---|---|---|---|---|---|
| Pixel 7a | 29.58 / 21.30 / 29.54 | 29.54 (21.30..29.58) | 17.71 / 17.79 / 19.01 | **17.79** (17.71..19.01) | -40 % |
| Samsung | 7.22 / 9.98 / 10.37 | 9.98 (7.22..10.37) | 8.35 / 10.50 / 9.91 | 9.91 (8.35..10.50) | -1 % (noise) |

- **Gate:** passed. On the Pixel B is lower and the spreads do not overlap (19.01 < 21.30); on the Samsung B's median
  is not higher than A's (9.91 vs 9.98), well inside A's spread. The declarations stay.
- **Where it comes from, Pixel:** the second submission's (stages 4-7) flush falls from 1.47..1.71 ms to 0.0007 ms and
  its invalidate from 1.30..1.41 ms to 0.04 ms. The walked bytes are unchanged (15.53 MiB flushed, 14.36 MiB
  invalidated); in coherent memory the VMA calls return at once. A's runs 1 and 3 also ran the GPU slower (submission 2
  GPU time 7.45 ms against 5.0 ms in every other run, as if at a lower clock level). Run 2, at the same GPU time as B,
  is the clock-matched comparison: 21.30 -> 17.79 ms (-16 %). Both are below the all-coherent 20.78 ms of
  2026-10-16, which was measured before tree-app-improvements changed stages 3 and 7, so the two are not comparable.
- **Samsung:** the GPU-local class is host-cached there, so the invalidate is unchanged (0.41..0.44 ms) and so is the
  chunk, as expected.
- **Checks on the Pixel** now read the octree through an uncached mapping. Not timed against a before figure: in the
  task 2.1 phone gate, `test-pipeline-e2e-vk` (checks every item's octree) took 64.5 s and
  `test-pipeline-chained-vk` 4.8 s. It affects test time only (Risks).

## Risks / Trade-offs

- **[Host reads of an uncached count are slow]** → One element per item; measured inside the whole
  chunk, so the gate already includes the cost.
- **[A check that scans the whole octree on the Pixel gets slower]** (uncached reads) → It only affects
  test time, not the pipeline. It is noted in the decision record if it is noticeable.
- **[Someone later adds a frequent host reader without updating the declaration]** → The marker's
  comment says what qualifies. Correctness does not depend on it, only speed, because every class is
  host-mapped and the visibility rule holds in each.

## Migration Plan

Additive and opt-in per buffer. Rollback is restoring `0` in the tree declarations.
