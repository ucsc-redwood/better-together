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
