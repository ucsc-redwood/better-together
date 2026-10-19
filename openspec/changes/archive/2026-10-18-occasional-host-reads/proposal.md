# Proposal

## Why

On the Pixel 7a, the tree app's GPU chunk still pays host cache maintenance for buffers the host
barely reads. Since the data-flow fixes (#28), a whole tree chunk (stages 1-7, host thread on the big
cores) takes 27.51 ms. With every buffer in coherent memory it took 20.78 ms
(`docs/reports-for-human/2026-10-16-vk-data-flow-fixes.md`). The difference is the buffers that stay
in the host-cached boundary class only because host code outside any stage reads them:

- the final octree (five buffers, 8.2 MiB), read only by checks;
- the two counts the dispatcher reads back (`u_out_idx[n-1]` after stage 3 and
  `u_edge_offset_s6[n_brt_nodes-1]` after stage 6), one element per item from 1.2 MiB buffers.

All of them are invalidated whole after every item.

The placement contract does not require this. "Boundary buffers keep host-cached placement" covers
buffers a CPU chunk reads. Host code outside the stages goes to the boundary class only as a
conservative implementation choice: the placement treats accessor 0 like a CPU stage. Every class is
host-mapped, and in coherent memory the completion wait already makes the GPU's writes visible to the
host. So such a buffer can live in GPU-side memory and still read correctly, just more slowly per
element on an uncached mapping.

## What Changes

- **An AppData buffer can declare that host code outside any stage reads it only sparsely:** either
  rarely (checks, final results) or a few elements per item (a count). That host access does not, by
  itself, keep the buffer in the boundary class: the placement decides from the stages that touch it,
  as it does for buffers no host code touches.
- **Host access stays exactly as today when it is declared without the new marker.** Today's
  declarations keep today's placement; the marker is opt-in per buffer.
- **The tree app declares** its five final-octree buffers and its two read-back count buffers that way.
- **The tree declarations ship only if they are measured faster** on the Pixel and not slower on the
  Samsung (whole tree GPU chunk, as in the data-flow fixes).
- No engine change: the classes, their placements and the completion-wait visibility rule are
  unchanged.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `vulkan-engine/memory-placement`: adds that a buffer whose only non-GPU accessor is declared sparse
  host reading is placed by its GPU accessors, that host reads of it stay correct in every class, and
  that undeclared host access keeps the boundary class. Existing requirements are unchanged.

## Impact

- **Code:**
  - `platform/util/buffer_placement.hpp` (the declaration marker);
  - `runtime/placement.hpp` (the placement rule);
  - `apps/tree/{tree_appdata.cpp,vulkan/vk_appdata.hpp}` (the tree declarations);
  - `runtime/tests/placement_check.hpp` and the tree placement tests.
- **Gates:**
  - local `ctest -L omp` (the hardware-free placement test);
  - rocky correctness;
  - the phone gate;
  - a whole-chunk A/B of the tree on both phones.
  - Nothing is profiled on rocky.
- **Out of scope:**
  - lazy or on-demand invalidation (it would change the completion-wait rule);
  - the cifar apps, whose final outputs are small;
  - the two-submission tree chunk (indirect dispatch).
