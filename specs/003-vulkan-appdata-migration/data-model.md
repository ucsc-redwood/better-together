# Data Model: Vulkan Genuinely-Chained AppData Migration

## `tree::vulkan::VkAppData` (new — the chained struct)

The Vulkan analog of `tree::AppData`: single buffer per pipeline field (no golden/`_out`
split), plus the Vulkan-specific scratch buffers the compute-shader algorithms need.

- **Inherits**: `tree::AppData` (gets `n_input`/`n_unique`/`n_brt_nodes`/
  `n_octree_nodes` + their get/set accessors, and every `u_*_sN` pipeline field, all
  by the plain field name — no `_out` suffix, matching what stage `N+1`'s dispatch
  will read as its real input).
- **Adds** (mirrors `VkAppData_Safe`'s scratch fields, sized identically):
  - `u_contributes`, `u_out_idx` — stage 3 dedup scratch.
  - `u_sums`, `u_prefix_sums` — generic prefix-sum scratch.
  - `u_sort_tmp`, `u_sort_histograms` (`kRadixBins * kRadixNumWorkgroups`) — stage 2
    multi-workgroup LSD radix sort ping-pong buffer + per-workgroup histograms.
  - `u_scan_block_sums` (`ceil(n_input / kScanElementsPerWg) + 1`) — stage 3/6
    device-wide scan per-block totals.
- **Constructed from**: `kiss_vk::VulkanMemoryResource::memory_resource*` + `n_input`,
  same shape as `VkAppData_Safe`'s constructor.
- **Relationship to `VkAppData_Safe`**: siblings, not parent/child — both extend the
  same scratch-buffer *shape* but over different base types (`tree::AppData` vs
  `tree::SafeAppData`). No inheritance between the two; kept independent exactly like
  `tree::AppData` and `tree::SafeAppData` themselves are independent (Phase 3 finding
  this session).

## `VulkanDispatcher` chained overloads (new, alongside existing `VkAppData_Safe` ones)

Not a data entity, but the contract surface this feature adds — see
`contracts/vulkan-dispatch-chained.md`.

## Device Target (verification tracking, not code)

One of the three required hardware targets from the spec — Jetson devkit, Pixel 7a,
Samsung Galaxy — each carrying its own independent pass/fail verdict per user story.
Tracked in `tasks.md`/verification runs, not a runtime data structure.
