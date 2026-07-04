# Research: Vulkan Genuinely-Chained AppData Migration

All findings below came from directly reading the current Vulkan dispatcher
(`apps/tree/vulkan/dispatchers.{hpp,cpp}`, `vk_appdata.hpp`) and comparing against the
already-completed CUDA equivalent (`apps/tree/cuda/dispatchers.{cuh,cu}`), not from
assumption — the CUDA precedent turned out to only partially transfer.

## Finding 1: Vulkan's dispatcher reads the CONST golden, never its own `_out`, for every stage

**Decision**: the chained Vulkan path needs new `record_stage_N` bodies for every one of
the 7 stages, not a thin wrapper.

**Rationale**: grepping every `appdata.u_*` reference in `dispatchers.cpp` shows a
uniform pattern across all 7 stages: each stage's shader writes its own `_out`-suffixed
field, but reads the **previous** stage's plain (golden, `const`) field name — never the
`_out` field the previous stage's own dispatch just wrote. Example: `record_stage_2`
reads `appdata.u_morton_keys_s1` (the immutable golden copy built once at
`SafeAppData` construction via the OMP-computed `HostTreeManager`), not
`appdata.u_morton_keys_s1_out` (what stage 1's own Vulkan dispatch just wrote). This is
consistent with `SafeAppData`'s documented golden-decoupled design (see
`tree_diff_oracle.hpp`'s header comment and the `safe-appdata-debt` project memory) — it
is not a bug, but it does mean genuinely chaining Vulkan requires touching every stage's
buffer bindings, unlike CUDA where `tree::AppData`'s field names already matched
`SafeAppData`'s golden names 1:1 and the existing kernels could be reused via a generic
overload.

**Alternatives considered**: assuming a thin `dispatch_stage(VkAppData&, int)` wrapper
around the existing `record_stage_N(VkAppData_Safe&, ...)` bodies would suffice (as
CUDA's `dispatch_multi_stage` cores-overload was) — rejected once the grep showed the
buffer-binding calls are hardcoded to `VkAppData_Safe`'s specific field names/no
`_out` for a chained struct's writes, so the record bodies themselves must change.

## Finding 2: Vulkan needs extra scratch buffers a plain `tree::AppData` doesn't have

**Decision**: the new chained struct (`tree::vulkan::VkAppData`) must inherit from
`tree::AppData` and add the same Vulkan-specific scratch fields `VkAppData_Safe`
already carries: `u_contributes`, `u_out_idx` (dedup), `u_sums`/`u_prefix_sums` (prefix
sum), `u_sort_tmp`/`u_sort_histograms` (multi-workgroup LSD radix sort ping-pong +
histograms), `u_scan_block_sums` (device-wide scan block totals).

**Rationale**: Vulkan's compute-shader stage 2 (sort) and stage 3/6 (scan) use explicit
multi-workgroup parallel algorithms (Mirco Werner / Embree-style LSD radix sort, a
device-wide scan) that need host-visible scratch buffers CUDA's `thrust`/`cub`-backed
calls and OMP's direct in-process calls don't need. `tree::AppData` has no equivalent
fields — they are Vulkan-specific, confirmed by `VkAppData_Safe`'s own field list.

**Alternatives considered**: adding these scratch fields directly to `tree::AppData`
itself — rejected as unnecessary bloat for the OMP/CUDA paths (Principle I, Simplicity
First) and inconsistent with `VkAppData_Safe`'s existing precedent of extending the base
struct in a Vulkan-specific subclass rather than polluting the shared one.

## Finding 3: genuine chaining forces two new host-readback synchronization points

**Decision**: `dispatch_multi_stage` for the chained struct must split a `[start, end]`
stage range into up to three sub-batches — `[start..3]`, `[4..6]`, `[7..end]` (only the
sub-ranges actually present in `[start, end]`) — each ending in its own
`submit()`/`wait_for_fence()`, with a host read of the relevant count in between. Stage
ranges that don't cross stage 3→4 or 6→7 stay a single command buffer/submit, same as
today.

**Rationale**: `n_unique`/`n_brt_nodes` (needed to size stage 4's workgroup count) and
`n_octree_nodes` (bookkeeping value, read after stage 6) are `const` fields on
`VkAppData_Safe`, fixed at construction from the golden — so today's dispatcher never
needs a host readback mid-chunk, and batches an entire `[start, end]` range into one
command buffer. A genuinely chained struct has no golden to fall back on: `n_unique`
only becomes known once stage 3's dedup shader has actually run, and the host must read
it back before it can correctly size stage 4's dispatch call (`vkCmdDispatch`'s
workgroup count is a host-side argument, not something the GPU can compute for itself
mid-command-buffer). Confirmed this is exactly the same constraint CUDA's own chained
path already has — `apps/tree/cuda/dispatchers.cu`'s `run_stage_4`(`_async`) has a
`// REQUIRED sync: the host reads u_num_selected_out below to size every later stage's
launch -- a true device->host data dependency, not framework tax` comment immediately
after stage 3, and an equivalent required sync after stage 6 for `n_octree_nodes`. CUDA
can do this cheaply because its buffers are zero-copy pinned host-visible memory
(`cudaDeviceSynchronize()` is enough); Vulkan's non-coherent `HOST_CACHED` memory needs
the existing `flush_touched()`/`invalidate_touched()` cache-maintenance calls already
implemented in `platform/engine/vulkan/sequence.cpp` (built for the Mali coherency
defect fix) — `wait_for_fence()` already calls `invalidate_touched()`, so the mechanism
to make a post-fence host read see fresh GPU writes already exists; the new work is
purely in the record-scheduling logic (rearranging the sub-batch/submit *AND* the
readback and setter call at the right place), not inventing new cache maintenance.

**Alternatives considered**: keeping the single-submit-per-chunk model and using a
GPU-side indirect-dispatch scheme (so the count never needs to reach the host mid-chunk)
— rejected as out of scope for this migration; it would be a genuine algorithm redesign
of stages 4 and 7's dispatch mechanism (indirect compute dispatch, a different GLSL/host
API surface) rather than "do the same [chaining] Vulkan already does for CUDA/OMP," and
the CUDA precedent this feature explicitly mirrors uses the same synchronous
host-readback approach.

## Finding 4: the existing `BT_DECLARE_TREE_DIFF_TESTS_APPDATA` harness needs no changes

**Decision**: `apps/tree/tree_diff_oracle.hpp`'s `tree::AppData`-based `CheckStageN`
functions, `RunAndCheckStageAppData`, and the `BT_DECLARE_TREE_DIFF_TESTS_APPDATA` macro
(added this session for CUDA/OMP) are already generic over any `Runner` whose `AppData`
type is (or derives from) `tree::AppData` — a Vulkan `Runner` supplying
`tree::vulkan::VkAppData` and a `RunStage` that calls the new Vulkan dispatch overloads
is a drop-in use of the existing macro, no header changes required.

**Rationale**: confirmed by reading the macro/template definitions directly — they are
parametrized purely on `Runner::AppData`/`Runner::Available()`/`Runner::RunStage()`
and construct the OMP reference (`ref`) independently of which backend `out` came from.

## Finding 5: `VkAppData_Safe` and the golden-decoupled test/profiler paths stay untouched

**Decision**: no existing file's `SafeAppData`/`VkAppData_Safe` code path is modified;
every change is additive (new struct, new overloads, new test suite declared alongside
the existing one, new `AppTraits` specialization only where the migrated test file
requires it).

**Rationale**: matches this session's established precedent for CUDA/OMP (Phase 2/3) and
the spec's explicit Assumption that `VkAppData_Safe` is not deleted here.
