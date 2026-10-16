# Design

## Context

See proposal.md (Why) for the per-stage numbers. The current state of the two kernels:

- **`new_cifar_sparse_conv2d_k3s1p1.comp`:** one invocation per output (n, out_c, oh, ow), workgroup
  256, grid `div_ceil(N*K*H*W, 256)` at five dispatch sites in
  `apps/cifar-sparse/vulkan/dispatchers.cpp`.
  - Each invocation walks its output channel's CSR row. Per nonzero it loads `col_idx`, decodes
    `(in_c, ky, kx)` with literal 9/3, loads the weight, and loads one input value.
  - When `subgroupAllEqual(out_c)` holds, the row bounds come from `subgroupBroadcastFirst`, making the
    walk scalar on AMD. On Mali the same instructions run per lane. The fallback is the per-lane walk.
  - No shared memory is used.
- **`new_cifar_dense_conv2d_k3s1p1.comp`:** one invocation per (n, 4-channel block, oh, ow), grid
  `div_ceil(N*(K/4)*H*W, 256)`.
  - It is already `subgroup_size_dependent`. `SUBGROUP_SIZE >= 32` takes the branchless,
    uniform-weight path; below 32 it takes `accumulate_branchy`, which is the previous kernel verbatim.
  - Per input channel, each invocation loads 9 taps and 4 x 9 weights from global memory, so the
    weights are fetched once per lane, not once per workgroup.
- **Where the cost is, as far as is known:** both kernels re-load the same value in many lanes. In the
  dense kernel every lane of a workgroup that shares an output-channel block loads the same 36
  weights. In the sparse kernel every lane loads the same `col_idx` and weight. Each output pixel loads
  its 9-tap window alone, although neighbours overlap. AMD hides the first through scalar loads; Mali
  has no scalar unit (the measured -42 % / 0 % split and the +12 % slowdown of branchless loads on the
  Pixel are consistent with this). Inferred, not measured on Mali: no `malioc` report or Mali counter
  data exists yet.
- **Tests:** `SparseConvK3S1P1.*` and `DenseConvK3S1P1.*` compare against a double-precision reference
  at 6x6 and 8x8. The differential suites run every stage against the OMP oracle. On the Pixel they
  exercise the Mali path.

## Goals / Non-Goals

**Goals:**
- Each value loaded from global memory on the Mali path feeds several outputs, without relying on
  subgroup-uniform (scalar) loads.
- One shader source per kernel, with the Mali path selected at pipeline creation. No new shader names,
  no device lists.
- A measured decision per kernel and per technique, recorded with its numbers.

**Non-Goals:**
- Changing the subgroup >= 32 paths (spec: "The subgroup-32-and-wider path does not change").
- Reduced precision, cooperative matrix, conv+pool fusion, the FC layers.
- Tuning for a specific Mali generation beyond the Pixel 7a's G710. The path is selected by subgroup
  size, so any other subgroup-16 Mali also takes it, and its correctness tests run there.

## Decisions

### D1. Selection: the existing `SUBGROUP_SIZE` specialization constant
Both kernels select the Mali path with `if (SUBGROUP_SIZE < 32)` on spec constant 0. The dense kernel
already does this. The sparse kernel's algorithm becomes `subgroup_size_dependent()` in its dispatcher
and in `SparseConvK3S1P1`'s test.

The branch is resolved at pipeline creation, so the dead path costs nothing at run time.

*Alternatives:* a second shader per kernel chosen by the dispatcher from `compute_limits()` (two
sources to keep in step, and two pipeline objects); vendor-ID lists (rejected by the pipeline-config
spec's "subgroup size comes from the running device").

### D2. Reuse through output blocking first, shared memory second
Techniques, in the order they are tried. Each is measured before the next one is added:

1. **Pixel blocking along W (both kernels).** One invocation computes `PX` adjacent output pixels of a
   row (start with `PX = 4`; all model maps are 8, 16 or 32 wide).
   - Dense: per input channel, the 3 x (PX+2) input window is loaded once (18 loads for 4 outputs
     instead of 36) and each weight is reused PX times.
   - Sparse: each nonzero is decoded once, its weight is loaded once, and it feeds PX inputs that are
     contiguous in memory (vectorizable when aligned).
   - This needs no shared memory and no barrier.
2. **Weights / CSR row in shared memory, per workgroup.** The workgroup is mapped so that it owns one
   output channel (sparse) or one channel block (dense), as `gl_WorkGroupID.y`. It stages that row in
   shared memory in bounded chunks, and the lanes read it from there instead of each fetching it from
   global memory. This is the Mali counterpart of the AMD scalar load.
3. **Input tile with halo in shared memory (dense only).** Only if 1 and 2 leave the dense kernel
   load-bound. The sparse kernel's CSR row touches arbitrary input channels, so an input tile cannot be
   reused across them.

*Alternatives considered:*
- Vectorizing the channel loop with vec4 over `in_c`: the data layout is NCHW, so channels are not
  contiguous.
- An implicit-GEMM rewrite: large, and its payoff on Mali without matrix units is unclear.
- FP16: out of scope (proposal).

### D3. Dispatch geometry from one contract per kernel
Pixel blocking (D2.1) changes the invocation count, and the count depends on the path, so the
dispatcher must know which path the running GPU takes. A small header next to the shaders (the
`linear_bt_contract.hpp` pattern) gives each kernel's outputs per invocation for a given subgroup size.
Both the dispatchers and the tests size the grid from it.

A mismatch must fail a test, not silently leave outputs uncomputed. The existing `-1` output fill in the
conv tests already catches that.

### D4. Ragged shapes are handled in the kernel, not by the dispatcher
A row whose width is not a multiple of `PX` runs a masked tail. The masked lanes still take part in any
barrier (D2.2) and never write. A workgroup's last channel chunk may be partial.

The tests add a 7x5 map (odd width, not a multiple of 4 or of the subgroup size) to the existing 6x6
and 8x8, and a CSR row whose nonzeros all fall in the padded border (spec scenarios).

### D5. Measurement method
- **Device A/B:** `bm-prof-cifar-{dense,sparse}-vk`, isolated scenario, on the Pixel 7a and the
  Samsung. Previous and new binaries are run 3 rounds interleaved with a 20 s cooldown, using the same
  scripts as the 2026-10-15/16 sessions (build both binaries from two worktrees with the Android
  preset, push both, and alternate). The gate is the spec's "A Mali path ships only if it is measured
  faster".
- **Static:** `malioc --format json` on the compiled Mali path, if Arm Performance Studio is installed.
  It gives register pressure (spills, and work registers against the occupancy limit), the
  arithmetic / load-store / texture cycle split, and the bound unit. It is used to pick D2 steps and to
  explain results, not as a gate.
- **Not on rocky:** rocky-ryzen runs only the correctness suites.

## Risks / Trade-offs

- **[A gain alone does not survive the pipeline]** (thermals, interference) → The gate is per stage,
  and the end-to-end effect is reported separately with a `bm-gen-logs` schedule run. It is not
  required to pass.
- **[Pixel blocking raises register use and lowers occupancy]** → Keep `PX` and `OC_TILE` as
  compile-time constants, try `PX = 2` and `4`, and check spills with `malioc`.
- **[Shared-memory staging adds barriers]**; with only 16-wide subgroups, small workgroups pay for them
  → Stage in chunks sized to the workgroup, and measure against D2.1 alone. Keep D2.2 only if it adds
  beyond D2.1.
- **[Real-weight CSR rows have very different lengths]** (25 % density per layer, uneven per channel),
  so workgroup-per-channel loads are unbalanced → Acceptable for the gate. The differential suite with
  real weights checks correctness. The imbalance is recorded if it shows.
- **[No `malioc`]** → Device A/B only. The design does not depend on it.
- **[A second subgroup-16 Mali in the future behaves differently]** → It takes the same path and its
  correctness tests run. Speed is only claimed for the Pixel 7a.

## Migration Plan

The Mali path ships per kernel, behind the spec's gate. If a technique fails the gate, the shader keeps
the previous kernel for `SUBGROUP_SIZE < 32`, as the dense kernel does today, so rollback means "do not
enable the new branch". No data, schedule or API format changes.

## Open Questions

- Whether Arm Performance Studio can be installed on the build box (licence acceptance, offline
  install). It does not change the approach (D5).
