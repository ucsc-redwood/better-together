# Code review — 2026-06-18

> From-scratch adversarial review of BetterTogether, weighted toward **build/portability**
> and **test-methodology/maintainability**. Every finding below survived per-finding
> adversarial verification by two independent verifiers (CONFIRMED = both agree real).
> Severities here are the **post-verification** ratings (verifiers downgraded several
> "high"s to medium/low once reachability was checked); the original author severity is
> noted in each section where it differed.

## Resolution status (updated 2026-06-18)

**34 of 37 findings fixed, full-fleet green** (OMP pc 10/10 · Vulkan rocky diff+engine+runtime ·
CUDA Jetson diff+runtime · optimizer 14 · #6 profiler smoke-runs valid JSONL on Jetson). See the
branch history (`refactor/component-structure`) — each fix commit names its finding number.

- **DONE (33/34 confirmed + D2):** #1–#28, #30–#34, and D2. Highlights: the two real z3 scheduling
  bugs (#1/#2), the Vulkan `Sequence` leak (#5), CUDA pinned guard (#15), the worker-terminate
  hardening (D2), bm_prof dedup −346 LoC (#6), per-backend tolerance (#24), the tree leaf-edge
  oracle (#7), the fleet-coverage manifest (#8).
- **DEFERRED — #29 (SafeAppData split):** architectural, not a quick fix. The 3 backends' dispatchers
  read SafeAppData's golden buffers AS STAGE INPUTS, so the naive lean/golden split starves them; it
  needs a dispatcher restructuring (a separate effort, outside P0–P6). Left as backlog.
- **NEEDS A HUMAN CALL — D1, D3 (disputed):** verifiers split. D1 (`target_sources` can silently
  drop a new kernel source) and D3 (`run_pipeline` empty core vector) — both cheap to harden
  defensively or to declare "by design"; not yet actioned.

## Known issue surfaced during fleet verification (out of scope, pre-existing)

**`tree`-on-Vulkan fails on the Adreno 540 (`ce0717178d7758b00b7e`).** This phone is NOT a
documented test target (the matrix is Pixel-Mali-16 + Samsung-Mali-32), but running it
revealed: `cifar-dense-vk` passes, while `tree-vk`'s radix-sort chain (stages 2/3/4/5/7)
produces an all-zero sorted buffer (`ref=4441, out=0` at index 0). Root cause: the tree radix
sort selects a fixed-subgroup shader variant `multi_radixsort_warp{16,32,64}` from the device
JSON's declared `gpu.subgroup_size`, and these SPIR-V variants (validated on Mali) are
incompatible with Adreno's wave semantics — Adreno runs wave64 internally even when subgroup=32
is requested, unless pinned via `VK_EXT_subgroup_size_control`. Tested both 32 (morton+stage6
pass, sort chain zeros) and 64 (all 7 stages fail) — neither works, so it is NOT a JSON value
fix; it's a genuine Adreno GPU-porting task. **Not a regression** from this branch (the tree
Vulkan kernels/shaders are byte-identical; this device was simply never tested). Deferred as a
separate effort; only relevant if Adreno phones become a test target.

## TL;DR

30 confirmed findings, 3 disputed. The only **reachable, manifesting correctness bugs**
are in the **z3 optimizer**: the `gapness` objective is degenerate (it rewards the single
*slowest* chunk, gap=0, over any pipelined schedule), and the `UNAVAILABLE=1e9` sentinel
that is supposed to keep z3 off absent hardware **does not protect that objective** — so
the default mode can emit schedules that pin every stage to a CPU tier the device lacks,
yielding an unrunnable schedule / dead worker. Every shipped `schedules_*_gapness.json`
is built on this. **Fix these two first** (they share one root cause: gap=0 single-chunk
degeneracy).

After that it is mostly **build/portability robustness** and **test-methodology gaps**,
none of which break the current fleet but several of which silently erode guarantees as
the codebase grows:
- A real **VkCommandPool/VkFence leak** in `~Sequence()` plus a device-destroyed-with-live-children
  spec violation (bounded, teardown-only).
- The everyday gate (`ctest -L omp`) has **the least independent power exactly for the GPU
  backends** — GPU differentials skip-to-green off-fleet, and several structural checks are
  self-consistency tautologies or throw-away-the-result.
- The single wall protecting the refactor's headline invariant (`guard_runtime_agnostic.py`)
  **hardcodes the app vocabulary** instead of reading `vocab.json`, so app #4's leaks pass
  silently.
- A pile of **duplication/dead-code** debt: `bm_prof` main() copy-pasted 6× (~820 lines),
  dead CUDA/Vulkan resource classes, stale doc paths pointing at the deleted `builtin-apps/`.

Several originally-"high" portability findings (CUDA pinned-free pointer mismatch, kiss-vk
iGPU hard-throw) are **real but dormant** — they cannot trigger on any current or planned
fleet hardware — so they sit in the table at low.

## Severity-sorted findings

| # | Title | Severity | Category | File:line | One-line fix |
|---|---|---|---|---|---|
| 1 | Gapness objective is degenerate: rewards the slowest single chunk (gap=0) | high | correctness (z3) | `optimizer/smt/constraints.py:68-73` | Minimize `T_max` (makespan); use gap only as a post-makespan tie-breaker, or drop it |
| 2 | `UNAVAILABLE=1e9` sentinel does not protect gapness → unrunnable schedules on absent HW | high | correctness (z3) | `optimizer/smt/data_loader.py:12,53` | Make absent tiers structurally impossible: omit/`Not()` `x[(i,c)]` for any PU with no data |
| 3 | Device-spec codegen embeds raw JSON with no schema validation in build/test loop | high → medium | build/codegen | `scripts/embed_device_specs.py:33-37` | Register `validate_devices.py` as a `ctest` (`LABELS omp;guard`) under `if(BT_PYTHON)` |
| 4 | `guard_runtime_agnostic.py` hardcodes the app vocabulary, decoupled from `vocab.json` | high → medium | maintainability | `scripts/guard_runtime_agnostic.py:19` | Derive the forbidden app-namespace list from `vocab.json app_stages`; keep `#include "apps/"` backstop |
| 5 | `~Sequence()` leaks `VkCommandPool`/`VkFence`; device destroyed with live children | high → medium | memory-safety | `platform/engine/vulkan/sequence.cpp:29-33` | Destroy `command_pool_` + `fence_` in `~Sequence()` before `BaseEngine` destroys the device |
| 6 | `bm_prof` main() duplicated verbatim across all 6 (app×backend) cells (~820 lines) | high | duplication | `profiler/tree-cu/bm_prof.cu:34-145` | Hoist driver into `bm_prof_common.hpp` templated on a GPU-timer policy; cells shrink to ~10 lines |
| 7 | Tree stage-7 leaf-edge topology is never differentially validated on any backend | high → medium | test-gap | `apps/tree/tree_diff_oracle.hpp:177-180` | Fail on residual leaf-edge diff (minus multiply-written slots); port leaf oracle into a backend-parametrized anchor |
| 8 | OMP-as-oracle gate gives no signal on CUDA/Vulkan; GPU coverage skip-exits-0 | medium | test-gap | `cmake/bt_targets.cmake:38-56` | Emit per-cell ran/skipped markers + checked-in fleet-coverage manifest; add OMP anchors for tree stages 2/3/6 |
| 9 | Concurrent-pipeline oracle checks only the terminal stage | medium → low/med | test-gap | `apps/cifar-dense/omp/test_pipeline_main.cpp:48` | In `CheckItem`, also assert a couple of intermediate stage buffers at per-stage (1e-3) tolerance |
| 10 | Vulkan `Runner::Available()` hardcodes `true` → absent iGPU is a hard crash, not a skip | medium | test-gap | `apps/tree/vulkan/test_main.cpp:24` | Probe-enumerate for `eIntegratedGpu` in try/catch; return false so cases `GTEST_SKIP` like CUDA |
| 11 | 32-bit Android is built but undeployable: no preset; deploy scripts hardcode aarch64 libc++ | medium → low | build/portability | `scripts/run-on-android.sh:37-39` | Derive libc++ arch from build ABI; add an `android32` preset |
| 12 | `bt_add_*_run`/`_app` forward `ARGN` as sources vs libs — silent footgun | medium → low | build | `cmake/bt_targets.cmake:6-32` | Add a one-line `ARGN = extra sources` comment to each `_run` helper (or use `cmake_parse_arguments`) |
| 13 | kiss-vk hard-selects `eIntegratedGpu` with fatal throw, no fallback, opaque message | medium → low | portability | `platform/engine/vulkan/base_engine.cpp:153-176` | On throw, log every enumerated device's `deviceType`+name (keep iGPU preference) |
| 14 | `baselines.py` hardcodes `_CPU_TIERS`, duplicating generated vocab; no drift guard | medium → low/med | duplication | `optimizer/smt/baselines.py:22` | Replace literal with `from .bt_vocab import CPU_TIERS as _CPU_TIERS` (one line) |
| 15 | `CudaPinnedResource::do_deallocate` frees the device ptr, not the host ptr | high → low | portability | `platform/engine/cuda/cu_mem_resource.cu:89-113` | Track/return the host pointer and `cudaFreeHost` that (UMA aliasing hides it today) |
| 16 | `CudaManagedResource` (~35 lines) no longer instantiated anywhere — dead | medium → low | dead-code | `platform/engine/cuda/cu_mem_resource.cu:47-90` | Flag for removal (class + impl + `requires`-clause alternative in `manager.cuh:15`) |
| 17 | `Sequence::launch_kernel_async()` and `sync()` are `[[deprecated]]` with zero callers | medium → low | dead-code | `platform/engine/vulkan/sequence.cpp:158-179` | Remove both from `sequence.hpp:24-25` and `sequence.cpp:158-179` |
| 18 | `Sequence::destroy()` declared-but-never-defined, never-called | medium → low | dead-code | `platform/engine/vulkan/sequence.hpp:41` | Delete the `void destroy();` declaration |
| 19 | Per-app pipeline-contract constants restated in every backend test TU | medium → low | duplication | `apps/tree/omp/test_pipeline_main.cpp:26` | Hoist `kNumStages`/`kNumToProcess` into shared `apps/<app>/traits.hpp`, seeded from `bt_vocab.hpp` |
| 20 | Codegen `custom_command` OUTPUTs write into the source tree → read-only CI fails | low | build | `platform/CMakeLists.txt:28-35` | Generate into `${CMAKE_BINARY_DIR}/generated` (mirror `bt_git_sha`); keep committed copies as no-python fallback |
| 21 | vocab codegen duplicates device-specs codegen inline instead of reusing `bt_codegen` | low | duplication | `platform/CMakeLists.txt:26-47` | Route device-specs through `bt_codegen` (pass globbed list as `DEPENDS`) |
| 22 | `profiler` `run-pipe-*-cu` hand-enumerated next to a foreach over the same app list | low | duplication | `profiler/CMakeLists.txt:7-19` | Hoist `set(BT_PROFILED_APPS …)` and drive both loops from it |
| 23 | Codegen scripts use locale-dependent `read_text()/write_text()` | low | build | `scripts/embed_device_specs.py:34,41` | Pass `encoding="utf-8"` to every `read_text()/write_text()` |
| 24 | GPU per-backend tolerance advertised but not applied; one shared `kRtol/kAtol` | low | test-gap | `apps/cifar-dense/cifar_dense_diff_oracle.hpp:23-32` | Parameterize tolerance via the `Runner`; tight for OMP/CUDA, relaxed only for Vulkan |
| 25 | Empty-range comparisons pass vacuously; no nonzero count guard in per-stage tree checks | low | test-gap | `apps/tree/tree_diff_oracle.hpp:67-89` | `ASSERT_GT(n, 0)` so a zero-length compare is a broken test, not a pass |
| 26 | `vocab.json` `tree=7` contradicts canonical AlexNetCIFAR(11); dead `cifar-sparse` key | low | maintainability | `vocab.json:17-21` | Drop/document the unused `cifar-sparse` key; `_comment` these as legacy SmallAlexNet counts |
| 27 | Codegen source-of-truth files document a stale `builtin-apps/` output path (post-P5) | medium → low | maintainability | `vocab.json:2` | Retarget the three comments to `platform/vocab/generated/bt_vocab.hpp` |
| 28 | instruction-for-ai docs cite pre-refactor paths (`common/kiss-vk/`, `pipeline/`, `app.hpp`) | medium → low | maintainability | `docs/instruction-for-ai/05-profiling.md:37` | Rewrite to post-refactor paths; optional CI grep that fails on unresolvable path tokens |
| 29 | `SafeAppData` fuses the differential-oracle golden into the production pooled type | medium | maintainability | `apps/tree/safe_tree_appdata.hpp:13` | Split a lean `tree::AppData` (input+out) from a test-only golden subclass; pool the lean type |
| 30 | `submit()` logs the wrong method name (`"launch_kernel_async()"`) | low | duplication | `platform/engine/vulkan/sequence.cpp:185` | Change the string to `"Sequence::submit()"` |
| 31 | Stale commented-out experiment blocks left in `sequence.{hpp,cpp}` | low | dead-code | `platform/engine/vulkan/sequence.cpp:137-156,226-240` | Delete the commented bodies + decls + dead `waitForFences` block |
| 32 | `main.cu` demo entry duplicated across the 3 CUDA cells | low | duplication | `profiler/tree-cu/main.cu:6-46` | Move `run()`/`main()` into a shared header taking the stage-1 OMP dispatch as a lambda |
| 33 | "Sorted by Max Time" summary prints mismatched UID (sorted metrics, unsorted UID) | low | correctness | `optimizer/smt/solver.py:102-104` | Read the uid from the sorted tuple (carry uid inside the tuple) |
| 34 | GPU chunks lack schema-required `hardware`; validity depends on orchestrator patch | low | maintainability | `optimizer/smt/solution_analyzer.py:142-169` | Set `hardware` inside `get_detailed_solution`, threading the gpu backend token through |

---

## z3 optimizer correctness (the only manifesting bugs)

### 1. Gapness objective is degenerate — high

**Problem.** The `gapness` objective minimizes `T_max − T_min` over maximal chunks. A
single-chunk schedule (all stages on one PU) trivially hits `gapness=0` regardless of how
slow that PU is, and gap=0 is the global minimum. So z3 systematically selects the
**slowest single-PU schedule** and ignores the pipelining the whole framework exists to
exploit.

**Evidence.** `constraints.py:68-73`: `opt.add(Gapness == T_max - T_min)` then
`if minimize_mode=='gapness': opt.minimize(Gapness)`. `T_min` is only bounded `>0` plus
`seg_sum >= T_min` for maximal segments, so a single chunk forces `T_min=T_max=sum → gap 0`.
Reproduced live by both verifiers: fixture `[Little=10ms, GPU=1ms]×4` → gapness picks
`['Little']×4` (makespan 40, gap 0) while `max_time` correctly picks `['GPU']×4` (makespan 4).
On real 4-tier vocab, the absent/slow-tier single-chunk solution is co-optimal at gap=0 and
is the first solution z3 returns. Every shipped `schedules_*_gapness.json` is built on this;
one verifier found a `jetson/tree/cu` gapness pick with makespan `7e9` (the UNAVAILABLE
sentinel). This is **distinct from** the previously-fixed "minimize_mode not threaded" bug —
the mode is now correctly threaded; threading merely *exposed* that the objective itself is
unsound. `test_minimize_mode.py:48` even enshrines the ~108ms single-PU gapness optimum as
expected.

**Fix.** Drop gapness as a primary objective. Minimize `T_max` (makespan); if load-balance
is wanted, apply it lexicographically *after* fixing makespan, or normalize gap by makespan.
At minimum, document that `schedules_*_gapness.json` are not throughput-optimal and exclude
them from selection. `max_time` is already the runner default (`03_run_schedule.py:93`), so
degenerate gapness schedules are opt-in today.

### 2. `UNAVAILABLE=1e9` sentinel does not protect the gapness objective — high

**Problem.** `data_loader` encodes a CPU tier the device physically lacks as
`UNAVAILABLE=1e9` so that z3 "never assigns a stage to hardware that does not exist." That
reasoning only holds for a **cost-minimizing** objective. Under gapness, a single
all-on-an-absent-tier chunk has `gap=0` — exactly as optimal as a single real-tier chunk —
so z3 is free to pick the absent tier and emit an **unrunnable** schedule.

**Evidence.** `data_loader.py:12,53`. Both verifiers reproduced live: with the real 4-tier
vocab on the docstring's own Big-only-machine example, gapness returns the absent-tier
schedule as the first co-optimal (gap=0) solution. Downstream: a chunk on an absent tier
maps via `config_reader.hpp` to e.g. `kMediumCore`; `get_cores_by_type`
(`device_registry.hpp:65`) returns an empty affinity vector → `omp_dispatch` runs with 0
cores (dead worker). The `schedule_unrunnable_reason()` guard exists but is **not** invoked
on the `pipeline_runner` dispatch path. gapness is the default `minimize_mode`, and
tier-absent devices are real, so this is the common path.

**Fix.** Make absent tiers structurally impossible rather than merely expensive: omit the
decision variable `x[(i,c)]` (or add `opt.add(Not(x[(i,c)]))`) for any PU with no measured
data. This protects every objective regardless of what is minimized. (Fixing #1 removes this
specific manifestation, but the structural guard is the robust fix.)

### 33. "Sorted by Max Time" summary prints a mismatched UID — low

**Problem / evidence.** `solver.py:102-104`: the loop iterates `sorted_solutions`
(re-sorted by max_time) but reads `detailed_solutions[i]['uid']`, which is still in
solver-discovery order — so the printed gap/max_time and the printed UID belong to different
solutions. Console-output-only; the JSON written to disk (`detailed_solutions`, unchanged
order) is unaffected, so it cannot corrupt schedules — only mislead a human reading solver
logs. **Fix.** Carry the uid inside the tuple and read it from the sorted tuple.

### 34. GPU chunks lack the schema-required `hardware` field — low (maintainability)

**Problem / evidence.** `get_detailed_solution` (`solution_analyzer.py:142-169`) builds GPU
chunks without the `hardware` key, but `schedule.schema.json:58-66` requires it iff
`core_type=='GPU'`. The only thing making output valid is `02_gen_schedule_merged.py:173-176`
injecting `chunk['hardware']=gpu_backend` before validation — and `test_schedule_contract.py`
has to mirror the same injection, so the landmine is already duplicated at two sites. Any
future caller of `solve_optimization_problem + dump_solutions_as_json` that skips the patch
fails schema validation. **Fix.** Set `hardware` inside `get_detailed_solution` itself,
threading the gpu backend token through `solve_optimization_problem`; then dump is
self-validating and 02's patch loop can be removed.

---

## Build & portability

### 3. Device-spec codegen embeds raw JSON with no schema validation — high → medium

**Problem.** `embed_device_specs.py` bakes each `devices/*.json` into
`device_specs_embedded.hpp` verbatim with zero validation. The repo *has* a validator
(`validate_devices.py`: structural schema checks + a golden core-topology characterization
that locks affinity maps against silent drift) but grep shows it is wired into **nothing** —
no CMake, no `.cmake`, no `ctest`, no CI. So the schema and golden guard are dead weight.

**Evidence.** `grep validate_devices` across CMake/`.cmake`/`.sh` → 0 hits.
`embed_device_specs.py:34,37` reads and raw-string-wraps with no `json.loads`/schema. The
embed step *is* wired into the build (`platform/CMakeLists.txt:30`). An established
`BT_PYTHON` ctest-guard precedent exists (`guard-runtime-agnostic`, `CMakeLists.txt:138`).

**Severity note.** Downgraded because the *loud* failure modes (malformed JSON, unknown core
`type`) throw at `DeviceRegistry()` construction — a confusing but deterministic on-device
exception, not silent. The genuinely **silent** class is narrower: a structurally-valid but
*wrong* value (flipped `pinnable`, wrong tier `type`, topology drift) parses cleanly and
silently yields a wrong affinity map — exactly what the golden check guards and what nothing
currently runs.

**Fix.** Register `validate_devices.py` as a ctest under `if(BT_PYTHON)` with
`LABELS "omp;guard"` (mirroring `guard-runtime-agnostic`), and/or `json.loads`+schema-validate
in `embed_device_specs.py` before embedding.

### 11. 32-bit Android is built but undeployable — medium → low

**Problem.** `android-arm32.cmake` exists for the documented 32-bit-only device and 15 arm32
binaries are producible, but (1) `CMakePresets.json` has only the arm64 `android` preset, and
(2) both deploy scripts locate libc++ with `find … -path '*aarch64*'`
(`run-on-android.sh:38`, `run-mali-oracle.sh:28`). On an arm32 build that pushes the aarch64
`libc++_shared.so` next to an armeabi-v7a binary → load-time ABI failure.

**Severity note.** Downgraded: the gate does **not** build or deploy arm32 (the inventory
file is a hand-produced, gitignored snapshot; the gate only builds the `pc` preset and never
calls the deploy scripts). The defect surfaces only if a human manually configures the arm32
toolchain *and* repoints a deploy script at it — undocumented, currently-unexercised tooling
debt, not active breakage.

**Fix.** Derive the libc++ arch from the build ABI; add an `android32` preset so the live
path is reproducible and CI-gated.

### 12. `bt_add_*_run`/`_app` forward `ARGN` as sources vs libs — medium → low

**Problem / evidence.** `cmake/bt_targets.cmake`: `bt_add_omp_run`/`bt_add_vk_run` (lines
7/25) splice `${ARGN}` into `add_executable()` as **sources**, while
`bt_add_omp_app`/`_cuda_app`/`_vk_app` (lines 13/21/31) splice `${ARGN}` into
`target_link_libraries()` as **libraries** — same slot, opposite meaning, near-identical
names. (The real split is run/test vs app.) A future contributor copying the `_app` pattern
into a `_run` call would mis-fire.

**Severity note.** Downgraded from the "silent miscompile" framing: both confusion cases are
**loud** fail-fast errors (a lib target as a source → configure "cannot find source file"; a
`.cpp` as a lib → link error), and every current call site follows the convention correctly.
Latent maintainability nit, not a live defect. **Fix.** Add a one-line `ARGN = extra sources`
comment to each `_run` helper (mirroring the existing `_app` comment), or use
`cmake_parse_arguments` with `SOURCES`/`LIBS`.

### 13. kiss-vk hard-selects `eIntegratedGpu` with fatal throw — medium → low

**Problem / evidence.** `base_engine.cpp:165-171` takes the first device exactly matching the
requested type (default `eIntegratedGpu`, `base_engine.hpp:36`) and throws
`std::runtime_error("No integrated GPU found")` with no fallback and no enumeration of what
*was* found. On a discrete-only host (the RTX build box) this throws at engine construction
(corroborated by CLAUDE.md). The iGPU restriction is **deliberate and documented** (and
load-bearing for the unified-memory/coherency assumptions), so silently falling back to a
discrete GPU is the wrong fix. **Fix (sound, low-cost part only).** On the throw, enumerate
and log every device's `deviceType`+name so the failure is diagnosable instead of opaque.

### 15. `CudaPinnedResource::do_deallocate` frees the device pointer — high → low

**Problem / evidence.** `do_allocate()` (`cu_mem_resource.cu`) calls `cudaHostAlloc` then
`cudaHostGetDevicePointer` and **returns the device pointer** (line 107) as the allocation
handle; `do_deallocate` passes that device pointer to `cudaFreeHost` (line 112), whose
contract requires the original *host* pointer. On Jetson Orin UMA the two pointers alias, so
the free succeeds — which is why it has never surfaced. `CudaPinnedResource` is the live
allocator for all three CUDA dispatchers.

**Severity note.** Downgraded from high to **low**: per CLAUDE.md, CUDA only ever runs
cross-compiled on the Jetson (a Tegra UMA device); the one non-UMA NVIDIA GPU (the discrete
RTX) is explicitly CUDA build-only because CUDA 13 breaks the build, and never executes this
code. So this is a genuine contract violation but **dormant** — no abort, no leak on any
current or planned target. **Fix.** Track the host pointer for the allocation's lifetime and
free that (on UMA `h_ptr==d_ptr`, so kernels are unaffected).

### 20. Codegen `custom_command` OUTPUTs write into the source tree — low

**Problem / evidence.** Both codegens declare `add_custom_command(OUTPUT <file-inside-source-tree>)`
(`platform/CMakeLists.txt:29` for `device_specs_embedded.hpp`; `bt_codegen.cmake:17` for
`bt_vocab.hpp`/`bt_vocab.py`) and are wired as build-time dependencies of `bt_core`. When
python is present (the common case), CMake re-runs the generator and the script does an
unconditional `OUT.write_text(...)` — the "identical bytes" mitigation prevents git churn but
not the `open()`-for-write. On a read-only source checkout (some CI sandboxes, packaging
builds, shared read-only worktrees) the write fails and the build errors despite a valid
committed header. One verifier reproduced this (`git archive` → out-of-source build →
`chmod -R a-w` source → `ninja bt_core` fails with `PermissionError [Errno 13]`). Narrow but
real. **Fix.** Generate into `${CMAKE_BINARY_DIR}/generated` and add it to the include path
(mirroring `bt_git_sha`), keeping committed copies only as the no-python fallback; or gate
regeneration behind an opt-in target like `bake-shaders`.

### 23. Codegen scripts use locale-dependent `read_text()/write_text()` — low

**Problem / evidence.** Both codegen scripts read JSON and write the generated header via
`read_text()/write_text()` with no `encoding=` (`embed_device_specs.py:34,41`;
`embed_vocab.py:63,82`), defaulting to `locale.getpreferredencoding()` and, on Windows,
universal-newline translation. That can break the load-bearing clean-tree invariant the
committed-fallback design rests on, or corrupt a future non-ASCII device `description`/`name`.
All `devices/*.json` are pure ASCII today and the fleet is Linux, so the real-world blast
radius is small — environment-conditional robustness gap, not a live bug. **Fix.** Pass
`encoding="utf-8"` to every `read_text()/write_text()` (also `embed_vocab.py:23`).

---

## Test methodology & coverage

### 7. Tree stage-7 leaf-edge topology never differentially validated — high → medium

**Problem.** `CheckStage7Topology` canonicalizes octree downward edges into INTERNAL and LEAF
sets, then deliberately **returns `AssertionSuccess()` whenever only leaf edges differ**. The
leaf-slot resolution (which `child[]` entries point at point indices — the entire output of
`process_link_leaf`) is compared and the result thrown away. The only independent leaf-link
validation is `TreeAnchorOmp.Stage7_TinyBruteForceOctree`, compiled **only** into the OMP
binary; CUDA/Vulkan `test_main` only emit `BT_DECLARE_TREE_DIFF_TESTS`. So a GPU kernel with a
wrong `child[]` index, wrong leaf-mask bit, or off-by-one leaf code produces an identical pass.

**Evidence.** `tree_diff_oracle.hpp:177-180` (the early `AssertionSuccess()` on leaf diff);
the brute-force at `omp/test_main.cpp:360-457` is OMP-only; `cuda/test_main.cu:30` and
`vulkan/test_main.cpp:30` only declare.

**Severity note.** Downgraded: it is not literally *zero* coverage — the leaf-mask
OR-reduction and per-node `cell_size`/`corner`/`node_mask` *are* checked, so leaf bugs that
perturb those are caught. The genuinely uncovered surface is the specific `child[]` leaf-slot
index. Also, naive "fail on any leaf diff" would false-positive on legitimate last-writer-wins
octant collisions. It is a real but bounded **GPU test-gap**, with no demonstrated kernel bug.

**Fix.** Make leaf-edge divergence a real failure once the multiset of multiply-written octant
slots is subtracted; better, port the geometry-keyed leaf oracle into a backend-parametrized
anchor so CUDA/Vulkan leaf linking is independently verified.

### 8. OMP-as-oracle gate gives no signal on GPU correctness — medium

**Problem.** The everyday gate is `ctest -L omp`, and a `GTEST_SKIP` counts as a pass. The
CUDA Runners `GTEST_SKIP` when `cudaGetDeviceCount==0`, so on any runner without the fleet the
entire CUDA/Vulkan differential matrix skips → green. Combined with the fact that the OMP diff
for the integer/structural tree stages 2/3/6 compares `_out` against a golden produced by the
**same OMP kernel** (a self-consistency tautology — stated outright at
`omp/test_main.cpp:38-45`), the only genuine cross-backend correctness evidence comes from
physical Jetson/rocky/phone runs. There is **no in-repo mechanism that fails if those fleet
runs are skipped** — an operator can ship believing "tests pass" while every GPU differential
silently no-op-skipped. The routinely-run label has the least independent power exactly for
the GPU backends.

**Evidence.** `cmake/bt_targets.cmake:38-56` attaches plain `LABELS omp/cuda/vulkan`; CUDA
`Available()` skips when no device; independent OMP anchors exist only for tree stages 1/4/5/7
(`omp/test_main.cpp:63-142`). (One verifier notes stage 2/sort is partially anchored via the
Morton ref; 3/unique and 6/prefix-sum genuinely lack a standalone anchor.) No fleet-coverage
manifest exists.

**Fix.** Emit a per-cell ran/skipped marker and a checked-in fleet-coverage manifest checked
in CI, so a skipped GPU differential is visibly a hole; add independent OMP anchors for the
un-anchored structural stages so the gate is not tautological for the structural stages the
GPU tests inherit as ground truth.

### 10. Vulkan `Runner::Available()` hardcodes `true` — medium

**Problem.** Every Vulkan `test_main` defines `static bool Available() { return true; }`,
unlike CUDA which probes `cudaGetDeviceCount`. The Vulkan engine constructor hard-selects
`eIntegratedGpu` and throws on a discrete-only box. Because `Available()` never returns false
and the dispatcher is constructed as a member when the Runner is instantiated, the
`if(!Available()) GTEST_SKIP()` path is **dead code for Vulkan** — running `ctest -L vulkan`
on the wrong host throws out of the test body and each affected case reports FAILURE instead
of an honest skip. So the vulkan label cannot be a uniform gate the way omp can.

**Evidence.** `apps/tree/vulkan/test_main.cpp:24` (identical at `cifar-dense/vulkan:20`),
contrast `cuda/test_main.cu:21-24`. (Correction to the finding: gtest catches per-test
exceptions by default, so it is N failures, not a single hard abort — operationally still
failures, not skips.) Bounded because CLAUDE.md already documents that vulkan must run only on
iGPU hosts. **Fix.** Make Vulkan `Available()` enumerate physical devices for `eIntegratedGpu`
in try/catch and return false if none.

### 9. Concurrent-pipeline oracle checks only the terminal stage — medium (verifiers split low/med)

**Problem / evidence.** The runtime/pipeline e2e tests validate each item with a single
per-item check on the **final** output (`test_pipeline_main.cpp:48` →
`CheckFinalPipeline`, `kE2eRtol=kE2eAtol=5e-3` — the loosest tolerance in the suite). The
ring's whole purpose is correct cross-chunk hand-off of intermediate `AppData` buffers, yet no
intermediate buffer is asserted against its per-stage reference in the concurrent path, even
though `CheckStage` references already exist. A fully dropped stage *is* caught (the final
reference is recomputed from the seed), but a subtle intermediate corruption numerically
swamped before the final 10-way linear can pass. Verifiers split: one notes pointer-handoff/
pool/visibility bugs tend to fail all-or-nothing (gross, already caught) so leans low; the
other holds the masking scenario is real and sits exactly on the component under test at the
loosest tolerance, so medium. **Fix.** In `CheckItem`, also assert a couple of intermediate
stage buffers (chunk-boundary + one mid-chunk) via `CheckStage` at the 1e-3 per-stage
tolerance.

### 24. GPU per-backend tolerance advertised but not applied — low

**Problem / evidence.** `cifar_dense_diff_oracle.hpp:23-32` defines `kRtol=1e-3`/`kAtol=1e-4`
once with a comment "tighten/loosen per backend," but all three backends include the same
header and call `CheckConv/CheckPool/CheckLinear` with the defaults — no per-backend override.
So the bound is whatever passes the worst backend (Vulkan fp32), and that loose bound is
applied to OMP and CUDA too, weakening their differential power (the reference accumulates in
double, so a tighter bound is feasible). Test-discriminating-power issue, not a correctness
defect. **Fix.** Parameterize tolerance through the `Runner`; OMP/CUDA tight (~1e-5..1e-4),
Vulkan relaxed to 1e-3.

### 25. Empty-range comparisons pass vacuously — low

**Problem / evidence.** `ExactEqual/NearEqual` return `AssertionSuccess()` when the compared
length `n` is 0 (`oracle.hpp:47-55,76-88`); the per-stage tree checks pass counts straight
from `get_n_unique()/get_n_brt_nodes()/get_n_octree_nodes()` with no `>0` guard
(`tree_diff_oracle.hpp:69,73,184`). The concurrent tree test *does* guard with `all_zero` +
`ASSERT_GT`, so the guard is inconsistent across harnesses. **Reachability caveat (both
verifiers):** the count is a `const` copied from the OMP golden, *not* from the
backend-under-test — so a no-op backend would compare nonzero golden vs zeroed output and
**fail**, not pass vacuously; the true `n==0` path requires the OMP oracle itself to yield
zero on the fixed canonical input (not reachable). A defense-in-depth/harness-consistency nit,
not a live blind spot. **Fix.** Add `ASSERT_GT(n, 0)`. (Do **not** add an
`EXPECT count != ref.size()` check — a count smaller than `ref.size()` is the intended
over-allocated-prefix design.)

---

## Maintainability & duplication

### 4. `guard_runtime_agnostic.py` hardcodes the app vocabulary — high → medium

**Problem.** The single build-time enforcement of the runtime↔app decoupling (the root
`CMakeLists.txt:17-18` comment states decoupling is enforced by *this guard*, not by include
scoping) is a regex with a hand-written closed list of app identifiers
(`tree::|cifar_dense|cifar_sparse|cifar::|octree::|morton|#include "apps/`). The repo has a
single source of truth for the app vocabulary (`vocab.json app_stages`) codegen'd into
C++/Python, yet the guard does not read it. When app #4 is added — the "mechanical, localized"
workflow the P0–P6 refactor was built to enable — a runtime header referencing `app4::` passes
the guard silently. The one wall protecting the refactor's headline invariant degrades as the
codebase grows.

**Severity note.** Downgraded because the generic `#include "apps/"` backstop is
vocab-independent and catches the **dominant** leak vector (the historically-actual leak,
`tree::SafeAppData`, flowed through an `apps/` include). The genuine residual gap is narrow: a
runtime header that forward-declares `namespace app4` and uses the type opaquely by
pointer/ref with no `apps/` include. Real but latent (all 3 current apps covered today; bites
only at app #4). **Fix.** Derive the forbidden app-namespace tokens from `vocab.json app_stages`
(a key→token mapping, since `octree`/`morton` are tree-internal and have no vocab key); keep
the `#include "apps/"` backstop; add a self-test that plants a known leak and asserts the guard
fails.

### 6. `bm_prof` main() duplicated verbatim across all 6 (app×backend) cells — high

**Problem.** The `bm_prof` driver is the one benchmark family in `profiler/` that was **not**
un-forked. Its sibling drivers (`bm_baseline`/`bm_fully_vs_normal`/`bm_gen_log`) were collapsed
to ~15-line thin wrappers passing the OMP-dispatch namespace as a lambda into a templated
`run()` in their `*_common.hpp`. `bm_prof` skipped this: the full ~140-line `main()` (env-knob
parsing, cudaEvent setup, the `time_once` lambda, the entire interference background-thread
block, the `RegisterBenchmark` loop, shutdown) is copy-pasted into 6 files (837 lines total).
`bm_prof_common.hpp` even *falsely* narrates that each cell "owns only its MEASURED LOOP plus
the app/backend strings and the OMP dispatch namespace" — in reality it owns the whole driver.

**Evidence.** `diff tree-cu/bm_prof.cu cifar-dense-cu/bm_prof.cu` → only 4 lines differ (doc
comment; two `tree::omp` vs `cifar_dense::omp`; the `emit_jsonl` app string). Same 4-line delta
for cifar-sparse and for all 3 VK cells. The siblings already template on a Timer policy
(`bm_fully_common.hpp` — `WallTimer` for VK, `CudaEventTimer` for CU), and the only genuine
per-backend difference in `bm_prof` is exactly the GPU-timing path. This is the single largest
copy-paste in the tree and a drift hazard (interference fixes must be hand-applied 6×).

**Fix.** Apply the sibling pattern: add
`bt_prof::run<AppDataT,DispatcherT,GpuTimer>(argc,argv,kNumStages,"tree","cuda",omp_lambda)`
to `bm_prof_common.hpp`, templating the GPU-timing path. Each cell shrinks to a ~10-line
`main()`, collapsing ~820 lines to ~120.

### 5. `~Sequence()` leaks `VkCommandPool`/`VkFence` — high → medium

**Problem.** `~Sequence()` destroys only `query_pool_`. The `command_pool_` (created at
`sequence.cpp:57`) and `fence_` (created at `sequence.cpp:64`) are never destroyed, and the
declared `Sequence::destroy()` (`sequence.hpp:41`) is never defined or called. At teardown
`BaseEngine::~BaseEngine()` calls `device_.destroy()` (`base_engine.cpp:69`) with a comment
asserting the sequence pools are "already destroyed" — false for the command pool and fence.
Destroying a `VkDevice` with live children is a spec violation
(VUID-vkDestroyDevice-device-05137).

**Severity note.** Downgraded high→medium by both verifiers: the leak is bounded (one command
pool + one fence per process, OS-reclaimed at exit), occurs only at teardown (not accumulating),
and the validation-error impact is conditional (the validation layer silently continues if
unavailable; tests are green). Member-declaration order (`engine` before `seq`) means
`~Sequence()` runs while the device is still valid, so the fix is straightforward.

**Fix.** In `~Sequence()` add
`if (command_pool_) device_ref_.destroyCommandPool(command_pool_); if (fence_) device_ref_.destroyFence(fence_);`
alongside the existing query-pool destroy, making the `base_engine.cpp` ordering claim actually
hold.

### 14. `baselines.py` hardcodes `_CPU_TIERS` — medium → low (verifiers split)

**Problem / evidence.** `vocab.json` + `embed_vocab.py` exist to kill the ~6 hand-duplicated
CPU-tier sites; `data_loader.py:6` correctly does `from .bt_vocab import CPU_TIERS as _CPU_TIERS`.
But `baselines.py:17` imports `APP_STAGES` from the same generated module yet `baselines.py:22`
re-hardcodes `_CPU_TIERS = ("little", "medium", "big")`. The drift guard `test_vocab.py` checks
`CPU_TIERS` via the `data_loader` path but **never** asserts `baselines._CPU_TIERS == CPU_TIERS`.
If a tier is renamed, baselines' OMP whole-pipeline baseline silently uses the wrong tier set.
Verifiers split low/medium: values match today (no current incorrect output) and the failure is
benign (a missed tier in a `min()`, never a crash), so it is a latent drift hazard. **Fix.**
One line: replace the literal with `from .bt_vocab import CPU_TIERS as _CPU_TIERS`.

### 19. Per-app pipeline-contract constants restated in every backend test TU — medium → low

**Problem / evidence.** The target-structure plan specified one `apps/<app>/traits.hpp` per
app, but `AppTraits` is specialized inline inside each per-backend test TU, so `kNumStages=7`
and `kNumToProcess=100` are physically duplicated across tree's omp/cuda/vulkan
(`omp/test_pipeline_main.cpp:26`, `cuda/...:38`, `vulkan/...:40`); `kNumStages` is also
authoritative in `vocab.json`. The `BtRuntimeApp` concept only checks `kNumStages` is
convertible-to-int, not that backends agree. **Severity note.** Downgraded: these are
**test-only** TUs (no runtime/optimizer consumes the C++ constant — they consume `vocab.json`,
whose ownership is intact); a mismatch surfaces as a failing differential test, not a
production defect. Keying `AppTraits` on Dispatcher is also a documented, justified deviation
(backends genuinely differ in `AppData`/`Queue`/`kGpuExecModel`/`kPoolSize`). Note `kPoolSize`
is genuinely per-backend (vk=16 vs omp/cuda=32), *not* duplicated — only `kNumStages`/
`kNumToProcess` are uniform. **Fix.** Hoist those two app-level fields into a shared
`apps/<app>/traits.hpp`, ideally seeded from generated `bt_vocab.hpp`.

### 29. `SafeAppData` fuses the oracle golden into the production pooled type — medium

**Problem / evidence.** `tree::SafeAppData` (the `AppData` pooled by the runtime ring and the
profiler) holds 16 `const` golden members (the precomputed reference for every stage, built by
`HostTreeManager`'s full sequential OMP pipeline) alongside 16 `_out` work buffers. The golden
is a **test oracle**, but the same type is what the perf path instantiates
(`profiler/tree-cu/const.hpp:14 using AppDataT = tree::SafeAppData;`; `bm_main.cpp` constructs
it 7×, pooled 32× in the profiler). So every pooled item carries ~2× the memory plus a one-time
full reference-pipeline run — none of which profiling intends to measure, and runtime-overhead
profiling is the project's current main thrust. Decoupling from `runtime/` held (runtime hits
are comment-only), so the debt is now contained to `apps/tree` + `profiler`.

**Clarification (verifiers).** The startup reference run is amortized once via a lazy singleton,
and the golden copy happens in the *untimed* dataset-build phase — so this does **not** corrupt
the measured numbers; the cost is setup time + doubled pooled-memory footprint, not measurement
error. Still a real design smell (test oracle fused into the production pooled type) and matches
the pre-flagged "SafeAppData debt." **Fix.** Split a lean `tree::AppData` (input + out only) for
the runtime/profiler hot path from a test-only golden subclass instantiated solely in the diff
tests. `VkAppData_Safe` already subclasses `SafeAppData`, so the split is tractable.

### 16/17/18/30/31. Vulkan `Sequence`/CUDA dead-code & cosmetic cluster

These share a file (`platform/engine/vulkan/sequence.{hpp,cpp}` + CUDA `cu_mem_resource.cu`)
and are cheap surgical cleanups; grouped because a single pass clears them.

- **16 — `CudaManagedResource` dead (medium → low).** After the managed→pinned migration, every
  live CUDA dispatcher uses `CudaManager<CudaPinnedResource>`. `CudaManagedResource`
  (`cu_mem_resource.cu:47-90`) is referenced only by its own definition and as a permitted
  template arg in `manager.cuh:15`'s `requires` clause; repo-wide grep finds zero instantiations.
  Its impl compiles but is unreachable. Per CLAUDE.md ("leave pre-existing dead code — mention,
  don't delete"), **flag** for removal: drop the class, its impl, and the `is_same_v<…,
  CudaManagedResource>` alternative once confirmed no out-of-tree consumer needs it.
- **17 — `launch_kernel_async()`/`sync()` deprecated, zero callers (medium → low).** Both are
  `[[deprecated]]` (`sequence.hpp:24-25`) pointing to `submit()`/`wait_for_fence()`, with no
  call site anywhere (`sequence.cpp:158-179`). Note `launch_kernel_async()` also lacks the
  `flush_all()` cache maintenance `submit()` added, so it is a latent coherency trap if
  resurrected on non-coherent memory. **Fix.** Remove from header and source.
- **18 — `Sequence::destroy()` declared-but-undefined (medium → low).** `sequence.hpp:41`
  declares `void destroy();` with no definition and no caller (the `.destroy()` calls in
  `base_engine.cpp` are on `device_`/`instance_` handles). It advertises an API that does not
  exist. **Fix.** Delete the declaration. (Note: one verifier observes the real underlying gap
  is that `~Sequence()` doesn't free the command pool/fence — see finding #5 — so prefer
  fixing #5 over merely deleting the decl.)
- **30 — `submit()` logs wrong method name (low).** `sequence.cpp:185` emits
  `spdlog::trace("Sequence::launch_kernel_async()")` inside `Sequence::submit()` — copy-paste
  from the deprecated method, never relabeled. Cosmetic but confusing in the exact area where
  the Mali coherency / shared-command-buffer bugs were chased. **Fix.** Change the string to
  `"Sequence::submit()"`.
- **31 — Stale commented-out experiment blocks (low).** Two commented method bodies
  (`insert_compute_memory_barrier()` `sequence.cpp:137-156`, `record_commands()` 226-240),
  matching commented decls (`sequence.hpp:21,23`), and a dead commented `waitForFences` variant
  (`sequence.cpp:202-204`) duplicating the live code below it. **Fix.** Delete all four regions.

### 21/22/32. Profiler / CMake duplication cluster

- **21 — vocab codegen duplicates device-specs codegen inline (low).** `bt_codegen.cmake` is the
  generic committed-fallback helper, used for `bt_vocab` (`platform/CMakeLists.txt:43`), but the
  `device_specs_embedded.hpp` generation directly above (lines 26-37) reimplements the same
  pattern by hand and the two already drift (device-specs globs `devices/*.json` with
  `CONFIGURE_DEPENDS`; the helper takes an explicit `DEPENDS`). **Fix.** Route device-specs
  through `bt_codegen` (glob in the caller, pass the list as `DEPENDS`).
- **22 — profiler `run-pipe-*-cu` hand-enumerated beside a foreach (low).** Three explicit CUDA
  runners (`profiler/CMakeLists.txt:7-9`) sit next to `foreach(app tree cifar-dense cifar-sparse)`
  (line 12) over the identical app list, so the set is encoded twice and can drift. (The runners
  and `bm-*` drivers use different sources, so the fix is two loops over a shared variable, and
  the CUDA-only run-pipe set is intentional.) **Fix.** Hoist
  `set(BT_PROFILED_APPS tree cifar-dense cifar-sparse)` and drive both loops from it.
- **32 — `main.cu` demo duplicated across 3 CUDA cells (low).** The standalone demo
  (`run()` + `main()`, 138 lines across cells) differs only at line 18's `<app>::omp::dispatch_stage`
  namespace. Lower priority than `bm_prof` (a demo, not a profiler), same un-parameterized fork.
  **Fix.** Move `run()`/`main()` into a shared `bm_demo_common.hpp` taking the stage-1 OMP
  dispatch as a lambda.

### 26/27/28. Source-of-truth / doc drift cluster

- **26 — `vocab.json tree=7` contradicts canonical AlexNetCIFAR(11) (low).** `vocab.json`
  encodes `tree=7, cifar-dense=9, cifar-sparse=9`, but `get_num_stages_for_app` collapses any
  `cifar-*` to `cifar-dense`, so the `cifar-sparse=9` key is decorative/dead (harmless only
  because it equals dense). More importantly the docs make AlexNetCIFAR canonical at 11 stages
  while the C++ still implements the 9-stage SmallAlexNet, and `vocab.json` hardcodes 9 with no
  marker that this is the legacy count. (`vocab.json`'s `_comment` already says values are
  "CURRENT … behavior-preserving" but does not flag the 9-vs-11 mismatch.) **Fix.** Drop/document
  the unused `cifar-sparse` key; add a `_comment` noting these are legacy SmallAlexNet counts
  pending migration.
- **27 — Codegen files document a stale `builtin-apps/` output path (medium → low).** The P5
  split moved vocab codegen output to `platform/vocab/generated/bt_vocab.hpp`, but three
  self-documenting comments still point at the deleted `builtin-apps/generated/`: `vocab.json:2`
  `_comment`, `embed_vocab.py:28` banner, `bt_codegen.cmake:9` usage example. This is doc-vs-code
  drift *inside* code/config (not a `.md`, so not caught by a reader's skepticism of stale docs).
  The actual codegen path is correct everywhere it matters, so impact is three misleading comment
  lines. **Fix.** Retarget the three comments to `platform/vocab/generated/bt_vocab.hpp`.
- **28 — instruction-for-ai docs cite pre-refactor paths (medium → low).** CLAUDE.md makes
  `docs/instruction-for-ai/` the canonical how-to an agent reads first; after P5/P6 several cited
  paths are dead: `05-profiling.md:37` `common/kiss-vk/sequence.cpp` (now
  `platform/engine/vulkan/sequence.cpp`) and `:40` `pipeline/spsc_queue.hpp` (now
  `runtime/spsc_queue.hpp`); `06-end-to-end-scheduling.md:132` `app.hpp::first_present_cpu_type()`
  (`app.hpp` is gone; the symbol now lives in `platform/registry/device_registry.hpp`). An agent
  grepping these tokens hits nothing. Doc-only, low blast radius, but high-visibility entry-point
  docs. **Fix.** Rewrite the three citations to their post-refactor locations; optionally add a
  CI grep that fails if `instruction-for-ai/*.md` references an unresolvable path token.

---

## Disputed / needs human call

These three findings split the verifiers (one "real", one "not real"). All hinge on
**reachability**: the code described is accurate, but whether the hazard can actually
manifest is contested. A human should decide whether to fix defensively or accept the risk.

### D1. Hand-listed kernel sources in `target_sources` can silently drop new `.cu`/`.cpp` — medium (one verifier: not real)

`apps/tree/CMakeLists.txt:11-25` enumerates kernel sources by hand into STATIC libs, so a
forgotten `08_*.cu` causes no configure error; `profiler/CMakeLists.txt:3-4` documents the
"dropped line builds fine and only fails at deploy" hazard. **The split:** the dissenting
verifier notes that each kernel is invoked from `dispatchers.cu` (in the same lib), so a
*referenced* dropped kernel produces a **loud local undefined-symbol link error**, not a
silent omission or deploy-only failure — truly silent omission requires a kernel nothing
calls (dead code). Also `cifar` uses a single `all_kernels.cu`, so only `tree` enumerates
per-kernel (small blast radius). **Human call:** is a `file(GLOB … CONFIGURE_DEPENDS)` or a
`build-all` aggregate target (wired into `refactor_gate.sh`) worth it as cheap robustness, or
is the link-error backstop sufficient?

### D2. `worker()` throws "App is nullptr" outside its try/catch → `std::terminate` — medium (one verifier: not real)

`pipeline.hpp:66-68` (and `worker_with_record` 107-109) throws **before** the try block
(line 71/112), and the header comment (49-53) documents that a throw escaping a `std::thread`
body calls `std::terminate` — exactly the failure mode the surrounding try/catch was written
to prevent. **The split:** the dissenting verifier traced every enqueue site and found the
rings are only ever fed non-null `unique_ptr::get()` values and workers only re-enqueue what
they dequeued, with `SPSCQueue::dequeue` unable to synthesize a null — so the throw branch is
**unreachable defensive code today**, contradicting the stated invariant only in principle.
The other verifier holds it is a genuine, accurately-located contradiction worth fixing.
**Human call:** log-and-skip/re-enqueue (mirror the catch blocks) or debug-only assert — cheap
to harden, but currently dead.

### D3. `run_pipeline` OMP path can dispatch with an empty core vector — low (one verifier: not real)

`pipeline_runner.hpp:117-121` validates only `first_concurrent_gpu_chunk` before spawning
workers; the OMP branch does `omp_dispatch(cores, cores.size(), …)` where `get_cores_by_type`
returns an empty vector for an absent tier, and `first_present_cpu_type` falls through to
`kLittleCore` unconditionally — so on a pathological zero-core topology a 0-core dispatch is
reachable. `schedule_unrunnable_reason()`/`first_unavailable_pu()` exist for exactly this but
are not wired into the runtime test runner. **The split:** the dissenting verifier notes the
OMP tests `GTEST_SKIP` on `has_*_cores()` before building schedules, `first_present_cpu_type`
is used only on the Vulkan path, reaching an empty tier needs a CPU with zero cores across all
four tiers (no such device exists), and the consequence (n=0) is a silent no-op that the
per-item golden differential would catch loudly anyway — so not a reachable bug. **Human
call:** wire `first_unavailable_pu` into `run_pipeline` (defense-in-depth, already used on the
production profiler path) or accept it as unreachable. **Note:** this is the same downstream
sink as confirmed finding #2 — if #2 is fixed structurally (absent tiers unselectable by z3),
the schedule can never carry an absent tier in the first place.
