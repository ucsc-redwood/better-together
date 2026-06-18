# BetterTogether — Target Project Structure & Migration TODO

Status: **proposed / next TODO** · Derived from a multi-agent structural audit (2026-06-17)
· Realizes the vision in [`rearchitecture.md`](rearchitecture.md) at the file/directory level.

This document is the concrete, directory-level plan for reorganizing the repo so that
**every top-level directory is exactly one component**, the runtime is provably
application-agnostic, and adding an app / a backend / a device is a localized, mechanical
change. It is the operational counterpart to [`rearchitecture.md`](rearchitecture.md): that
doc defines the 5-layer *vision* (contracts → core runtime → apps-as-plugins → optimizer →
CLI); this doc says *which files move where* and *in what order*.

It was produced by a fan-out/critique/synthesis workflow (6 examiner agents → 3 candidate
structures → 3 adversarial-lens critics → 1 synthesizer). The numbers below (`162` relative
includes, `66` prefixed, `app.hpp` 33 consumers, etc.) are grounded measurements from that
audit, not estimates.

---

## 1. Verdict

The **component-first** backbone wins (clean / robust / scalable = 5/5/5): directory name ==
paper component, each component is its own CMake target with a PRIVATE include scope so the
decoupling is **build-checkable**, the fragile magic-typedef pipeline contract becomes a real
`AppTraits<App>` struct, and one vocabulary manifest collapses the ~6 hand-duplicated enum
sites.

A "minimal-churn, keep `builtin-apps/` as the anchor" alternative was rejected as having a
**fatal, verified flaw**: its premise that staying inside `builtin-apps/` is cheap is false.
The repo has **162 relative `../` includes** vs only **66 repo-root-prefixed** ones; moving
`pipeline/` → `runtime/`, `conf.cpp` → `registry/`, `config_reader` → `io/` *internally*
breaks the relative includes anyway (`app.hpp` alone has 33 consumers, `pipeline/` 26). It
also missed that `mr_ptr.hpp` lives in `pipe/` and is pulled by `pipeline_common.hpp` — its
"move the worker" step would not compile.

**Decision:** take the component backbone, but graft in the minimal-churn *phasing
discipline* — land every cheap, separable, independently-shippable win first (each gated
green), and isolate the one atomic include-rewrite + CMake-split to the last phase.

---

## 2. Target directory tree

```
better-together/
├── CMakeLists.txt                  # thin root: add_subdirectory() per component + options/preset glue
├── CMakePresets.json               # pc / jetson / vulkan / android — UNCHANGED (backend gating already clean)
├── vocab.json                      # NEW single source of truth: PUs, backends, apps→stage_count
├── cmake/
│   ├── CPM.cmake  write_git_sha.cmake
│   ├── bt_codegen.cmake            # NEW generic embed helper (reuses the devices/*.json codegen pattern)
│   ├── bt_add_app.cmake            # NEW  bt_add_app(NAME tree BACKENDS omp cuda vulkan)
│   └── toolchains/{jetson-aarch64,android-arm64}.cmake
├── schemas/                        # device-spec / profiling-table / schedule .schema.json — UNCHANGED
├── devices/                        # device-registry DATA, one JSON per host — UNCHANGED (the device axis)
├── tests/fixtures/schedule.contract.json   # cross-tool round-trip fixture — UNCHANGED
│
├── platform/                       # COMPONENT: shared substrate (registry + engines + utils + vocab)
│   ├── CMakeLists.txt              #   → bt::core   (always built, CPU-only; no app, no runtime deps)
│   ├── registry/                   #   ← builtin-apps/conf.* + app.* + affinity.hpp
│   │   ├── conf.{hpp,cpp}          #       ProcessorType / Core / Device
│   │   ├── device_registry.{hpp,cpp}   #   ← app.cpp/app.hpp (it is device SELECTION, not an app):
│   │   │                           #       GlobalDeviceRegistry + --device + g_*_cores
│   │   ├── affinity.hpp
│   │   └── generated/device_specs_embedded.hpp   # codegen output (committed fallback kept)
│   ├── vocab/generated/{bt_vocab.hpp, bt_vocab.py}   # codegen output from vocab.json
│   ├── engine/
│   │   ├── vulkan/                 #   ← common/kiss-vk/  ENGINE ONLY (algorithm/base_engine/sequence/vk)
│   │   └── cuda/                   #   ← common/cuda/     (cu_mem_resource, manager.cuh, helpers.cuh)
│   ├── mem/mr_ptr.hpp              #   ← pipe/mr_ptr.hpp  (the CUDA-ref vs Vulkan-pointer get_mr() seam)
│   └── util/                       #   ndarray, base_appdata, load_npy, hex_dump.*, resources_path, debug_logger
│                                   #   (cache.hpp DELETED — 0 refs, confirmed dead)
│
├── runtime/                        # COMPONENT: BT-Implementer (app-agnostic; owns the ring plumbing)
│   ├── CMakeLists.txt              #   → bt::runtime  (links bt::core ONLY; links ZERO app targets)
│   ├── schedule.hpp  record.hpp  spsc_queue.hpp  task.hpp
│   ├── config_reader.hpp  schedule_source.hpp        # JSON-ingest path (already app-agnostic)
│   ├── app_traits.hpp                                # NEW: AppTraits<App> primary template + C++20 concept
│   ├── pipeline.hpp                                  # ← pipe/pipeline_common.hpp : worker()/make_dataset()
│   ├── pipeline_runner.hpp                           # ← pipeline_test_runner.hpp, now traits-driven
│   └── tests/test_schedule.cpp                       # pure-logic unit, LABEL "unit" (no hardware)
│
├── apps/                           # COMPONENT: APP KERNELS only (one self-contained dir per app)
│   ├── CMakeLists.txt              #   foreach(app) add_subdirectory(${app})
│   ├── tree/
│   │   ├── CMakeLists.txt          #   bt_add_app(NAME tree BACKENDS omp cuda vulkan)
│   │   ├── appdata.hpp  safe_appdata.hpp  traits.hpp    # traits.hpp = AppTraits<tree> specialization
│   │   ├── omp/    dispatchers.{hpp,cpp}  test_diff.cpp  test_pipeline.cpp  bm_kernel.cpp
│   │   ├── cuda/   all_kernels.cu  dispatchers.cu  test_diff.cu  test_pipeline.cu
│   │   └── vulkan/ dispatchers.cpp  test_diff.cpp  test_pipeline.cpp
│   │            └── shaders/ *.comp  generated/*_spv.h     # CO-LOCATED with the app (symmetric w/ cuda/)
│   ├── cifar-dense/  (same shape)
│   ├── cifar-sparse/ (same shape)
│   └── octree/       (omp/ only — honest stub; declared via BACKENDS omp, not pretend-symmetric)
│
├── profiler/                       # COMPONENT: BT-Profiler (← pipe/)
│   ├── CMakeLists.txt              #   links bt::runtime  (ONE-WAY: profiler → runtime)
│   ├── common/  bm_{prof,baseline,fully,gen_log}_common.hpp  table.hpp
│   └── cells/<app>-<backend>/  const.hpp  main.*  bm_*.*     # thin; emitted by bt_add_app
│
├── optimizer/                      # COMPONENT: BT-Optimizer (z3) + orchestration + analysis (← scripts/collect)
│   ├── pyproject.toml              #   makes it a real package + a pytest target
│   ├── smt/         solver.py constraints.py baselines.py data_loader.py solution_analyzer.py profiling_loader.py
│   ├── orchestrate/ 02_gen_schedule.py … 05_timeline.py  case.py  export_btpm_csv.py
│   │                run-on-{jetson,rocky,android}.sh  cross-build-jetson.sh
│   ├── analysis/    results/  render_isolated_table.py  coverage.py  figures/figure_*.py
│   └── tests/       test_baselines.py test_minimize_mode.py test_schedule_contract.py test_case.py
│
├── scripts/                        # codegen ONLY (referenced by CMake)
│   └── embed_device_specs.py  embed_vocab.py  validate_devices.py
│
├── tools/                          # ← utility/  (probes RENAMED probe_* not test_*)
│   └── probes/  query_cpuinfo.cpp  check_vulkan/  probe_affinity.cpp  probe_omp.cu  stress/
│
└── resources/ saved_params/        # input data — UNCHANGED  (data/ already out of git)
```

### Mapping to the `rearchitecture.md` 5-layer vision

| rearchitecture.md layer | realized here as |
|---|---|
| Layer 0 — Contracts | `schemas/` + `devices/` + `vocab.json` (UNCHANGED / NEW vocab) |
| Layer 1 — Core runtime | `platform/` (`bt::core`) + `runtime/` (`bt::runtime`) |
| Layer 2 — Apps as plugins | `apps/<app>/` with `AppTraits<app>` registration |
| Layer 3 — Optimizer | `optimizer/` (now a real Python package) |
| Layer 4 — CLI / orchestration | `optimizer/orchestrate/` + thin root scripts |

---

## 3. Rationale — what makes it robust + scalable

- **Dir-name == paper component.** `runtime/` = BT-Implementer, `profiler/` = BT-Profiler,
  `optimizer/` = BT-Optimizer, `apps/` = kernels, `platform/` = shared substrate. The two
  naming lies are gone: `builtin-apps/` (which held FIVE components) is dissolved, and
  `app.cpp` (which is device *selection*, not an app) becomes
  `platform/registry/device_registry.cpp`.

- **One-way dependency arrow matching the paper's tool order:**
  `bt::core (platform) ← bt::runtime ← bt::engine::{cuda,vulkan} ← bt::app::<x> ← bt::profiler`.
  `worker()` / `make_dataset()` (today in `pipe/pipeline_common.hpp`, a *profiler* dir) move
  into `runtime/pipeline.hpp`, severing the **11 verified reverse `../../pipe/pipeline_common.hpp`
  includes** and the runtime→profiler cycle. `mr_ptr.hpp` moves to `platform/mem/` so both
  runtime and profiler depend *down* on it instead of into each other.

- **App-agnostic runtime, enforced three ways** (belt-and-suspenders, because the leak is
  invisible today):
  1. `bt::runtime` links zero app targets — so `pipeline_test_executor.hpp` hardcoding
     `tree::SafeAppData` (verified lines 22–23 / 51) becomes a **link error**, not a silent
     compile. That file is deleted.
  2. `app_traits.hpp` replaces the fragile pre-`#include` magic-typedef contract (verified:
     `pipeline_common.hpp` references `DispatcherT / AppDataT / AppDataPtr / QueueT / LocalQueue`
     by bare name) with `AppTraits<App>{ using AppData; using OmpDispatcher; static constexpr
     int kNumStages; … }` + a C++20 `concept` — the compiler now checks the contract.
  3. A CTest `guard-runtime-agnostic` greps `runtime/` for `tree|cifar|octree|morton` and fails
     on any hit (catches header-only slips that link-scoping cannot).

- **One vocabulary, proven mechanism.** `vocab.json` (PU tiers incl. the orphaned `super`
  tier, backend short/long names, apps→stage_count) → `embed_vocab.py` → `bt_vocab.hpp` +
  `bt_vocab.py`, via the *same* `add_custom_command` + `file(GLOB CONFIGURE_DEPENDS)` pattern
  already working for `devices/*.json` (CMakeLists.txt:110–117). Collapses the verified ~6
  hand-duplicated enum sites (`config_reader` maps, `schedule.schema.json`, `data_loader.py`
  core_types, `baselines.py` stage dict, `conf.cpp` tiers) and structurally fixes a **latent
  super-core bug** caused by that drift. JSON (not YAML) on purpose — no new build dependency;
  committed generated headers are the fallback, exactly as `device_specs_embedded.hpp` is today.

- **Shader bake moved into CMake (DONE).** The `glslc → .spv → xxd` bake moved from the standalone
  root `Makefile` (which CMake never invoked) into `cmake/bt_shaders.cmake`, exposed as an opt-in
  `bake-shaders` target, guarded by `BT_GLSLC`/`BT_XXD` cache vars + committed-header fallback (the
  Android/NDK preset with no `glslc` on PATH builds the committed `*_spv.h` directly). Two findings
  forced the shape: (1) shaders are **engine-shared, keyed by name** — they did *not* co-locate per
  app (`apps/<app>/vulkan/shaders/`); they stayed flat under `platform/engine/vulkan/shaders/`.
  (2) **glslc is not byte-reproducible across versions** (measured: 5 of 29 `.spv` differ on a local
  re-bake), so `.comp→.spv` must stay off the ALL target — `bake-shaders` is explicit, run only when
  a `.comp` changes. Also fixed a **latent P5a bug**: the move left the committed `*_spv.h` carrying
  stale `builtin_apps_common_kiss_vk_shaders_spv_*` (old-path) variable names that `all_shaders.hpp`
  hard-referenced; re-baking would have emitted new names and broken the build. Variable names are
  now the bare `<name>_spv` basename (`xxd -i` run from `spv/`), path-independent. (Vulkan compiled
  green; clean-tree verified; omp gate 9/9.)

- **CTest gains a "kind" axis** as a second label (no new infra): `"omp;unit"`,
  `"omp;differential"`, `"omp;runtime"`, `"vulkan;engine"`. `ctest -L omp` stays the everyday
  gate; `-L unit` runs the hardware-free `test_schedule` alone; `-L runtime` runs the pipeline
  suite (which carries the deadlock watchdog + the open `AlternatingBoundary` segfault) without
  blocking the correctness gate.

---

## 4. What a developer touches to add an app / backend / device

### Add an app `foo`  (was ~40 scattered edits → now ~3 sites)
1. `mkdir apps/foo/` with `appdata.hpp`, `traits.hpp` (specialize `AppTraits<foo>`),
   `{omp,cuda,vulkan}/dispatchers.*`, `vulkan/shaders/*.comp`, and `<backend>/test_diff.*` +
   `<backend>/test_pipeline.*`.
2. One line in **`vocab.json`**: `"foo": {"stages": N}` → propagates the stage-count to
   `bt_vocab.{hpp,py}`, the profiling-table schema enum, and `baselines.py` (no more 3-place
   hand-sync).
3. One call in `apps/foo/CMakeLists.txt`: `bt_add_app(NAME foo BACKENDS omp cuda vulkan)` →
   wires the OMP dispatcher into `bt::core`, the per-backend libs/runners, the per-stage
   `bm_kernel`, the differential + runtime tests with correct kind labels, the shader bake,
   **and** the `profiler/cells/foo-{cu,vk}` cells.

   **Runtime needs zero edits** — `pipeline_runner<foo>()` works because `AppTraits<foo>` exists.
   No copy-the-executor step.

### Add a backend `edgetpu`  (localized to vocab + one engine + one CMake block)
1. One entry in `vocab.json` (short/long name, `ExecutionModel` / `ProcessorType` token) →
   regenerates the schedule-schema `hardware` enum, the `config_reader` mapping, `data_loader.py`
   core_types, and the C++ enum from one place.
2. `platform/engine/edgetpu/` (mirrors `engine/cuda`) with its `mr_ptr` `get_mr()` specialization
   in `platform/mem/`.
3. `apps/<app>/edgetpu/dispatchers.*` per app (real new kernel code — unavoidable).
4. One `if(BT_ENABLE_EDGETPU)` block in `platform/CMakeLists.txt` defining `bt::edgetpu`; extend
   `bt_add_app` to emit the `edgetpu` column; add a preset + toolchain if cross-compiled. The
   CTest kind/backend label auto-extends.

### Add a device  (already the best axis — unchanged)
Drop `devices/<id>.json` (schema-validated by `validate_devices.py`). The codegen GLOB re-runs
`embed_device_specs.py` → `platform/registry/generated/device_specs_embedded.hpp`. Select with
`--device`. **Zero C++/CMake edits** — unless the device introduces a brand-new CPU tier, which
is now **one `vocab.json` line** (was: `conf.hpp` enum + `parse_core_type` + `CoreTypeName` +
every `g_*_cores` switch by hand).

---

## 5. How decoupling becomes build-checkable

| Leak (verified today) | Enforcement mechanism after migration |
|---|---|
| Runtime → app (`pipeline_test_executor.hpp` hardcodes `tree::SafeAppData`) | `bt::runtime` links **only** `bt::core`; its PRIVATE include scope does not expose `apps/`. `#include "apps/tree/…"` fails at configure; a `tree::` symbol fails at link. Executor file deleted. |
| Runtime ↔ profiler cycle (11 reverse `../../pipe/` includes; `pipeline_common.hpp` pulls `pipe/mr_ptr.hpp`) | `worker()`/`make_dataset()` move to `runtime/pipeline.hpp`; `mr_ptr.hpp` moves to `platform/mem/`. `profiler/` links `bt::runtime` one-way; runtime has no `profiler/` on its path. |
| Magic-typedef ODR contract (bare names `#define`d before include) | `AppTraits<App>` + C++20 `concept` → a wrong/missing field is a named compile diagnostic, not a cryptic template/ODR error. |
| Residual same-target header leak | `guard-runtime-agnostic` CTest greps `runtime/` for app names. |
| Vulkan stale-`.spv` links green | shader bake is a build-graph node via `add_custom_command` (was an out-of-band `Makefile`). |
| Vocabulary drift across 6 sites | single `vocab.json` → codegen; hand-edits to generated headers are overwritten at build. |

The decisive lever vs. status quo: **the include root stops being the repo root** (verified
`$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>` at CMakeLists.txt:99/174/253). Each component
exports only its own dir as PUBLIC, so boundary violations fail the build instead of compiling
green.

---

## 6. Phased migration plan (the TODO — low-risk-first, each phase gated)

Total churn is real and large but ~70% mechanical: **66 prefixed + 162 relative includes** (the
relative ones are the true driver — `app.hpp` 33 consumers, `pipeline/` 26). The order below
lands every cheap, separable win first; the one atomic rewrite is isolated to Phase 5.

> **Gate legend.** **G-omp** = `ctest --test-dir build/pc -L omp --output-on-failure` (local,
> the everyday gate). **G-gpu** = `-L cuda` on Jetson + `-L vulkan` on rocky-ryzen. **G-runtime**
> = `-L runtime` on all three machines.

- [x] **P0 — Dead-code & cosmetics (near-zero risk).** ✅ DONE (commit `ac07ed9`, gate green).
      Deleted `cache.hpp` (0 refs); renamed `utility/` probes `test_*` → `probe_*` (sources +
      the 2 `add_executable` paths; target names `test-affinity`/`test-omp` kept for the P5
      `utility/` → `tools/probes/` move). **Dropped the planned "scratch `tmp_*.comp`" deletion:**
      `tmp_single_radixsort_warp{16,32,64}.comp` are NOT scratch — they are live entries in the
      Vulkan shader registry (`all_shaders.hpp` `#include`s their baked `_spv.h` and registers
      them via `SHADER_ENTRY`). Removing them mutates the shader table → a *semantic* change, out
      of scope for this behavior-preserving refactor. If they are truly dead at runtime, excise
      them in a standalone "remove dead experimental shaders" commit with a Vulkan T4 run, not
      here. **Gate: G-omp.**

- [x] **P1 — CTest kind axis (additive, zero infra).** ✅ DONE (commit `e229923`, gate green).
      Added a SECOND label refining each test by purpose (`unit`/`differential`/`runtime`/`engine`)
      while keeping the backend label so `ctest -L omp` stays the everyday gate. Implemented via
      per-test-group `set_tests_properties` (not by mutating the helper `LABELS` lines as sketched
      here — each helper builds tests of *multiple* kinds, so a helper-level kind would mislabel).
      Verified: `ctest -L unit -N` → exactly `test-schedule-omp`; `-L omp` unchanged (8); vulkan
      `-L engine` → exactly `test-kiss-vk`. **Gate: G-omp.**

- [x] **P2 — Surface the optimizer (additive).** ✅ DONE (commit gated; pytest 6 passed,
      C++ inventory `+test-optimizer` only). `git mv scripts/collect/*` into
      `optimizer/{smt,orchestrate,analysis,tests}`; added `pyproject.toml` (pytest rootdir +
      `pythonpath`); rewrote intra-package imports to absolute (`orchestrate.case`,
      `smt.profiling_loader`, `analysis.results.*`); CTest `test-optimizer` under `LABEL
      "optimizer"` (not in any backend gate). Kept `embed_device_specs.py`/`validate_devices.py`
      in `scripts/`. **Observation (not fixed, behavior-preserving):** `smt.*` imports
      `orchestrate.case` — the SMT layer depends *up* onto orchestrate; `case.py` is really a
      shared model. Candidate for a later move. **Gate: G-omp + `pytest optimizer/tests`.**

- [x] **P3 — Break the cycle + kill the executor (the load-bearing C++ change, contained).** ✅ DONE.
      Moved `pipe/pipeline_common.hpp` → `runtime/pipeline.hpp`, `pipe/mr_ptr.hpp` →
      `platform/mem/mr_ptr.hpp`, `pipeline_test_runner.hpp` → `runtime/pipeline_runner.hpp`. Added
      `runtime/app_traits.hpp` (`AppTraits<Dispatcher>` + C++20 `BtRuntimeApp` concept); **deleted
      `pipeline_test_executor.hpp`** (generic `OmpStubDispatcher<AppData>` replaces its tree-hardcoded
      stub). The 4 free functions are now templates (the magic-typedef ODR contract is gone); the SPSC
      ring body of `run_pipeline()` is byte-identical. The 9 test TUs shrink to an `AppTraits`
      specialization + `run_runtime_test<Dispatcher>()`. **Keying decision (diverges from the plan's
      `AppTraits<App>`):** keyed on the **Dispatcher** type — tree's OMP+CUDA cells share
      `tree::SafeAppData` and all of cifar's backends share one `cifar_*::AppData`, so AppData/App
      can't identify a cell (see [[safe-appdata-debt]]). **Gate: G-omp + G-gpu + G-runtime all
      green** — Jetson CUDA + rocky Vulkan, every differential AND pipeline-e2e test passed.

- [x] **P4 — Vocabulary codegen (additive, proven pattern).** ✅ DONE (G-omp green + pytest 11
      passed). Added `vocab.json` + `scripts/embed_vocab.py` + `cmake/bt_codegen.cmake`; codegen →
      `builtin-apps/generated/bt_vocab.hpp` (ProcessorType/CoreTypeName/ParseCoreType, included by
      `conf.hpp`) + `optimizer/smt/bt_vocab.py` (CPU_TIERS/CORE_TYPES/APP_STAGES, read by
      `data_loader`/`baselines`). **Super-core bug NOT "fixed" — behavior-preserving:** `vocab.json`
      encodes today's values verbatim (super stays out of the solver tier list; enum values
      preserved), so z3's cost-matrix shape is unchanged. `test_vocab.py` guards drift across
      vocab.json / data_loader / baselines / the schema enum. **← This is the off-ramp: P0–P4 deliver
      every robustness/decoupling win except build-enforcement, all gated green.**

- [x] **P5 — The component split (DONE).** Landed across P5a (the move) + P5b (CMake split).
  - [x] **P5a — the move (DONE, full-fleet green).** `git mv` builtin-apps/pipe/utility into
        `platform/ runtime/ apps/ profiler/ tools/` (app.*→`device_registry.*`); 228 includes rewritten
        (relative→repo-root-prefixed→component paths; bare cross-boundary includes resolved by basename);
        CMakeLists source paths + device/vocab codegen outputs + shader-bake Makefile remapped. Behavior-
        preserving; gated **G-omp + G-gpu + G-runtime** (Jetson CUDA + rocky Vulkan, all differential +
        pipeline-e2e green). Repo root is still the PUBLIC include dir (move correct, not yet enforced).
  - [x] **P5b — the enforcement (DONE).**
        - [x] **Shader bake into CMake** — `cmake/bt_shaders.cmake` + opt-in `bake-shaders` target
              replaces the dead root `Makefile`; `BT_GLSLC`/`BT_XXD` + committed-header fallback; fixed the
              stale-variable-name P5a bug. See the §5 bullet above. (Vulkan green, clean-tree, omp 9/9.)
        - [x] **CMakeLists split** — the ~390-line root became a thin root (options/deps/helpers/language
              setup) + one `add_subdirectory` per component (`platform/ apps/<app>/ runtime/ profiler/ tools/`);
              each app contributes its kernels to the backend libs via `target_sources()`. **Deliberate
              deviation from the literal plan:** kept the backend libs BUNDLED (not per-app libs) and did NOT
              narrow include scope — the only payoff of per-app-libs + scope-narrowing was compile-time
              enforcement, which `guard-runtime-agnostic` already delivers (the plan's explicit OR-branch:
              repo-root export + link-scoping + the guard). Avoids the ~228-include component-relative rewrite
              and link-graph churn for zero added enforcement. Gated: target-list diff old==new (79==79, no
              target lost) for vulkan+bench and jetson configs; full fleet green (omp 9/9 pc; Vulkan rocky
              diff 7/10/10 + engine 2 + runtime 3/2/2; both Mali phones 7/10/10; CUDA Jetson diff 7/10/10 +
              runtime 2/2/2); PC preset invokes no `nvcc`; `bt::vulkan` keeps its `VulkanHeaders` SYSTEM include.

  <!-- original P5 text retained below for reference -->
  `git mv builtin-apps/{pipeline→runtime, conf.*+app.*+affinity→platform/registry,
      common/kiss-vk→platform/engine/vulkan, common/cuda→platform/engine/cuda, util→platform/util,
      <apps>→apps/}`, `pipe/→profiler/`, shaders into `apps/<app>/vulkan/shaders/`. Then one scripted
      `sed` pass rewriting the 66 prefixed + 162 relative includes to component paths, and **split the
      ~386-line `CMakeLists.txt` into ~6 per-component sub-files + a thin root** with
      `add_subdirectory`, narrowing each target's include scope to its own dir (this is what makes
      P3's decoupling *enforced*). Move the shader bake into `bt_add_app` with the `BT_GLSLC` +
      committed-`.spv` fallback. ~228 include rewrites + CMake split. **Must land atomically.**
      **Gate: G-omp AND G-gpu AND G-runtime**; verify the PC preset still never invokes `nvcc`
      (`enable_language(CUDA)` stays inside `if(BT_ENABLE_CUDA)`) and `bt::vulkan` keeps its
      `VulkanHeaders` SYSTEM include.

- [x] **P6 — target-helper consolidation (DONE).**
  - [x] **`guard-runtime-agnostic` grep CTest.** `scripts/guard_runtime_agnostic.py` greps `runtime/` for
        app namespaces/identifiers in CODE (comments stripped) + `#include "apps/"`; wired as a CTest
        labelled `omp;guard` (rides the everyday omp gate). Build-enforces the P3 decoupling. (Also fixed a
        `refactor_gate.sh` inventory() sed bug that undercounted any dir >10 tests.)
  - [x] **Target-helper consolidation.** The 8 `bt_add_{omp,cuda,vk}_{run,app,test}` helpers (extracted to
        `cmake/bt_targets.cmake`) ARE the per-app target consolidation — each component's CMakeLists.txt now
        calls them once per target. **Deliberately did NOT build a `bt_add_app(NAME … BACKENDS …)` helper
        that reads stage-count from `vocab.json`:** the stage count is not needed to define any target, and
        the apps are irregular enough (tree has no cuda bm_main, octree is omp+vk only, cifar splits
        all_kernels vs per-stage) that a single generic helper would be more complex than the explicit
        per-app calls — an abstraction for no real duplication. Gate met: target-list diff (no target lost).

**Off-ramp:** Phases 0–4 deliver every robustness/decoupling win *except* build-enforcement and are
independently shippable. If the deadline bites, stop after P4 with the cycle broken, the executor
gone, the contract compiler-checked, the optimizer gated, and the vocabulary unified — deferring
only the atomic P5 rewrite.

---

## 7. Coordination & prerequisites

- **Already done (do not redo):** schedule JSON schema + 1-based start/end format + removed
  `stage_assignments` + round-trip contract test + `Case` path-builder; `data/` removed from git
  (regenerable, not versioned); `conf.cpp` data-driven from `devices/*.json` (rearchitecture
  Phase 2). See the schedule-contract memory and [`rearchitecture.md`](rearchitecture.md) §5.
- **Coordinate P5 with the in-flight CMake-migration RFC Phase C3** (`pipe/` "duplication farm"
  deletion, flagged at CMakeLists.txt:181–182, and [`rearchitecture.md`](rearchitecture.md) Phase
  4) — both touch the same directories. Sequence them, do not run in parallel branches.
- This doc supersedes the directory-level guesses in earlier planning notes; the *conceptual*
  layering still lives in [`rearchitecture.md`](rearchitecture.md), and the review-findings
  execution order in [`code-improvement-plan.md`](code-improvement-plan.md).

---

## 8. Execution protocol (for the coder agent doing the refactor)

**This is a behavior-preserving refactor: it adds NO behavior.** So the gate is *equivalence
to a pre-refactor baseline*, not new tests. The risk is ~entirely in the build graph (include
paths, targets, link scope), not in logic. Internalize the five rules below before touching a
file.

### Mandate
- **Surgical & behavior-preserving.** Do not fix bugs, improve kernels, or touch logic. If you
  find a real bug mid-refactor, **write it down and leave it** — fixing it here pollutes a
  reviewable mechanical diff. (CLAUDE.md: surgical changes.)
- **Never mix a mechanical move with a semantic change in one commit.** Moves (`git mv` +
  scripted `sed` on include paths) are reviewed as renames. Semantic changes (P3 `AppTraits` /
  delete executor, P4 vocab codegen) are reviewed as real edits. If a *moved* file needs edits
  beyond include-path rewrites to compile, **STOP and report** — a hidden semantic dependency
  surfaced; do not silently fix it inside the move.
- **One commit per phase** on `refactor/component-structure`, so a breakage bisects to one
  phase. P4 is the off-ramp (see §6).

### The gate ladder (this is how "e2e is slow" becomes tractable)
For a structural refactor, **compiling** a preset catches ~90% of the breakage; you do **not**
need to *run* on a device to know the CUDA/Vulkan build didn't break — you need to *compile* it.
Reserve true on-device e2e for the two load-bearing phases only.

| Tier | What | When | Cost |
|---|---|---|---|
| **T1** | `cmake --preset pc` + `ctest -L unit` | every edit (exists after P1) | seconds |
| **T2** | `cmake --build --preset pc` + `ctest -L omp` | end of every phase | minutes |
| **T3** | **cross-compile** jetson (CUDA, via `scripts/cross-build-jetson.sh`) + configure/compile the `vulkan` preset (host clang+glslc, builds on *any* box) | end of every phase | build-only, no device |
| **T4** | real hardware: Jetson `-L cuda` + rocky `-L vulkan` + `-L runtime` | **P3 and P5 only** | slow, full fleet |

Most phases gate on T1–T3. Only P3 (break cycle + delete executor) and P5 (atomic move) pay T4.

### The gate tool: `scripts/refactor_gate.sh`
```
scripts/refactor_gate.sh capture-baseline   # run ONCE on a clean tree before P0
scripts/refactor_gate.sh check              # at the end of every phase; exits !=0 on regression
```
It builds the local preset (build failure = gate failure), runs `ctest`, and asserts
**equivalence to the baseline**: same test inventory per preset, non-zero & unchanged test count,
and no test flipped PASS→FAIL or RUN→SKIP. Pre-existing failures are captured in the baseline so
they are not blamed on the refactor. Run it on the fleet too (Jetson/rocky) to gate the GPU rows;
baseline store is `.refactor-gate/` (gitignored). Knobs: `RUN_PRESET`, `RUN_LABELS`, `BUILD_DIRS`.

### "Green" lies — three traps the tool guards, but you must respect
1. **Non-zero test count.** `ctest` reporting "100% passed" over **0 tests** (a target failed to
   build, or a label matched nothing) is the #1 false-green. The tool fails if count is 0 or
   changed.
2. **Skip-count diff.** A test that flips RUN→SKIP (a target got dropped from a link line) is a
   regression wearing a green hat — CLAUDE.md counts `GTEST_SKIP` as pass. The tool diffs the
   skip set.
3. **Known failures live on the baseline.** The open `AlternatingBoundary` segfault is captured
   at `capture-baseline` time, so it does not trip `check`. Do **not** "fix" it here.

### Repo-specific landmines (you will lose hours to these otherwise)
- **PC cannot build native CUDA** (CUDA 13 / CUB removal). "Compile the CUDA target" means the
  **NVIDIA-container cross-build** (`scripts/cross-build-jetson.sh` / `Dockerfile.cross`), not a
  bare `cmake --preset jetson` on the PC.
- **Vulkan compiles anywhere but runs only on rocky/phones** (`kiss-vk` hard-selects the iGPU;
  the discrete RTX throws "No integrated GPU found"). So the T3 *compile* gate works on the PC;
  the T4 *run* gate needs rocky-ryzen.
- **rocky's login shell is fish** → wrap remote commands in `bash -lc '…'`.
- **Stale `.spv` links green.** After P5 moves shaders into `apps/<app>/vulkan/shaders/`, force a
  **clean** rebuild of the SPIR-V so you prove the new bake path actually regenerates — otherwise
  an old baked header keeps the Vulkan tests passing against dead shaders.
- Fixed differential seed `114514`; `data/` is out of git (regenerable); `resources/*.npy` are
  required at runtime.

### Per-phase loop
1. `git checkout -b refactor/component-structure` (first phase only); `refactor_gate.sh
   capture-baseline` on a clean tree **before P0**.
2. Do exactly one phase. 3. `refactor_gate.sh check` (T1/T2) + the T3 cross-compile; for P3/P5
   also T4 on the fleet. 4. Green → commit the phase. Red → fix or revert; never commit red.
