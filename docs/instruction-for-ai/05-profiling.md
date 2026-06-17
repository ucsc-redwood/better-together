# Profiling — measuring the runtime's overhead (CLI / agent-driven)

> This is the **actionable** profiling doc: which CLI tools to run on each target,
> the exact commands, and what their output means. Every tool here is **command-line
> and emits structured text (JSON / CSV / SQL)** so an agent can drive it and parse
> the result — GUI-only profilers (Tracy, RGP, Streamline, VTune front-ends) are out
> of scope on purpose. Device hosts / serials / access shells:
> [`01-hardware.md`](01-hardware.md). How to build the binaries you profile:
> [`02-building.md`](02-building.md).

## What we are actually measuring

BetterTogether is a **runtime** (SPSC-queue dispatchers + UMA buffers + pipeline). The
first-order question is **not** "how fast is this kernel" — it's **"how much time does
the runtime itself burn that isn't useful kernel work."** Kernel micro-optimization is
real but lower priority. Frame every measurement around two KPIs:

- **Framework overhead** = `T_wall_per_task − T_kernel` → drive toward **0 ns/task**.
- **Bottleneck-PU utilization** = `T_kernel_on_bottleneck / T_wall` → drive toward **1.0**.

Two cheap, standard ways to get those numbers (both reuse the existing google-benchmark
harness — see [`03-unit-testing.md`](03-unit-testing.md) for how the `bm_*` targets are
built and run):

1. **Null-kernel ceiling** (EPCC-style empty-task microbenchmark). Replace the kernel
   with a no-op; the SPSC queue / dispatch / submit / fence machinery runs unchanged, so
   everything measured is **pure framework tax**. `1.8M tasks/s` empty ⇒ `~555 ns/task`
   overhead. This is the headline number and the regression guard.
2. **Differential** = `T_pipeline(fully) − Σ(kernel-only stage times)`. Both terms
   already exist (`pipe/*/bm_fully_vs_normal` dumps them in the `### PYTHON_DATA ###`
   block); subtract to get overhead + parallelism net.

Where the overhead hides in this codebase (the suspects worth instrumenting first):

| Overhead source | Location | Why it's a suspect |
|---|---|---|
| Per-stage Vulkan `submit` + `wait_for_fence` | `common/kiss-vk/sequence.cpp` | 7 stages = 7 CPU↔GPU round-trips/task; fence latency dominates on iGPU/Mali |
| Command-buffer re-record | `cmd_begin()` uses `eOneTimeSubmit` | re-recorded every task incl. descriptor binding |
| UMA flush/invalidate | Mali `HOST_CACHED` path | correctness fix has a cache-maintenance cost — measure it |
| SPSC queue atomics / false sharing | `pipeline/spsc_queue.hpp` | head/tail on one cache line bounces between cores |
| Per-task pmr allocation | `make_dataset` / AppData ctor | allocator in the hot path if not pooled |

## The agent rule: prefer tools whose output is JSON / CSV / SQL

## Tier S — works today, zero install, output is already structured

| Tool | Command | What the agent parses | Targets |
|---|---|---|---|
| **google-benchmark JSON** | `./bm_xxx --benchmark_format=json --benchmark_out=r.json` | per-stage time + variance, JSON | all |
| **built-in VK timestamp / CUDA event** | run `bm_fully_vs_normal`, read the `### PYTHON_DATA ###` stdout block | device-side per-stage time, CSV | all |
| **A/B regression** | `compare.py benchmarks a.json b.json` | %-delta + U-test | google-benchmark ships it |

Use this layer first: null-kernel ceiling, per-stage times, and before/after deltas all
come from here with no setup. The built-in Vulkan timestamps already work on Mali, so the
per-stage GPU time is available on the phones via the same JSON.

## Tier A — CLI tools that need install or a permission, per target

### Linux CPU side (pc / Jetson / rocky)

| Tool | Agent-friendly command | Output | Note |
|---|---|---|---|
| `perf stat` | `perf stat -x, -e ... -- ./bm` | **CSV** via `-x,` | needs `paranoid<=1` or sudo |
| `perf record`→report | `perf record -g -- ./bm` then `perf report --stdio` | parseable text | same |
| flame graph (folded) | `perf script \| stackcollapse-perf.pl` | **folded stacks are plain text** — read directly, skip the SVG | needs FlameGraph scripts |
| off-CPU (the sync tax) | `bpftrace` one-liner / BCC `offcputime` | histogram text | **finds `waitForFences` / queue-empty blocking** |
| TMA top-down | `perf stat --topdown` or `toplev -lN --csv` | CSV | top-down beats roofline for the dispatch path |

### Jetson — CUDA (`ssh duck-naughty`; login shell is fish → `ssh … bash -s` for multi-line)

`nsys`/`ncu` are **not deprecated** — they replaced `nvprof`/`nvvp`. Both have headless
CLI export. For *this* runtime, the valuable signal is the **gap between the CPU submit
thread and GPU execution** = dispatch overhead + driver launch latency + sync blocking.

```bash
# capture (osrt = OS runtime/thread blocking; nvtx projects your stage ranges)
ssh duck-naughty 'cd <bindir> && nsys profile \
  --trace=cuda,nvtx,osrt --cpuctxsw=process-tree --sample=none \
  --capture-range=cudaProfilerApi --force-overwrite=true -o /tmp/bt ./bm_tree_cu'

# export to CSV the agent can parse
ssh duck-naughty 'nsys stats --format csv \
  --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,nvtx_pushpop_sum /tmp/bt.nsys-rep'

# custom "submit→execute gap" analysis: export sqlite and run SQL
ssh duck-naughty 'nsys export --type sqlite -o /tmp/bt.sqlite /tmp/bt.nsys-rep'
```

Read for the runtime: `cuda_api_sum` → `cudaStreamSynchronize`/`cudaDeviceSynchronize`
cumulative wall = the **CUDA fence-wait tax**; `nvtx_gpu_proj_sum` → your dispatch range
projected onto the GPU; kernel-to-kernel gaps in `cuda_gpu_trace` → GPU starved.

| Tool | Command | Output |
|---|---|---|
| Nsight Compute | `ncu --csv --metrics ... ./bm` | CSV (occupancy, bandwidth, stall) — needs `sudo` for counters |
| `tegrastats` | `tegrastats --interval 100` | text (per-engine util, power, **temp → throttle**) |

> Tegra is **UMA**: no separate H2D/D2H copy timeline, so watch kernel gaps + sync, not
> copy/compute overlap. GPU context-switch trace needs `sudo nsys`.

### Rocky MiniPC — Vulkan / RADV (`ssh rocky-ryzen`, fish → `ssh … bash -s` / `bash -lc`)

| Tool | Command | Output |
|---|---|---|
| built-in VK timestamp | Tier S | prefer this — zero setup |
| AMD uProf CLI | `AMDuProfCLI collect …` then `AMDuProfCLI report --format csv` | CSV |
| `radeontop` | `radeontop -d - -l 1` | text (iGPU busy / bandwidth) |

### Android — Mali phones (both Pixel + Samsung adb-attached to rocky; all adb-scriptable)

| Tool | Agent command | Output | Note |
|---|---|---|---|
| **Perfetto** | `adb shell perfetto -c cfg -o /d/t`, pull, then `trace_processor_shell -q q.sql t` | **SQL → CSV/JSON** | Android's production standard; no GUI needed |
| **ATrace** (NVTX analog) | `ATrace_beginSection("dispatch_stage_3")` in dispatch code (`<android/trace.h>`, link `-landroid`) | named slices in Perfetto | cheap; shows *your* segments |
| **simpleperf** | push, `simpleperf record -g --trace-offcpu …` then `simpleperf report` | text / folded stacks | `--trace-offcpu` = CPU-side sync tax |
| **malioc** | `malioc --format json shaders/comp/<x>.comp` | **JSON** | offline, on the build box, **CI-friendly**; compute-vs-bandwidth bound + register pressure; compares the subgroup-16 (Pixel) vs 32 (Samsung) variants |

A Perfetto capture of `sched/sched_switch` + `power/cpu_frequency` + `power/gpu_frequency`
lets `trace_processor_shell` compute, in SQL: **GPU idle-gap total** (runtime not feeding
it), **dispatch-thread off-CPU blocking** on fence/queue, and **thermal-throttle windows**
— exactly the three overhead questions.

## The shared prerequisite: name your hot-path ranges

Both `nsys` (Jetson) and Perfetto (Android) only see *driver-level* API without
instrumentation. To see **your runtime's segments**, put named ranges in the
dispatch/queue path: **NVTX** (`nvtxRangePush/Pop`) on Jetson, **ATrace**
(`ATrace_beginSection/EndSection`) on Android — one thin platform-dispatched macro.
NVTX is already linked in the **xmake** build (`-lnvToolsExt`) but **not in CMake**, and
no ranges are placed in the code yet — wire this first or both profilers stay blind to
the runtime.

## Environment gaps on this fleet (by ROI)

1. **`sudo sysctl kernel.perf_event_paranoid=1`** — the build box ships at `=4` (most
   restricted), which blocks unprivileged `perf`/`bpftrace`. Zero-cost, instant; persist
   via `/etc/sysctl.d/`.
2. **`malioc`** — in ARM Performance Studio (free). Static Mali shader analysis, JSON,
   runs on the build box, no device — highest-ROI Android tool, good for CI.
3. **`hyperfine`** — whole-binary A/B with `--export-json`.
4. **FlameGraph scripts + pmu-tools (`toplev`)** — git clone, unlocks flame graphs + TMA.
5. **`$ANDROID_HOME`** — set it so the NDK `simpleperf` resolves.

`perf` and `bpftrace` are installed; `adb` is installed with the Pixel attached.

## Measurement hygiene (or the numbers lie)

- **Pin frequency / kill DVFS & turbo** (`cpupower frequency-set`, disable boost). On the
  phones **thermal throttling is the #1 confound** — trace `cpu_frequency` (Perfetto) and
  discard throttled samples.
- **Separate cold-start from steady-state** (the `bm_*` targets already warm up).
- **Report distributions (p50/p99), not means** — per-task overhead is long-tailed from
  scheduling jitter; an HdrHistogram is the right shape.
- **Hot-path timestamps must be cheap**: `cntvct_el0` / `rdtsc` (the harness already has
  `bt_bm::host_cycles()` in [`bm_manual_time.hpp`](../../builtin-apps/common/bm_manual_time.hpp)),
  never `clock_gettime` in the inner loop.
- **Pin threads** (affinity is already done) so migration doesn't masquerade as overhead.

## Recommended order for an agent

1. **Tier S today** — null-kernel ceiling + per-stage JSON + A/B deltas. No setup.
2. Name the hot-path ranges (NVTX / ATrace) — the shared prerequisite.
3. **Jetson**: `nsys profile --trace=cuda,nvtx,osrt` → `nsys stats --format csv`; read
   `cudaStreamSynchronize` cumulative + kernel gaps.
4. **Android**: ATrace + Perfetto `sched`/`freq` capture → `trace_processor_shell` SQL for
   GPU-gap / off-CPU tables; `malioc --format json` in CI to rule out shader-bound.
5. **Linux CPU**: `perf stat -x,` + `bpftrace` off-CPU for the sync tax; `toplev` for TMA.
