# Kernel-optimization wave & the definitive post-wave baseline (2026-07-03)

**TL;DR** — We profiler-audited the runtime (verdict: SPSC/handoff is ~60 ns — near
zero), then ran a five-branch kernel-optimization campaign across all three backends
(every step gated by the differential suites + real-weight accuracy), and re-ran the
full-fleet benchmark on the frozen result. Hybrid pipelining now wins **up to 2.5×
against single-PU baselines that are themselves optimized to the hardware's limits** —
smaller numbers than the naive-kernel era, but unimpeachable ones.

## 1. Kernel wave (all merged into dev)

| Branch | Scope | Result (isolated, locked/perf clocks) |
|---|---|---|
| perf/cuda-kernels | dense+sparse CUDA | dense 129→21.2 ms (6.1×); sparse 421→242 ms; fc1 42→2.0 ms (21×) |
| perf/vk-kernels | dense Vulkan | minipc 773→8.4 ms (**92×**; fc1 352→0.85, 417×); pixel 588→186 ms (3.2×) |
| perf/sparse-kernels | sparse VK + OMP FC | sparse VK 2994→115 ms (26×; fc1 207×); OMP FC 1.9× |
| perf/omp-kernels | dense OMP | CPU 116.5→17.5 ms (6.6×); convs 8.6× |
| perf/omp-fc-gemm | OMP FC microkernel | dense fc 2.15→1.23 ms — **91% of measured DRAM ceiling**; sparse fc 2.77× |

Techniques: coalesced warp-row access, batch-tiled GEMM (weight reuse across images),
3×3/s1/p1 specialization + register blocking (4 oc/thread), CSR strength reduction,
XNNPACK-style mr×nr register tiles on CPU. Negative results kept on record: CUDA
Graphs (launch tax hides behind ms-kernels), device weight mirror on UMA (same DRAM),
FC accumulator chains on GCC (regressed). Reference-architecture note: our
OMP-as-oracle + optimized-backends split mirrors ExecuTorch portable/XNNPACK.

## 2. Definitive fleet baseline (21/21 cells, rounds A+B)

Method: fresh full-fleet profile (real weights, default governors) → plain z3 round →
`fit_overhead` refit (per-chunk tax: duck GPU 0.6–0.9 ms, CPU 1.2–1.3 ms; minipc CPU
4.1 ms) → overhead-aware z3 + union sweep, top-4 measured. Data in `data/` stores on
the benchmarking host; summary reproduced by `00_run_fleet.py --phases summary`.

| Device | App | Best hybrid vs best single PU |
|---|---|---|
| duck-stable / naughty | tree ×CUDA | **2.25× / 2.53×** |
| samsung | cifar-dense | **1.84×** (VK 31.1 → 16.8 ms) |
| samsung | tree | **1.83×** |
| pixel | cifar-dense | **1.75×** (125.9 → 72.0 ms) |
| duck ×2 | cifar-sparse ×VK | 1.32–1.34× |
| duck ×2 | cifar-sparse ×CUDA | 1.20× |
| pixel tree / minipc sparse | | 1.18× / 1.13× |
| duck ×2 dense (GPU dominant) | | 1.03–1.15× (solver correctly near-refuses) |

Sub-1.0 rows and how to read them:
- **tree×VK / minipc (0.78–0.91)**: µs-scale apps expose the measured-vs-profiled
  definitional gap — baseline is a pure stage-sum with no pipeline pool tax; measured
  makespan ≈ baseline + fitted per-chunk tax (duck tree VK: 2.40+0.6 ≈ 2.65 observed).
- **phone sparse (0.58–0.88)**: long-running cells under thermal throttling with
  caps=1 (one candidate measured). Needs cooldown-aware re-measurement to be fair.

## 3. The structural finding (paper §discussion material)

Hybrid pipelining pays **iff PU speeds are comparable** on the workload. The naive-
kernel era's larger speedups (up to 5.8×) were partly an artifact of inefficient GPU
kernels making CPUs look competitive. After equal-effort optimization on both sides,
the wins concentrate where the balance survives: tree (integer, GPU:CPU ≈ 1:1.2),
dense on phones (Mali vs big cores ≈ 1:1–3), sparse on Jetson (CSR irregularity taxes
the GPU). The overhead-aware cost model (P2) is what converts this into behavior:
round A (no overhead term) lost cells like samsung dense at 0.93×; round B wins it at
1.84×, and correctly near-refuses pipelining where the GPU dominates.

Also formalized on the way: the CV quality gate must not drop the GPU target PU
(sub-ms kernels + one preempted sample → cv>1; p50 is spike-robust — exempt + warn).

## 4. What's parked / next

- XNNPACK as a pluggable CPU kernel provider (per-tier pinned pthreadpool maps to the
  PU model; OMP stays the oracle; 1-h spike first). See memory: xnnpack-backend-plan.
- Phone-sparse thermal-fair re-measurement (cooldown intervals or caps>1).
- Mali-specific conv tuning (register pressure; only ~1.5× of the RDNA3 9–12× landed).
- exp/cuda-graphs archived (no win); bt-octree / bt-classify streaming demos designed
  but deliberately not started (user call).
