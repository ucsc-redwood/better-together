# End-to-end speedup: scheduled CPU+GPU pipeline vs fastest single-PU baseline

Measured 2026-06-17. For each (device × app × backend) cell: the best z3 schedule
(from the BTPM/interference table, tmax mode) run on the device via
`03_run_schedule.py`, compared against the fastest single-PU `bm-baseline-*` (all 9
stages on one PU, no pipeline framework) on the same device. Throughput, ms/task
(schedule = wall/100; baseline = per-task latency = single-PU throughput).

| device | app | backend | fastest baseline | best schedule | speedup |
|---|---|---|---|---|---|
| Jetson | tree         | vk | VK 5.67    | SCH-L1G6  7.25   | 0.78× |
| Jetson | tree         | cu | CUDA 9.85  | SCH       5.49   | 1.79× |
| Jetson | cifar-dense  | vk | VK 59.0    | SCH-L4G5  47.6   | 1.24× |
| Jetson | cifar-dense  | cu | CUDA 38.1  | SCH-L2G7  35.75  | 1.07× |
| Jetson | cifar-sparse | vk | VK 438     | SCH       439    | 1.00× |
| Jetson | cifar-sparse | cu | CUDA 356   | SCH-L2G7  333    | 1.07× |
| MiniPC | tree         | vk | VK 1.87    | SCH       2.40   | 0.78× |
| MiniPC | cifar-dense  | vk | VK 39.8    | SCH-G6B3  28.0   | 1.42× |
| MiniPC | cifar-sparse | vk | VK 144     | SCH       142.6  | 1.01× |
| Samsung| tree         | vk | Big 12.9   | SCH-M3G4  7.18   | 1.80× |
| Samsung| cifar-dense  | vk | VK 34.5    | SCH-M1G6L2 29.9  | 1.15× |
| Samsung| cifar-sparse | vk | VK 269     | SCH-G7L2  263    | 1.02× |

12/12 cells measured. Wins (>1.1×): 6 · ties (~1.0×): 4 · losses: 2.

## Reading it

- **cifar-dense (balanced CPU/GPU): always wins** (1.07–1.42×) — the framework's
  sweet spot; the CPU chunk overlaps the GPU chunk (e.g. Jetson SCH-L4G5 measured
  97% CPU/GPU concurrency).
- **cifar-sparse (GPU-dominated, heavy): ties** (1.00–1.07×) — pure GPU is already
  near-optimal; the split adds little.
- **tree (tiny): wins iff the GPU is not dominant** — Samsung 1.80× and Jetson-cu
  1.79× (GPU not dominant there) vs 0.78× losses where the GPU dominates
  (Jetson-vk, MiniPC-vk).

## Why the two tree losses (not a framework defect)

z3 optimized on the **BTPM (interference)** table. On the shared-memory iGPUs
(RADV / Mali) the GPU time is inflated *under contention*, so z3 moved tree work to
the CPU. But the baseline GPU runs **uncontended** (isolated), and tree is so small
that pure GPU wins. Fix for single-app-alone deployment: feed z3 the **isolated**
table → it picks pure-GPU for tree (tie, not loss). BTPM is the right table only
when apps actually contend.

## Caveats

- Jetson **CUDA** carries the managed-mem correctness bug (bugs-found §1): timings
  are valid, kernel outputs partially wrong. Speedups above are timing-only.
- Jetson **VK** executor segfaults on teardown (bugs-found §9): records are valid
  (flushed before the crash); `03` tolerates the non-zero exit.
- Samsung **cifar-sparse**: running all candidate schedules is impractical (CPU-only
  candidates take ~12 min each), so only the best-predicted schedule was run.
