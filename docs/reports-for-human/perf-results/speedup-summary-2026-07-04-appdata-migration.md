# Measured pipeline speedups (BetterTogether)

Measured speedup = fastest single-PU whole-pipeline baseline (ms/task, summed from isolated profiling) / best measured pipeline makespan (max-chunk steady-state, min over the z3 btpm/isolated tmax candidates that were run).

| Device | App | Backend | Baseline | Best | Speedup |
|---|---|---|---|---|---|
| duck-stable | tree | CUDA | CUDA 5.83 | isolated 2.62 | 2.23x |
| duck-stable | tree | VK | VK 2.45 | btpm 2.70 | 0.91x |
| duck-stable | cifar-dense | CUDA | CUDA 22.13 | isolated 21.62 | 1.02x |
| duck-stable | cifar-dense | VK | VK 20.93 | isolated 18.07 | 1.16x |
| duck-stable | cifar-sparse | CUDA | CUDA 255.67 | btpm 212.61 | 1.20x |
| duck-stable | cifar-sparse | VK | VK 379.13 | isolated 284.63 | 1.33x |
| samsung | tree | VK | OMP 9.69 | btpm 7.46 | 1.30x |
| samsung | cifar-dense | VK | VK 125.12 | btpm 37.97 | 3.30x |
| samsung | cifar-sparse | VK | VK 1857.10 | isolated 736.76 | 2.52x |
| minipc | tree | VK | VK 1.49 | isolated 2.03 | 0.73x |
| minipc | cifar-dense | VK | VK 517.64 | isolated 437.66 | 1.18x |
| minipc | cifar-sparse | VK | OMP 638.34 | isolated 460.32 | 1.39x |
| pixel | tree | VK | VK 25.22 | btpm 41.52 | 0.61x |
| pixel | cifar-dense | VK | OMP 570.81 | btpm 175.92 | 3.24x |
| pixel | cifar-sparse | VK | VK 4225.75 | btpm 1840.74 | 2.30x |
| jetson | tree | CUDA | CUDA 9.02 | btpm 3.66 | 2.46x |
| jetson | tree | VK | VK 5.27 | btpm 4.29 | 1.23x |
| jetson | cifar-dense | CUDA | CUDA 827.70 | btpm 690.02 | 1.20x |
| jetson | cifar-dense | VK | VK 1171.56 | btpm 896.12 | 1.31x |
| jetson | cifar-sparse | CUDA | CUDA 2033.43 | btpm 1188.62 | 1.71x |
| jetson | cifar-sparse | VK | OMP 2238.62 | isolated 1428.43 | 1.57x |

## Reading the table
- Baseline is the *fastest single processing unit* running the whole pipeline alone (OMP = the fastest CPU tier; VK/CUDA = the GPU); the cell names that PU and its ms/task.
- Best is the best *measured* pipelined makespan across the z3 candidate schedules that were run, and which profiling table (btpm/isolated) z3 solved on.
- Speedup > 1 means software-pipelining across CPU+GPU beat the best single PU.

## Tree losses
tree is a tiny integer pipeline (sub-ms stages); per-task framework overhead (per-stage GPU submit + fence round-trips) is a large fraction of its kernel work, so on devices with higher GPU overhead the pipelined makespan can exceed the fastest single PU (speedup < 1). This is a framework-overhead property of a tiny workload, not a kernel bug.

## Caveats
- Phone (samsung/pixel) cifar-sparse: only the best-predicted schedule(s) were swept where CPU-only candidates were too slow to run all ten.
