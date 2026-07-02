# Measured pipeline speedups (BetterTogether)

Measured speedup = fastest single-PU whole-pipeline baseline (ms/task, summed from isolated profiling) / best measured pipeline makespan (max-chunk steady-state, min over the z3 btpm/isolated tmax candidates that were run).

| Device | App | Backend | Baseline | Best | Speedup |
|---|---|---|---|---|---|
| duck-stable | tree | CUDA | CUDA 5.82 | isolated 3.07 | 1.90x |
| duck-stable | tree | VK | VK 2.41 | isolated 2.64 | 0.91x |
| duck-stable | cifar-dense | CUDA | CUDA 25.95 | btpm 20.35 | 1.28x |
| duck-stable | cifar-dense | VK | VK 32.48 | btpm 25.43 | 1.28x |
| duck-stable | cifar-sparse | CUDA | CUDA 216.37 | isolated 133.29 | 1.62x |
| duck-stable | cifar-sparse | VK | VK 263.87 | isolated 164.50 | 1.60x |
| duck-naughty | tree | CUDA | CUDA 5.71 | isolated 2.60 | 2.19x |
| duck-naughty | tree | VK | VK 2.51 | isolated 2.59 | 0.97x |
| duck-naughty | cifar-dense | CUDA | CUDA 23.67 | btpm 18.66 | 1.27x |
| duck-naughty | cifar-dense | VK | VK 32.11 | btpm 25.11 | 1.28x |
| duck-naughty | cifar-sparse | CUDA | CUDA 214.98 | isolated 133.03 | 1.62x |
| duck-naughty | cifar-sparse | VK | VK 264.67 | btpm 164.37 | 1.61x |
| samsung | tree | VK | OMP 7.40 | btpm 4.21 | 1.76x |
| samsung | cifar-dense | VK | VK 23.79 | btpm 15.80 | 1.51x |
| samsung | cifar-sparse | VK | VK 385.50 | isolated 342.72 | 1.12x |
| minipc | tree | VK | VK 1.52 | btpm 2.03 | 0.75x |
| minipc | cifar-dense | VK | VK 21.86 | btpm 17.56 | 1.24x |
| minipc | cifar-sparse | VK | OMP 118.63 | isolated 96.29 | 1.23x |
| pixel | tree | VK | OMP 9.32 | btpm 7.93 | 1.18x |
| pixel | cifar-dense | VK | VK 102.08 | btpm 128.73 | 0.79x |
| pixel | cifar-sparse | VK | VK 815.57 | isolated 692.84 | 1.18x |

## Reading the table
- Baseline is the *fastest single processing unit* running the whole pipeline alone (OMP = the fastest CPU tier; VK/CUDA = the GPU); the cell names that PU and its ms/task.
- Best is the best *measured* pipelined makespan across the z3 candidate schedules that were run, and which profiling table (btpm/isolated) z3 solved on.
- Speedup > 1 means software-pipelining across CPU+GPU beat the best single PU.

## Tree losses
tree is a tiny integer pipeline (sub-ms stages); per-task framework overhead (per-stage GPU submit + fence round-trips) is a large fraction of its kernel work, so on devices with higher GPU overhead the pipelined makespan can exceed the fastest single PU (speedup < 1). This is a framework-overhead property of a tiny workload, not a kernel bug.

## Caveats
- Phone (samsung/pixel) cifar-sparse: only the best-predicted schedule(s) were swept where CPU-only candidates were too slow to run all ten.
