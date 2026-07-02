# Measured pipeline speedups (BetterTogether)

Measured speedup = fastest single-PU whole-pipeline baseline (ms/task, summed from isolated profiling) / best measured pipeline makespan (max-chunk steady-state, min over the z3 btpm/isolated tmax candidates that were run).

| Device | App | Backend | Baseline | Best | Speedup |
|---|---|---|---|---|---|
| duck-stable | tree | CUDA | CUDA 5.82 | btpm 2.83 | 2.06x |
| duck-stable | tree | VK | VK 2.41 | btpm 2.63 | 0.91x |
| duck-stable | cifar-dense | CUDA | CUDA 25.95 | btpm 20.35 | 1.28x |
| duck-stable | cifar-dense | VK | VK 32.48 | btpm 25.46 | 1.28x |
| duck-stable | cifar-sparse | CUDA | CUDA 216.37 | btpm 133.25 | 1.62x |
| duck-stable | cifar-sparse | VK | VK 263.87 | isolated 164.35 | 1.61x |
| duck-naughty | tree | CUDA | CUDA 5.71 | isolated 2.77 | 2.06x |
| duck-naughty | tree | VK | VK 2.51 | isolated 2.60 | 0.96x |
| duck-naughty | cifar-dense | CUDA | CUDA 23.67 | btpm 18.64 | 1.27x |
| duck-naughty | cifar-dense | VK | VK 32.11 | isolated 25.08 | 1.28x |
| duck-naughty | cifar-sparse | CUDA | CUDA 214.98 | isolated 132.99 | 1.62x |
| duck-naughty | cifar-sparse | VK | VK 264.67 | isolated 164.85 | 1.61x |
| samsung | tree | VK | OMP 7.40 | btpm 7.53 | 0.98x |
| samsung | cifar-dense | VK | VK 23.79 | btpm 15.90 | 1.50x |
| samsung | cifar-sparse | VK | VK 385.50 | isolated 270.67 | 1.42x |
| minipc | tree | VK | VK 1.52 | btpm 2.02 | 0.75x |
| minipc | cifar-dense | VK | VK 21.86 | btpm 17.53 | 1.25x |
| minipc | cifar-sparse | VK | OMP 118.63 | isolated 94.10 | 1.26x |
| pixel | tree | VK | OMP 9.32 | btpm 11.03 | 0.85x |
| pixel | cifar-dense | VK | VK 102.08 | btpm 81.79 | 1.25x |
| pixel | cifar-sparse | VK | VK 815.57 | isolated 595.51 | 1.37x |

## Reading the table
- Baseline is the *fastest single processing unit* running the whole pipeline alone (OMP = the fastest CPU tier; VK/CUDA = the GPU); the cell names that PU and its ms/task.
- Best is the best *measured* pipelined makespan across the z3 candidate schedules that were run, and which profiling table (btpm/isolated) z3 solved on.
- Speedup > 1 means software-pipelining across CPU+GPU beat the best single PU.

## Tree losses
tree is a tiny integer pipeline (sub-ms stages); per-task framework overhead (per-stage GPU submit + fence round-trips) is a large fraction of its kernel work, so on devices with higher GPU overhead the pipelined makespan can exceed the fastest single PU (speedup < 1). This is a framework-overhead property of a tiny workload, not a kernel bug.

## Caveats
- Phone (samsung/pixel) cifar-sparse: only the best-predicted schedule(s) were swept where CPU-only candidates were too slow to run all ten.
