# Measured pipeline speedups (BetterTogether)

Measured speedup = fastest single-PU whole-pipeline baseline (ms/task, summed from isolated profiling) / best measured pipeline makespan (max-chunk steady-state, min over the z3 btpm/isolated tmax candidates that were run).

| Device | App | Backend | Baseline | Best | Speedup |
|---|---|---|---|---|---|
| duck-stable | tree | CUDA | CUDA 5.82 | btpm 3.06 | 1.90x |
| duck-stable | tree | VK | VK 2.41 | btpm 2.67 | 0.90x |
| duck-stable | cifar-dense | CUDA | CUDA 25.95 | btpm 20.34 | 1.28x |
| duck-stable | cifar-dense | VK | VK 32.48 | btpm 25.24 | 1.29x |
| duck-stable | cifar-sparse | CUDA | CUDA 216.37 | isolated 133.31 | 1.62x |
| duck-stable | cifar-sparse | VK | VK 263.87 | isolated 164.03 | 1.61x |
| duck-naughty | tree | CUDA | CUDA 5.71 | isolated 2.67 | 2.14x |
| duck-naughty | tree | VK | VK 2.51 | isolated 2.60 | 0.96x |
| duck-naughty | cifar-dense | CUDA | CUDA 23.67 | isolated 18.70 | 1.27x |
| duck-naughty | cifar-dense | VK | VK 32.11 | btpm 25.32 | 1.27x |
| duck-naughty | cifar-sparse | CUDA | CUDA 214.98 | btpm 133.04 | 1.62x |
| duck-naughty | cifar-sparse | VK | VK 264.67 | isolated 164.28 | 1.61x |
| samsung | tree | VK | OMP 7.40 | btpm 4.95 | 1.50x |
| samsung | cifar-dense | VK | VK 23.79 | btpm 31.68 | 0.75x |
| samsung | cifar-sparse | VK | VK 385.50 | btpm 402.98 | 0.96x |
| minipc | tree | VK | VK 1.52 | btpm 2.03 | 0.75x |
| minipc | cifar-dense | VK | VK 21.86 | isolated 17.90 | 1.22x |
| minipc | cifar-sparse | VK | OMP 118.63 | isolated 96.52 | 1.23x |
| pixel | tree | VK | OMP 9.32 | isolated 7.52 | 1.24x |
| pixel | cifar-dense | VK | VK 102.08 | btpm 101.31 | 1.01x |
| pixel | cifar-sparse | VK | VK 815.57 | isolated 803.45 | 1.02x |

## Reading the table
- Baseline is the *fastest single processing unit* running the whole pipeline alone (OMP = the fastest CPU tier; VK/CUDA = the GPU); the cell names that PU and its ms/task.
- Best is the best *measured* pipelined makespan across the z3 candidate schedules that were run, and which profiling table (btpm/isolated) z3 solved on.
- Speedup > 1 means software-pipelining across CPU+GPU beat the best single PU.

## Tree losses
tree is a tiny integer pipeline (sub-ms stages); per-task framework overhead (per-stage GPU submit + fence round-trips) is a large fraction of its kernel work, so on devices with higher GPU overhead the pipelined makespan can exceed the fastest single PU (speedup < 1). This is a framework-overhead property of a tiny workload, not a kernel bug.

## Caveats
- Phone (samsung/pixel) cifar-sparse: only the best-predicted schedule(s) were swept where CPU-only candidates were too slow to run all ten.
