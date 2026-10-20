# Measured pipeline speedups (BetterTogether)

Measured speedup = fastest single-PU whole-pipeline baseline (ms/task, summed from isolated profiling) / best measured pipeline makespan (max-chunk steady-state, min over the z3 btpm/isolated tmax candidates that were run).

| Device | App | Backend | Baseline | Best | Speedup |
|---|---|---|---|---|---|
| samsung | tree | VK | VK 10.09 | isolated 4.02 | 2.51x |
| samsung | cifar-dense | VK | VK 16.96 | btpm 9.77 | 1.74x |
| samsung | cifar-sparse | VK | VK 118.79 | btpm 153.65 | 0.77x |
| pixel | tree | VK | VK 13.10 | isolated 8.83 | 1.48x |
| pixel | cifar-dense | VK | OMP 98.07 | isolated 37.78 | 2.60x |
| pixel | cifar-sparse | VK | VK 604.44 | btpm 419.43 | 1.44x |

## Reading the table
- Baseline is the *fastest single processing unit* running the whole pipeline alone (OMP = the fastest CPU tier; VK/CUDA = the GPU); the cell names that PU and its ms/task.
- Best is the best *measured* pipelined makespan across the z3 candidate schedules that were run, and which profiling table (btpm/isolated) z3 solved on.
- Speedup > 1 means software-pipelining across CPU+GPU beat the best single PU.

## Tree losses
tree is a tiny integer pipeline (sub-ms stages); per-task framework overhead (per-stage GPU submit + fence round-trips) is a large fraction of its kernel work, so on devices with higher GPU overhead the pipelined makespan can exceed the fastest single PU (speedup < 1). This is a framework-overhead property of a tiny workload, not a kernel bug.

## Caveats
- Phone (samsung/pixel) cifar-sparse: only the best-predicted schedule(s) were swept where CPU-only candidates were too slow to run all ten.

## Prediction accuracy

Measured period = per run log, each chunk's mean duration summed per PU, maxed over PUs; the median over the logs (spread [min-max]). Predicted = the schedule's metrics.max_time. Error = |measured / predicted - 1|. A run is matched to every schedule file of its cell that proposed the same chunk assignment, and an assignment's logs are pooled over the cell's log folders (Logs). Cost model and table are the ones the schedule records (metrics.cost_model / metrics.table); an assignment priced the same way in several files (both minimize modes, a per-stage fallback) counts once.

| Device | App | Backend | Cost model | Table | Executed | Median abs error | Best measured ms [min-max] |
|---|---|---|---|---|---|---|---|
| samsung | tree | VK | per-stage | btpm | 10 | 0.445 | 4.32 [4.15-4.32] |
| samsung | tree | VK | per-stage | isolated | 10 | 0.171 | 4.05 [4.02-4.07] |
| samsung | cifar-dense | VK | per-stage | btpm | 10 | 0.051 | 9.74 [9.72-9.85] |
| samsung | cifar-dense | VK | per-stage | isolated | 10 | 0.067 | 10.84 [10.69-11.06] |
| samsung | cifar-sparse | VK | per-stage | btpm | 6 | 0.049 | 153.27 [153.10-154.57] |
| samsung | cifar-sparse | VK | per-stage | isolated | 6 | 0.662 | 153.27 [153.10-154.57] |
| pixel | tree | VK | per-stage | btpm | 10 | 0.487 | 8.96 [8.15-9.64] |
| pixel | tree | VK | per-stage | isolated | 10 | 0.853 | 8.92 [8.62-8.95] |
| pixel | cifar-dense | VK | per-stage | btpm | 10 | 0.122 | 40.68 [40.65-40.72] |
| pixel | cifar-dense | VK | per-stage | isolated | 10 | 0.052 | 37.80 [37.42-38.10] |
| pixel | cifar-sparse | VK | per-stage | btpm | 5 | 0.108 | 419.52 [419.24-457.35] |
| pixel | cifar-sparse | VK | per-stage | isolated | 5 | 0.092 | 419.52 [419.24-457.35] |

### Executed schedules

| Device | App | Backend | Cost model | Table | Mode | Schedule | Predicted ms | Measured ms [min-max] | Logs | Measured / predicted |
|---|---|---|---|---|---|---|---|---|---|---|
| samsung | tree | VK | per-stage | btpm | tmax | SCH-B1G6-G248-7d95 | 3.55 | 9.05 [7.56-9.41] | 3 | 2.547 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-L1G6-G020-e923 | 3.55 | 6.27 [6.21-6.58] | 3 | 1.765 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-M1G6-G287-e417 | 3.55 | 8.70 [8.66-8.78] | 3 | 2.450 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-G7-G000-ad7d | 3.74 | 11.19 [10.33-11.50] | 3 | 2.991 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-B1G4L1M1-G421-3984 | 4.69 | 5.80 [5.77-6.07] | 3 | 1.236 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-G5L1M1-G421-dabd | 4.69 | 6.21 [5.71-6.33] | 3 | 1.322 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-G4L2M1-G257-aa42 | 4.69 | 4.71 [4.42-4.90] | 3 | 1.004 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-B1G3L2M1-G362-310b | 4.69 | 4.32 [4.15-4.32] | 3 | 0.920 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-G5B1M1-G453-78a7 | 4.69 | 7.36 [6.92-7.39] | 3 | 1.568 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-L1G4B1M1-G453-2437 | 4.69 | 5.03 [5.01-5.15] | 3 | 1.071 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2L1M3B1-G294-eec3 | 3.62 | 4.05 [4.02-4.07] | 3 | 1.118 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2M2L2B1-G222-e530 | 3.62 | 4.31 [4.25-4.51] | 3 | 1.188 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-L1G2M3B1-G104-0b04 | 3.62 | 4.38 [4.26-4.51] | 3 | 1.209 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2M3L1B1-G321-24e8 | 3.69 | 4.08 [3.88-4.11] | 3 | 1.103 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2L1M2B2-G310-dd59 | 3.79 | 4.24 [4.22-4.26] | 3 | 1.120 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2M2L1B2-G287-f0a3 | 3.79 | 4.69 [4.34-4.83] | 3 | 1.240 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-L1G1M3B2-G249-8ab8 | 3.79 | 4.64 [4.48-4.94] | 3 | 1.225 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2M3B2-G091-723e | 3.79 | 4.34 [4.15-4.40] | 3 | 1.147 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-L1G2M2B2-G120-9f42 | 3.79 | 4.45 [4.44-4.81] | 3 | 1.176 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-L1G1M4B1-G258-d56e | 3.87 | 4.51 [4.07-4.53] | 3 | 1.165 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-L2G8B1-G1033-9e89 | 10.36 | 10.20 [10.17-10.33] | 3 | 0.984 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-L1M1G8B1-G1033-1b40 | 10.36 | 9.85 [9.83-9.97] | 3 | 0.951 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-M1L1G8B1-G1033-e93c | 10.36 | 9.74 [9.72-9.85] | 3 | 0.941 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-M2G8B1-G1033-e0b2 | 10.36 | 11.25 [11.19-11.28] | 3 | 1.087 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-B2G8M1-G1034-f8d1 | 10.36 | 11.33 [11.22-11.39] | 3 | 1.094 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-B1L1G8M1-G1034-0acf | 10.36 | 9.93 [9.91-10.05] | 3 | 0.959 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-L2G8M1-G1034-276f | 10.36 | 10.07 [9.78-10.31] | 3 | 0.972 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-L1B1G8M1-G1034-a3e1 | 10.36 | 9.81 [9.64-9.91] | 3 | 0.947 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-M2G8L1-G1029-c17d | 10.36 | 10.05 [10.05-10.15] | 3 | 0.971 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-M1B1G8L1-G1029-c38a | 10.36 | 9.78 [9.73-9.84] | 3 | 0.945 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2G5M2B2-G462-ed31 | 10.26 | 10.87 [10.81-11.75] | 3 | 1.059 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2G5B2M2-G471-abbf | 10.35 | 11.14 [11.09-11.18] | 3 | 1.077 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2M1G6B2-G477-5287 | 10.41 | 11.76 [11.27-12.20] | 3 | 1.130 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1G6M2B2-G821-567d | 10.56 | 11.26 [10.62-11.76] | 3 | 1.066 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1G6B2M2-G821-7f9c | 10.56 | 11.31 [11.14-11.50] | 3 | 1.071 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2G6M1B2-G494-7791 | 10.58 | 11.63 [10.70-12.21] | 3 | 1.099 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2G6B1M2-G494-62d4 | 10.58 | 11.06 [10.66-11.19] | 3 | 1.045 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1G7M1B2-G854-cb45 | 10.89 | 11.39 [11.15-12.35] | 3 | 1.046 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1G7B1M2-G854-1782 | 10.89 | 10.84 [10.69-11.06] | 3 | 0.996 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2M2G5B2-G544-052d | 11.08 | 11.84 [11.83-12.12] | 3 | 1.068 |
| samsung | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L2G5M2B2-G6627-c9b4 | 147.21 | 154.25 [153.26-158.24] | 6 | 1.048 |
| samsung | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L2G5B2M2-G5605-da4e | 147.21 | 154.78 [152.97-156.61] | 6 | 1.051 |
| samsung | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L2G6B1M2-G5640-04fc | 147.55 | 155.07 [152.02-157.69] | 6 | 1.051 |
| samsung | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L2G6M1B2-G6782-041f | 147.55 | 155.14 [154.13-155.97] | 6 | 1.051 |
| samsung | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L1G6M2B2-G8876-1877 | 148.30 | 153.27 [153.10-154.57] | 3 | 1.034 |
| samsung | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L1G6B2M2-G8876-f2a1 | 148.30 | 153.88 [152.18-155.83] | 3 | 1.038 |
| samsung | cifar-sparse | VK | per-stage | isolated | tmax | SCH-L2G5B2M2-G4345-c2d0 | 92.55 | 154.78 [152.97-156.61] | 6 | 1.672 |
| samsung | cifar-sparse | VK | per-stage | isolated | tmax | SCH-L2G5M2B2-G4345-e8bc | 92.55 | 154.25 [153.26-158.24] | 6 | 1.667 |
| samsung | cifar-sparse | VK | per-stage | isolated | tmax | SCH-L2G6B1M2-G4423-26ee | 93.33 | 155.07 [152.02-157.69] | 6 | 1.662 |
| samsung | cifar-sparse | VK | per-stage | isolated | tmax | SCH-L2G6M1B2-G4423-cf3c | 93.33 | 155.14 [154.13-155.97] | 6 | 1.662 |
| samsung | cifar-sparse | VK | per-stage | isolated | tmax | SCH-L1G6B2M2-G6255-6d59 | 94.91 | 153.88 [152.18-155.83] | 3 | 1.621 |
| samsung | cifar-sparse | VK | per-stage | isolated | tmax | SCH-L1G6M2B2-G6255-5e00 | 94.91 | 153.27 [153.10-154.57] | 3 | 1.615 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-L1G1B4M1-G432-dae3 | 7.57 | 8.96 [8.15-9.64] | 6 | 1.184 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G2B4M1-G203-28d4 | 7.57 | 11.71 [11.43-11.83] | 6 | 1.546 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G2B3L1M1-G698-a93f | 7.57 | 11.34 [11.08-11.61] | 6 | 1.498 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G2L1B3M1-G635-30b3 | 7.57 | 11.18 [9.70-11.49] | 6 | 1.476 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G2B2L2M1-G530-d0ee | 7.57 | 8.99 [8.45-9.34] | 6 | 1.188 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-L1G2B3M1-G432-63d8 | 7.57 | 10.73 [9.94-10.83] | 3 | 1.417 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G3B1L2M1-G530-2976 | 7.57 | 9.45 [9.24-9.62] | 3 | 1.248 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G3B2L1M1-G698-4ac6 | 7.57 | 12.24 [9.75-12.37] | 3 | 1.616 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G3B3M1-G140-e4b1 | 7.57 | 12.26 [12.14-12.81] | 3 | 1.620 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-L1G3B2M1-G705-c24a | 7.67 | 13.03 [12.65-13.18] | 3 | 1.700 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1G1B4M1-G284-f467 | 5.33 | 8.96 [8.15-9.64] | 6 | 1.681 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-G2B2L2M1-G378-6d6f | 5.46 | 8.99 [8.45-9.34] | 6 | 1.647 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-G2L1B3M1-G417-662f | 5.46 | 11.18 [9.70-11.49] | 6 | 2.046 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-G2B3L1M1-G487-6d2a | 5.46 | 11.34 [11.08-11.61] | 6 | 2.076 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-G2B4M1-G071-11c1 | 5.46 | 11.71 [11.43-11.83] | 6 | 2.143 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1G1B3M2-G306-25c9 | 5.55 | 9.39 [9.28-9.58] | 3 | 1.691 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-G2L1B2M2-G426-873b | 5.55 | 11.18 [11.06-11.50] | 3 | 2.014 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-G2B3M2-G098-e694 | 5.55 | 11.61 [11.41-11.89] | 3 | 2.091 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-G2B2L1M2-G447-3e69 | 5.55 | 9.04 [9.04-9.23] | 3 | 1.629 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-G2M2L2B1-G392-7c0f | 5.60 | 8.92 [8.62-8.95] | 3 | 1.593 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-L2B2M1G6-G3337-e2f7 | 46.52 | 40.82 [40.60-40.86] | 3 | 0.878 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-L2B1M2G6-G3337-b4ef | 46.52 | 40.80 [40.75-41.15] | 3 | 0.877 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-L1B2M2G6-G3896-f330 | 46.52 | 40.87 [40.82-40.91] | 3 | 0.878 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-L1B3M1G6-G3896-4a8d | 46.52 | 40.83 [40.81-40.85] | 3 | 0.878 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-B3M2G6-G2091-d0f0 | 46.52 | 41.40 [41.33-41.46] | 3 | 0.890 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-B4M1G6-G1924-0a7f | 46.52 | 41.41 [41.38-41.59] | 3 | 0.890 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-B3L1M1G6-G4274-f16a | 46.52 | 40.69 [40.64-40.71] | 3 | 0.875 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-L1M2B2G6-G3896-8780 | 46.52 | 40.68 [40.65-40.72] | 3 | 0.874 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-M3B2G6-G2124-de91 | 46.52 | 41.53 [41.36-41.64] | 3 | 0.893 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-L2M1B2G6-G3337-3cbf | 46.52 | 40.89 [40.81-41.09] | 3 | 0.879 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2B3G2M4-G2977-9a59 | 37.12 | 38.48 [38.32-38.54] | 3 | 1.037 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-B5G2L1M3-G3636-cbbf | 37.12 | 41.72 [41.23-42.50] | 3 | 1.124 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-B5G2M4-G817-cc51 | 37.12 | 38.93 [38.55-39.25] | 3 | 1.049 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-B5G2M3L1-G3683-96fa | 37.12 | 42.57 [41.63-42.76] | 3 | 1.147 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1B4G2M4-G3285-0f4b | 37.12 | 39.96 [39.83-40.16] | 3 | 1.077 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2B3G3M3-G2992-ee8d | 37.26 | 37.86 [37.68-38.62] | 3 | 1.016 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-B5G3M3-G868-dd79 | 37.26 | 38.94 [38.63-40.17] | 3 | 1.045 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-B5G3M2L1-G3698-8a3f | 37.26 | 41.92 [41.87-41.92] | 3 | 1.125 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1B4G3M3-G3299-c2bc | 37.26 | 39.34 [38.96-39.57] | 3 | 1.056 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-M3L1G2B5-G3569-cf1d | 37.98 | 37.80 [37.42-38.10] | 3 | 0.995 |
| pixel | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L1B3G3M4-G41505-5d3f | 470.88 | 419.52 [419.24-457.35] | 6 | 0.891 |
| pixel | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L2B2G3M4-G38859-8ce6 | 470.88 | 419.70 [419.40-420.78] | 6 | 0.891 |
| pixel | cifar-sparse | VK | per-stage | btpm | tmax | SCH-B4G3L1M3-G46569-dc4d | 470.88 | 420.17 [419.95-455.09] | 6 | 0.892 |
| pixel | cifar-sparse | VK | per-stage | btpm | tmax | SCH-B3L1G3M4-G45527-6db4 | 470.88 | 429.00 [419.55-451.33] | 6 | 0.911 |
| pixel | cifar-sparse | VK | per-stage | btpm | tmax | SCH-B4G3M3L1-G46839-8907 | 470.88 | 425.61 [419.79-442.86] | 6 | 0.904 |
| pixel | cifar-sparse | VK | per-stage | isolated | tmax | SCH-L2B2G3M4-G41214-7dc1 | 462.63 | 419.70 [419.40-420.78] | 6 | 0.907 |
| pixel | cifar-sparse | VK | per-stage | isolated | tmax | SCH-L1B3G3M4-G42693-b03e | 462.63 | 419.52 [419.24-457.35] | 6 | 0.907 |
| pixel | cifar-sparse | VK | per-stage | isolated | tmax | SCH-B4G3L1M3-G45826-aaa6 | 462.63 | 420.17 [419.95-455.09] | 6 | 0.908 |
| pixel | cifar-sparse | VK | per-stage | isolated | tmax | SCH-B3L1G3M4-G45077-74fa | 462.63 | 429.00 [419.55-451.33] | 6 | 0.927 |
| pixel | cifar-sparse | VK | per-stage | isolated | tmax | SCH-B4G3M3L1-G46130-4844 | 462.63 | 425.61 [419.79-442.86] | 6 | 0.920 |
