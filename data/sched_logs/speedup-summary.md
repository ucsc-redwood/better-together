# Measured pipeline speedups (BetterTogether)

Measured speedup = fastest single-PU whole-pipeline baseline (ms/task, summed from isolated profiling) / best measured pipeline makespan (max-chunk steady-state, min over the z3 btpm/isolated tmax candidates that were run).

| Device | App | Backend | Baseline | Best | Speedup |
|---|---|---|---|---|---|
| samsung | tree | VK | VK 8.98 | btpm 3.91 | 2.30x |
| samsung | cifar-dense | VK | VK 20.39 | isolated 10.98 | 1.86x |
| samsung | cifar-sparse | VK | VK 284.62 | isolated 183.72 | 1.55x |
| pixel | tree | VK | OMP 16.36 | isolated 6.86 | 2.38x |
| pixel | cifar-dense | VK | VK 81.75 | isolated 39.59 | 2.06x |
| pixel | cifar-sparse | VK | VK 580.43 | btpm 421.31 | 1.38x |

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
| samsung | tree | VK | per-stage | btpm | 10 | 0.575 | 3.95 [3.83-4.23] |
| samsung | tree | VK | per-stage | isolated | 10 | 0.161 | 3.94 [3.78-4.51] |
| samsung | cifar-dense | VK | per-stage | btpm | 10 | 0.060 | 11.26 [11.22-11.45] |
| samsung | cifar-dense | VK | per-stage | isolated | 10 | 0.049 | 10.99 [10.92-11.02] |
| samsung | cifar-sparse | VK | per-stage | btpm | 5 | 0.344 | 213.92 [211.10-215.12] |
| samsung | cifar-sparse | VK | per-stage | isolated | 5 | 0.162 | 183.20 [182.14-194.23] |
| pixel | tree | VK | per-stage | btpm | 10 | 0.701 | 8.60 [8.46-8.98] |
| pixel | tree | VK | per-stage | isolated | 10 | 0.027 | 6.87 [6.85-6.87] |
| pixel | cifar-dense | VK | per-stage | btpm | 10 | 0.116 | 43.19 [42.90-43.43] |
| pixel | cifar-dense | VK | per-stage | isolated | 10 | 0.120 | 39.59 [39.55-39.63] |
| pixel | cifar-sparse | VK | per-stage | btpm | 6 | 0.009 | 431.24 [419.81-451.62] |
| pixel | cifar-sparse | VK | per-stage | isolated | 6 | 0.080 | 431.24 [419.81-451.62] |

### Executed schedules

| Device | App | Backend | Cost model | Table | Mode | Schedule | Predicted ms | Measured ms [min-max] | Logs | Measured / predicted |
|---|---|---|---|---|---|---|---|---|---|---|
| samsung | tree | VK | per-stage | btpm | tmax | SCH-M1G6-G271-ed0c | 3.44 | 8.69 [8.58-8.72] | 3 | 2.526 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-B1G6-G229-dec2 | 3.44 | 8.66 [8.52-8.76] | 3 | 2.517 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-G7-G000-766f | 3.63 | 11.17 [10.26-11.32] | 3 | 3.079 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-L1G6-G038-0ff0 | 3.82 | 6.42 [6.23-6.67] | 3 | 1.679 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-G6M1-G224-136a | 4.80 | 7.73 [7.57-7.93] | 3 | 1.612 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-G4L2M1-G271-4f2f | 4.80 | 4.96 [4.80-5.11] | 3 | 1.033 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-G3B1L2M1-G319-90a3 | 4.80 | 4.27 [4.25-4.29] | 3 | 0.890 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-G2B2L2M1-G353-179e | 4.80 | 4.48 [4.45-4.51] | 3 | 0.934 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-G4B2M1-G432-1266 | 4.80 | 7.37 [7.03-7.40] | 3 | 1.537 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-L1G3B2M1-G432-3b83 | 4.80 | 3.95 [3.83-4.23] | 6 | 0.823 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2M2L2B1-G226-3f74 | 3.66 | 4.24 [4.21-4.81] | 3 | 1.161 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2L1M3B1-G297-2279 | 3.66 | 4.08 [4.04-4.11] | 3 | 1.116 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-L1G2M3B1-G121-a8fc | 3.66 | 4.55 [4.23-4.60] | 3 | 1.246 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2M3L1B1-G323-2fd1 | 3.71 | 3.98 [3.93-4.13] | 3 | 1.071 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-B1G3L1M2-G291-7bf8 | 3.82 | 4.52 [4.42-4.58] | 3 | 1.185 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-B1G3L2M1-G291-d80b | 3.82 | 3.94 [3.78-4.51] | 3 | 1.033 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-L1G3B2M1-G335-370b | 3.82 | 3.95 [3.83-4.23] | 6 | 1.034 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-L1G3B1M2-G351-b36e | 3.82 | 4.43 [3.64-4.55] | 3 | 1.161 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-L1G3M3-G123-ec79 | 3.82 | 4.99 [4.81-5.05] | 3 | 1.308 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-B1G3M3-G291-a7df | 3.82 | 6.54 [6.40-6.86] | 3 | 1.713 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-L2M1G7B1-G1169-f0bd | 12.28 | 11.52 [11.41-11.85] | 3 | 0.938 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-B2M1G7L1-G1165-89b6 | 12.28 | 11.65 [11.36-11.68] | 3 | 0.949 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-L2M1B1G7-G1060-853e | 12.30 | 11.26 [11.22-11.45] | 3 | 0.915 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-B2M1L1G7-G888-07f4 | 12.30 | 11.67 [11.47-11.96] | 3 | 0.949 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-L2M1G8-G338-deae | 12.47 | 11.41 [11.40-11.62] | 3 | 0.914 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-L1B1M1G8-G1047-05b4 | 12.47 | 11.43 [11.39-11.54] | 3 | 0.916 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-B1L1M1G8-G1050-6b3b | 12.47 | 11.46 [11.24-11.48] | 3 | 0.919 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-B2M1G8-G905-715d | 12.47 | 12.29 [11.63-12.72] | 3 | 0.985 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-L2G8M1-G1243-b8b8 | 13.01 | 12.25 [12.16-12.47] | 3 | 0.941 |
| samsung | cifar-dense | VK | per-stage | btpm | tmax | SCH-B1L1G8M1-G1243-921a | 13.01 | 12.33 [12.19-12.49] | 3 | 0.948 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2G5B2M2-G482-8ed4 | 11.49 | 11.22 [11.06-11.40] | 3 | 0.977 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2G5M2B2-G482-2198 | 11.49 | 11.27 [10.99-11.66] | 3 | 0.981 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2G6B1M2-G514-c9ab | 11.81 | 10.99 [10.92-11.02] | 3 | 0.930 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2G6M1B2-G514-be52 | 11.81 | 11.31 [11.24-11.56] | 3 | 0.957 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2M2G5B2-G566-1baf | 12.32 | 12.00 [11.89-12.40] | 3 | 0.974 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2M1G6B2-G569-e8e8 | 12.36 | 11.78 [11.58-11.84] | 3 | 0.953 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1M2G6B2-G941-3b92 | 12.56 | 11.93 [11.91-12.37] | 3 | 0.949 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1G6B2M2-G945-0dbd | 12.61 | 11.71 [11.49-11.97] | 3 | 0.929 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1G6M2B2-G945-d0a6 | 12.61 | 11.16 [11.04-11.25] | 3 | 0.885 |
| samsung | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1G7M1B2-G978-b8af | 12.93 | 11.24 [10.98-11.27] | 3 | 0.869 |
| samsung | cifar-sparse | VK | per-stage | btpm | tmax | SCH-B4G3M3L1-G33527-01cd | 336.31 | 220.70 [219.15-221.08] | 3 | 0.656 |
| samsung | cifar-sparse | VK | per-stage | btpm | tmax | SCH-B4G3M4-G9089-8315 | 336.31 | 221.88 [219.91-222.81] | 3 | 0.660 |
| samsung | cifar-sparse | VK | per-stage | btpm | tmax | SCH-B4G3L1M3-G32679-4677 | 336.31 | 226.01 [223.52-226.50] | 3 | 0.672 |
| samsung | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L2B2G3M4-G25650-260f | 336.31 | 213.92 [211.10-215.12] | 3 | 0.636 |
| samsung | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L1B3G3M4-G29037-8f0c | 336.31 | 218.84 [217.72-219.60] | 3 | 0.651 |
| samsung | cifar-sparse | VK | per-stage | isolated | tmax | SCH-M4G4B2L1-G21875-23f8 | 219.67 | 183.68 [183.54-183.96] | 3 | 0.836 |
| samsung | cifar-sparse | VK | per-stage | isolated | tmax | SCH-M3G5B2L1-G21875-8522 | 219.67 | 185.26 [183.65-186.51] | 3 | 0.843 |
| samsung | cifar-sparse | VK | per-stage | isolated | tmax | SCH-M4G4B3-G1607-12f5 | 219.88 | 183.20 [182.14-194.23] | 3 | 0.833 |
| samsung | cifar-sparse | VK | per-stage | isolated | tmax | SCH-M3G5B3-G2029-c7b7 | 219.88 | 184.34 [182.46-186.80] | 3 | 0.838 |
| samsung | cifar-sparse | VK | per-stage | isolated | tmax | SCH-M3G4L1B3-G21390-d58f | 219.88 | 187.55 [185.67-193.19] | 3 | 0.853 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G2L1M3B1-G474-6538 | 5.97 | 11.07 [11.04-11.51] | 3 | 1.854 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G2M2L2B1-G386-46ae | 5.97 | 8.60 [8.46-8.98] | 3 | 1.440 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G2M3L1B1-G542-647a | 5.97 | 11.29 [11.16-11.39] | 3 | 1.891 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-L1G2M3B1-G303-4143 | 5.97 | 9.01 [8.60-9.84] | 3 | 1.509 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G3M3B1-G031-240f | 5.97 | 10.99 [10.63-10.99] | 3 | 1.840 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G3M1L2B1-G386-b48b | 5.97 | 9.33 [9.07-9.34] | 3 | 1.562 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G3M2L1B1-G542-bc97 | 5.97 | 11.14 [10.95-11.42] | 3 | 1.866 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G2B2L2M1-G386-804e | 5.98 | 8.85 [8.43-9.12] | 3 | 1.482 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G2B3L1M1-G543-2f33 | 5.98 | 11.13 [11.07-11.16] | 3 | 1.863 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-L1G1B4M1-G303-8435 | 5.98 | 9.19 [8.23-9.96] | 3 | 1.538 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1B1G2M3-G428-9f2d | 6.74 | 7.03 [6.93-7.07] | 3 | 1.043 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1B1G3M2-G428-cfdd | 6.74 | 6.87 [6.85-6.87] | 3 | 1.019 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1B1G4M1-G428-3a07 | 6.74 | 7.21 [7.01-7.84] | 3 | 1.070 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1B1M2G3-G428-fcdf | 6.74 | 6.99 [6.96-7.00] | 3 | 1.036 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1B1M3G2-G428-1081 | 6.74 | 6.91 [6.85-6.92] | 3 | 1.025 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1B1M4G1-G428-25e4 | 6.74 | 6.92 [6.91-6.94] | 3 | 1.026 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1B2M3G1-G446-087c | 6.92 | 7.12 [7.10-7.15] | 3 | 1.028 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1B2M2G2-G446-1278 | 6.92 | 7.02 [7.02-7.15] | 3 | 1.015 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1B2M1G3-G446-ed30 | 6.92 | 7.04 [7.02-7.09] | 3 | 1.018 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1B2G1M3-G446-d188 | 6.92 | 7.18 [7.15-7.20] | 3 | 1.038 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-M4B1G5L1-G4821-09d0 | 48.73 | 43.19 [42.90-43.43] | 6 | 0.886 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-M3B2G5L1-G4821-014f | 48.73 | 43.30 [42.89-43.31] | 6 | 0.889 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-M4B1G6-G850-500d | 49.62 | 44.07 [44.04-44.14] | 3 | 0.888 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-L1M3B1G6-G4389-4809 | 49.62 | 43.85 [43.76-43.89] | 3 | 0.884 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-M3L1B1G6-G4384-2541 | 49.62 | 43.46 [43.43-43.74] | 3 | 0.876 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-L1M2B2G6-G4389-1518 | 49.62 | 43.49 [43.49-43.50] | 3 | 0.876 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-M3B2G6-G1019-c487 | 49.62 | 43.90 [43.85-43.91] | 3 | 0.885 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-L2M2B1G6-G3928-2936 | 49.62 | 43.54 [43.46-43.79] | 3 | 0.877 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-L2M1B2G6-G3928-753b | 49.62 | 43.48 [43.30-43.50] | 3 | 0.876 |
| pixel | cifar-dense | VK | per-stage | btpm | tmax | SCH-M4G3L1B3-G4813-6fd2 | 50.38 | 48.60 [48.30-48.67] | 3 | 0.965 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2G4M1B4-G4112-7b5b | 48.84 | 42.98 [42.84-43.22] | 3 | 0.880 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2G4M2B3-G4112-be37 | 48.84 | 43.05 [43.01-43.15] | 3 | 0.882 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2G4B3M2-G4112-1815 | 48.84 | 43.08 [43.02-43.14] | 3 | 0.882 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-M4B1G5L1-G4899-d082 | 49.28 | 43.19 [42.90-43.43] | 6 | 0.876 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-M3B2G5L1-G4899-158c | 49.28 | 43.30 [42.89-43.31] | 6 | 0.879 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1G5M1B4-G4486-4dbf | 49.31 | 43.47 [43.39-43.48] | 3 | 0.882 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1G5M2B3-G4486-6354 | 49.31 | 43.34 [43.25-43.35] | 3 | 0.879 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L1G5B3M2-G4486-4a0f | 49.31 | 43.35 [43.17-43.48] | 3 | 0.879 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2B3M1G5-G4187-5a0c | 49.58 | 43.81 [43.54-45.14] | 3 | 0.883 |
| pixel | cifar-dense | VK | per-stage | isolated | tmax | SCH-L2B3G4M2-G4187-d33c | 49.58 | 39.59 [39.55-39.63] | 3 | 0.799 |
| pixel | cifar-sparse | VK | per-stage | btpm | tmax | SCH-B4G3L1M3-G44949-fd1f | 454.26 | 431.24 [419.81-451.62] | 6 | 0.949 |
| pixel | cifar-sparse | VK | per-stage | btpm | tmax | SCH-B3L1G3M4-G43834-da93 | 454.26 | 450.49 [440.63-470.22] | 6 | 0.992 |
| pixel | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L2B2G3M4-G37961-d258 | 454.26 | 463.69 [452.27-472.82] | 6 | 1.021 |
| pixel | cifar-sparse | VK | per-stage | btpm | tmax | SCH-L1B3G3M4-G40134-2c76 | 454.26 | 449.48 [433.81-466.99] | 6 | 0.989 |
| pixel | cifar-sparse | VK | per-stage | btpm | tmax | SCH-B4G3M4-G8252-e373 | 454.26 | 457.13 [437.55-463.21] | 3 | 1.006 |
| pixel | cifar-sparse | VK | per-stage | btpm | tmax | SCH-B4G3M3L1-G45193-d593 | 454.26 | 454.38 [450.26-463.40] | 3 | 1.000 |
| pixel | cifar-sparse | VK | per-stage | isolated | tmax | SCH-L2B2G3M4-G36089-38bf | 418.93 | 463.69 [452.27-472.82] | 6 | 1.107 |
| pixel | cifar-sparse | VK | per-stage | isolated | tmax | SCH-L1B3G3M4-G37804-574e | 418.93 | 449.48 [433.81-466.99] | 6 | 1.073 |
| pixel | cifar-sparse | VK | per-stage | isolated | tmax | SCH-B4G3L1M3-G41421-a525 | 418.93 | 431.24 [419.81-451.62] | 6 | 1.029 |
| pixel | cifar-sparse | VK | per-stage | isolated | tmax | SCH-B3L1G3M4-G40554-59e9 | 418.93 | 450.49 [440.63-470.22] | 6 | 1.075 |
| pixel | cifar-sparse | VK | per-stage | isolated | tmax | SCH-B4G3M3L1-G41766-e841 | 418.93 | 454.38 [450.26-463.40] | 3 | 1.085 |
| pixel | cifar-sparse | VK | per-stage | isolated | tmax | SCH-B4G3M4-G22502-6b93 | 418.93 | 457.13 [437.55-463.21] | 3 | 1.091 |
