# Measured pipeline speedups (BetterTogether)

Measured speedup = fastest single-PU whole-pipeline baseline (ms/task, summed from isolated profiling) / best measured pipeline makespan (max-chunk steady-state, min over the z3 btpm/isolated tmax candidates that were run).

| Device | App | Backend | Baseline | Best | Speedup |
|---|---|---|---|---|---|
| samsung | tree | VK | VK 9.47 | btpm 4.18 | 2.26x |
| samsung | cifar-dense | VK | VK 20.39 | isolated 10.98 | 1.86x |
| samsung | cifar-sparse | VK | VK 284.62 | isolated 183.72 | 1.55x |
| pixel | tree | VK | VK 17.76 | btpm 12.42 | 1.43x |
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
| samsung | tree | VK | per-stage | btpm | 10 | 0.085 | 3.88 [3.83-4.84] |
| samsung | tree | VK | per-stage | isolated | 10 | 1.403 | 7.11 [6.84-9.55] |
| samsung | cifar-dense | VK | per-stage | btpm | 10 | 0.060 | 11.26 [11.22-11.45] |
| samsung | cifar-dense | VK | per-stage | isolated | 10 | 0.049 | 10.99 [10.92-11.02] |
| samsung | cifar-sparse | VK | per-stage | btpm | 5 | 0.344 | 213.92 [211.10-215.12] |
| samsung | cifar-sparse | VK | per-stage | isolated | 5 | 0.162 | 183.20 [182.14-194.23] |
| pixel | tree | VK | per-stage | btpm | 10 | 0.615 | 11.54 [10.44-21.44] |
| pixel | tree | VK | per-stage | isolated | 10 | 2.818 | 26.55 [24.96-26.70] |
| pixel | cifar-dense | VK | per-stage | btpm | 10 | 0.116 | 43.19 [42.90-43.43] |
| pixel | cifar-dense | VK | per-stage | isolated | 10 | 0.120 | 39.59 [39.55-39.63] |
| pixel | cifar-sparse | VK | per-stage | btpm | 6 | 0.009 | 431.24 [419.81-451.62] |
| pixel | cifar-sparse | VK | per-stage | isolated | 6 | 0.080 | 431.24 [419.81-451.62] |

### Executed schedules

| Device | App | Backend | Cost model | Table | Mode | Schedule | Predicted ms | Measured ms [min-max] | Logs | Measured / predicted |
|---|---|---|---|---|---|---|---|---|---|---|
| samsung | tree | VK | per-stage | btpm | tmax | SCH-M1G6-G150-d41e | 3.80 | 5.46 [5.21-5.46] | 3 | 1.435 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-L1G6-G175-d87b | 3.80 | 3.88 [3.83-4.84] | 3 | 1.019 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-B1G6-G146-6d6e | 3.80 | 5.53 [4.05-5.82] | 3 | 1.456 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-G7-G000-360c | 3.96 | 11.16 [11.13-11.59] | 3 | 2.822 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-B1M1G5-G446-141d | 6.81 | 16.23 [7.13-19.25] | 3 | 2.386 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-B1M1L1G4-G605-5f5e | 6.81 | 7.15 [7.10-7.48] | 3 | 1.050 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-L1M1G5-G475-a3b6 | 6.81 | 7.15 [5.75-7.56] | 3 | 1.050 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-M1L1G5-G465-fc35 | 6.95 | 7.53 [7.38-10.46] | 3 | 1.084 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-B1L1G5-G461-fd27 | 6.95 | 7.55 [7.11-16.67] | 3 | 1.086 |
| samsung | tree | VK | per-stage | btpm | tmax | SCH-M1L2G4-G526-cefc | 7.15 | 7.69 [7.25-16.87] | 3 | 1.075 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2M2L2B1-G282-1e19 | 3.83 | 9.68 [9.42-9.70] | 3 | 2.527 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2M2B2L1-G252-2cc2 | 3.90 | 9.73 [6.85-9.86] | 3 | 2.493 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2L2B1M2-G303-5245 | 3.93 | 8.28 [7.54-8.78] | 3 | 2.106 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2L2B2M1-G256-e8cd | 3.93 | 8.92 [8.28-9.46] | 3 | 2.267 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2L2M2B1-G286-87e3 | 3.93 | 9.43 [8.81-9.49] | 3 | 2.397 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2M2B1L2-G310-a60a | 4.01 | 8.28 [8.25-10.28] | 3 | 2.067 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2M1B2L2-G335-881a | 4.03 | 9.72 [6.60-9.80] | 3 | 2.410 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-M1G2B2L2-G247-6bad | 4.03 | 12.86 [7.49-12.98] | 3 | 3.189 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-G2L1B2M2-G334-d4eb | 4.03 | 7.11 [6.84-9.55] | 3 | 1.763 |
| samsung | tree | VK | per-stage | isolated | tmax | SCH-L1G2B2M2-G247-9bdd | 4.03 | 9.88 [9.49-12.86] | 3 | 2.450 |
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
| pixel | tree | VK | per-stage | btpm | tmax | SCH-L1M1G5-G652-537e | 8.26 | 15.37 [15.36-16.04] | 3 | 1.860 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-B1M1G5-G622-fadf | 8.26 | 20.89 [16.99-24.37] | 3 | 2.529 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-M1L1G5-G598-6cd4 | 8.26 | 24.60 [10.35-24.62] | 3 | 2.977 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-B1L1G5-G622-af2d | 8.26 | 13.98 [10.08-18.65] | 3 | 1.691 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-L1B1G5-G652-dc10 | 8.26 | 19.68 [13.42-23.71] | 3 | 2.382 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-M1B1G5-G598-bf19 | 8.26 | 12.71 [10.28-20.03] | 3 | 1.538 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-L2G5-G160-4fd8 | 9.86 | 11.54 [10.44-21.44] | 3 | 1.170 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-M2G5-G191-ca75 | 10.17 | 13.00 [10.26-28.27] | 3 | 1.278 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-B2G5-G195-63a8 | 10.21 | 12.85 [10.36-28.48] | 3 | 1.258 |
| pixel | tree | VK | per-stage | btpm | tmax | SCH-G6M1-G019-7a83 | 10.23 | 15.08 [14.89-15.23] | 3 | 1.473 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1M1B3G2-G589-6026 | 7.04 | 27.19 [26.91-27.84] | 3 | 3.862 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1M1B2G3-G589-acdb | 7.04 | 27.08 [26.49-27.38] | 3 | 3.847 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-L1M1B4G1-G589-b0af | 7.04 | 27.08 [26.86-27.66] | 3 | 3.847 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-B1M1L3G2-G590-3301 | 7.04 | 26.78 [26.69-27.25] | 3 | 3.804 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-B1M1L2G3-G590-40f7 | 7.04 | 26.93 [26.76-27.77] | 3 | 3.826 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-B1M1L4G1-G590-c1af | 7.04 | 26.55 [24.96-26.70] | 3 | 3.771 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-M1L1B4G1-G593-25ba | 7.08 | 26.99 [26.23-27.50] | 3 | 3.810 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-M1L1B3G2-G593-fac9 | 7.08 | 26.97 [25.75-28.07] | 3 | 3.807 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-M1L1B2G3-G593-e636 | 7.08 | 26.69 [26.50-27.64] | 3 | 3.768 |
| pixel | tree | VK | per-stage | isolated | tmax | SCH-B1L1M4G1-G594-3839 | 7.08 | 27.56 [26.84-27.70] | 3 | 3.890 |
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
