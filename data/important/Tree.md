# Tree

## Google

### Run 1

### Run 2

## OnePlus

### Run 1

Normal Benchmark Results Table (ms per task):
| Stage | Little Core | Medium Core | Big Core | Vulkan   |
| ----- | ----------- | ----------- | -------- | -------- |
| 1     | 8.8938      | 3.1053      | 3.8798   | 1.8556   |
| 2     | 5.3069      | 2.8764      | 6.3544   | 132.7500 |
| 3     | 2.8254      | 1.2133      | 2.2344   | 0.7493   |
| 4     | 27.1892     | 23.2791     | 16.3065  | 4.8599   |
| 5     | 2.3302      | 0.7851      | 0.9737   | 1.3481   |
| 6     | 1.1532      | 0.9018      | 1.7668   | 0.6349   |
| 7     | 13.1429     | 6.9722      | 7.4627   | 4.2723   |

Normal Benchmark - Sum of stages 1-7:
Little Core: 60.8415 ms
Medium Core: 39.1332 ms
Big Core: 38.9783 ms
Vulkan: 146.4701 ms

Fully Benchmark Results Table (ms per task):
| Stage | Little Core | Medium Core | Big Core | Vulkan   |
| ----- | ----------- | ----------- | -------- | -------- |
| 1     | 10.6915     | 3.2276      | 2.0894   | 1.0349   |
| 2     | 6.2308      | 2.8383      | 4.9670   | 131.5000 |
| 3     | 2.8305      | 2.0809      | 1.2322   | 1.6480   |
| 4     | 32.6774     | 23.8837     | 15.8906  | 2.8022   |
| 5     | 4.2385      | 1.1401      | 0.8218   | 0.7175   |
| 6     | 1.4178      | 1.9513      | 1.3656   | 1.1990   |
| 7     | 23.4419     | 7.8295      | 5.4728   | 3.8779   |

Fully Benchmark - Sum of stages 1-7:
Little Core: 81.5284 ms
Medium Core: 42.9513 ms
Big Core: 31.8395 ms
Vulkan: 142.7795 ms

Performance Comparison (Fully vs Normal):
| Processor   | Normal (ms) | Fully (ms) | Ratio |
| ----------- | ----------- | ---------- | ----- |
| Little Core | 60.84       | 81.53      | 1.34x |
| Medium Core | 39.13       | 42.95      | 1.10x |
| Big Core    | 38.98       | 31.84      | 0.82x |
| Vulkan      | 146.47      | 142.78     | 0.97x |

```
### PYTHON_DATA_START ###
# NORMAL_BENCHMARK_DATA
stage,little,medium,big,vulkan
1,8.89,3.11,3.88,1.86
2,5.31,2.88,6.35,132.75
3,2.83,1.21,2.23,0.75
4,27.19,23.28,16.31,4.86
5,2.33,0.79,0.97,1.35
6,1.15,0.90,1.77,0.63
7,13.14,6.97,7.46,4.27
# FULLY_BENCHMARK_DATA
stage,little,medium,big,vulkan
1,10.69,3.23,2.09,1.03
2,6.23,2.84,4.97,131.50
3,2.83,2.08,1.23,1.65
4,32.68,23.88,15.89,2.80
5,4.24,1.14,0.82,0.72
6,1.42,1.95,1.37,1.20
7,23.44,7.83,5.47,3.88
# RAW_NORMAL_TABLE_DATA
Stage 1: 8.89 3.11 3.88 1.86
Stage 2: 5.31 2.88 6.35 132.75
Stage 3: 2.83 1.21 2.23 0.75
Stage 4: 27.19 23.28 16.31 4.86
Stage 5: 2.33 0.79 0.97 1.35
Stage 6: 1.15 0.90 1.77 0.63
Stage 7: 13.14 6.97 7.46 4.27
# RAW_FULLY_TABLE_DATA
Stage 1: 10.69 3.23 2.09 1.03
Stage 2: 6.23 2.84 4.97 131.50
Stage 3: 2.83 2.08 1.23 1.65
Stage 4: 32.68 23.88 15.89 2.80
Stage 5: 4.24 1.14 0.82 0.72
Stage 6: 1.42 1.95 1.37 1.20
Stage 7: 23.44 7.83 5.47 3.88
### PYTHON_DATA_END ###
```

### Run 2

## Jetson

### Run 1

### Run 2

## Jetson Lowpower

### Run 1

### Run 2


