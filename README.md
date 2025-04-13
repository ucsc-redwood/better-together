# Redwood All-in-One

```
===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 C Header               32         7148         7148            0            0
 C++                    48        31439        25738         1025         4676
 C++ Header             54        10042         6693         1025         2324
 GLSL                   32         2016         1426          196          394
 JSON                  339       134285       134285            0            0
 Lua                    18         1201          619          369          213
 Makefile                1           38           20            9            9
 Python                 15         2567         2024          151          392
 Plain Text           1202      6668012            0      6668012            0
-------------------------------------------------------------------------------
 Markdown                2           46            0           20           26
 |- BASH                 2            8            8            0            0
 (Total)                             54            8           20           26
===============================================================================
 Total                1743      6856794       177953      6670807         8034
===============================================================================
```

## Overview

This repository contains the source code for the Redwood project, which is a collection of benchmarks and tools for evaluating the performance of various programming models. And creating pipeline execution for applications on heterogeneous systems. 


## Developer Notes

### Mini Benchmark

```bash
    xmake r bm-mini-cifar-sparse-vk -l off | tee tmp.txt
```

you will get something like

```
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
VK/CifarSparse/Mini/Baseline           37.0 ms         3.83 ms          100
VK/CifarSparse/Mini/Stage/1            6.33 ms        0.383 ms         1000
VK/CifarSparse/Mini/Stage/2            9.54 ms        0.395 ms         1382
VK/CifarSparse/Mini/Stage/3            5.33 ms        0.451 ms         1034
...
OMP/CifarSparse/Mini/Baseline/4        18.2 ms         18.1 ms           39
OMP/CifarSparse/Mini/Baseline/5        69.8 ms         69.3 ms           10
OMP/CifarSparse/Mini/Baseline/6        57.1 ms         56.7 ms           12
OMP/CifarSparse/Mini/Baseline/7        49.4 ms         49.1 ms           14
OMP/CifarSparse/Mini/Baseline/8        44.7 ms         44.1 ms           16
...
OMP/CifarSparse/Mini/Stage/1/2/2       4.38 ms         4.35 ms          165 Big, 2 cores
OMP/CifarSparse/Mini/Stage/1/2/3 SKIPPED: ''
OMP/CifarSparse/Mini/Stage/1/2/4 SKIPPED: ''
...
OMP/CifarSparse/Mini/Stage/2/0/3       30.3 ms         29.9 ms           23 Little, 3 cores
...
```

summary

VK

```bash
awk '/^VK\/CifarSparse\/Mini\/Stage/ {
  split($1, a, "/"); 
  printf("VK Stage %s: %s %s\n", a[5], $2, $3)
}' tmp.txt
```
omp 

```bash
grep -E '^OMP/CifarSparse/Mini/Stage' tmp.txt | grep -iv 'skipped' | \
awk '{ 
  split($1, a, "/"); 
  printf("OMP Stage %s Core %s Threads %s: %s %s\n", a[5], a[6], a[7], $2, $3)
}'
```

### Real Benchmark

#### fully occupied

```bash
xmake r bm-real-cifar-sparse-vk --stage 1 -l off | tee non_full_stage.txt
xmake r bm-real-cifar-sparse-vk --stage 2 -l off | tee non_full_stage.txt
xmake r bm-real-cifar-sparse-vk --stage 3 -l off | tee non_full_stage.txt
xmake r bm-real-cifar-sparse-vk --stage 4 -l off | tee non_full_stage.txt
```

```
cat non_full_stage.txt | rg PROCESSOR=
```

you will get

```
PROCESSOR=Little|COUNT=100|TOTAL=2317.8|AVG=23.178|GEOMEAN=22.4267|MEDIAN=21.0945|MIN=18.8645|MAX=63.3621|STDDEV=7.47547|CV=0.322524|P90=26.2698|P95=44.926|P99=63.3621
PROCESSOR=Medium|COUNT=100|TOTAL=615.677|AVG=6.15677|GEOMEAN=5.96631|MEDIAN=5.51851|MIN=5.45117|MAX=15.5101|STDDEV=1.9455|CV=0.315993|P90=7.03516|P95=12.2372|P99=15.5101
PROCESSOR=Big|COUNT=100|TOTAL=538.806|AVG=5.38806|GEOMEAN=5.16501|MEDIAN=4.97194|MIN=4.61068|MAX=31.9001|STDDEV=2.84837|CV=0.528646|P90=5.38371|P95=5.46724|P99=31.9001
PROCESSOR=GPU|COUNT=100|TOTAL=463.311|AVG=4.63311|GEOMEAN=4.44775|MEDIAN=3.95111|MIN=3.41825|MAX=14.7775|STDDEV=1.60296|CV=0.34598|P90=6.35421|P95=7.96151|P99=14.7775
```

#### non-fully occupied

#### comparing


```bash
just compare-all > full_output.raw
grep 'PROCESSOR=' full_output.raw > full_output.txt
py analy.py
```


```
Optimal chunk time: 577227/50000 = 11.54454 ms
Stage 0: core type Big with time 7.89115
Stage 1: core type GPU with time 4.95056
Stage 2: core type GPU with time 4.36582
Stage 3: core type Medium with time 6.49812
Stage 4: core type Medium with time 1.66993
Stage 5: core type Medium with time 1.69134
Stage 6: core type Medium with time 1.68515
Stage 7: core type Little with time 6.29989
Stage 8: core type Little with time 0.182615

Math model summary:
chunk 0 (Big): 7.89115 ms
chunk 1 (GPU): 9.31638 ms
chunk 2 (Medium): 11.54454 ms
chunk 3 (Little): 6.48251 ms

Real measurements:
  All Tasks Average: 141961.49 cycles (5.776428 ms)
  All Tasks Average: 166560.19 cycles (6.777352 ms)
  All Tasks Average: 275456.24 cycles (11.208343 ms)
  All Tasks Average: 157681.26 cycles (6.416067 ms)
  Widest Task: 11.208343 ms

diff of optimal and widest real measurement:
  Raw: 11.208343 - 11.54454 = -0.336197 ms
  Percentage: -0.336197 / 11.54454 = -2.91%

```

