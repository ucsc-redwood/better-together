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

### Collecting benchmarks

Baseline

```bash
xmake r bm-cifar-dense-omp --benchmark_repetitions=5 --benchmark_filter=Baseline --benchmark_format=json > ./data/raw_benchmarks/cifar-dense-baseline.json
```









