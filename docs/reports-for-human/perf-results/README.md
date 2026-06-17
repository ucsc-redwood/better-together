# Perf results

Concrete, dated performance measurements on the test fleet — the *decided*
results, not the how-to. For **how** to profile (tools per backend/target, exact
commands), see [`../../instruction-for-ai/05-profiling.md`](../../instruction-for-ai/05-profiling.md).

Each file is a single measurement campaign: method (reproducible commands),
results, findings, insights, and suggestions.

| Doc | Target | Tool | What it found |
|---|---|---|---|
| [`radeon-780m-rga-static.md`](radeon-780m-rga-static.md) | Radeon 780M (gfx1103, RADV) | RGA 2.14 static | No kernel spills or is register/LDS-bound → bottleneck is runtime overhead, not shader codegen |
