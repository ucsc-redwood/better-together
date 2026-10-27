# Perf results

Concrete, dated performance measurements on the test fleet — the *decided*
results, not the how-to. For **how** to profile (tools per backend/target, exact
commands), see [`../../instruction-for-ai/05-profiling.md`](../../instruction-for-ai/05-profiling.md).

Each file is a single measurement campaign: method (reproducible commands),
results, findings, insights, and suggestions.

| Doc | Target | Tool | What it found |
|---|---|---|---|
| [`radeon-780m-rga-static.md`](radeon-780m-rga-static.md) | Radeon 780M (gfx1103, RADV) | RGA 2.14 static | No kernel spills or is register/LDS-bound → bottleneck is runtime overhead, not shader codegen |
| [`2026-10-20-cifar-sparse-conv.md`](2026-10-20-cifar-sparse-conv.md) | Samsung Xclipse 940, Pixel 7a (Mali-G710) | `bm-prof-cifar-sparse-vk` A/B, ACO (780M) | One output channel per workgroup row: Samsung conv2-conv5 23-48 % faster (stage sum -30 %), Pixel unchanged; the CSR loads were already scalar, the gain is 8 input loads in flight |
| [`2026-10-20-cifar-dense-conv.md`](2026-10-20-cifar-dense-conv.md) | Samsung Xclipse 940, Pixel 7a (Mali-G710) | `bm-prof-cifar-dense-vk` A/B with GPU clock, ACO (780M) | Channel block per workgroup and a register tile per conv: Samsung conv2-conv5 -16 % (cycles -27 %), Pixel unchanged; the Samsung's GPU clock settles lower for kernels that save cycles |
| [`2026-10-20-cifar-fc.md`](2026-10-20-cifar-fc.md) | Samsung Xclipse 940, Pixel 7a (Mali-G710) | `bm-prof-cifar-{dense,sparse}-vk` A/B with GPU clock, malioc | Clustered FC reduction: cifar-sparse fc1+fc2 -31 % Pixel, -7 % Samsung; opt-in INT8 weights 0.45-0.66 of FP32 at unchanged accuracy; once-per-batch layouts and FP16 rejected; Pixel cifar-dense FP32 fc slower above 471 MHz |
| [`../2026-10-20-end-to-end-refresh.md`](../2026-10-20-end-to-end-refresh.md) | Pixel 7a, Samsung Galaxy | `bm-prof` store, z3 per-stage schedules, `bm-gen-logs` executions, `bm-baseline` | The October campaign end to end, every (phone, app, table) cell before (2026-10-17) vs now: no best period worse, 7 of 12 better beyond noise (tree isolated -68 % / -82 %, Samsung cifar-sparse -35 %); measured speedup over the best single PU 0.95x (Samsung cifar-dense isolated) to 2.5x; the profile-summed Step 5 baseline is off by up to 1.6x; the largest error left is the GPU-chunk price (GPU clock) |
