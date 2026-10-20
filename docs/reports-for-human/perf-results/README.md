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
