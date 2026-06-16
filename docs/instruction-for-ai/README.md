# Instructions for AI agents

Canonical, load-bearing knowledge an AI agent needs to work on BetterTogether
**correctly**. Read these in order; each is meant to be actionable, not narrative.

> **Behavioral coding rules** (think before coding, simplicity, surgical changes,
> goal-driven execution) are **not here** — they live in the repo-root
> [`CLAUDE.md`](../../CLAUDE.md) and are loaded automatically.

| # | Doc | Read it to learn |
|---|---|---|
| 00 | [`00-project-goal.md`](00-project-goal.md) | What the project is, the 3-tools-through-files model, the apps, what "done" means |
| 01 | [`01-hardware.md`](01-hardware.md) | Every test target: specs, role, subgroup size, **and how to ssh/adb in and deploy** |
| 02 | [`02-building.md`](02-building.md) | How to build: CMake presets `pc`/`jetson`/`vulkan`/`android`, cross-compile recipes, overrides |
| 03 | [`03-unit-testing.md`](03-unit-testing.md) | How to run & write tests: the OMP-as-oracle differential method, per-backend commands, CI labels |
| 04 | [`04-alexnet-cifar-spec.md`](04-alexnet-cifar-spec.md) | The canonical AlexNet/CIFAR model — exact shapes to match when writing kernels |
| 05 | [`05-profiling.md`](05-profiling.md) | How to profile **runtime overhead** (CLI / agent-driven): which tool per backend & target, exact commands, what the output means |

For **status, audits, decision logs, and roadmaps** (the "why" and "where we are"),
see [`../reports-for-human/`](../reports-for-human/README.md).
