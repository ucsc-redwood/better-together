# Contributing & branch model

BetterTogether uses a two-branch model with a hardware-in-the-loop gate.

## Branches

| Branch | Role |
|--------|------|
| **`main`** | **Stable, verified, public.** Every change here has passed the full test matrix — the OMP-as-oracle differential gate **and** the real-hardware fleet (CUDA on Jetson, Vulkan on the Rocky iGPU box + Android phones). This is the branch external researchers should clone, build, and cite. |
| **`dev`** | Active development / integration. All work lands here first. |

`main` only ever moves by **promotion from `dev`** (via pull request) — never a direct push.

## For external researchers

Clone `main`. It is the default branch and is kept stable: it builds and its
tests pass on every supported backend/target. Build instructions are in
[`docs/instruction-for-ai/02-building.md`](docs/instruction-for-ai/02-building.md);
the quickstart (CPU/OpenMP) is in the root `README.md` / `CLAUDE.md`.

## For contributors

1. Branch off `dev` and open your pull request **against `dev`** (not `main`).
2. CI (GitHub Actions, `.github/workflows/ci.yml`) must be green on the PR. The
   hosted gate runs:
   - `format` — `just fmt-check` (clang-format / ruff / gersemi / shfmt / prettier)
   - `build + ctest -L omp` — the OpenMP differential oracle
   - `optimizer pytest`
   Run them locally first with `just fmt-check` and `ctest --test-dir build/pc -L omp`.
3. The CUDA / Vulkan / Android tests need the physical fleet and run out-of-band
   via `just test` (and, once wired, a scheduled self-hosted CI job) — they are
   not part of the per-PR hosted gate.

## Promotion: `dev` → `main`

A maintainer promotes `dev` to `main` only after **both** gates are green:
1. Tier-0 hosted CI on `dev`.
2. The full fleet matrix (`just test` across Jetson / Rocky iGPU / phones).

Promotion is a pull request `dev` → `main`; `main` keeps a linear history.
