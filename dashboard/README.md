# BetterTogether Dashboard

A static, offline, self-contained dashboard for exploring the project's
**devices**, **applications/kernels**, and **profiling matrix**. No server, no
build step — a Python generator inlines everything into one `bundle.js` and the
site opens by double-clicking.

## Generate & open

```bash
# uses the repo's uv environment (jsonschema + the reused loaders)
uv run python dashboard/generate.py
# then open the generated site (no server needed):
xdg-open dashboard/site/index.html        # or just double-click it
```

Re-run the generator whenever `devices/*.json`, the kernels, or
`data/profiling/**` change. `dashboard/site/` is gitignored (it's a build
artifact derived from the gitignored `data/`).

## What it shows

1. **Devices** — every `devices/*.json`: heterogeneous CPU tiers
   (little/medium/big/super, per-core pinnable flags), GPU (backend + name +
   subgroup size), supported backends. Greyed = no profiling data yet.
2. **Applications** — `tree` (7), `cifar-dense` (9), `cifar-sparse` (9): the
   stage→backend support matrix, each stage's kernel source for OMP/CUDA/Vulkan
   (syntax-highlighted, real source sliced from the repo), and the per-stage
   AppData layout (buffers / tensor shapes).
3. **Profiling** — the `device × app × backend × scenario × stage × PU` matrix
   (p50/p95/p99/mean/cv/min/max): an interactive detail table, a stage×PU
   heatmap, a per-stage cross-device/PU bar chart, and an interference/isolated
   ratio heatmap. Empty cells are marked; high-CV cells (excluded by the
   solver's `cv>0.1` gate) are flagged but their raw values still shown.

## How it's wired

- `generate.py` — reads `vocab.json`, `devices/*.json`, the kernel manifests,
  the kernel source, and the JSONL profiling store; reuses
  `optimizer/smt/profiling_loader.py` (record read + schema validate) and
  `optimizer/orchestrate/case.py` (path layout). Emits `site/bundle.js`.
- `manifest/kernels.<app>.json` — curated stage → {op, description,
  per-backend source locators, AppData rows}. Source locators are sliced by the
  generator: `{path}`=whole file, `{path,symbol}`=that function, `{path,lines}`=range.
- `manifest/device_freqs.json` — optional CPU/GPU clock enrichment for the four
  SoCs in the IISWC'25 paper (devices/*.json carries tiers, not clocks).
- `static/` — the SPA (`index.html`, `css/app.css`, `js/app.js`) + vendored
  `highlight.js` (cpp+glsl) and `ECharts`. Copied verbatim into `site/`.
