# Test-run logs

Verbatim stdout of the `scripts/run-on-*.sh` deploy helpers, one log per fleet
sweep; each helper exits non-zero on any non-skip failure.

> **Fresh start 2026-07-01:** every log collected before this date — the whole
> 2026-06-17 full-matrix sweep, including the retired JetPack-6 `jetson` device —
> is archived in [`archive-pre-2026-07/`](archive-pre-2026-07/) (see its README for
> what each log covered and why the Jetson numbers are not comparable). New sweeps
> on the current fleet (`duck-stable`, `duck-naughty`, `minipc`, `pixel`, `samsung`)
> land directly in this folder.

| log | host | backend | what ran |
|---|---|---|---|
| `duck-stable-cuda.log` | duck-stable (JetPack 7.2) | CUDA | per-stage differential ×3 apps (tree 7 / dense 10 / sparse 10) — all green, 2026-07-02, `bt-cross:6.1` binaries |
| `duck-naughty-cuda.log` | duck-naughty (JetPack 7.2) | CUDA | same — all green, 2026-07-02 |
