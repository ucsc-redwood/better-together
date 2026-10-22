8-token decode trials of plan_corun_*_et_overlap.txt as first run, WITHOUT --warmup (2026-10-21 10:31-11:14 PDT).
Not used: a fresh llama_main's first decode steps dominate an 8-token window (Vulkan 1B solo: Pixel 4.85 tok/s,
Samsung 8.77, against 14.4 and 37 in a 512-token window). Rerun with --warmup in corun-et/. The driver was
stopped by PID after these trials; the trial in flight on each phone was never pulled.
