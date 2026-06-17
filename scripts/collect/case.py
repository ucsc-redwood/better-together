"""The data-layout single source of truth.

`<root>/<device>/<app>/<backend>/...` was a stringly-typed schema rebuilt by hand in
02_gen_schedule_merged.py, 03_run_schedule.py, and export_btpm_csv.py — change the
layout and they silently diverge (e.g. 02 writes data/schedules/ while 03 defaulted
to data/schedules_btpm/). A `Case` (device, app, backend) owns the layout once.

Backend naming: the canonical-JSONL profiling store keys the GPU by its long name
(`cuda`/`vulkan`); everything downstream (CSV, schedules, z3 `--backend`, the C++
consumer) uses the short name (`cu`/`vk`). `Case.backend` is the short name; the
profiling helpers translate.
"""
import os
from dataclasses import dataclass

# short (consumer) <-> long (profiling-store) GPU backend names.
_LONG = {"cu": "cuda", "vk": "vulkan"}
_SHORT = {v: k for k, v in _LONG.items()}

# scenario -> the legacy wide-CSV filename the z3 consumer reads.
_CSV_NAME = {"isolated": "isolated.csv", "interference": "btpm.csv"}


def to_short_backend(backend: str) -> str:
    """Accept either naming, return the short (cu/vk) form."""
    return _SHORT.get(backend, backend)


@dataclass(frozen=True)
class Case:
    """One (device, app, backend) profiling/scheduling cell. backend is short: cu|vk."""

    device: str
    app: str
    backend: str  # "cu" | "vk"

    @property
    def backend_long(self) -> str:
        return _LONG.get(self.backend, self.backend)

    # --- profiling JSONL store (long backend name, per-scenario dir) ---
    def profiling_dir(self, root: str, scenario: str) -> str:
        return os.path.join(root, self.device, self.app, self.backend_long, scenario)

    def profiling_glob(self, root: str, scenario: str) -> str:
        return os.path.join(self.profiling_dir(root, scenario), "run-*.jsonl")

    # --- exported wide CSV the SMT solver reads (short backend name) ---
    def csv_dir(self, root: str) -> str:
        return os.path.join(root, self.device, self.app, self.backend)

    def csv_path(self, root: str, scenario_or_table: str) -> str:
        """scenario_or_table: 'isolated'|'interference' (-> isolated/btpm.csv) or a
        table_type 'isolated'|'btpm' (-> <name>.csv)."""
        name = _CSV_NAME.get(scenario_or_table, f"{scenario_or_table}.csv")
        return os.path.join(self.csv_dir(root), name)

    # --- z3 schedule JSON (short backend name) ---
    def schedule_path(self, root: str, table_type: str, minimize_mode: str) -> str:
        return os.path.join(
            root, self.device, self.app, self.backend,
            f"schedules_{table_type}_{minimize_mode}.json",
        )

    @classmethod
    def from_profiling_relpath(cls, rel_parts) -> "Case":
        """rel_parts = (device, app, backend_long, scenario, ...) as split from a
        profiling glob match. Returns the Case (backend normalised to short)."""
        device, app, backend = rel_parts[0], rel_parts[1], rel_parts[2]
        return cls(device, app, to_short_backend(backend))
