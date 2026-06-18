"""The data-layout single source of truth.

`<root>/<device>/<app>/<backend>/...` was a stringly-typed schema rebuilt by hand in
02_gen_schedule_merged.py and 03_run_schedule.py — change the layout and they silently
diverge (e.g. 02 writes data/schedules/ while 03 defaulted to data/schedules_btpm/).
A `Case` (device, app, backend) owns the layout once.

Backend naming: the canonical-JSONL profiling store keys the GPU by its long name
(`cuda`/`vulkan`); everything downstream (schedules, z3 `--backend`, the C++ consumer)
uses the short name (`cu`/`vk`). `Case.backend` is the short name; the profiling helpers
translate.
"""
import os
from dataclasses import dataclass

# short (consumer) <-> long (profiling-store) GPU backend names.
_LONG = {"cu": "cuda", "vk": "vulkan"}
_SHORT = {v: k for k, v in _LONG.items()}

# z3 CLI table_type -> profiling-store scenario dir. The schedule filenames keep the
# table_type token ("btpm"); the JSONL store keys it as the scenario ("interference").
_TABLE_SCENARIO = {"isolated": "isolated", "btpm": "interference"}


def to_short_backend(backend: str) -> str:
    """Accept either naming, return the short (cu/vk) form."""
    return _SHORT.get(backend, backend)


def table_to_scenario(table_type: str) -> str:
    """Map a z3 --table_type ('isolated'|'btpm') to its profiling-store scenario dir."""
    return _TABLE_SCENARIO[table_type]


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
