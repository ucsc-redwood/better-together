# Device Specifications

Each `*.json` file in this directory declares one target SoC's CPU core topology
and pinning affinity, validated against
[`../schemas/device-spec.schema.json`](../schemas/device-spec.schema.json).

These files are the **single source of truth** for device configuration. They
were extracted from the hardcoded `DeviceRegistry` in
[`../builtin-apps/conf.cpp`](../builtin-apps/conf.cpp); the C++ runtime will be
migrated to load them at runtime (see
[`../docs/reports-for-human/rearchitecture.md`](../docs/reports-for-human/rearchitecture.md), Phase 2), after which
the hardcoded table is deleted.

> Note: where the top-level `README.md` and `conf.cpp` disagreed about a
> device's topology, **`conf.cpp` (the code that produced the published
> results) is authoritative** and is what these files reflect.

## Adding a new device

1. Find the device's id (for Android: `adb devices`).
2. Determine its core tiers (e.g. with the `bm-check-core-types` utility).
3. Create `devices/<id>.json`:

   ```json
   {
     "id": "<id>",
     "description": "Model / notes",
     "cores": [
       { "id": 0, "type": "little", "pinnable": true },
       { "id": 1, "type": "big",    "pinnable": true }
     ]
   }
   ```

   `type` is one of `little`, `medium`, `big`, `super`. `pinnable` marks whether
   the scheduler may pin a worker thread to that core.
4. Add the device's golden topology to `GOLDEN` in
   [`../scripts/validate_devices.py`](../scripts/validate_devices.py) so it is
   locked against future regressions.
5. Validate:

   ```bash
   uv run scripts/validate_devices.py
   ```
