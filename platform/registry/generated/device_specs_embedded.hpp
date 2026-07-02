#pragma once
// AUTO-GENERATED from devices/*.json by scripts/embed_device_specs.py -- DO NOT EDIT.
// Regenerate after changing any devices/*.json.

#include <string_view>
#include <vector>

namespace bt::device_specs {

// One raw-JSON device spec per registered device (schema: schemas/device-spec.schema.json).
inline const std::vector<std::string_view> kEmbedded = {
    // 3A021JEHN02756.json
    R"DEVSPEC({
  "id": "3A021JEHN02756",
  "description": "Google Pixel 7a (Tensor G2): 4 little + 2 medium + 2 big.",
  "cores": [
    { "id": 0, "type": "little", "pinnable": true },
    { "id": 1, "type": "little", "pinnable": true },
    { "id": 2, "type": "little", "pinnable": true },
    { "id": 3, "type": "little", "pinnable": true },
    { "id": 4, "type": "medium", "pinnable": true },
    { "id": 5, "type": "medium", "pinnable": true },
    { "id": 6, "type": "big", "pinnable": true },
    { "id": 7, "type": "big", "pinnable": true }
  ],
  "gpu": { "backend": "vulkan", "name": "Mali-G710", "subgroup_size": 16 }
})DEVSPEC",
    // R5CY21Y3VEV.json
    R"DEVSPEC({
  "id": "R5CY21Y3VEV",
  "description": "Samsung Galaxy (device-reported model SM-S926B): 4 little + 3 medium + 2 big + 1 super.",
  "cores": [
    { "id": 0, "type": "little", "pinnable": true },
    { "id": 1, "type": "little", "pinnable": true },
    { "id": 2, "type": "little", "pinnable": true },
    { "id": 3, "type": "little", "pinnable": true },
    { "id": 4, "type": "medium", "pinnable": true },
    { "id": 5, "type": "medium", "pinnable": true },
    { "id": 6, "type": "medium", "pinnable": true },
    { "id": 7, "type": "big", "pinnable": true },
    { "id": 8, "type": "big", "pinnable": true },
    { "id": 9, "type": "super", "pinnable": true }
  ],
  "gpu": {
    "backend": "vulkan",
    "name": "Samsung Xclipse (Galaxy S24, SM-S926B)",
    "subgroup_size": 32
  }
})DEVSPEC",
    // duck-naughty.json
    R"DEVSPEC({
  "id": "duck-naughty",
  "description": "NVIDIA Jetson Orin Nano Devkit Super, host duck-naughty (JetPack 7.2 / L4T R39.2.0, CUDA 13.2, Ubuntu 24.04, MAXN_SUPER): 6 cores, single tier. Same hardware/software as duck-stable; separate id so per-unit profiling stores never mix. Replaces the retired JetPack-6 device id 'jetson' (reflashed 2026-07-01).",
  "cores": [
    { "id": 0, "type": "little", "pinnable": true },
    { "id": 1, "type": "little", "pinnable": true },
    { "id": 2, "type": "little", "pinnable": true },
    { "id": 3, "type": "little", "pinnable": true },
    { "id": 4, "type": "little", "pinnable": true },
    { "id": 5, "type": "little", "pinnable": true }
  ],
  "gpu": {
    "backend": "cuda",
    "name": "NVIDIA Tegra Orin (nvgpu)",
    "subgroup_size": 32
  }
})DEVSPEC",
    // duck-stable.json
    R"DEVSPEC({
  "id": "duck-stable",
  "description": "NVIDIA Jetson Orin Nano Devkit Super, host duck-stable (JetPack 7.2 / L4T R39.2.0, CUDA 13.2, Ubuntu 24.04, MAXN_SUPER): 6 cores, single tier. Replaces the retired JetPack-6 device id 'jetson' (reflashed 2026-07-01).",
  "cores": [
    { "id": 0, "type": "little", "pinnable": true },
    { "id": 1, "type": "little", "pinnable": true },
    { "id": 2, "type": "little", "pinnable": true },
    { "id": 3, "type": "little", "pinnable": true },
    { "id": 4, "type": "little", "pinnable": true },
    { "id": 5, "type": "little", "pinnable": true }
  ],
  "gpu": {
    "backend": "cuda",
    "name": "NVIDIA Tegra Orin (nvgpu)",
    "subgroup_size": 32
  }
})DEVSPEC",
    // minipc.json
    R"DEVSPEC({
  "id": "minipc",
  "description": "16-core x86_64 mini PC, single tier.",
  "cores": [
    { "id": 0, "type": "big", "pinnable": true },
    { "id": 1, "type": "big", "pinnable": true },
    { "id": 2, "type": "big", "pinnable": true },
    { "id": 3, "type": "big", "pinnable": true },
    { "id": 4, "type": "big", "pinnable": true },
    { "id": 5, "type": "big", "pinnable": true },
    { "id": 6, "type": "big", "pinnable": true },
    { "id": 7, "type": "big", "pinnable": true },
    { "id": 8, "type": "big", "pinnable": true },
    { "id": 9, "type": "big", "pinnable": true },
    { "id": 10, "type": "big", "pinnable": true },
    { "id": 11, "type": "big", "pinnable": true },
    { "id": 12, "type": "big", "pinnable": true },
    { "id": 13, "type": "big", "pinnable": true },
    { "id": 14, "type": "big", "pinnable": true },
    { "id": 15, "type": "big", "pinnable": true }
  ],
  "gpu": { "backend": "vulkan", "name": "AMD Radeon 780M", "subgroup_size": 64 }
})DEVSPEC",
    // pc.json
    R"DEVSPEC({
  "id": "pc",
  "description": "Generic x86_64 desktop: 8 big cores + 16 little cores.",
  "cores": [
    { "id": 0, "type": "big", "pinnable": true },
    { "id": 1, "type": "big", "pinnable": true },
    { "id": 2, "type": "big", "pinnable": true },
    { "id": 3, "type": "big", "pinnable": true },
    { "id": 4, "type": "big", "pinnable": true },
    { "id": 5, "type": "big", "pinnable": true },
    { "id": 6, "type": "big", "pinnable": true },
    { "id": 7, "type": "big", "pinnable": true },
    { "id": 8, "type": "little", "pinnable": true },
    { "id": 9, "type": "little", "pinnable": true },
    { "id": 10, "type": "little", "pinnable": true },
    { "id": 11, "type": "little", "pinnable": true },
    { "id": 12, "type": "little", "pinnable": true },
    { "id": 13, "type": "little", "pinnable": true },
    { "id": 14, "type": "little", "pinnable": true },
    { "id": 15, "type": "little", "pinnable": true },
    { "id": 16, "type": "little", "pinnable": true },
    { "id": 17, "type": "little", "pinnable": true },
    { "id": 18, "type": "little", "pinnable": true },
    { "id": 19, "type": "little", "pinnable": true },
    { "id": 20, "type": "little", "pinnable": true },
    { "id": 21, "type": "little", "pinnable": true },
    { "id": 22, "type": "little", "pinnable": true },
    { "id": 23, "type": "little", "pinnable": true }
  ],
  "gpu": {
    "backend": "vulkan",
    "name": "Intel UHD Graphics 770",
    "subgroup_size": 32
  }
})DEVSPEC",
};

}  // namespace bt::device_specs

