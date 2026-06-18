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
    // 9b034f1b.json
    R"DEVSPEC({
  "id": "9b034f1b",
  "description": "OnePlus 11 (CPH2451, Snapdragon 8 Gen 2): 3 little + 2 medium + 3 big; big cores are not pinnable.",
  "cores": [
    { "id": 0, "type": "little", "pinnable": true },
    { "id": 1, "type": "little", "pinnable": true },
    { "id": 2, "type": "little", "pinnable": true },
    { "id": 3, "type": "medium", "pinnable": true },
    { "id": 4, "type": "medium", "pinnable": true },
    { "id": 5, "type": "big", "pinnable": false },
    { "id": 6, "type": "big", "pinnable": false },
    { "id": 7, "type": "big", "pinnable": false }
  ],
  "gpu": { "backend": "vulkan", "name": "Adreno 740", "subgroup_size": 64 }
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
  "gpu": { "backend": "vulkan", "name": "Samsung Xclipse (Galaxy S24, SM-S926B)", "subgroup_size": 32 }
})DEVSPEC",
    // R9TR30814KJ.json
    R"DEVSPEC({
  "id": "R9TR30814KJ",
  "description": "Samsung tablet: 4 little + 4 big; only the last big core (id 7) is pinnable.",
  "cores": [
    { "id": 0, "type": "little", "pinnable": true },
    { "id": 1, "type": "little", "pinnable": true },
    { "id": 2, "type": "little", "pinnable": true },
    { "id": 3, "type": "little", "pinnable": true },
    { "id": 4, "type": "big", "pinnable": false },
    { "id": 5, "type": "big", "pinnable": false },
    { "id": 6, "type": "big", "pinnable": false },
    { "id": 7, "type": "big", "pinnable": true }
  ],
  "gpu": { "backend": "vulkan", "name": "Adreno 610", "subgroup_size": 64 }
})DEVSPEC",
    // ZY22FLDDK7.json
    R"DEVSPEC({
  "id": "ZY22FLDDK7",
  "description": "Motorola moto g pure, MediaTek MT6762G (32-bit armeabi-v7a userspace): 8x Cortex-A53 in two freq bins -- cores 0-3 @2.0GHz (cap 1024) -> big, cores 4-7 @1.5GHz (cap 768) -> little. GPU PowerVR Rogue GE8320: cifar kernels (no subgroup) verified numerically correct, but the subgroup-arithmetic radix sort (tree stage 2) gives all-zero output for every subgroup_size (16/32/64) -- a PowerVR subgroup limitation, so tree is unsupported here. subgroup_size kept at 32 (irrelevant to the working cifar path).",
  "cores": [
    { "id": 0, "type": "big", "pinnable": true },
    { "id": 1, "type": "big", "pinnable": true },
    { "id": 2, "type": "big", "pinnable": true },
    { "id": 3, "type": "big", "pinnable": true },
    { "id": 4, "type": "little", "pinnable": true },
    { "id": 5, "type": "little", "pinnable": true },
    { "id": 6, "type": "little", "pinnable": true },
    { "id": 7, "type": "little", "pinnable": true }
  ],
  "gpu": { "backend": "vulkan", "name": "PowerVR Rogue GE8320", "subgroup_size": 32 }
})DEVSPEC",
    // ce0717178d7758b00b7e.json
    R"DEVSPEC({
  "id": "ce0717178d7758b00b7e",
  "description": "Samsung Galaxy Note 8 (SM-N950U), Snapdragon 835: 4 little (cores 0-3 @1.9GHz) + 4 big (cores 4-7 @2.36GHz). Tiers verified by cpufreq cluster + max_freq (the previous spec had little/big swapped).",
  "cores": [
    { "id": 0, "type": "little", "pinnable": true },
    { "id": 1, "type": "little", "pinnable": true },
    { "id": 2, "type": "little", "pinnable": true },
    { "id": 3, "type": "little", "pinnable": true },
    { "id": 4, "type": "big", "pinnable": true },
    { "id": 5, "type": "big", "pinnable": true },
    { "id": 6, "type": "big", "pinnable": true },
    { "id": 7, "type": "big", "pinnable": true }
  ],
  "gpu": { "backend": "vulkan", "name": "Adreno 540", "subgroup_size": 32 }
})DEVSPEC",
    // jetson.json
    R"DEVSPEC({
  "id": "jetson",
  "description": "NVIDIA Jetson Orin Nano (normal power mode): 6 cores, single tier.",
  "cores": [
    { "id": 0, "type": "little", "pinnable": true },
    { "id": 1, "type": "little", "pinnable": true },
    { "id": 2, "type": "little", "pinnable": true },
    { "id": 3, "type": "little", "pinnable": true },
    { "id": 4, "type": "little", "pinnable": true },
    { "id": 5, "type": "little", "pinnable": true }
  ],
  "gpu": { "backend": "cuda", "name": "NVIDIA Tegra Orin (nvgpu)", "subgroup_size": 32 }
})DEVSPEC",
    // jetsonlowpower.json
    R"DEVSPEC({
  "id": "jetsonlowpower",
  "description": "NVIDIA Jetson Orin Nano (7W low-power mode): 4 cores, single tier.",
  "cores": [
    { "id": 0, "type": "little", "pinnable": true },
    { "id": 1, "type": "little", "pinnable": true },
    { "id": 2, "type": "little", "pinnable": true },
    { "id": 3, "type": "little", "pinnable": true }
  ],
  "gpu": { "backend": "cuda", "name": "NVIDIA Tegra Orin (nvgpu)", "subgroup_size": 32 }
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
  "gpu": { "backend": "vulkan", "name": "Intel UHD Graphics 770", "subgroup_size": 32 }
})DEVSPEC",
};

}  // namespace bt::device_specs

