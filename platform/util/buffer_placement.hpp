#pragma once
// Buffer placement (memory-placement spec, design B2/B3). An AppData that supports it builds
// each buffer through a PlaceBuffer, declaring what touches the buffer: the stages that read
// or write it, with 0 standing for host code outside any stage (e.g. the input written at
// construction, the final output read by a check). The caller decides the memory resource;
// a GPU backend maps a BufferClass to memory, others ignore it.

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory_resource>

namespace bt {

enum class BufferKind : std::uint8_t { kActivation, kWeights };

// Boundary: host-cached, today's placement (anything host code or a CPU chunk touches).
// Weights / GPU-local: may live in another host-visible memory type on a device where that
// was measured to be faster.
enum class BufferClass : std::uint8_t { kBoundary, kWeights, kGpuLocal };

using PlaceBuffer =
    std::function<std::pmr::memory_resource*(BufferKind kind, std::initializer_list<int> stages)>;

// Every buffer from one resource: the AppData(mr) behaviour.
[[nodiscard]] inline PlaceBuffer place_all_in(std::pmr::memory_resource* mr) {
  return [mr](BufferKind, std::initializer_list<int>) { return mr; };
}

}  // namespace bt
