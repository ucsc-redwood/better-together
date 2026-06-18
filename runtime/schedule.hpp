#pragma once

#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "platform/registry/conf.hpp"

// ---------------------------------------------------------------------
// Execution model
// ---------------------------------------------------------------------

enum class ExecutionModel { kOMP, kVulkan, kCuda };

// Define the configuration and execution model.
// This is the configuration for a single chunk.
struct ChunkConfig {
  ExecutionModel exec_model;  // e.g., kOMP, kVulkan, kCuda
  int start_stage;
  int end_stage;
  std::optional<ProcessorType> cpu_proc_type;  // e.g., kLittleCore, kMediumCore, kBigCore
};

// TODO: add CUDA
static inline ProcessorType get_processor_type_from_chunk_config(const ChunkConfig& chunk_config) {
  // if is CPU
  if (chunk_config.exec_model == ExecutionModel::kOMP) {
    return chunk_config.cpu_proc_type.value();
  }

  // if is CUDA
  if (chunk_config.exec_model == ExecutionModel::kCuda) {
    return ProcessorType::kCuda;
  }

  // if is Vulkan
  if (chunk_config.exec_model == ExecutionModel::kVulkan) {
    return ProcessorType::kVulkan;
  }

  throw std::invalid_argument("Invalid execution model");
}

struct Schedule {
  std::string uid;
  std::vector<ChunkConfig> chunks;

  [[nodiscard]] size_t n_chunks() const { return chunks.size(); }

  [[nodiscard]] size_t start_stage(const size_t chunk_id) const {
    return chunks[chunk_id].start_stage;
  }

  [[nodiscard]] size_t end_stage(const size_t chunk_id) const { return chunks[chunk_id].end_stage; }

  void print(const size_t id) const {
    std::cout << "Schedule " << id << " [UID: " << uid << "]:\n";
    const std::string indent = "  ";

    for (size_t i = 0; i < chunks.size(); ++i) {
      const auto& chunk = chunks[i];

      // Get processor type string
      std::string cpu_proc_type;
      if (chunk.exec_model == ExecutionModel::kOMP) {
        switch (chunk.cpu_proc_type.value()) {
          case ProcessorType::kLittleCore:
            cpu_proc_type = "Little";
            break;
          case ProcessorType::kMediumCore:
            cpu_proc_type = "Medium";
            break;
          case ProcessorType::kBigCore:
            cpu_proc_type = "Big   ";
            break;
          default:
            cpu_proc_type = "?";
        }
      }

      // Print chunk header with execution model and processor type
      std::cout << indent << "Chunk " << i << " [";
      switch (chunk.exec_model) {
        case ExecutionModel::kOMP:
          std::cout << "OMP/" << cpu_proc_type;
          break;
        case ExecutionModel::kVulkan:
          std::cout << "Vulkan    ";
          break;
        case ExecutionModel::kCuda:
          std::cout << "CUDA      ";
          break;
      }
      std::cout << "]: ";

      // Print stages included in this chunk
      for (int stage = chunk.start_stage; stage <= chunk.end_stage; ++stage) {
        std::cout << stage;
        if (stage != chunk.end_stage) {
          std::cout << ", ";
        }
      }
      std::cout << std::endl;
    }
  }
};

// ---------------------------------------------------------------------
// Validate that a schedule's chunks contiguously cover stages [1, n_stages]
// (1-based, inclusive — the convention config_reader produces and the executor
// dispatches verbatim). Throws std::runtime_error on any gap, overlap, missing
// first/last stage, out-of-range chunk, or empty schedule.
//
// This is the pipeline's correctness gate: it catches a malformed / incomplete
// schedule before the executor runs the pipeline on partial data — e.g. the
// "dropped first stage of every chunk" class of bug (a chunk starting at 2
// instead of 1), which previously survived because the pipeline path had no
// correctness assertions (only timing).
// ---------------------------------------------------------------------
inline void validate_schedule_coverage(const Schedule& schedule, const size_t n_stages) {
  const int n = static_cast<int>(n_stages);

  if (schedule.chunks.empty()) {
    throw std::runtime_error("Schedule [" + schedule.uid + "] has no chunks");
  }

  int expected = 1;  // the next stage that must be covered (1-based)
  for (size_t i = 0; i < schedule.chunks.size(); ++i) {
    const ChunkConfig& c = schedule.chunks[i];

    if (c.start_stage != expected) {
      throw std::runtime_error(
          "Schedule [" + schedule.uid + "] chunk " + std::to_string(i) +
          ": expected start_stage " + std::to_string(expected) + " but got " +
          std::to_string(c.start_stage) +
          " (gap, overlap, or dropped stage — chunks must contiguously cover [1, " +
          std::to_string(n) + "])");
    }
    if (c.end_stage < c.start_stage) {
      throw std::runtime_error("Schedule [" + schedule.uid + "] chunk " + std::to_string(i) +
                               ": end_stage " + std::to_string(c.end_stage) +
                               " < start_stage " + std::to_string(c.start_stage));
    }
    expected = c.end_stage + 1;
  }

  const int covered = expected - 1;  // last stage covered
  if (covered != n) {
    throw std::runtime_error("Schedule [" + schedule.uid + "] covers stages [1, " +
                             std::to_string(covered) + "] but the application has " +
                             std::to_string(n) + " stages");
  }
}

// ---------------------------------------------------------------------
// Validate that every chunk's processing unit is actually PRESENT on the device
// we're about to run on. A chunk targeting an absent PU (e.g. a Little-core chunk
// on the Big-only MiniPC) would otherwise throw deep inside an unguarded worker
// thread → std::terminate; checking up front lets the caller skip+warn.
//
// `pu_present(ProcessorType) -> bool` answers "does this device have that PU".
// Returns the first offending chunk's reason, or std::nullopt when every chunk's
// PU is available. Kept predicate-driven (not tied to the device globals) so it is
// unit-testable without a real device; app.hpp wraps it with the live capabilities.
// ---------------------------------------------------------------------
template <typename PuPresentFn>
[[nodiscard]] inline std::optional<std::string> first_unavailable_pu(const Schedule& schedule,
                                                                     PuPresentFn pu_present) {
  for (size_t i = 0; i < schedule.chunks.size(); ++i) {
    const ChunkConfig& c = schedule.chunks[i];
    if (c.exec_model == ExecutionModel::kOMP && !c.cpu_proc_type.has_value()) {
      return "chunk " + std::to_string(i) + " is OMP but names no CPU tier";
    }
    const ProcessorType pu = get_processor_type_from_chunk_config(c);
    if (!pu_present(pu)) {
      return "chunk " + std::to_string(i) + " targets absent PU '" + CoreTypeName(pu) + "'";
    }
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------
// A GPU backend (Vulkan / CUDA) owns a SINGLE dispatcher — one engine, one command
// buffer / Sequence, one fence. Every schedule chunk runs on its own worker thread,
// so a schedule that puts a GPU backend in MORE THAN ONE chunk spawns multiple
// threads that concurrently record into that one shared command buffer:
// `vkBeginCommandBuffer` is called on a buffer still in the recording/pending state
// of another thread (VUID-vkBeginCommandBuffer-commandBuffer-00049) → corruption →
// VK_ERROR_DEVICE_LOST → crash.
//
// Diagnosed 2026-06-17 with GPU-assisted validation on rocky/RADV — this SUPERSEDES
// the earlier "stale octree count / GPU re-entry" triage: the octree is a red
// herring, the bug is concurrent command-buffer reuse, and it tracks the number of
// concurrent GPU chunks (≥2), not octree re-entry. The z3 solver only ever assigns
// ONE contiguous chunk per PU, so it never emits such a schedule; this guard rejects
// it cleanly instead of racing the GPU into a device loss.
//
// Returns the offending chunk's reason, or std::nullopt when every GPU backend
// appears in at most one chunk.
[[nodiscard]] inline std::optional<std::string> first_concurrent_gpu_chunk(
    const Schedule& schedule) {
  std::set<ExecutionModel> gpu_seen;
  for (size_t i = 0; i < schedule.chunks.size(); ++i) {
    const ExecutionModel m = schedule.chunks[i].exec_model;
    if (m != ExecutionModel::kVulkan && m != ExecutionModel::kCuda) continue;
    if (!gpu_seen.insert(m).second) {
      return "Schedule [" + schedule.uid + "] places a GPU backend in >1 chunk (chunk " +
             std::to_string(i) +
             "): the GPU dispatcher's single command buffer/fence is shared across the "
             "per-chunk worker threads and is not safe under concurrent recording "
             "(VK_ERROR_DEVICE_LOST). A GPU backend must run in exactly one chunk — the z3 "
             "solver assigns one contiguous chunk per PU and never emits otherwise.";
    }
  }
  return std::nullopt;
}
