#pragma once

#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "conf.hpp"
#include "pipeline/schedule.hpp"

// ---------------------------------------------------------------------
// Helper function: Reads chunks from a JSON configuration file.
// It ignores any static/global fields and only extracts the "chunks" array
// from the "schedule" object.
// ---------------------------------------------------------------------
inline std::vector<Schedule> readSchedulesFromJson(const nlohmann::json& j) {
  std::vector<Schedule> schedules;

  // Check if the JSON is an array (multiple solutions)
  if (!j.is_array()) {
    spdlog::error("JSON data is not an array of solutions");
    return schedules;
  }

  // Process each solution in the array
  for (const auto& solution : j) {
    Schedule schedule;

    // Skip if no chunks field
    if (!solution.contains("chunks")) {
      spdlog::warn("Solution missing 'chunks' field, skipping");
      continue;
    }

    const auto& chunks_json = solution["chunks"];
    if (!chunks_json.is_array()) {
      spdlog::warn("Chunks field is not an array, skipping solution");
      continue;
    }

    // Process each chunk in the solution
    for (const auto& chunk_json : chunks_json) {
      ChunkConfig chunk;

      // Skip if required fields are missing
      if (!chunk_json.contains("core_type") || !chunk_json.contains("stages")) {
        spdlog::warn("Chunk missing required fields, skipping");
        continue;
      }

      // Get the core type string and convert to ExecutionModel and ProcessorType
      std::string core_type = chunk_json["core_type"].get<std::string>();

      // Set execution model based on core type
      if (core_type == "GPU") {
        chunk.exec_model = ExecutionModel::kVulkan;
        chunk.cpu_proc_type = std::nullopt;
      } else {
        chunk.exec_model = ExecutionModel::kOMP;

        // Map the core type string to ProcessorType
        if (core_type == "Little") {
          chunk.cpu_proc_type = ProcessorType::kLittleCore;
        } else if (core_type == "Medium") {
          chunk.cpu_proc_type = ProcessorType::kMediumCore;
        } else if (core_type == "Big") {
          chunk.cpu_proc_type = ProcessorType::kBigCore;
        } else {
          spdlog::warn("Unknown core type: {}, defaulting to LittleCore", core_type);
          chunk.cpu_proc_type = ProcessorType::kLittleCore;
        }
      }

      // Get the stages array and set start and end stages
      const auto& stages = chunk_json["stages"];
      if (!stages.is_array() || stages.empty()) {
        spdlog::warn("Stages is not an array or is empty, skipping chunk");
        continue;
      }

      // Get first and last elements of the stages array for start and end
      chunk.start_stage = stages[0].get<int>();
      chunk.end_stage = stages[stages.size() - 1].get<int>();

      // Add the chunk to the schedule
      schedule.chunks.push_back(chunk);
    }

    // Only add schedules that have at least one valid chunk
    if (!schedule.chunks.empty()) {
      schedules.push_back(schedule);
    }
  }

  spdlog::info("Loaded {} valid schedules from JSON", schedules.size());
  return schedules;
}
