#pragma once

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

#include "conf.hpp"

namespace fs = std::filesystem;

// ---------------------------------------------------------------------
// Execution model
// ---------------------------------------------------------------------

enum class ExecutionModel { kOMP, kGPU, kVulkan };

// Define the configuration and execution model.
struct ChunkConfig {
  ExecutionModel exec_model;  // e.g., kOMP, kGPU, kVulkan
  int start_stage;
  int end_stage;
  std::optional<ProcessorType> proc_type;  // e.g., kLittleCore, kMediumCore, kBigCore
  std::optional<int> num_threads;
};

// ---------------------------------------------------------------------
// Helper function: Reads chunks from a JSON configuration file.
// It ignores any static/global fields and only extracts the "chunks" array
// from the "schedule" object.
// ---------------------------------------------------------------------
inline std::vector<ChunkConfig> readChunksFromJson(const nlohmann::json &j) {
  std::vector<ChunkConfig> chunks;

  // The JSON is expected to have a "schedule" object with a "chunks" array.
  const auto &jChunks = j.at("schedule").at("chunks");
  for (const auto &jchunk : jChunks) {
    ChunkConfig config;
    std::string hardware = jchunk.at("hardware").get<std::string>();

    // Determine the execution model and CPU details if applicable.
    if (hardware == "gpu_cuda") {
      config.exec_model = ExecutionModel::kGPU;
    } else if (hardware == "vulkan") {
      config.exec_model = ExecutionModel::kVulkan;
    } else if (hardware == "little") {
      config.exec_model = ExecutionModel::kOMP;
      config.proc_type = ProcessorType::kLittleCore;
      config.num_threads = jchunk.at("threads").get<int>();
    } else if (hardware == "medium") {
      config.exec_model = ExecutionModel::kOMP;
      config.proc_type = ProcessorType::kMediumCore;
      config.num_threads = jchunk.at("threads").get<int>();
    } else if (hardware == "big") {
      config.exec_model = ExecutionModel::kOMP;
      config.proc_type = ProcessorType::kBigCore;
      config.num_threads = jchunk.at("threads").get<int>();
    } else {
      throw std::runtime_error("Unknown hardware type: " + hardware);
    }

    // Extract the stage range from the "stages" array.
    const auto &jStages = jchunk.at("stages");
    if (jStages.empty()) throw std::runtime_error("Empty stages array in chunk");
    config.start_stage = jStages.front().get<int>();
    config.end_stage = jStages.back().get<int>();

    chunks.push_back(config);

    // print out for debugging
    spdlog::info("Chunk config: {}", jchunk.at("name").get<std::string>());
    spdlog::info("\tStart stage: {}", config.start_stage);
    spdlog::info("\tEnd stage: {}", config.end_stage);

    // Convert enum to string
    std::string exec_model_str;
    if (config.exec_model == ExecutionModel::kOMP) {
      exec_model_str = "OMP";
    } else if (config.exec_model == ExecutionModel::kGPU) {
      exec_model_str = "GPU";
    } else if (config.exec_model == ExecutionModel::kVulkan) {
      exec_model_str = "Vulkan";
    }
    spdlog::info("\tExecution model: {}", exec_model_str);

    // Only log processor type if it's available
    if (config.proc_type.has_value()) {
      std::string proc_type_str;
      if (config.proc_type.value() == ProcessorType::kLittleCore) {
        proc_type_str = "Little Core";
      } else if (config.proc_type.value() == ProcessorType::kMediumCore) {
        proc_type_str = "Medium Core";
      } else if (config.proc_type.value() == ProcessorType::kBigCore) {
        proc_type_str = "Big Core";
      }
      spdlog::info("\tProcessor type: {}", proc_type_str);
    }

    // Only log thread count if it's available
    if (config.num_threads.has_value()) {
      spdlog::info("\tNumber of threads: {}", config.num_threads.value());
    }
  }

  return chunks;
}

// // ---------------------------------------------------------------------
// // Helper function: Reads chunks from a JSON configuration file.
// // It ignores any static/global fields and only extracts the "chunks" array
// // from the "schedule" object.
// // ---------------------------------------------------------------------
// inline std::vector<ChunkConfig> readChunksFromJson(const fs::path &filepath) {
//   std::ifstream file(filepath);
//   if (!file) throw std::runtime_error("Could not open file: " + filepath.string());

//   nlohmann::json j;
//   file >> j;

//   std::vector<ChunkConfig> chunks;

//   // The JSON is expected to have a "schedule" object with a "chunks" array.
//   const auto &jChunks = j.at("schedule").at("chunks");
//   for (const auto &jchunk : jChunks) {
//     ChunkConfig config;
//     std::string hardware = jchunk.at("hardware").get<std::string>();

//     // Determine the execution model and CPU details if applicable.
//     if (hardware == "gpu_cuda") {
//       config.exec_model = ExecutionModel::kGPU;
//     } else if (hardware == "vulkan") {
//       config.exec_model = ExecutionModel::kVulkan;
//     } else if (hardware == "little") {
//       config.exec_model = ExecutionModel::kOMP;
//       config.proc_type = ProcessorType::kLittleCore;
//       config.num_threads = jchunk.at("threads").get<int>();
//     } else if (hardware == "medium") {
//       config.exec_model = ExecutionModel::kOMP;
//       config.proc_type = ProcessorType::kMediumCore;
//       config.num_threads = jchunk.at("threads").get<int>();
//     } else if (hardware == "big") {
//       config.exec_model = ExecutionModel::kOMP;
//       config.proc_type = ProcessorType::kBigCore;
//       config.num_threads = jchunk.at("threads").get<int>();
//     } else {
//       throw std::runtime_error("Unknown hardware type: " + hardware);
//     }

//     // Extract the stage range from the "stages" array.
//     const auto &jStages = jchunk.at("stages");
//     if (jStages.empty()) throw std::runtime_error("Empty stages array in chunk");
//     config.start_stage = jStages.front().get<int>();
//     config.end_stage = jStages.back().get<int>();

//     chunks.push_back(config);

//     // print out for debugging
//     spdlog::info("Chunk config: {}", jchunk.at("name").get<std::string>());
//     spdlog::info("\tStart stage: {}", config.start_stage);
//     spdlog::info("\tEnd stage: {}", config.end_stage);

//     // Convert enum to string
//     std::string exec_model_str;
//     if (config.exec_model == ExecutionModel::kOMP) {
//       exec_model_str = "OMP";
//     } else if (config.exec_model == ExecutionModel::kGPU) {
//       exec_model_str = "GPU";
//     } else if (config.exec_model == ExecutionModel::kVulkan) {
//       exec_model_str = "Vulkan";
//     }
//     spdlog::info("\tExecution model: {}", exec_model_str);

//     // Only log processor type if it's available
//     if (config.proc_type.has_value()) {
//       std::string proc_type_str;
//       if (config.proc_type.value() == ProcessorType::kLittleCore) {
//         proc_type_str = "Little Core";
//       } else if (config.proc_type.value() == ProcessorType::kMediumCore) {
//         proc_type_str = "Medium Core";
//       } else if (config.proc_type.value() == ProcessorType::kBigCore) {
//         proc_type_str = "Big Core";
//       }
//       spdlog::info("\tProcessor type: {}", proc_type_str);
//     }

//     // Only log thread count if it's available
//     if (config.num_threads.has_value()) {
//       spdlog::info("\tNumber of threads: {}", config.num_threads.value());
//     }
//   }

//   return chunks;
// }
