#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include "benchmarks/argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "generated_code.cuh"

__global__ void kernel_test() {}

void warmup() {
  kernel_test<<<1, 1>>>();
  CheckCuda(cudaDeviceSynchronize());
}

struct DeviceInfo {
  const generated_schedules::ScheduleRecord *table;
  size_t count;
};

// Now create a lookup map from the string device ID to the device's table info
static std::map<std::string, DeviceInfo> g_device_map = {
    {"jetson", {device_jetson::schedule_table, device_jetson::schedule_count}},
    {"jetsonlowpower",
     {device_jetsonlowpower::schedule_table, device_jetsonlowpower::schedule_count}},
};

// Helper function to register a single benchmark for a given device & schedule index
void register_single_benchmark(const std::string &device_id, int schedule_index) {
  // Look up the device
  auto it = g_device_map.find(device_id);
  if (it == g_device_map.end()) {
    throw std::runtime_error("Invalid device ID: " + device_id);
  }

  // Confirm the schedule index is valid
  const DeviceInfo &dev_info = it->second;
  if (schedule_index < 0 || schedule_index >= static_cast<int>(dev_info.count)) {
    throw std::runtime_error("Invalid schedule index " + std::to_string(schedule_index) +
                             " for device " + device_id + ". Must be between 0 and " +
                             std::to_string(dev_info.count - 1));
  }

  // Grab the chosen schedule record
  const auto &rec = dev_info.table[schedule_index];

  // rec.name is the schedule_id string
  // rec.func is the function pointer
  benchmark::RegisterBenchmark(rec.name, rec.func)->Unit(benchmark::kMillisecond)->Iterations(10);
}

// Register either all or one schedule from a device
void register_benchmarks(const std::string &device_id, int index) {
  auto it = g_device_map.find(device_id);
  if (it == g_device_map.end()) {
    throw std::runtime_error("Invalid device ID: " + device_id);
  }
  const DeviceInfo &dev_info = it->second;

  if (index == -1) {
    // Register all schedules for this device
    for (int i = 0; i < static_cast<int>(dev_info.count); i++) {
      register_single_benchmark(device_id, i);
    }
  } else {
    // Register just one
    register_single_benchmark(device_id, index);
  }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
  PARSE_ARGS_BEGIN;

  int schedule_index = 0;  // Default to first schedule
  app.add_option("-i,--index", schedule_index, "Schedule index (0-9, or -1 for all schedules)")
      ->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::off);

  warmup();

  auto [new_argc, new_argv] = sanitize_argc_argv_for_pipe_benchmark(argc, argv);
  benchmark::Initialize(&new_argc, new_argv.data());

  // Register the benchmark based on the device ID and index
  try {
    register_benchmarks(g_device_id, schedule_index);
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
