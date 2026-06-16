#pragma once

#include <spdlog/spdlog.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

// ----------------------------------------------------------------------------
// Schedule source: the optimizer's schedule JSON is delivered to each target as
// a plain file over the channel that already exists -- `adb push` for the phones,
// `scp` for Jetson/rocky -- and read from the local filesystem here. No HTTP
// server, no libcurl, no hardcoded IP. This is the literal "three tools talking
// through files" model.
//
// Convention when a directory is used to stage many schedules:
//   <base_dir>/<device_id>/<app_name>/schedule_<NNN>.json
// (the harness builds that path; this loader just takes the final path).
// ----------------------------------------------------------------------------

[[nodiscard]]
static inline nlohmann::json load_schedule_json(const std::string& path) {
  spdlog::info("Loading schedule JSON from file: {}", path);
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open schedule file: " + path);
  }
  return nlohmann::json::parse(in);
}
