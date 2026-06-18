#pragma once
// AUTO-GENERATED from vocab.json by scripts/embed_vocab.py -- DO NOT EDIT.
// Regenerate after changing vocab.json.

#include <stdexcept>
#include <string>

// Processing-unit class. Values are load-bearing (schedules index them).
enum class ProcessorType {
  kLittleCore = 0,
  kMediumCore = 1,
  kBigCore = 2,
  kVulkan = 3,
  kCuda = 4,
  kSuperCore = 5,
};

inline std::string CoreTypeName(const ProcessorType core_type) {
  switch (core_type) {
    case ProcessorType::kLittleCore:
      return "little";
    case ProcessorType::kMediumCore:
      return "medium";
    case ProcessorType::kBigCore:
      return "big";
    case ProcessorType::kSuperCore:
      return "super";
    default:
      return "unknown";
  }
}

inline ProcessorType ParseCoreType(const std::string& s) {
  if (s == "little") return ProcessorType::kLittleCore;
  if (s == "medium") return ProcessorType::kMediumCore;
  if (s == "big") return ProcessorType::kBigCore;
  if (s == "super") return ProcessorType::kSuperCore;
  throw std::runtime_error("unknown core type '" + s + "'");
}
