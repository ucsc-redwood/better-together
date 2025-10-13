#pragma once

#include <cstdint>

struct BaseAppData {
  static uint32_t next_uid;
  uint32_t uid;
  uint32_t initial_uid;

  BaseAppData() : uid(next_uid++), initial_uid(uid) {}

  void reset() { uid = next_uid++; }

  [[nodiscard]] uint32_t get_uid() const { return uid; }

  [[nodiscard]] uint32_t get_initial_uid() const { return initial_uid; }
};

inline uint32_t BaseAppData::next_uid = 0;
