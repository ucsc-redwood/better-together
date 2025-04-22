#pragma once

#include <cstdint>

struct BaseAppData {
  static uint32_t next_uid;
  uint32_t uid;

  BaseAppData() : uid(next_uid++) {}

  void reset() { uid = next_uid++; }

  [[nodiscard]] uint32_t get_uid() const { return uid; }
};
