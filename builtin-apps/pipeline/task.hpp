#pragma once

#include <memory_resource>

template <typename AppDataT>
struct Task {
  static uint32_t uid_counter;
  uint32_t uid;

  // ----------------------------------
  AppDataT appdata;
  // ----------------------------------

  explicit Task(std::pmr::memory_resource* mr) : appdata(mr) { uid = uid_counter++; }

  void reset() {
    // appdata.reset();
    uid = uid_counter++;
  }
};

template <typename AppDataT>
uint32_t Task<AppDataT>::uid_counter = 0;
