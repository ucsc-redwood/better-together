#pragma once

#include "task.hpp"
#include <cmath>

inline void process_task_stage_A(Task& task) {
#pragma omp for
  for (size_t i = 0; i < task.data.size(); ++i) {
    float x = task.data[i];
    for (int j = 0; j < 100; j++) {
      x = std::sin(x) * std::cos(x) + std::sqrt(std::abs(x));
    }
    task.data[i] = x;
  }
}

inline void process_task_stage_B(Task& task) {
#pragma omp for
  for (size_t i = 0; i < task.data.size(); ++i) {
    float x = task.data[i];
    for (int j = 0; j < 100; j++) {
      x = std::exp(std::sin(x)) + std::log(std::abs(x) + 1.0f);
    }
    task.data[i] = x;
  }
}

inline void process_task_stage_C(Task& task) {
#pragma omp for
  for (size_t i = 0; i < task.data.size(); ++i) {
    float x = task.data[i];
    for (int j = 0; j < 100; j++) {
      x = std::pow(std::abs(std::sin(x)), 0.5f) + std::tan(x / 10.0f);
    }
    task.data[i] = x;
  }
}
