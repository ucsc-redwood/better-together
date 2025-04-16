#pragma once

#include "builtin-apps/app.hpp"
#include "builtin-apps/pipeline/record.hpp"
#include "builtin-apps/pipeline/worker.hpp"

#define BIG(...)                                                \
  {                                                             \
    .core = ProcessorType::kBigCore, .indices = { __VA_ARGS__ } \
  }
#define MED(...)                                                   \
  {                                                                \
    .core = ProcessorType::kMediumCore, .indices = { __VA_ARGS__ } \
  }
#define LIT(...)                                                   \
  {                                                                \
    .core = ProcessorType::kLittleCore, .indices = { __VA_ARGS__ } \
  }
#define GPU(...)                                               \
  {                                                            \
    .core = ProcessorType::kVulkan, .indices = { __VA_ARGS__ } \
  }

#define SCHEDULE(...)         \
  {                           \
    .chunks = { __VA_ARGS__ } \
  }

// ----------------------------------------------------------------------------
// Device 3A021JEHN02756
// ----------------------------------------------------------------------------

namespace device_3A021JEHN02756 {
// clang-format off
  static std::array<Schedule, 10> schedules = {{
      SCHEDULE(
          BIG(0),
          GPU(1, 2, 3, 4, 5),
          MED(6, 7, 8)
      ),
      SCHEDULE(
          BIG(0),
          GPU(1, 2, 3, 4, 5),
          MED(6, 7),
          LIT(8)
      ),
      SCHEDULE(
          BIG(0),
          GPU(1, 2, 3, 4),
          LIT(5),
          MED(6, 7, 8)
      ),
      SCHEDULE(
          MED(0),
          GPU(1, 2, 3, 4, 5),
          BIG(6, 7),
          LIT(8)
      ),
      SCHEDULE(
          MED(0),
          GPU(1, 2, 3, 4),
          LIT(5),
          BIG(6, 7, 8)
      ),
      SCHEDULE(
          MED(0),
          GPU(1, 2, 3, 4, 5),
          BIG(6, 7, 8)
      ),
      SCHEDULE(
          GPU(0, 1, 2, 3),
          BIG(4),
          LIT(5),
          MED(6, 7, 8)
      ),
      SCHEDULE(
          GPU(0, 1, 2, 3),
          MED(4),
          LIT(5),
          BIG(6, 7, 8)
      ),
      SCHEDULE(
          GPU(0, 1, 2, 3),
          LIT(4),
          MED(5, 6),
          BIG(7, 8)
      ),
      SCHEDULE(
          GPU(0, 1, 2, 3),
          MED(4, 5, 6),
          BIG(7, 8)
      ),
  }};
// clang-format on
}  // namespace device_3A021JEHN02756

// ----------------------------------------------------------------------------
// Device 9b034f1b
// ----------------------------------------------------------------------------

namespace device_9b034f1b {}

// ----------------------------------------------------------------------------
// Device jetson
// ----------------------------------------------------------------------------

namespace device_jetson {}

namespace device_jetsonlowpower {}