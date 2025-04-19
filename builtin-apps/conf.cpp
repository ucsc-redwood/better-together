#include "conf.hpp"

DeviceRegistry::DeviceRegistry() {
  // For "pc": 8 P cores and 12 E cores
  devices_.emplace(
      "pc",
      Device("pc",
             std::vector<Core>{
                 {0, ProcessorType::kBigCore, true},     {1, ProcessorType::kBigCore, true},
                 {2, ProcessorType::kBigCore, true},     {3, ProcessorType::kBigCore, true},
                 {4, ProcessorType::kBigCore, true},     {5, ProcessorType::kBigCore, true},
                 {6, ProcessorType::kBigCore, true},     {7, ProcessorType::kBigCore, true},
                 {8, ProcessorType::kLittleCore, true},  {9, ProcessorType::kLittleCore, true},
                 {10, ProcessorType::kLittleCore, true}, {11, ProcessorType::kLittleCore, true},
                 {12, ProcessorType::kLittleCore, true}, {13, ProcessorType::kLittleCore, true},
                 {14, ProcessorType::kLittleCore, true}, {15, ProcessorType::kLittleCore, true},
                 {16, ProcessorType::kLittleCore, true}, {17, ProcessorType::kLittleCore, true},
                 {18, ProcessorType::kLittleCore, true}, {19, ProcessorType::kLittleCore, true},
                 {20, ProcessorType::kLittleCore, true}, {21, ProcessorType::kLittleCore, true},
                 {22, ProcessorType::kLittleCore, true}, {23, ProcessorType::kLittleCore, true},
             }));

  // For "jetson": 6 cores all of one type.

  // 2025-03-20T09:43:23-07:00
  // Running /home/yanwen/Desktop/redwood-aio/build/linux/arm64/releasedbg/bm-check-core-types
  // Run on (6 X 1510.4 MHz CPU s)
  // CPU Caches:
  //   L1 Data 64 KiB (x6)
  //   L1 Instruction 64 KiB (x6)
  //   L2 Unified 256 KiB (x6)
  //   L3 Unified 2048 KiB (x1)
  // Load Average: 0.45, 0.40, 0.19
  // ***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and
  // will incur extra overhead.
  // ---------------------------------------------------------------
  // Benchmark                     Time             CPU   Iterations
  // ---------------------------------------------------------------
  // HeavyFloat/CoreID0/0       47.1 ms         47.0 ms           15
  // HeavyFloat/CoreID1/1       47.0 ms         47.0 ms           14
  // HeavyFloat/CoreID2/2       47.0 ms         47.0 ms           14
  // HeavyFloat/CoreID3/3       47.3 ms         47.2 ms           13
  // HeavyFloat/CoreID4/4       47.0 ms         47.0 ms           15
  // HeavyFloat/CoreID5/5       47.7 ms         47.6 ms           13
  // GraphBFS/CoreID0/0         6.52 ms         6.50 ms          107
  // GraphBFS/CoreID1/1         6.62 ms         6.61 ms          104
  // GraphBFS/CoreID2/2         6.40 ms         6.39 ms          107
  // GraphBFS/CoreID3/3         6.64 ms         6.62 ms          105
  // GraphBFS/CoreID4/4         6.13 ms         6.12 ms          107
  // GraphBFS/CoreID5/5         6.45 ms         6.43 ms          108
  devices_.emplace("jetson",
                   Device("jetson",
                          std::vector<Core>{
                              {0, ProcessorType::kLittleCore, true},
                              {1, ProcessorType::kLittleCore, true},
                              {2, ProcessorType::kLittleCore, true},
                              {3, ProcessorType::kLittleCore, true},
                              {4, ProcessorType::kLittleCore, true},
                              {5, ProcessorType::kLittleCore, true},
                          }));

  // ----------------------------------------------------------------------------
  // For "jetsonlowpower": 4 cores all of one type.
  // ----------------------------------------------------------------------------

  // bm-check-core-types 2025-03-20T10:04:46-07:00 Running
  // /home/yanwen/Desktop/redwood-aio/build/linux/arm64/releasedbg/bm-check-core-types Run on (4 X
  // 1510.4 MHz CPU s) CPU Caches:
  //   L1 Data 64 KiB (x4)
  //   L1 Instruction 64 KiB (x4)
  //   L2 Unified 256 KiB (x4)
  //   L3 Unified 2048 KiB (x1)
  // Load Average: 0.31, 0.20, 0.08
  // ***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and
  // will incur extra overhead.
  // ---------------------------------------------------------------
  // Benchmark                     Time             CPU   Iterations
  // ---------------------------------------------------------------
  // HeavyFloat/CoreID0/0       74.4 ms         74.2 ms            9
  // HeavyFloat/CoreID1/1       75.2 ms         75.0 ms            9
  // HeavyFloat/CoreID2/2       74.5 ms         74.4 ms            9
  // HeavyFloat/CoreID3/3       73.9 ms         73.9 ms            9
  // GraphBFS/CoreID0/0         29.0 ms         28.9 ms           20
  // GraphBFS/CoreID1/1         29.8 ms         29.7 ms           21
  // GraphBFS/CoreID2/2         29.7 ms         29.6 ms           21
  // GraphBFS/CoreID3/3         29.9 ms         29.8 ms           22
  devices_.emplace("jetsonlowpower",
                   Device("jetsonlowpower",
                          std::vector<Core>{
                              {0, ProcessorType::kLittleCore, true},
                              {1, ProcessorType::kLittleCore, true},
                              {2, ProcessorType::kLittleCore, true},
                              {3, ProcessorType::kLittleCore, true},
                          }));

  // For "3A021JEHN02756": 8 cores in 3 groups.
  devices_.emplace("3A021JEHN02756",
                   Device("3A021JEHN02756",
                          std::vector<Core>{
                              {0, ProcessorType::kLittleCore, true},
                              {1, ProcessorType::kLittleCore, true},
                              {2, ProcessorType::kLittleCore, true},
                              {3, ProcessorType::kLittleCore, true},
                              {4, ProcessorType::kMediumCore, true},
                              {5, ProcessorType::kMediumCore, true},
                              {6, ProcessorType::kBigCore, true},
                              {7, ProcessorType::kBigCore, true},
                          }));

  // For "9b034f1b": 8 cores, only cores 0-4 are pinnable.
  devices_.emplace("9b034f1b",
                   Device("9b034f1b",
                          std::vector<Core>{
                              {0, ProcessorType::kLittleCore, true},
                              {1, ProcessorType::kLittleCore, true},
                              {2, ProcessorType::kLittleCore, true},
                              {3, ProcessorType::kMediumCore, true},
                              {4, ProcessorType::kMediumCore, true},
                              {5, ProcessorType::kBigCore, false},
                              {6, ProcessorType::kBigCore, false},
                              {7, ProcessorType::kBigCore, false},
                          }));

  // For "ce0717178d7758b00b7e": 8 cores split into LITTLE and BIG.
  devices_.emplace("ce0717178d7758b00b7e",
                   Device("ce0717178d7758b00b7e",
                          std::vector<Core>{
                              {4, ProcessorType::kLittleCore, true},
                              {5, ProcessorType::kLittleCore, true},
                              {6, ProcessorType::kLittleCore, true},
                              {7, ProcessorType::kLittleCore, true},
                              {0, ProcessorType::kBigCore, true},
                              {1, ProcessorType::kBigCore, true},
                              {2, ProcessorType::kBigCore, true},
                              {3, ProcessorType::kBigCore, true},
                          }));

  // For "R9TR30814KJ": Samsung Tablet 8 cores,
  devices_.emplace("R9TR30814KJ",
                   Device("R9TR30814KJ",
                          std::vector<Core>{
                              {0, ProcessorType::kLittleCore, true},
                              {1, ProcessorType::kLittleCore, true},
                              {2, ProcessorType::kLittleCore, true},
                              {3, ProcessorType::kLittleCore, true},
                              {4, ProcessorType::kBigCore, false},
                              {5, ProcessorType::kBigCore, false},
                              {6, ProcessorType::kBigCore, true},
                              {7, ProcessorType::kBigCore, true},
                          }));

  // For "minipc": 16 cores all of the same type.
  devices_.emplace("minipc",
                   Device("minipc",
                          std::vector<Core>({
                              {0, ProcessorType::kBigCore, true},
                              {1, ProcessorType::kBigCore, true},
                              {2, ProcessorType::kBigCore, true},
                              {3, ProcessorType::kBigCore, true},
                              {4, ProcessorType::kBigCore, true},
                              {5, ProcessorType::kBigCore, true},
                              {6, ProcessorType::kBigCore, true},
                              {7, ProcessorType::kBigCore, true},
                              {8, ProcessorType::kBigCore, true},
                              {9, ProcessorType::kBigCore, true},
                              {10, ProcessorType::kBigCore, true},
                              {11, ProcessorType::kBigCore, true},
                              {12, ProcessorType::kBigCore, true},
                              {13, ProcessorType::kBigCore, true},
                              {14, ProcessorType::kBigCore, true},
                              {15, ProcessorType::kBigCore, true},
                          })));
}