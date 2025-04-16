#include <benchmark/benchmark.h>
#include <omp.h>

#include "builtin-apps/app.hpp"  // for 'g_big_cores', 'g_medium_cores', 'g_little_cores'
#include "common.hpp"

namespace device_3A021JEHN02756 {

// little | medium | big | gpu

// stage_timings = [
//     [28.94, 7.69, 10.68, 5.28],
//     [55.03, 27.91, 33.87, 3.41],
//     [25.15, 6.53, 6.73, 3.09],
//     [27.07, 14.11, 17.14, 2.43],
//     [13.78, 3.85, 3.87, 1.93],
//     [13.79, 3.96, 3.89, 1.99],
//     [14.34, 4.03, 3.94, 2.15],
//     [15.57, 7.46, 10.15, 1.43],
//     [0.08, 0.03, 0.03, 0.40],
// ]

// ----------------------------------------------------------------------------
// Schedule 1
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0] 
// GPU = [1, 2, 3, 4, 5] 
// Medium = [6, 7, 8] 
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_1(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 6);
    auto t2 = create_thread(q_2, q_0, g_medium_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 2
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3, 4, 5]
// Medium = [6, 7]
// Little = [8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_2(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 6);
    auto t2 = create_thread(q_2, q_3, g_medium_cores, 7, 8);
    auto t3 = create_thread(q_3, q_0, g_little_cores, 9, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 3
// ----------------------------------------------------------------------------
// Stage assignments:
// Big = [0]
// GPU = [1, 2, 3, 4]
// Little = [5]
// Medium = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_3(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_big_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 5);
    auto t2 = create_thread(q_2, q_3, g_little_cores, 6, 6);
    auto t3 = create_thread(q_3, q_0, g_medium_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 4
// ----------------------------------------------------------------------------
// Stage assignments:
// Medium = [0]
// GPU = [1, 2, 3, 4, 5]
// Big = [6, 7]
// Little = [8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_4(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_medium_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 6);
    auto t2 = create_thread(q_2, q_3, g_big_cores, 7, 8);
    auto t3 = create_thread(q_3, q_0, g_little_cores, 9, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_4)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 5
// ----------------------------------------------------------------------------
// Stage assignments:
// Medium = [0]
// GPU = [1, 2, 3, 4]
// Little = [5]
// Big = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_5(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_medium_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 5);
    auto t2 = create_thread(q_2, q_3, g_little_cores, 6, 6);
    auto t3 = create_thread(q_3, q_0, g_big_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_5)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 6
// ----------------------------------------------------------------------------
// Stage assignments:
// Medium = [0]
// GPU = [1, 2, 3, 4, 5]
// Big = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_6(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, g_medium_cores, 1, 1);
    auto t1 = create_thread(q_1, q_2, disp, 2, 6);
    auto t2 = create_thread(q_2, q_0, g_big_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_6)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 7
// ----------------------------------------------------------------------------
// Stage assignments:
// GPU = [0, 1, 2, 3]
// Big = [4]
// Little = [5]
// Medium = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_7(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, disp, 1, 4);
    auto t1 = create_thread(q_1, q_2, g_big_cores, 5, 5);
    auto t2 = create_thread(q_2, q_3, g_little_cores, 6, 6);
    auto t3 = create_thread(q_3, q_0, g_medium_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_7)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 8
// ----------------------------------------------------------------------------
// Stage assignments:
// GPU = [0, 1, 2, 3]
// Medium = [4]
// Little = [5]
// Big = [6, 7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_8(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, disp, 1, 4);
    auto t1 = create_thread(q_1, q_2, g_medium_cores, 5, 5);
    auto t2 = create_thread(q_2, q_3, g_little_cores, 6, 6);
    auto t3 = create_thread(q_3, q_0, g_big_cores, 7, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_8)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 9
// ----------------------------------------------------------------------------
// Stage assignments:
// GPU = [0, 1, 2, 3]
// Little = [4]
// Medium = [5, 6]
// Big = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_9(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, disp, 1, 4);
    auto t1 = create_thread(q_1, q_2, g_little_cores, 5, 5);
    auto t2 = create_thread(q_2, q_3, g_medium_cores, 6, 7);
    auto t3 = create_thread(q_3, q_0, g_big_cores, 8, 9);

    t0.join();
    t1.join();
    t2.join();
    t3.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_9)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

// ----------------------------------------------------------------------------
// Schedule 10
// ----------------------------------------------------------------------------
// Stage assignments:
// GPU = [0, 1, 2, 3]
// Medium = [4, 5, 6]
// Big = [7, 8]
// ----------------------------------------------------------------------------
static void BM_pipe_cifar_sparse_vk_schedule_10(benchmark::State& state) {
  SETUP_DATA;

  for (auto _ : state) {
    auto t0 = create_thread(q_0, q_1, disp, 1, 4);
    auto t1 = create_thread(q_1, q_2, g_medium_cores, 5, 7);
    auto t2 = create_thread(q_2, q_0, g_big_cores, 8, 9);

    t0.join();
    t1.join();
    t2.join();
  }
}

BENCHMARK(BM_pipe_cifar_sparse_vk_schedule_10)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1)
    ->Repetitions(5);

}  // namespace device_3A021JEHN02756

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != "3A021JEHN02756") {
    return 0;
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();  // Ensure proper cleanup
  return 0;
}
