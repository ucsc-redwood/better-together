#include <benchmark/benchmark.h>
#include <concurrentqueue.h>

#include <queue>

#include "../../lpipe/spsc_queue.hpp"

// ------------------------------------------------------------------------------------------------
// SPSCQueue
// ------------------------------------------------------------------------------------------------

static void BM_SPSCQueue_Enqueue(benchmark::State& state) {
  SPSCQueue<int> queue;
  for (auto _ : state) {
    queue.enqueue(1);
  }
}

BENCHMARK(BM_SPSCQueue_Enqueue);

static void BM_SPSCQueue_Dequeue(benchmark::State& state) {
  SPSCQueue<int> queue;
  for (auto _ : state) {
    state.PauseTiming();
    queue.enqueue(1);
    state.ResumeTiming();

    int item;
    queue.dequeue(item);
    benchmark::DoNotOptimize(item);
  }
}

BENCHMARK(BM_SPSCQueue_Dequeue);

// ------------------------------------------------------------------------------------------------
// moodycamel::ConcurrentQueue
// ------------------------------------------------------------------------------------------------

static void BM_MoodyConcurrentQueue_Enqueue(benchmark::State& state) {
  moodycamel::ConcurrentQueue<int> queue;
  for (auto _ : state) {
    queue.enqueue(1);
  }
}

BENCHMARK(BM_MoodyConcurrentQueue_Enqueue);

static void BM_MoodyConcurrentQueue_Dequeue(benchmark::State& state) {
  moodycamel::ConcurrentQueue<int> queue;
  for (auto _ : state) {
    state.PauseTiming();
    queue.enqueue(1);
    state.ResumeTiming();

    int item;
    queue.try_dequeue(item);
    benchmark::DoNotOptimize(item);
  }
}

BENCHMARK(BM_MoodyConcurrentQueue_Dequeue);

// ------------------------------------------------------------------------------------------------
// std::queue
// ------------------------------------------------------------------------------------------------

static void BM_StdQueue_Enqueue(benchmark::State& state) {
  std::queue<int> queue;
  for (auto _ : state) {
    queue.push(1);
  }
}

BENCHMARK(BM_StdQueue_Enqueue);

static void BM_StdQueue_Dequeue(benchmark::State& state) {
  std::queue<int> queue;
  for (auto _ : state) {
    state.PauseTiming();
    queue.push(1);
    state.ResumeTiming();

    int item = queue.front();
    queue.pop();
    benchmark::DoNotOptimize(item);
  }
}

BENCHMARK(BM_StdQueue_Dequeue);

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();
  return 0;
}
