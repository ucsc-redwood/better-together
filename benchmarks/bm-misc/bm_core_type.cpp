#include <benchmark/benchmark.h>

#include <cmath>
#include <queue>
#include <random>
#include <thread>
#include <vector>

#include "builtin-apps/affinity.hpp"

//------------------------------------------------------------------------------
// Global (or static) data for BFS benchmark
//------------------------------------------------------------------------------
static std::vector<std::vector<int>> g_adjacency_list;
static constexpr int NUM_NODES = 20000;
static constexpr int EDGES_PER_NODE = 30;

// Precompute a random adjacency list for BFS
static void InitializeGraph() {
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> dist(0, NUM_NODES - 1);

  g_adjacency_list.resize(NUM_NODES);
  for (int i = 0; i < NUM_NODES; ++i) {
    g_adjacency_list[i].reserve(EDGES_PER_NODE);
    for (int e = 0; e < EDGES_PER_NODE; ++e) {
      int neighbor = dist(rng);
      if (neighbor != i) {
        g_adjacency_list[i].push_back(neighbor);
      }
    }
  }
}

//------------------------------------------------------------------------------
// Heavy Floating Point Benchmark
//------------------------------------------------------------------------------
static void BM_HeavyFloatingPoint(benchmark::State& state) {
  // Pin the thread to the core specified by the range (arg).
  int core_id = static_cast<int>(state.range(0));
  try {
    bind_thread_to_cores({core_id});
  } catch (const std::exception& ex) {
    // If we can't pin to the core, skip the benchmark
    state.SkipWithError(
        ("Failed to pin to core " + std::to_string(core_id) + ": " + ex.what()).c_str());
    // Return immediately so we do no further work in this benchmark.
    return;
  } catch (...) {
    // Catch any other kind of exception
    state.SkipWithError(
        ("Failed to pin to core " + std::to_string(core_id) + " with unknown error").c_str());
    return;
  }

  // If pinning succeeded, proceed with the heavy floating-point math.
  for (auto _ : state) {
    double sum = 0.0;
    for (int i = 0; i < 1'000'000; ++i) {
      sum += std::sin(i) * std::cos(i);
    }
    benchmark::DoNotOptimize(sum);
  }
}

//------------------------------------------------------------------------------
// Graph BFS Benchmark
//------------------------------------------------------------------------------
static void BM_GraphBFS(benchmark::State& state) {
  // Pin the thread to the core specified by the range (arg).
  int core_id = static_cast<int>(state.range(0));
  try {
    bind_thread_to_cores({core_id});
  } catch (const std::exception& ex) {
    state.SkipWithError(
        ("Failed to pin to core " + std::to_string(core_id) + ": " + ex.what()).c_str());
    return;
  } catch (...) {
    state.SkipWithError(
        ("Failed to pin to core " + std::to_string(core_id) + " with unknown error").c_str());
    return;
  }

  // If pinning succeeded, proceed with the BFS.
  for (auto _ : state) {
    std::vector<bool> visited(NUM_NODES, false);
    std::queue<int> q;
    q.push(0);
    visited[0] = true;

    int visited_count = 0;

    while (!q.empty()) {
      int node = q.front();
      q.pop();
      visited_count++;

      for (int neighbor : g_adjacency_list[node]) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          q.push(neighbor);
        }
      }
    }
    benchmark::DoNotOptimize(visited_count);
  }
}

//------------------------------------------------------------------------------
// main() - Register benchmarks for each core
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // Initialize our graph data once
  InitializeGraph();

  const auto num_cores = std::thread::hardware_concurrency();

  // Register all HeavyFloat benchmarks first
  for (unsigned int core_id = 0; core_id < num_cores; ++core_id) {
    benchmark::RegisterBenchmark(("HeavyFloat/CoreID" + std::to_string(core_id)).c_str(),
                                 BM_HeavyFloatingPoint)
        ->Arg(core_id)
        ->Unit(benchmark::kMillisecond);
  }

  // Then register all GraphBFS benchmarks
  for (unsigned int core_id = 0; core_id < num_cores; ++core_id) {
    benchmark::RegisterBenchmark(("GraphBFS/CoreID" + std::to_string(core_id)).c_str(), BM_GraphBFS)
        ->Arg(core_id)
        ->Unit(benchmark::kMillisecond);
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
