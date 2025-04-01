#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "base_appdata.hpp"
#include "common/vulkan/engine.hpp"

// ----------------------------------------------------------------------------
// globals
// ----------------------------------------------------------------------------

std::string device_id;

struct LocalPushConstants {
  uint32_t g_num_elements;
};

struct GlobalPushConstants {
  uint32_t g_num_blocks;
};

class Singleton {
 public:
  // Delete copy constructor and assignment operator to prevent copies
  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

  static Singleton &getInstance() {
    static Singleton instance;
    return instance;
  }

  vulkan::Engine &get_engine() { return engine; }
  const vulkan::Engine &get_engine() const { return engine; }

 private:
  Singleton() { spdlog::info("Singleton instance created."); }
  ~Singleton() { spdlog::info("Singleton instance destroyed."); }

  vulkan::Engine engine;
};

class VK_Misc : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State &) override {}

  void TearDown(benchmark::State &) override {}
};

BENCHMARK_DEFINE_F(VK_Misc, PrefixSum)
(benchmark::State &state) {
  const uint32_t n = state.range(0);
  const uint32_t n_blocks = (n + 255) / 256;

  auto &engine = Singleton::getInstance().get_engine();

  auto mr = engine.get_mr();

  std::pmr::vector<uint32_t> u_elements_in(n, mr);
  std::pmr::vector<uint32_t> u_elements_out(n, mr);
  std::pmr::vector<uint32_t> u_sums(n_blocks, mr);
  std::pmr::vector<uint32_t> u_prefix_sums(n_blocks, mr);

  std::ranges::fill(u_elements_in, 1);

  auto local_inclusive_scan = engine
                                  .algorithm("tmp_local_inclusive_scan.comp",
                                             {
                                                 engine.get_buffer(u_elements_in.data()),
                                                 engine.get_buffer(u_elements_out.data()),
                                                 engine.get_buffer(u_sums.data()),
                                             })
                                  ->set_push_constants<LocalPushConstants>({
                                      .g_num_elements = n,
                                  })
                                  ->build();

  auto global_exclusive_scan = engine
                                   .algorithm("tmp_global_exclusive_scan.comp",
                                              {
                                                  engine.get_buffer(u_sums.data()),
                                                  engine.get_buffer(u_prefix_sums.data()),
                                              })
                                   ->set_push_constants<GlobalPushConstants>({
                                       .g_num_blocks = n_blocks,
                                   })
                                   ->build();

  auto add_base = engine
                      .algorithm("tmp_add_base.comp",
                                 {
                                     engine.get_buffer(u_elements_out.data()),
                                     engine.get_buffer(u_prefix_sums.data()),
                                 })
                      ->set_push_constants<LocalPushConstants>({
                          .g_num_elements = n,
                      })
                      ->build();

  auto seq = engine.sequence();

  for (auto _ : state) {
    seq->record_commands_with_blocks(local_inclusive_scan.get(), n_blocks);
    seq->launch_kernel_async();
    seq->sync();

    seq->record_commands_with_blocks(global_exclusive_scan.get(), 1);
    seq->launch_kernel_async();
    seq->sync();

    seq->record_commands_with_blocks(add_base.get(), n_blocks);
    seq->launch_kernel_async();
    seq->sync();
  }
}

BENCHMARK_REGISTER_F(VK_Misc, PrefixSum)
    ->Args({1024})
    ->Args({64 * 1024})
    ->Args({640 * 480})
    ->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

static void BM_PrefixSumCPU(benchmark::State &state) {
  const auto n = state.range(0);
  std::vector<uint32_t> elements(n, 1);
  std::vector<uint32_t> prefix_sums(n);

  for (auto _ : state) {
    std::partial_sum(elements.begin(), elements.end(), prefix_sums.begin());
  }
}

BENCHMARK(BM_PrefixSumCPU)
    ->Args({1024})
    ->Args({64 * 1024})
    ->Args({640 * 480})
    ->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------
int main(int argc, char **argv) {
  CLI::App app{"default"};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.allow_extras();

  spdlog::set_level(spdlog::level::off);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}