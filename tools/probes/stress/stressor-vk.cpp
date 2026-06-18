#include <chrono>

#include "platform/engine/vulkan/engine.hpp"

int main() {
  kiss_vk::Engine engine;

  constexpr auto n = 1024;
  std::pmr::vector<float> input(n, engine.get_mr());

  struct Ps {
    uint n;
    float time_limit;
    float start_time;
  } pc{
      .n = n,
      .time_limit = 1.0f,
      .start_time = 0.0f,
  };

  auto algo = engine.make_algo("stressor")
                  ->work_group_size(256, 1, 1)
                  ->num_sets(1)
                  ->num_buffers(1)
                  ->push_constant_size(12)
                  ->build();

  algo->update_push_constant(pc);

  algo->update_descriptor_set(0, {engine.get_buffer_info(input)});

  auto seq = engine.make_seq();

  // Start timing
  auto start_time = std::chrono::high_resolution_clock::now();
  constexpr double total_duration = 10.0;  // 10 seconds
  int iteration_count = 0;

  spdlog::info("Starting GPU stress test for {} seconds...", total_duration);

  while (true) {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(current_time - start_time).count();

    if (elapsed >= total_duration) {
      break;
    }

    // Update push constants with current time
    pc.start_time = static_cast<float>(elapsed);
    algo->update_push_constant(pc);

    // Record and submit the kernel
    seq->cmd_begin();
    algo->record_bind_core(seq->get_handle(), 0);
    algo->record_bind_push(seq->get_handle());
    algo->record_dispatch(seq->get_handle(), {kiss_vk::div_ceil(n, 256), 1, 1});
    seq->cmd_end();

    seq->submit();
    seq->wait_for_fence();
    seq->reset_fence();

    iteration_count++;

    // Log progress every second
    if (iteration_count % 10 == 0) {
      spdlog::info("Stress test running... {:.1f}s elapsed, {} iterations completed",
                   elapsed,
                   iteration_count);
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_elapsed = std::chrono::duration<double>(end_time - start_time).count();

  spdlog::info("GPU stress test completed!");
  spdlog::info("Total time: {:.2f} seconds", total_elapsed);
  spdlog::info("Total iterations: {}", iteration_count);
  spdlog::info("Average iteration time: {:.3f} ms", (total_elapsed * 1000.0) / iteration_count);

  return 0;
}