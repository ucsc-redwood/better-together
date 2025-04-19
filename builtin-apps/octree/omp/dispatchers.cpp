#include "dispatchers.hpp"

#include "../../app.hpp"
#include "../../debug_logger.hpp"
#include "kernels.hpp"
#include "sort.hpp"

namespace octree::omp {

// ----------------------------------------------------------------------------
// Stage 1 (compute morton codes)
// ----------------------------------------------------------------------------

void run_stage_1(AppData &app) {
  LOG_KERNEL(LogKernelType::kOMP, 1, &app);

  compute_morton_codes_with_range(app.u_positions.data(),
                                  app.n,
                                  glm::vec3(kMinCoord, kMinCoord, kMinCoord),
                                  glm::vec3(kMaxCoord, kMaxCoord, kMaxCoord),
                                  app.u_morton_codes_alt.data());

#pragma omp barrier
}

// ----------------------------------------------------------------------------
// Stage 2 (sort morton codes)
// ----------------------------------------------------------------------------

void run_stage_2(AppData &app) {
  LOG_KERNEL(LogKernelType::kOMP, 2, &app);

  // Get the number of threads in the current parallel region
  const int num_threads = omp_get_num_threads();

  // Static workspace vectors to avoid repeated allocations
  // These need to be properly synchronized when accessed by multiple threads
  static std::vector<size_t> local_hist;
  static std::vector<size_t> local_offset;
  static std::vector<size_t> global_hist;
  static std::vector<size_t> prefix;
  static int last_num_threads = 0;
  static std::vector<uint32_t> buffer_in;
  static std::vector<uint32_t> buffer_out;

  // Thread-safe buffer management - single thread manages shared resources
#pragma omp single
  {
    // Only resize if the number of threads has changed
    if (num_threads != last_num_threads) {
      last_num_threads = num_threads;
      local_hist.resize(num_threads * RADIX);
      local_offset.resize(num_threads * RADIX);
      global_hist.resize(RADIX);
      prefix.resize(RADIX);
      spdlog::debug("Resized radix sort buffers for {} threads", num_threads);
    }

    // Resize input/output buffers if necessary
    if (buffer_in.size() < app.n) {
      buffer_in.resize(app.n);
      buffer_out.resize(app.n);
      spdlog::debug("Resized input/output buffers to size {}", app.n);
    }

    // Zero out the workspace vectors
    std::fill(local_hist.begin(), local_hist.end(), 0);
    std::fill(local_offset.begin(), local_offset.end(), 0);
    std::fill(global_hist.begin(), global_hist.end(), 0);
    std::fill(prefix.begin(), prefix.end(), 0);

    // Copy the input data (single-threaded to avoid race conditions)
    std::ranges::copy(
        app.u_morton_codes_alt.begin(), app.u_morton_codes_alt.begin() + app.n, buffer_in.begin());
  }
  // Implicit barrier at the end of the omp single region

  // Perform the sort with all threads
  radix_sort_in_parallel(buffer_in.data(),
                         buffer_out.data(),
                         app.n,
                         local_hist.data(),
                         local_offset.data(),
                         global_hist.data(),
                         prefix.data());

  // Ensure all threads have completed the sort before copying results
#pragma omp barrier

  // One thread copies the result back to app data to avoid race conditions
#pragma omp single
  {
    std::ranges::copy(buffer_out.begin(), buffer_out.begin() + app.n, app.u_morton_codes.begin());
  }

  // Final barrier to ensure all threads see the updated app.u_morton_codes
#pragma omp barrier
}

// ----------------------------------------------------------------------------
// Stage 3 (unique morton codes)
// ----------------------------------------------------------------------------

void run_stage_3(AppData &app) {
  LOG_KERNEL(LogKernelType::kOMP, 3, &app);

#pragma omp single
  {
    auto end_it = std::unique(app.u_morton_codes.begin(), app.u_morton_codes.begin() + app.n);
    app.m = std::distance(app.u_morton_codes.begin(), end_it);

    // print m/n
    spdlog::info("m/n: {}/{}", app.m, app.n);

    assert(size_t(app.m) <= app.n);
  }

#pragma omp barrier
}

// ----------------------------------------------------------------------------
// Stage 4 (build radix tree)
// ----------------------------------------------------------------------------

void run_stage_4(AppData &app) {
  LOG_KERNEL(LogKernelType::kOMP, 4, &app);

  build_radix_tree(app.u_morton_codes.data(),
                   app.m,
                   app.u_parents.data(),
                   app.u_left_child.data(),
                   app.u_has_leaf_left.data(),
                   app.u_has_leaf_right.data(),
                   app.u_prefix_length.data());

#pragma omp barrier
}

// ----------------------------------------------------------------------------
// Stage 5 (edge count)
// ----------------------------------------------------------------------------

void run_stage_5(AppData &app) {
  LOG_KERNEL(LogKernelType::kOMP, 5, &app);

  assert(app.m != std::numeric_limits<size_t>::max());

  compute_edge_count_kernel(app.u_morton_codes.data(),
                            app.m,
                            app.u_left_child.data(),
                            app.u_prefix_length.data(),
                            app.u_edge_count.data());

#pragma omp barrier
}

// ----------------------------------------------------------------------------
// Stage 6 (offsets)
// ----------------------------------------------------------------------------

void run_stage_6(AppData &app) {
  LOG_KERNEL(LogKernelType::kOMP, 6, &app);

#pragma omp single
  {
    app.u_offsets[0] = 0;
    for (size_t i = 1; i < app.m; ++i) {
      app.u_offsets[i] = app.u_offsets[i - 1] + app.u_edge_count[i - 1];
    }
    app.total_children = app.u_offsets[app.m - 1] + app.u_edge_count[app.m - 1];
    // app.n_octree_nodes = size_t(1 + app.total_children);
  }
#pragma omp barrier
}

void run_stage_7(AppData &app) {
  LOG_KERNEL(LogKernelType::kOMP, 7, &app);

  build_octree_nodes_kernel(app.u_morton_codes.data(),
                            app.m,
                            app.u_left_child.data(),
                            app.u_prefix_length.data(),
                            app.u_edge_count.data(),
                            app.u_offsets.data(),
                            app.u_children.data());
}

}  // namespace octree::omp
