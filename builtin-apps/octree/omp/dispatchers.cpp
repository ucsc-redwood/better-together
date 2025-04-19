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

  // –– 1) Static workspace (un‑sized on declaration) ––//
  static std::vector<size_t> local_hist;
  static std::vector<size_t> local_offset;
  static std::vector<size_t> global_hist;
  static std::vector<size_t> prefix;

  // –– 2) One‑time allocation & sizing ––//
  //      We only ever allocate when sizes change.

  const auto num_threads = omp_get_num_threads();
  static int last_threads = 0;
  if (last_threads != num_threads) {
    last_threads = num_threads;
    local_hist.assign(num_threads * RADIX, 0);
    local_offset.assign(num_threads * RADIX, 0);
    global_hist.assign(RADIX, 0);
    prefix.assign(RADIX, 0);
  } else {
    // –– 3) Zero out existing buffers before EACH call ––//
    std::ranges::fill(local_hist, 0);
    std::ranges::fill(local_offset, 0);
    std::ranges::fill(global_hist, 0);
    std::ranges::fill(prefix, 0);
  }

  radix_sort_in_parallel(app.u_morton_codes_alt.data(),
                         app.u_morton_codes.data(),
                         app.n,
                         local_hist.data(),
                         local_offset.data(),
                         global_hist.data(),
                         prefix.data());

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

    assert(size_t(app.m) <= app.reserved_n);
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

  assert(app.n_brt_nodes != std::numeric_limits<size_t>::max());

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
