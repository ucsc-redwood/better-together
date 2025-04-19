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

  // const auto num_threads = omp_get_num_threads();
  // const int tid = omp_get_thread_num();
  // ::tree::omp::parallel_sort(app.u_morton_codes, app.u_morton_codes, tid, num_threads);

  


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
