// Schedule executor (emits the per-task `### Python ###` timing log) for CIFAR-sparse
// x VK. The warmup + run + main() live in ../bm_gen_log_common.hpp; this file supplies
// the cell types (const.hpp first), the GPU ExecutionModel, and the OMP dispatch.
#include "const.hpp"

#include "../bm_gen_log_common.hpp"

int main(int argc, char** argv) {
  return bt_gen_log::run(
      argc, argv, ExecutionModel::kVulkan,
      [](const std::vector<int>& cores, size_t n, AppDataT& app, int lo, int hi) {
        cifar_sparse::omp::dispatch_multi_stage(cores, n, app, lo, hi);
      });
}
