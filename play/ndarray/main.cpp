#include "builtin-apps/app.hpp"
#include "omp/dispatchers.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  auto mr = std::pmr::new_delete_resource();

  cifar_dense::AppDataBatch batched_appdata(mr);

  omp::dispatch_multi_stage<ProcessorType::kLittleCore>(4, batched_appdata, 1, 9);

  cifar_dense::print_batch_predictions(batched_appdata);

  return 0;
}
