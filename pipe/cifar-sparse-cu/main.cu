#include <thread>

#include "builtin-apps/app.hpp"
#include "const.hpp"

void run(const std::vector<AppDataPtr>& data, DispatcherT& disp) {
  QueueT q0;
  QueueT q1;

  for (const auto& item : data) {
    q0.enqueue(item.get());
  }

  std::thread t0(
      worker<QueueT>,
      std::ref(q0),
      std::ref(q1),
      [](AppDataT* app) { cifar_sparse::omp::dispatch_stage(*app, 1); },
      kNumToProcess,
      false);

  std::thread t1(
      worker<QueueT>,
      std::ref(q1),
      std::ref(q0),
      [&disp](AppDataT* app) { disp.dispatch_stage(*app, 2); },
      kNumToProcess,
      true);

  t0.join();
  t1.join();
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  DispatcherT disp;

  const std::vector<AppDataPtr> dataset = make_dataset(disp, 10);

  run(dataset, disp);

  spdlog::info("Done with vector");
  return 0;
}