#pragma once
// ---------------------------------------------------------------------------
// bm_demo_common -- the two-stage concurrent-pipeline DEMO runner shared by every
// <app>-cu/main.cu. It builds a small dataset, wires a stage-1 (OMP) worker into
// a stage-2 (GPU) worker through the SPSC ring, and joins. The ONLY per-app
// variation is the stage-1 OMP token (<app>::omp::dispatch_stage), which the cell
// passes as a callable; everything else is identical across the 3 CUDA cells.
// ---------------------------------------------------------------------------

#include <spdlog/spdlog.h>

#include <thread>
#include <vector>

#include "platform/registry/device_registry.hpp"

namespace bt_demo {

// Disp/App/AppPtr/Queue come from the cell's const.hpp; `stage1` is the app's
// OMP stage-1 token (e.g. [](App* a){ tree::omp::dispatch_stage(*a, 1); }).
template <class Disp, class App, class AppPtr, class Queue, class Stage1>
void run(const std::vector<AppPtr>& data, Disp& disp, size_t num_to_process, Stage1 stage1) {
  Queue q0;
  Queue q1;

  for (const auto& item : data) {
    q0.enqueue(item.get());
  }

  std::thread t0(worker<Queue, App>, std::ref(q0), std::ref(q1), stage1, num_to_process, false);

  std::thread t1(
      worker<Queue, App>,
      std::ref(q1),
      std::ref(q0),
      [&disp](App* app) { disp.dispatch_stage(*app, 2); },
      num_to_process,
      true);

  t0.join();
  t1.join();
}

template <class Disp, class App, class AppPtr, class Queue, class Stage1>
int run_main(int argc, char** argv, size_t num_to_process, Stage1 stage1) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  Disp disp;

  const std::vector<AppPtr> dataset = make_dataset<App>(disp, 10);

  run<Disp, App, AppPtr, Queue>(dataset, disp, num_to_process, stage1);

  spdlog::info("Done with vector");
  return 0;
}

}  // namespace bt_demo
