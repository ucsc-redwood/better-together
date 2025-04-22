#include <thread>

#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
#include "const.hpp"

[[nodiscard]] std::vector<AppDataPtr> make_dataset(DispatcherT& disp,
                                                   const size_t num_items = kNumToProcess) {
  std::vector<AppDataPtr> result;
  result.reserve(num_items);

  for (size_t i = 0; i < num_items; ++i) {
    auto app = std::make_shared<cifar_dense::AppData>(&disp.get_mr());
    result.push_back(app);
  }

  return result;
}

template <typename T>
std::queue<T> make_queue_from_vector(const std::vector<T>& vec) {
  return std::queue<T>(std::deque<T>(vec.begin(), vec.end()));
}

// template <typename TaskT, size_t kPoolSize, size_t kNumToProcess>

void worker(SPSCQueue<AppDataPtr, kPoolSize>& q_in,
            SPSCQueue<AppDataPtr, kPoolSize>& q_out,
            const bool is_last = false) {
  for (size_t i = 0; i < kNumToProcess; ++i) {
    AppDataPtr app;
    while (!q_in.dequeue(app)) {
      std::this_thread::yield();
    }

    if (app == nullptr) {
      throw std::runtime_error("App is nullptr");
    }

    // ------------------------------------------------------------------------
    spdlog::info(
        "Processing idx {}, uid {}, initial_uid {}", i, app->get_uid(), app->get_initial_uid());
    // cifar_dense::omp::dispatch_stage(*app, 1);
    // ------------------------------------------------------------------------

    if (is_last) {
      app->reset();
    }

    while (!q_out.enqueue(app)) {
      std::this_thread::yield();
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  DispatcherT disp;

  const std::vector<AppDataPtr> data = make_dataset(disp, 10);

  SPSCQueue<AppDataPtr, kPoolSize> q0;
  SPSCQueue<AppDataPtr, kPoolSize> q1;

  for (const auto& item : data) {
    q0.enqueue(item);
  }

  worker(q0, q0, true);

  spdlog::info("Done with vector");
  return 0;
}