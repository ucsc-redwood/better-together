#include "builtin-apps/app.hpp"
#include "const.hpp"

[[nodiscard]] std::vector<AppDataPtr> make_dataset(DispatcherT& disp, const size_t num_items) {
  std::vector<AppDataPtr> result;
  result.reserve(num_items);

  for (int i = 0; i < num_items; ++i) {
    auto app = std::make_shared<cifar_dense::AppData>(&disp.get_mr());
    result.push_back(app);
  }

  return result;
}

template <typename T>
std::queue<T> make_queue_from_vector(const std::vector<T>& vec) {
  return std::queue<T>(std::deque<T>(vec.begin(), vec.end()));
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  DispatcherT disp;

  auto appdatas = make_dataset(disp, 10);

  SPSCQueue<AppDataPtr, 1024> q;

  for (auto & app : appdatas) {
    q.enqueue(app);
  }

  spdlog::info("Done with vector");
  return 0;
}