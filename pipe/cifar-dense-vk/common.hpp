#pragma once

#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
#include "builtin-apps/cifar-dense/vulkan/dispatchers.hpp"
#include "builtin-apps/pipeline/task.hpp"
#include "builtin-apps/pipeline/worker.hpp"

using MyTask = Task<cifar_dense::v2::AppData>;

#define SETUP_DATA                                                            \
  cifar_dense::vulkan::v2::VulkanDispatcher disp;                             \
  std::vector<std::unique_ptr<MyTask>> preallocated_tasks;                    \
  SPSCQueue<MyTask*, kPoolSize> q_0;                                          \
  SPSCQueue<MyTask*, kPoolSize> q_1;                                          \
  SPSCQueue<MyTask*, kPoolSize> q_2;                                          \
  SPSCQueue<MyTask*, kPoolSize> q_3;                                          \
  for (size_t i = 0; i < kPoolSize; ++i) {                                    \
    preallocated_tasks.emplace_back(std::make_unique<MyTask>(disp.get_mr())); \
    q_0.enqueue(preallocated_tasks.back().get());                             \
  }

template <typename QueueT>
[[nodiscard]]
std::thread create_thread(
    QueueT& in, QueueT& out, const std::vector<int>& cores, const int start, const int end) {
  return std::thread(
      worker_thread<MyTask>, std::ref(in), std::ref(out), [&cores, start, end](MyTask& task) {
        cifar_dense::omp::v2::dispatch_multi_stage(cores, cores.size(), task.appdata, start, end);
      });
}

template <typename QueueT>
[[nodiscard]]
std::thread create_thread(QueueT& in,
                          QueueT& out,
                          cifar_dense::vulkan::v2::VulkanDispatcher& disp,
                          const int start,
                          const int end) {
  return std::thread(
      worker_thread<MyTask>, std::ref(in), std::ref(out), [&disp, start, end](MyTask& task) {
        disp.dispatch_multi_stage(task.appdata, start, end);
      });
}

template <typename QueueT>
[[nodiscard]]
std::thread create_thread_record(const int i,
                                 QueueT& in,
                                 QueueT& out,
                                 const std::vector<int>& cores,
                                 const int start,
                                 const int end) {
  return std::thread(worker_thread_record<MyTask>,
                     i,
                     std::ref(in),
                     std::ref(out),
                     [&cores, start, end](MyTask& task) {
                       cifar_dense::omp::v2::dispatch_multi_stage(
                           cores, cores.size(), task.appdata, start, end);
                     });
}

template <typename QueueT>
[[nodiscard]]
std::thread create_thread_record(const int i,
                                 QueueT& in,
                                 QueueT& out,
                                 cifar_dense::vulkan::v2::VulkanDispatcher& disp,
                                 const int start,
                                 const int end) {
  return std::thread(
      worker_thread_record<MyTask>,
      i,
      std::ref(in),
      std::ref(out),
      [&disp, start, end](MyTask& task) { disp.dispatch_multi_stage(task.appdata, start, end); });
}