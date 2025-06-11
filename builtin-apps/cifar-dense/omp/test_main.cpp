#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <queue>
#include <thread>

#include "../../app.hpp"
#include "../../pipeline/spsc_queue.hpp"
#include "../appdata.hpp"
#include "dispatchers.hpp"

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

constexpr int kTestBatchSize = 128;

TEST(Stage1Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv1_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv1_out.d1(), 16);
  EXPECT_EQ(appdata.u_conv1_out.d2(), 32);
  EXPECT_EQ(appdata.u_conv1_out.d3(), 32);

  // Check if the output buffers value is different before calling the kernel
  const std::vector<float> conv1_out_before(appdata.u_conv1_out.pmr_vec().begin(),
                                            appdata.u_conv1_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 1)) << "Stage 1 should not throw";

  const std::vector<float> conv1_out_after(appdata.u_conv1_out.pmr_vec().begin(),
                                           appdata.u_conv1_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < conv1_out_before.size(); ++i) {
    if (conv1_out_before[i] != conv1_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 2
// ----------------------------------------------------------------------------

TEST(Stage2Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 1));

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool1_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_pool1_out.d1(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d2(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d3(), 16);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> pool1_out_before(appdata.u_pool1_out.pmr_vec().begin(),
                                            appdata.u_pool1_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 2)) << "Stage 2 should not throw";

  const std::vector<float> pool1_out_after(appdata.u_pool1_out.pmr_vec().begin(),
                                           appdata.u_pool1_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < pool1_out_before.size(); ++i) {
    if (pool1_out_before[i] != pool1_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 3
// ----------------------------------------------------------------------------

TEST(Stage3Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(appdata, 1, 2));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv2_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv2_out.d1(), 32);
  EXPECT_EQ(appdata.u_conv2_out.d2(), 16);
  EXPECT_EQ(appdata.u_conv2_out.d3(), 16);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv2_out_before(appdata.u_conv2_out.pmr_vec().begin(),
                                            appdata.u_conv2_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 3)) << "Stage 3 should not throw";

  const std::vector<float> conv2_out_after(appdata.u_conv2_out.pmr_vec().begin(),
                                           appdata.u_conv2_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < conv2_out_before.size(); ++i) {
    if (conv2_out_before[i] != conv2_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 4
// ----------------------------------------------------------------------------

TEST(Stage4Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(appdata, 1, 3));

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool2_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_pool2_out.d1(), 32);
  EXPECT_EQ(appdata.u_pool2_out.d2(), 8);
  EXPECT_EQ(appdata.u_pool2_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> pool2_out_before(appdata.u_pool2_out.pmr_vec().begin(),
                                            appdata.u_pool2_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 4)) << "Stage 4 should not throw";

  const std::vector<float> pool2_out_after(appdata.u_pool2_out.pmr_vec().begin(),
                                           appdata.u_pool2_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < pool2_out_before.size(); ++i) {
    if (pool2_out_before[i] != pool2_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 5
// ----------------------------------------------------------------------------

TEST(Stage5Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(appdata, 1, 4));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv3_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv3_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv3_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv3_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv3_out_before(appdata.u_conv3_out.pmr_vec().begin(),
                                            appdata.u_conv3_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 5)) << "Stage 5 should not throw";

  const std::vector<float> conv3_out_after(appdata.u_conv3_out.pmr_vec().begin(),
                                           appdata.u_conv3_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < conv3_out_before.size(); ++i) {
    if (conv3_out_before[i] != conv3_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 6
// ----------------------------------------------------------------------------

TEST(Stage6Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(appdata, 1, 5));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv4_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv4_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv4_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv4_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv4_out_before(appdata.u_conv4_out.pmr_vec().begin(),
                                            appdata.u_conv4_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 6)) << "Stage 6 should not throw";

  const std::vector<float> conv4_out_after(appdata.u_conv4_out.pmr_vec().begin(),
                                           appdata.u_conv4_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < conv4_out_before.size(); ++i) {
    if (conv4_out_before[i] != conv4_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 7
// ----------------------------------------------------------------------------

TEST(Stage7Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(appdata, 1, 6));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv5_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv5_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv5_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv5_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv5_out_before(appdata.u_conv5_out.pmr_vec().begin(),
                                            appdata.u_conv5_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 7)) << "Stage 7 should not throw";

  const std::vector<float> conv5_out_after(appdata.u_conv5_out.pmr_vec().begin(),
                                           appdata.u_conv5_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < conv5_out_before.size(); ++i) {
    if (conv5_out_before[i] != conv5_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 8
// ----------------------------------------------------------------------------

TEST(Stage8Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(appdata, 1, 7));

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool3_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_pool3_out.d1(), 64);
  EXPECT_EQ(appdata.u_pool3_out.d2(), 4);
  EXPECT_EQ(appdata.u_pool3_out.d3(), 4);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> pool3_out_before(appdata.u_pool3_out.pmr_vec().begin(),
                                            appdata.u_pool3_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 8)) << "Stage 8 should not throw";

  const std::vector<float> pool3_out_after(appdata.u_pool3_out.pmr_vec().begin(),
                                           appdata.u_pool3_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < pool3_out_before.size(); ++i) {
    if (pool3_out_before[i] != pool3_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 9
// ----------------------------------------------------------------------------

TEST(Stage9Test, Basic) {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData appdata(mr);

  // Run previous stages
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(appdata, 1, 8));

  // Check output dimensions
  EXPECT_EQ(appdata.u_linear_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_linear_out.d1(), 10);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> linear_out_before(appdata.u_linear_out.pmr_vec().begin(),
                                             appdata.u_linear_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(cifar_dense::omp::dispatch_stage(appdata, 9)) << "Stage 9 should not throw";

  const std::vector<float> linear_out_after(appdata.u_linear_out.pmr_vec().begin(),
                                            appdata.u_linear_out.pmr_vec().end());

  bool is_different = false;
  for (size_t i = 0; i < linear_out_before.size(); ++i) {
    if (linear_out_before[i] != linear_out_after[i]) {
      is_different = true;
      break;
    }
  }

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// Test Queue environment
// ----------------------------------------------------------------------------

TEST(QueueTest, Basic) {
  auto mr = std::pmr::new_delete_resource();

  std::vector<std::shared_ptr<cifar_dense::AppData>> appdatas;
  appdatas.reserve(10);
  for (int i = 0; i < 10; i++) {
    appdatas.push_back(std::make_shared<cifar_dense::AppData>(mr));
  }

  std::queue<std::shared_ptr<cifar_dense::AppData>> queue;
  for (auto& appdata : appdatas) {
    queue.push(appdata);
  }

  while (!queue.empty()) {
    auto appdata = queue.front();
    queue.pop();

    EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(*appdata, 1, 9));
  }

  EXPECT_TRUE(queue.empty());
}

// ----------------------------------------------------------------------------
// Test Concurrent Queue environment
// ----------------------------------------------------------------------------

TEST(ConcurrentQueueTest, Basic) {
  auto mr = std::pmr::new_delete_resource();

  std::vector<std::shared_ptr<cifar_dense::AppData>> appdatas;
  appdatas.reserve(10);
  for (int i = 0; i < 10; i++) {
    appdatas.push_back(std::make_shared<cifar_dense::AppData>(mr));
  }

  SPSCQueue<std::shared_ptr<cifar_dense::AppData>> cpu_queue1;
  SPSCQueue<std::shared_ptr<cifar_dense::AppData>> cpu_queue2;

  // Initial push to first CPU queue
  for (auto& appdata : appdatas) {
    cpu_queue1.enqueue(std::move(appdata));
  }

  // Producer thread - first half processing
  std::thread producer([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_dense::AppData> appdata;
      while (!cpu_queue1.dequeue(appdata)) {
        std::this_thread::yield();
      }

      EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(*appdata, 1, 4));

      while (!cpu_queue2.enqueue(std::move(appdata))) {
        std::this_thread::yield();
      }
    }
  });

  // Consumer thread - second half processing
  std::thread consumer([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_dense::AppData> appdata;
      while (!cpu_queue2.dequeue(appdata)) {
        std::this_thread::yield();
      }

      // Process second half (stages 5-9)
      EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(*appdata, 5, 9));
    }
  });

  producer.join();
  consumer.join();
}

TEST(CifarDenseTest, InterleavedConcurrentStages) {
  auto mr = std::pmr::new_delete_resource();

  // Create queues for inter-thread communication
  SPSCQueue<std::shared_ptr<cifar_dense::AppData>> cpu_queue1;
  SPSCQueue<std::shared_ptr<cifar_dense::AppData>> cpu_queue2;
  SPSCQueue<std::shared_ptr<cifar_dense::AppData>> cpu_queue3;
  SPSCQueue<std::shared_ptr<cifar_dense::AppData>> cpu_queue4;

  // Create test data
  std::vector<std::shared_ptr<cifar_dense::AppData>> appdatas;
  appdatas.reserve(10);
  for (int i = 0; i < 10; i++) {
    appdatas.push_back(std::make_shared<cifar_dense::AppData>(mr));
  }

  // Initial push to first CPU queue
  for (auto& appdata : appdatas) {
    cpu_queue1.enqueue(std::move(appdata));
  }

  // Thread 1 - processes stages 1-2
  std::thread thread1([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_dense::AppData> appdata;
      while (!cpu_queue1.dequeue(appdata)) {
        std::this_thread::yield();
      }
      EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(*appdata, 1, 2));
      while (!cpu_queue2.enqueue(std::move(appdata))) {
        std::this_thread::yield();
      }
    }
  });

  // Thread 2 - processes stages 3-4
  std::thread thread2([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_dense::AppData> appdata;
      while (!cpu_queue2.dequeue(appdata)) {
        std::this_thread::yield();
      }
      EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(*appdata, 3, 4));
      while (!cpu_queue3.enqueue(std::move(appdata))) {
        std::this_thread::yield();
      }
    }
  });

  // Thread 3 - processes stages 5-6
  std::thread thread3([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_dense::AppData> appdata;
      while (!cpu_queue3.dequeue(appdata)) {
        std::this_thread::yield();
      }
      EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(*appdata, 5, 6));
      while (!cpu_queue4.enqueue(std::move(appdata))) {
        std::this_thread::yield();
      }
    }
  });

  // Thread 4 - processes stages 7-9
  std::thread thread4([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_dense::AppData> appdata;
      while (!cpu_queue4.dequeue(appdata)) {
        std::this_thread::yield();
      }
      EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(*appdata, 7, 9));
    }
  });

  // Wait for all threads to complete
  thread1.join();
  thread2.join();
  thread3.join();
  thread4.join();
}

int main(int argc, char** argv) {
  // Initialize Google Test
  ::testing::InitGoogleTest(&argc, argv);

  // Parse command-line arguments
  parse_args(argc, argv);

  // Set logging level to off
  spdlog::set_level(spdlog::level::off);

  // Run the tests
  return RUN_ALL_TESTS();
}
