#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <queue>

#include "../../app.hpp"
#include "../../pipeline/spsc_queue.hpp"
#include "../appdata.hpp"
#include "../omp/dispatchers.hpp"
#include "dispatchers.cuh"

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

constexpr int kTestBatchSize = 128;

TEST(Stage1Test, Basic) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv1_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv1_out.d1(), 16);
  EXPECT_EQ(appdata.u_conv1_out.d2(), 32);
  EXPECT_EQ(appdata.u_conv1_out.d3(), 32);

  // Check if the output buffers value is different before calling the kernel

  const std::vector<float> conv1_out_before(appdata.u_conv1_out.pmr_vec().begin(),
                                            appdata.u_conv1_out.pmr_vec().end());

  // Check No throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1)) << "Stage 1 should not throw";

  const std::vector<float> conv1_out_after(appdata.u_conv1_out.pmr_vec().begin(),
                                           appdata.u_conv1_out.pmr_vec().end());

  const bool is_different = !std::ranges::equal(conv1_out_before, conv1_out_after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 2
// ----------------------------------------------------------------------------

TEST(Stage2Test, Basic) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run previous stages
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool1_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_pool1_out.d1(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d2(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d3(), 16);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> pool1_out_before(appdata.u_pool1_out.pmr_vec().begin(),
                                            appdata.u_pool1_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2)) << "Stage 2 should not throw";

  const std::vector<float> pool1_out_after(appdata.u_pool1_out.pmr_vec().begin(),
                                           appdata.u_pool1_out.pmr_vec().end());

  const bool is_different = !std::ranges::equal(pool1_out_before, pool1_out_after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 3
// ----------------------------------------------------------------------------

TEST(Stage3Test, Basic) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run previous stages
  EXPECT_NO_THROW(disp.dispatch_multi_stage(appdata, 1, 2));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv2_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv2_out.d1(), 32);
  EXPECT_EQ(appdata.u_conv2_out.d2(), 16);
  EXPECT_EQ(appdata.u_conv2_out.d3(), 16);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv2_out_before(appdata.u_conv2_out.pmr_vec().begin(),
                                            appdata.u_conv2_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3)) << "Stage 3 should not throw";

  const std::vector<float> conv2_out_after(appdata.u_conv2_out.pmr_vec().begin(),
                                           appdata.u_conv2_out.pmr_vec().end());

  const bool is_different = !std::ranges::equal(conv2_out_before, conv2_out_after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 4
// ----------------------------------------------------------------------------

TEST(Stage4Test, Basic) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run previous stages
  EXPECT_NO_THROW(disp.dispatch_multi_stage(appdata, 1, 3));

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool2_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_pool2_out.d1(), 32);
  EXPECT_EQ(appdata.u_pool2_out.d2(), 8);
  EXPECT_EQ(appdata.u_pool2_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> pool2_out_before(appdata.u_pool2_out.pmr_vec().begin(),
                                            appdata.u_pool2_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 4)) << "Stage 4 should not throw";

  const std::vector<float> pool2_out_after(appdata.u_pool2_out.pmr_vec().begin(),
                                           appdata.u_pool2_out.pmr_vec().end());

  const bool is_different = !std::ranges::equal(pool2_out_before, pool2_out_after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 5
// ----------------------------------------------------------------------------

TEST(Stage5Test, Basic) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run previous stages
  EXPECT_NO_THROW(disp.dispatch_multi_stage(appdata, 1, 4));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv3_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv3_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv3_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv3_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv3_out_before(appdata.u_conv3_out.pmr_vec().begin(),
                                            appdata.u_conv3_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 5)) << "Stage 5 should not throw";

  const std::vector<float> conv3_out_after(appdata.u_conv3_out.pmr_vec().begin(),
                                           appdata.u_conv3_out.pmr_vec().end());

  const bool is_different = !std::ranges::equal(conv3_out_before, conv3_out_after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 6
// ----------------------------------------------------------------------------

TEST(Stage6Test, Basic) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run previous stages
  EXPECT_NO_THROW(disp.dispatch_multi_stage(appdata, 1, 5));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv4_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv4_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv4_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv4_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv4_out_before(appdata.u_conv4_out.pmr_vec().begin(),
                                            appdata.u_conv4_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 6)) << "Stage 6 should not throw";

  const std::vector<float> conv4_out_after(appdata.u_conv4_out.pmr_vec().begin(),
                                           appdata.u_conv4_out.pmr_vec().end());

  const bool is_different = !std::ranges::equal(conv4_out_before, conv4_out_after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 7
// ----------------------------------------------------------------------------

TEST(Stage7Test, Basic) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run previous stages
  EXPECT_NO_THROW(disp.dispatch_multi_stage(appdata, 1, 6));

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv5_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_conv5_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv5_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv5_out.d3(), 8);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> conv5_out_before(appdata.u_conv5_out.pmr_vec().begin(),
                                            appdata.u_conv5_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 7)) << "Stage 7 should not throw";

  const std::vector<float> conv5_out_after(appdata.u_conv5_out.pmr_vec().begin(),
                                           appdata.u_conv5_out.pmr_vec().end());

  const bool is_different = !std::ranges::equal(conv5_out_before, conv5_out_after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 8
// ----------------------------------------------------------------------------

TEST(Stage8Test, Basic) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run previous stages
  EXPECT_NO_THROW(disp.dispatch_multi_stage(appdata, 1, 7));

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool3_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_pool3_out.d1(), 64);
  EXPECT_EQ(appdata.u_pool3_out.d2(), 4);
  EXPECT_EQ(appdata.u_pool3_out.d3(), 4);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> pool3_out_before(appdata.u_pool3_out.pmr_vec().begin(),
                                            appdata.u_pool3_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 8)) << "Stage 8 should not throw";

  const std::vector<float> pool3_out_after(appdata.u_pool3_out.pmr_vec().begin(),
                                           appdata.u_pool3_out.pmr_vec().end());

  const bool is_different = !std::ranges::equal(pool3_out_before, pool3_out_after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// test Stage 9
// ----------------------------------------------------------------------------

TEST(Stage9Test, Basic) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run previous stages
  EXPECT_NO_THROW(disp.dispatch_multi_stage(appdata, 1, 8));

  // Check output dimensions
  EXPECT_EQ(appdata.u_linear_out.d0(), kTestBatchSize);
  EXPECT_EQ(appdata.u_linear_out.d1(), 10);

  // Check if the output buffer value is different before calling the kernel
  const std::vector<float> linear_out_before(appdata.u_linear_out.pmr_vec().begin(),
                                             appdata.u_linear_out.pmr_vec().end());

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 9)) << "Stage 9 should not throw";

  const std::vector<float> linear_out_after(appdata.u_linear_out.pmr_vec().begin(),
                                            appdata.u_linear_out.pmr_vec().end());

  const bool is_different = !std::ranges::equal(linear_out_before, linear_out_after);

  EXPECT_TRUE(is_different) << "Output buffer did not change after dispatch.";
}

// ----------------------------------------------------------------------------
// Test Mixing Omp and Cuda
// ----------------------------------------------------------------------------

TEST(MixingTest, CudaThenOmp) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_2(appdata));
}

TEST(MixingTest, OmpThenCuda) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_1(appdata));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2));
}

TEST(MixingTest, MultipleStages) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run first 3 stages with CUDA
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3));

  // Run next 2 stages with OMP
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_4(appdata));
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_5(appdata));

  // Run final stages with CUDA
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 6));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 7));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 8));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 9));
}

TEST(MixingTest, AlternatingStages) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Alternate between CUDA and OMP for each stage
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_2(appdata));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3));
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_4(appdata));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 5));
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_6(appdata));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 7));
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_8(appdata));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 9));
}

TEST(MixingTest, MixedBatch) {
  cifar_sparse::cuda::CudaDispatcher disp;
  cifar_sparse::AppData appdata(&disp.get_mr());

  // Run first half with CUDA
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 4));

  // Run second half with OMP
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_5(appdata));
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_6(appdata));
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_7(appdata));
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_8(appdata));
  EXPECT_NO_THROW(cifar_sparse::omp::run_stage_9(appdata));
}

// ----------------------------------------------------------------------------
// Test Queue environment
// ----------------------------------------------------------------------------

TEST(QueueTest, Basic) {
  cifar_sparse::cuda::CudaDispatcher disp;

  std::vector<std::shared_ptr<cifar_sparse::AppData>> appdatas;
  appdatas.reserve(10);
  for (int i = 0; i < 10; i++) {
    appdatas.push_back(std::make_shared<cifar_sparse::AppData>(&disp.get_mr()));
  }

  std::queue<std::shared_ptr<cifar_sparse::AppData>> queue;
  for (auto& appdata : appdatas) {
    queue.push(appdata);
  }

  while (!queue.empty()) {
    auto appdata = queue.front();
    queue.pop();

    EXPECT_NO_THROW(disp.dispatch_multi_stage(*appdata, 1, 9));
  }

  EXPECT_TRUE(queue.empty());
}

// ----------------------------------------------------------------------------
// Test Concurrent Queue environment
// ----------------------------------------------------------------------------

TEST(ConcurrentQueueTest, Basic) {
  cifar_sparse::cuda::CudaDispatcher disp;

  std::vector<std::shared_ptr<cifar_sparse::AppData>> appdatas;
  appdatas.reserve(10);
  for (int i = 0; i < 10; i++) {
    appdatas.push_back(std::make_shared<cifar_sparse::AppData>(&disp.get_mr()));
  }

  SPSCQueue<std::shared_ptr<cifar_sparse::AppData>> gpu_queue;
  SPSCQueue<std::shared_ptr<cifar_sparse::AppData>> cpu_queue;

  // Initial push to GPU queue
  for (auto& appdata : appdatas) {
    gpu_queue.enqueue(std::move(appdata));
  }

  // Producer thread - GPU processing
  std::thread producer([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_sparse::AppData> appdata;
      while (!gpu_queue.dequeue(appdata)) {
        std::this_thread::yield();
      }

      EXPECT_NO_THROW(disp.dispatch_multi_stage(*appdata, 1, 4));

      while (!cpu_queue.enqueue(std::move(appdata))) {
        std::this_thread::yield();
      }
    }
  });

  // Consumer thread - CPU processing
  std::thread consumer([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_sparse::AppData> appdata;
      while (!cpu_queue.dequeue(appdata)) {
        std::this_thread::yield();
      }

      // Process on CPU (stages 5-9)
      EXPECT_NO_THROW(cifar_sparse::omp::dispatch_multi_stage(LITTLE_CORES, *appdata, 5, 9));
    }
  });

  producer.join();
  consumer.join();
}

TEST(CifarSparseTest, InterleavedConcurrentStages) {
  cifar_sparse::cuda::CudaDispatcher disp;

  // Create queues for inter-thread communication
  SPSCQueue<std::shared_ptr<cifar_sparse::AppData>> gpu_queue1;
  SPSCQueue<std::shared_ptr<cifar_sparse::AppData>> cpu_queue1;
  SPSCQueue<std::shared_ptr<cifar_sparse::AppData>> gpu_queue2;
  SPSCQueue<std::shared_ptr<cifar_sparse::AppData>> cpu_queue2;

  // Create test data
  std::vector<std::shared_ptr<cifar_sparse::AppData>> appdatas;
  appdatas.reserve(10);
  for (int i = 0; i < 10; i++) {
    appdatas.push_back(std::make_shared<cifar_sparse::AppData>(&disp.get_mr()));
  }

  // Initial push to first GPU queue
  for (auto& appdata : appdatas) {
    gpu_queue1.enqueue(std::move(appdata));
  }

  // GPU thread 1 - processes stages 1-2
  std::thread gpu_thread1([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_sparse::AppData> appdata;
      while (!gpu_queue1.dequeue(appdata)) {
        std::this_thread::yield();
      }
      EXPECT_NO_THROW(disp.dispatch_multi_stage(*appdata, 1, 2));
      while (!cpu_queue1.enqueue(std::move(appdata))) {
        std::this_thread::yield();
      }
    }
  });

  // CPU thread 1 - processes stages 3-4
  std::thread cpu_thread1([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_sparse::AppData> appdata;
      while (!cpu_queue1.dequeue(appdata)) {
        std::this_thread::yield();
      }
      EXPECT_NO_THROW(cifar_sparse::omp::dispatch_multi_stage(LITTLE_CORES, *appdata, 3, 4));
      while (!gpu_queue2.enqueue(std::move(appdata))) {
        std::this_thread::yield();
      }
    }
  });

  // GPU thread 2 - processes stages 5-6
  std::thread gpu_thread2([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_sparse::AppData> appdata;
      while (!gpu_queue2.dequeue(appdata)) {
        std::this_thread::yield();
      }
      EXPECT_NO_THROW(disp.dispatch_multi_stage(*appdata, 5, 6));
      while (!cpu_queue2.enqueue(std::move(appdata))) {
        std::this_thread::yield();
      }
    }
  });

  // CPU thread 2 - processes stages 7-9
  std::thread cpu_thread2([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_sparse::AppData> appdata;
      while (!cpu_queue2.dequeue(appdata)) {
        std::this_thread::yield();
      }
      EXPECT_NO_THROW(cifar_sparse::omp::dispatch_multi_stage(LITTLE_CORES, *appdata, 7, 9));
      while (!gpu_queue1.enqueue(std::move(appdata))) {
        std::this_thread::yield();
      }
    }
  });

  // Wait for all threads to complete
  gpu_thread1.join();
  cpu_thread1.join();
  gpu_thread2.join();
  cpu_thread2.join();
}

// ----------------------------------------------------------------------------
// Test Mixing Omp and Cuda
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  return RUN_ALL_TESTS();
}
