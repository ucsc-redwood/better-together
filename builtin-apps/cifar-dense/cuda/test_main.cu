#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <queue>

#include "../../app.hpp"
#include "../../pipeline/spsc_queue.hpp"
#include "../appdata.hpp"
#include "../omp/dispatchers.hpp"
#include "dispatchers.cuh"

// ----------------------------------------------------------------------------
// test appdata initialization
// ----------------------------------------------------------------------------

TEST(AppdataTest, Initialization) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  EXPECT_EQ(appdata.u_conv1_w.d0(), 16);
  EXPECT_EQ(appdata.u_conv1_w.d1(), 3);
  EXPECT_EQ(appdata.u_conv1_w.d2(), 3);
  EXPECT_EQ(appdata.u_conv1_w.d3(), 3);
}

// ----------------------------------------------------------------------------
// test Stage 1
// ----------------------------------------------------------------------------

TEST(Stage1Test, Basic) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  // Run stage 1
  disp.dispatch_stage(appdata, 1);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv1_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv1_out.d1(), 16);
  EXPECT_EQ(appdata.u_conv1_out.d2(), 32);
  EXPECT_EQ(appdata.u_conv1_out.d3(), 32);

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
}

// ----------------------------------------------------------------------------
// test Stage 2
// ----------------------------------------------------------------------------

TEST(Stage2Test, Basic) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  // Run stage 2
  disp.dispatch_stage(appdata, 2);

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool1_out.d0(), 128);
  EXPECT_EQ(appdata.u_pool1_out.d1(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d2(), 16);
  EXPECT_EQ(appdata.u_pool1_out.d3(), 16);

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2));
}

// ----------------------------------------------------------------------------
// test Stage 3
// ----------------------------------------------------------------------------

TEST(Stage3Test, Basic) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  // Run stage 3
  disp.dispatch_stage(appdata, 3);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv2_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv2_out.d1(), 32);
  EXPECT_EQ(appdata.u_conv2_out.d2(), 16);
  EXPECT_EQ(appdata.u_conv2_out.d3(), 16);

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3));
}

// ----------------------------------------------------------------------------
// test Stage 4
// ----------------------------------------------------------------------------

TEST(Stage4Test, Basic) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  // Run stage 4
  disp.dispatch_stage(appdata, 4);

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool2_out.d0(), 128);
  EXPECT_EQ(appdata.u_pool2_out.d1(), 32);
  EXPECT_EQ(appdata.u_pool2_out.d2(), 8);
  EXPECT_EQ(appdata.u_pool2_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 4));
}

// ----------------------------------------------------------------------------
// test Stage 5
// ----------------------------------------------------------------------------

TEST(Stage5Test, Basic) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  // Run stage 5
  disp.dispatch_stage(appdata, 5);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv3_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv3_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv3_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv3_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 5));
}

// ----------------------------------------------------------------------------
// test Stage 6
// ----------------------------------------------------------------------------

TEST(Stage6Test, Basic) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  // Run stage 6
  disp.dispatch_stage(appdata, 6);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv4_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv4_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv4_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv4_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 6));
}

// ----------------------------------------------------------------------------
// test Stage 7
// ----------------------------------------------------------------------------

TEST(Stage7Test, Basic) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  // Run stage 7
  disp.dispatch_stage(appdata, 7);

  // Check output dimensions
  EXPECT_EQ(appdata.u_conv5_out.d0(), 128);
  EXPECT_EQ(appdata.u_conv5_out.d1(), 64);
  EXPECT_EQ(appdata.u_conv5_out.d2(), 8);
  EXPECT_EQ(appdata.u_conv5_out.d3(), 8);

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 7));
}

// ----------------------------------------------------------------------------
// test Stage 8
// ----------------------------------------------------------------------------

TEST(Stage8Test, Basic) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  // Run stage 8
  disp.dispatch_stage(appdata, 8);

  // Check output dimensions
  EXPECT_EQ(appdata.u_pool3_out.d0(), 128);
  EXPECT_EQ(appdata.u_pool3_out.d1(), 64);
  EXPECT_EQ(appdata.u_pool3_out.d2(), 4);
  EXPECT_EQ(appdata.u_pool3_out.d3(), 4);

  // Check no throw
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 8));
}

// ----------------------------------------------------------------------------
// Test Mixing Omp and Cuda
// ----------------------------------------------------------------------------

TEST(MixingTest, CudaThenOmp) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_2(appdata));
}

TEST(MixingTest, OmpThenCuda) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  EXPECT_NO_THROW(cifar_dense::omp::run_stage_1(appdata));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2));
}

TEST(MixingTest, MultipleStages) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  // Run first 3 stages with CUDA
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3));

  // Run next 2 stages with OMP
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_4(appdata));
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_5(appdata));

  // Run final stages with CUDA
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 6));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 7));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 8));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 9));
}

TEST(MixingTest, AlternatingStages) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  // Alternate between CUDA and OMP for each stage
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_2(appdata));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3));
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_4(appdata));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 5));
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_6(appdata));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 7));
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_8(appdata));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 9));
}

TEST(MixingTest, MixedBatch) {
  cifar_dense::cuda::CudaDispatcher disp;
  cifar_dense::AppData appdata(&disp.get_mr());

  // Run first half with CUDA
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 1));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 2));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 3));
  EXPECT_NO_THROW(disp.dispatch_stage(appdata, 4));

  // Run second half with OMP
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_5(appdata));
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_6(appdata));
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_7(appdata));
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_8(appdata));
  EXPECT_NO_THROW(cifar_dense::omp::run_stage_9(appdata));
}

// ----------------------------------------------------------------------------
// Test Queue environment
// ----------------------------------------------------------------------------

TEST(QueueTest, Basic) {
  cifar_dense::cuda::CudaDispatcher disp;

  std::vector<std::shared_ptr<cifar_dense::AppData>> appdatas;
  appdatas.reserve(10);
  for (int i = 0; i < 10; i++) {
    appdatas.push_back(std::make_shared<cifar_dense::AppData>(&disp.get_mr()));
  }

  std::queue<std::shared_ptr<cifar_dense::AppData>> queue;
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
  cifar_dense::cuda::CudaDispatcher disp;

  std::vector<std::shared_ptr<cifar_dense::AppData>> appdatas;
  appdatas.reserve(10);
  for (int i = 0; i < 10; i++) {
    appdatas.push_back(std::make_shared<cifar_dense::AppData>(&disp.get_mr()));
  }

  SPSCQueue<std::shared_ptr<cifar_dense::AppData>> gpu_queue;
  SPSCQueue<std::shared_ptr<cifar_dense::AppData>> cpu_queue;

  // Initial push to GPU queue
  for (auto& appdata : appdatas) {
    gpu_queue.enqueue(std::move(appdata));
  }

  // Producer thread - GPU processing
  std::thread producer([&]() {
    for (int i = 0; i < 10; i++) {
      std::shared_ptr<cifar_dense::AppData> appdata;
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
      std::shared_ptr<cifar_dense::AppData> appdata;
      while (!cpu_queue.dequeue(appdata)) {
        std::this_thread::yield();
      }

      // Process on CPU (stages 5-9)
      EXPECT_NO_THROW(cifar_dense::omp::dispatch_multi_stage(*appdata, 5, 9));
    }
  });

  producer.join();
  consumer.join();
}

// ----------------------------------------------------------------------------
// Test Mixing Omp and Cuda
// ----------------------------------------------------------------------------

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
