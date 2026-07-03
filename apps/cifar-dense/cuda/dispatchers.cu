#include "all_kernels.cuh"
#include "dispatchers.cuh"
#include "platform/util/debug_logger.hpp"

namespace cifar_dense::cuda {

constexpr bool kSync = false;

namespace {
float* upload(const float* src, size_t n) {
  float* d = nullptr;
  CheckCuda(cudaMalloc(&d, n * sizeof(float)));
  CheckCuda(cudaMemcpy(d, src, n * sizeof(float), cudaMemcpyHostToDevice));
  return d;
}
}  // namespace

CudaDispatcher::~CudaDispatcher() {
  for (auto& [app, w] : devw_) {
    for (auto* p : w.c_w) cudaFree(p);
    for (auto* p : w.c_b) cudaFree(p);
    for (auto* p : w.f_w) cudaFree(p);
    for (auto* p : w.f_b) cudaFree(p);
  }
}

const DeviceWeights& CudaDispatcher::dev_weights(const cifar_dense::AppData& a) {
  auto it = devw_.find(&a);
  if (it != devw_.end()) return it->second;

  auto sz4 = [](const auto& t) {
    return static_cast<size_t>(t.d0()) * t.d1() * t.d2() * t.d3();
  };
  auto sz2 = [](const auto& t) { return static_cast<size_t>(t.d0()) * t.d1(); };
  auto sz1 = [](const auto& t) { return static_cast<size_t>(t.d0()); };

  DeviceWeights w{};
  w.c_w[0] = upload(a.u_conv1_w.data(), sz4(a.u_conv1_w));
  w.c_b[0] = upload(a.u_conv1_b.data(), sz1(a.u_conv1_b));
  w.c_w[1] = upload(a.u_conv2_w.data(), sz4(a.u_conv2_w));
  w.c_b[1] = upload(a.u_conv2_b.data(), sz1(a.u_conv2_b));
  w.c_w[2] = upload(a.u_conv3_w.data(), sz4(a.u_conv3_w));
  w.c_b[2] = upload(a.u_conv3_b.data(), sz1(a.u_conv3_b));
  w.c_w[3] = upload(a.u_conv4_w.data(), sz4(a.u_conv4_w));
  w.c_b[3] = upload(a.u_conv4_b.data(), sz1(a.u_conv4_b));
  w.c_w[4] = upload(a.u_conv5_w.data(), sz4(a.u_conv5_w));
  w.c_b[4] = upload(a.u_conv5_b.data(), sz1(a.u_conv5_b));
  w.f_w[0] = upload(a.u_fc1_w.data(), sz2(a.u_fc1_w));
  w.f_b[0] = upload(a.u_fc1_b.data(), sz1(a.u_fc1_b));
  w.f_w[1] = upload(a.u_fc2_w.data(), sz2(a.u_fc2_w));
  w.f_b[1] = upload(a.u_fc2_b.data(), sz1(a.u_fc2_b));
  w.f_w[2] = upload(a.u_fc3_w.data(), sz2(a.u_fc3_w));
  w.f_b[2] = upload(a.u_fc3_b.data(), sz1(a.u_fc3_b));
  return devw_.emplace(&a, w).first->second;
}


void CudaDispatcher::run_stage_1_async(cifar_dense::AppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 1, &appdata);

  const int N = appdata.u_input.d0();
  const int inC = appdata.u_input.d1();
  const int inH = appdata.u_input.d2();
  const int inW = appdata.u_input.d3();
  const int outC = appdata.u_conv1_w.d0();
  const int outH = appdata.u_conv1_out.d2();
  const int outW = appdata.u_conv1_out.d3();

  conv2d_batch_cuda(appdata.u_input.data(),
                    dev_weights(appdata).c_w[0],
                    dev_weights(appdata).c_b[0],
                    appdata.u_conv1_out.data(),
                    N,
                    inC,
                    inH,
                    inW,
                    outC,
                    kKernelSize,
                    kKernelSize,
                    outH,
                    outW,
                    kStride,
                    kPadding,
                    kRelu);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_2_async(cifar_dense::AppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 2, &appdata);

  const int N = appdata.u_conv1_out.d0();     // 128
  const int C = appdata.u_conv1_out.d1();     // 64
  const int inH = appdata.u_conv1_out.d2();   // 32
  const int inW = appdata.u_conv1_out.d3();   // 32
  const int outH = appdata.u_pool1_out.d2();  // 16
  const int outW = appdata.u_pool1_out.d3();  // 16

  maxpool2d_batch_cuda(appdata.u_conv1_out.data(),
                       appdata.u_pool1_out.data(),
                       N,
                       C,
                       inH,
                       inW,
                       outH,
                       outW,
                       kPoolSize,
                       kPoolStride);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_3_async(cifar_dense::AppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 3, &appdata);

  const int N = appdata.u_pool1_out.d0();     // 128
  const int inC = appdata.u_pool1_out.d1();   // 64
  const int inH = appdata.u_pool1_out.d2();   // 16
  const int inW = appdata.u_pool1_out.d3();   // 16
  const int outC = appdata.u_conv2_w.d0();    // 192
  const int outH = appdata.u_conv2_out.d2();  // 16
  const int outW = appdata.u_conv2_out.d3();  // 16

  conv2d_batch_cuda(appdata.u_pool1_out.data(),
                    dev_weights(appdata).c_w[1],
                    dev_weights(appdata).c_b[1],
                    appdata.u_conv2_out.data(),
                    N,
                    inC,
                    inH,
                    inW,
                    outC,
                    kKernelSize,
                    kKernelSize,
                    outH,
                    outW,
                    kStride,
                    kPadding,
                    kRelu);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_4_async(cifar_dense::AppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 4, &appdata);

  const int N = appdata.u_conv2_out.d0();     // 128
  const int C = appdata.u_conv2_out.d1();     // 192
  const int inH = appdata.u_conv2_out.d2();   // 16
  const int inW = appdata.u_conv2_out.d3();   // 16
  const int outH = appdata.u_pool2_out.d2();  // 8
  const int outW = appdata.u_pool2_out.d3();  // 8

  maxpool2d_batch_cuda(appdata.u_conv2_out.data(),
                       appdata.u_pool2_out.data(),
                       N,
                       C,
                       inH,
                       inW,
                       outH,
                       outW,
                       kPoolSize,
                       kPoolStride);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_5_async(cifar_dense::AppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 5, &appdata);

  const int N = appdata.u_pool2_out.d0();     // 128
  const int inC = appdata.u_pool2_out.d1();   // 192
  const int inH = appdata.u_pool2_out.d2();   // 8
  const int inW = appdata.u_pool2_out.d3();   // 8
  const int outC = appdata.u_conv3_w.d0();    // 384
  const int outH = appdata.u_conv3_out.d2();  // 8
  const int outW = appdata.u_conv3_out.d3();  // 8

  conv2d_batch_cuda(appdata.u_pool2_out.data(),
                    dev_weights(appdata).c_w[2],
                    dev_weights(appdata).c_b[2],
                    appdata.u_conv3_out.data(),
                    N,
                    inC,
                    inH,
                    inW,
                    outC,
                    kKernelSize,
                    kKernelSize,
                    outH,
                    outW,
                    kStride,
                    kPadding,
                    kRelu);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_6_async(cifar_dense::AppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 6, &appdata);

  const int N = appdata.u_conv3_out.d0();     // 128
  const int inC = appdata.u_conv3_out.d1();   // 384
  const int inH = appdata.u_conv3_out.d2();   // 8
  const int inW = appdata.u_conv3_out.d3();   // 8
  const int outC = appdata.u_conv4_w.d0();    // 256
  const int outH = appdata.u_conv4_out.d2();  // 8
  const int outW = appdata.u_conv4_out.d3();  // 8

  conv2d_batch_cuda(appdata.u_conv3_out.data(),
                    dev_weights(appdata).c_w[3],
                    dev_weights(appdata).c_b[3],
                    appdata.u_conv4_out.data(),
                    N,
                    inC,
                    inH,
                    inW,
                    outC,
                    kKernelSize,
                    kKernelSize,
                    outH,
                    outW,
                    kStride,
                    kPadding,
                    kRelu);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_7_async(cifar_dense::AppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 7, &appdata);

  const int N = appdata.u_conv4_out.d0();     // 128
  const int inC = appdata.u_conv4_out.d1();   // 256
  const int inH = appdata.u_conv4_out.d2();   // 8
  const int inW = appdata.u_conv4_out.d3();   // 8
  const int outC = appdata.u_conv5_w.d0();    // 256
  const int outH = appdata.u_conv5_out.d2();  // 8
  const int outW = appdata.u_conv5_out.d3();  // 8

  conv2d_batch_cuda(appdata.u_conv4_out.data(),
                    dev_weights(appdata).c_w[4],
                    dev_weights(appdata).c_b[4],
                    appdata.u_conv5_out.data(),
                    N,
                    inC,
                    inH,
                    inW,
                    outC,
                    kKernelSize,
                    kKernelSize,
                    outH,
                    outW,
                    kStride,
                    kPadding,
                    kRelu);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_8_async(cifar_dense::AppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 8, &appdata);

  const int N = appdata.u_conv5_out.d0();     // 128
  const int C = appdata.u_conv5_out.d1();     // 256
  const int inH = appdata.u_conv5_out.d2();   // 8
  const int inW = appdata.u_conv5_out.d3();   // 8
  const int outH = appdata.u_pool3_out.d2();  // 4
  const int outW = appdata.u_pool3_out.d3();  // 4

  maxpool2d_batch_cuda(appdata.u_conv5_out.data(),
                       appdata.u_pool3_out.data(),
                       N,
                       C,
                       inH,
                       inW,
                       outH,
                       outW,
                       kPoolSize,
                       kPoolStride);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_9_async(cifar_dense::AppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 9, &appdata);

  const int N = appdata.u_pool3_out.d0();  // 128
  const int C = appdata.u_pool3_out.d1();  // 256
  const int H = appdata.u_pool3_out.d2();  // 4
  const int W = appdata.u_pool3_out.d3();  // 4
  const int inF = C * H * W;               // 4096
  const int outF = appdata.u_fc1_w.d0();   // 4096

  linear_batch_cuda(appdata.u_pool3_out.data(),
                    dev_weights(appdata).f_w[0],
                    dev_weights(appdata).f_b[0],
                    appdata.u_fc1_out.data(),
                    N,
                    inF,
                    outF,
                    kRelu);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_10_async(cifar_dense::AppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 10, &appdata);

  const int N = appdata.u_fc1_out.d0();    // 128
  const int inF = appdata.u_fc1_out.d1();  // 4096
  const int outF = appdata.u_fc2_w.d0();   // 4096

  linear_batch_cuda(appdata.u_fc1_out.data(),
                    dev_weights(appdata).f_w[1],
                    dev_weights(appdata).f_b[1],
                    appdata.u_fc2_out.data(),
                    N,
                    inF,
                    outF,
                    kRelu);

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

void CudaDispatcher::run_stage_11_async(cifar_dense::AppData& appdata) {
  LOG_KERNEL(LogKernelType::kCUDA, 11, &appdata);

  const int N = appdata.u_fc2_out.d0();    // 128
  const int inF = appdata.u_fc2_out.d1();  // 4096
  const int outF = appdata.u_fc3_w.d0();   // 10

  linear_batch_cuda(appdata.u_fc2_out.data(),
                    dev_weights(appdata).f_w[2],
                    dev_weights(appdata).f_b[2],
                    appdata.u_fc3_out.data(),
                    N,
                    inF,
                    outF,
                    false);  // FC3 emits raw logits: no ReLU

  if constexpr (kSync) {
    CheckCuda(cudaGetLastError());
    CheckCuda(cudaDeviceSynchronize());
  }
}

}  // namespace cifar_dense::cuda
