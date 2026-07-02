#pragma once

#include <algorithm>
#include <memory_resource>
#include <random>
#include <vector>

#include "platform/util/base_appdata.hpp"
#include "platform/util/ndarray.hpp"

namespace cifar_sparse {

// Convolution parameters
constexpr int kKernelSize = 3;
constexpr int kStride = 1;
constexpr int kPadding = 1;

// Pooling parameters
constexpr int kPoolSize = 2;
constexpr int kPoolStride = 2;

constexpr bool kRelu = true;

struct CSRMatrix {
  const int rows;
  const int cols;
  int nnz;  // set by build_from_dense() once the dense `values` are seeded
  std::pmr::vector<float> values;
  std::pmr::vector<int> row_ptr;
  std::pmr::vector<int> col_idx;

  // Basic constructor
  explicit CSRMatrix(const int r,
                     const int c,
                     std::pmr::memory_resource* mr = std::pmr::get_default_resource())
      : rows(r),
        cols(c),
        nnz(0),
        values(r * c, 0.0f, mr),
        row_ptr(r + 1, 0, mr),
        col_idx(r * c, 0, mr) {}

  // Build a valid CSR in place from the dense (rows*cols, row-major) weights that
  // were seeded into `values`. Without this the matrix ships with row_ptr/col_idx
  // all-zero (nnz=0) and the sparse kernels iterate empty rows -> the whole
  // pipeline computes zeros. Keeps a deterministic ~1/4 subset of entries per row
  // (matching the real magnitude-pruned AlexNetCIFAR weights' 0.25 density) so
  // rows have varying, non-trivial nnz that exercises the kernel's CSR decode,
  // compacting the kept weights into values[0..nnz) with matching col_idx/row_ptr.
  // Must be called exactly once, right after the dense values are written (it
  // overwrites the values layout in place).
  void build_from_dense() {
    std::vector<float> kept;
    kept.reserve(static_cast<size_t>(rows) * cols);
    int nz = 0;
    row_ptr[0] = 0;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        if (((r * 131 + c * 17) % 4) == 0) {  // deterministic keep pattern (~25% density)
          col_idx[nz] = c;
          kept.push_back(values[r * cols + c]);
          ++nz;
        }
      }
      row_ptr[r + 1] = nz;
    }
    for (int i = 0; i < nz; ++i) values[i] = kept[i];  // compact (aliasing-safe via `kept`)
    nnz = nz;
  }

  // Get raw pointers for compatibility with old code
  [[nodiscard]] const float* values_data() const { return values.data(); }
  [[nodiscard]] const int* row_ptr_data() const { return row_ptr.data(); }
  [[nodiscard]] const int* col_idx_data() const { return col_idx.data(); }
  [[nodiscard]] std::pmr::vector<float>& values_pmr_vec() { return values; }
  [[nodiscard]] std::pmr::vector<int>& row_ptr_pmr_vec() { return row_ptr; }
  [[nodiscard]] std::pmr::vector<int>& col_idx_pmr_vec() { return col_idx; }
  [[nodiscard]] const std::pmr::vector<float>& values_pmr_vec() const { return values; }
  [[nodiscard]] const std::pmr::vector<int>& row_ptr_pmr_vec() const { return row_ptr; }
  [[nodiscard]] const std::pmr::vector<int>& col_idx_pmr_vec() const { return col_idx; }
};

struct AppData final : public BaseAppData {
  // static constexpr size_t BATCH_SIZE = 512;  // sized for the old SmallAlexNet
  static constexpr size_t BATCH_SIZE = 128;

  // conv1: 64 output channels, 3×3×3 kernel = 27 inputs
  // conv2: 192 output channels, 64×3×3 kernel = 576 inputs
  // conv3: 384 output channels, 192×3×3 kernel = 1728 inputs
  // conv4: 256 output channels, 384×3×3 kernel = 3456 inputs
  // conv5: 256 output channels, 256×3×3 kernel = 2304 inputs
  // fc1: 4096 output channels, 4096 inputs (dense)
  // fc2: 4096 output channels, 4096 inputs (dense)
  // fc3: 10 output channels, 4096 inputs (dense)

  explicit AppData(std::pmr::memory_resource* mr)
      : BaseAppData(),
        u_input(BATCH_SIZE, 3, 32, 32, mr),
        u_conv1_out(BATCH_SIZE, 64, 32, 32, mr),
        u_pool1_out(BATCH_SIZE, 64, 16, 16, mr),
        u_conv2_out(BATCH_SIZE, 192, 16, 16, mr),
        u_pool2_out(BATCH_SIZE, 192, 8, 8, mr),
        u_conv3_out(BATCH_SIZE, 384, 8, 8, mr),
        u_conv4_out(BATCH_SIZE, 256, 8, 8, mr),
        u_conv5_out(BATCH_SIZE, 256, 8, 8, mr),
        u_pool3_out(BATCH_SIZE, 256, 4, 4, mr),
        u_fc1_out(BATCH_SIZE, 4096, mr),
        u_fc2_out(BATCH_SIZE, 4096, mr),
        u_fc3_out(BATCH_SIZE, 10, mr),
        u_conv1_b(64, mr),
        u_conv2_b(192, mr),
        u_conv3_b(384, mr),
        u_conv4_b(256, mr),
        u_conv5_b(256, mr),
        u_fc1_w(4096, 4096, mr),
        u_fc1_b(4096, mr),
        u_fc2_w(4096, 4096, mr),
        u_fc2_b(4096, mr),
        u_fc3_w(10, 4096, mr),
        u_fc3_b(10, mr),
        // Initialize CSR matrices
        conv1_sparse(64, 27, mr),
        conv2_sparse(192, 576, mr),
        conv3_sparse(384, 1728, mr),
        conv4_sparse(256, 3456, mr),
        conv5_sparse(256, 2304, mr) {
    std::mt19937 gen(114514);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::ranges::generate(u_input.pmr_vec(), [&]() { return dis(gen); });

    // Keep the synthetic weights small so 11 chained stages stay well inside float
    // range: convs ±0.05, the 4096-wide FCs ±0.02 (same regime as cifar-dense).
    // Biases are 0.0f (folded-BN convention).
    std::uniform_real_distribution<float> conv_weight_dis(-0.05f, 0.05f);
    std::ranges::generate(conv1_sparse.values_pmr_vec(), [&]() { return conv_weight_dis(gen); });
    std::ranges::generate(conv2_sparse.values_pmr_vec(), [&]() { return conv_weight_dis(gen); });
    std::ranges::generate(conv3_sparse.values_pmr_vec(), [&]() { return conv_weight_dis(gen); });
    std::ranges::generate(conv4_sparse.values_pmr_vec(), [&]() { return conv_weight_dis(gen); });
    std::ranges::generate(conv5_sparse.values_pmr_vec(), [&]() { return conv_weight_dis(gen); });

    std::uniform_real_distribution<float> fc_weight_dis(-0.02f, 0.02f);
    std::ranges::generate(u_fc1_w.pmr_vec(), [&]() { return fc_weight_dis(gen); });
    std::ranges::generate(u_fc2_w.pmr_vec(), [&]() { return fc_weight_dis(gen); });
    std::ranges::generate(u_fc3_w.pmr_vec(), [&]() { return fc_weight_dis(gen); });

    // Turn the dense seeded weights into a real CSR so the shipped sparse
    // pipeline computes actual convolutions instead of zeros (was a latent defect
    // -- the CSR index structure was never built; see CSRMatrix::build_from_dense).
    conv1_sparse.build_from_dense();
    conv2_sparse.build_from_dense();
    conv3_sparse.build_from_dense();
    conv4_sparse.build_from_dense();
    conv5_sparse.build_from_dense();

    std::ranges::fill(u_conv1_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_conv2_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_conv3_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_conv4_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_conv5_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_fc1_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_fc2_b.pmr_vec(), 0.0f);
    std::ranges::fill(u_fc3_b.pmr_vec(), 0.0f);
  }

  // Input and intermediate outputs
  Ndarray4D u_input;      // (128, 3, 32, 32)
  Ndarray4D u_conv1_out;  // (128, 64, 32, 32)
  Ndarray4D u_pool1_out;  // (128, 64, 16, 16)
  Ndarray4D u_conv2_out;  // (128, 192, 16, 16)
  Ndarray4D u_pool2_out;  // (128, 192, 8, 8)
  Ndarray4D u_conv3_out;  // (128, 384, 8, 8)
  Ndarray4D u_conv4_out;  // (128, 256, 8, 8)
  Ndarray4D u_conv5_out;  // (128, 256, 8, 8)
  Ndarray4D u_pool3_out;  // (128, 256, 4, 4)

  // Flatten would be (128, 4096), stored or created on-the-fly
  Ndarray2D u_fc1_out;  // (128, 4096)
  Ndarray2D u_fc2_out;  // (128, 4096)
  Ndarray2D u_fc3_out;  // shape = (128, 10) for final classification

  // Model parameters (the FC head is dense, as in cifar-dense)
  Ndarray1D u_conv1_b;  // (64)
  Ndarray1D u_conv2_b;  // (192)
  Ndarray1D u_conv3_b;  // (384)
  Ndarray1D u_conv4_b;  // (256)
  Ndarray1D u_conv5_b;  // (256)
  Ndarray2D u_fc1_w;    // (4096, 4096)
  Ndarray1D u_fc1_b;    // (4096)
  Ndarray2D u_fc2_w;    // (4096, 4096)
  Ndarray1D u_fc2_b;    // (4096)
  Ndarray2D u_fc3_w;    // (10, 4096)
  Ndarray1D u_fc3_b;    // (10)

  // Sparse matrices
  CSRMatrix conv1_sparse;  // (64, 27)
  CSRMatrix conv2_sparse;  // (192, 576)
  CSRMatrix conv3_sparse;  // (384, 1728)
  CSRMatrix conv4_sparse;  // (256, 3456)
  CSRMatrix conv5_sparse;  // (256, 2304)
};

}  // namespace cifar_sparse
