#pragma once
// Reduced-precision fully-connected weights (cifar-gpu-kernels F2): the precision mode and the
// host-side quantization, done once when the weights are loaded. Only the Vulkan FC kernel reads
// the copy; every other backend, and the Vulkan default, uses the FP32 weights. INT8 only: FP16
// was built and measured too, and was not faster than FP32 on the Samsung (task 3.8).
//
// Per-element round-trip error (FcWeights unit test): per output row r, with
// scale[r] = max_j |w[r, j]| / 127 and q = round(w / scale[r]), |w - scale[r] q| <= scale[r] / 2.
// A row of zeros has scale 0.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory_resource>
#include <stdexcept>
#include <string>
#include <string_view>

namespace bt {

enum class FcWeights : std::uint8_t { kFp32, kInt8 };

[[nodiscard]] inline const char* to_string(const FcWeights m) {
  return m == FcWeights::kInt8 ? "int8" : "fp32";
}

// BT_VK_FC_WEIGHTS=int8 (fp32, empty or unset: FP32, the default). Any other value throws.
[[nodiscard]] inline FcWeights fc_weights_from_env() {
  const char* v = std::getenv("BT_VK_FC_WEIGHTS");
  if (v == nullptr || std::string_view(v).empty() || std::string_view(v) == "fp32") {
    return FcWeights::kFp32;
  }
  if (std::string_view(v) == "int8") return FcWeights::kInt8;
  throw std::runtime_error(std::string("BT_VK_FC_WEIGHTS must be fp32 or int8, got ") + v);
}

// The INT8 copy of one [rows, cols] FC weight matrix, as the 32-bit words the Vulkan kernel reads
// (new_cifar_dense_linear_bt.comp, WEIGHTS = 1): rows * cols / 4 words of four two's-complement
// bytes each, row-major and lowest byte first, then rows words of FP32 scales. Sized for its mode
// at construction (so it allocates when its buffer is placed; empty in FP32 mode), filled by
// quantize(). cols must be a multiple of 4.
struct FcWeightCopy {
  FcWeightCopy(const FcWeights mode, const int rows, const int cols, std::pmr::memory_resource* mr)
      : rows(rows), cols(cols), words(size_for(mode, rows, cols), mr) {}

  // From the FP32 weights w ([rows, cols], row-major); a no-op in FP32 mode.
  void quantize(const float* w) {
    if (words.empty()) return;
    const size_t n = static_cast<size_t>(rows) * cols;
    for (int r = 0; r < rows; ++r) {
      const float* row = w + static_cast<size_t>(r) * cols;
      float m = 0.0f;
      for (int j = 0; j < cols; ++j) m = std::max(m, std::abs(row[j]));
      const float scale = m / 127.0f;
      words[n / 4 + r] = std::bit_cast<std::uint32_t>(scale);
      for (int j = 0; j < cols; j += 4) {
        std::uint32_t word = 0;
        for (int b = 0; b < 4; ++b) {
          const float q =
              scale > 0.0f ? std::clamp(std::nearbyint(row[j + b] / scale), -127.0f, 127.0f) : 0.0f;
          word |= std::uint32_t{static_cast<std::uint8_t>(static_cast<std::int8_t>(q))} << (8 * b);
        }
        words[(static_cast<size_t>(r) * cols + j) / 4] = word;
      }
    }
  }

  [[nodiscard]] float scale(const int r) const {
    return std::bit_cast<float>(words[static_cast<size_t>(rows) * cols / 4 + r]);
  }

  // The weight [r, j] as the kernel reads it.
  [[nodiscard]] float dequantized(const int r, const int j) const {
    const size_t i = static_cast<size_t>(r) * cols + j;
    const auto q =
        static_cast<std::int8_t>(static_cast<std::uint8_t>(words[i / 4] >> (8 * (i % 4))));
    return scale(r) * static_cast<float>(q);
  }

  int rows;
  int cols;
  std::pmr::vector<std::uint32_t> words;

 private:
  static size_t size_for(const FcWeights mode, const int rows, const int cols) {
    if (mode == FcWeights::kFp32) return 0;
    if (cols % 4 != 0) {
      throw std::runtime_error("FcWeightCopy: cols must be a multiple of 4, got " +
                               std::to_string(cols));
    }
    return static_cast<size_t>(rows) * cols / 4 + rows;
  }
};

}  // namespace bt
