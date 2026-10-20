#pragma once
// Reduced-precision fully-connected weights (cifar-gpu-kernels F2): the precision mode and the
// host-side quantization, done once when the weights are loaded. Only the Vulkan FC kernel reads
// the copies; every other backend, and the Vulkan default, uses the FP32 weights.
//
// Per-element round-trip error (FcWeights unit test):
//   fp16: IEEE binary16, round to nearest even. |w - w16| <= 2^-11 |w| for 2^-14 <= |w| < 65504,
//         and <= 2^-25 below (subnormals).
//   int8: per output row r, scale[r] = max_j |w[r, j]| / 127 and q = round(w / scale[r]), so
//         |w - scale[r] q| <= scale[r] / 2. A row of zeros has scale 0.

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

enum class FcWeights : std::uint8_t { kFp32, kFp16, kInt8 };

[[nodiscard]] inline const char* to_string(const FcWeights m) {
  return m == FcWeights::kFp16 ? "fp16" : m == FcWeights::kInt8 ? "int8" : "fp32";
}

// BT_VK_FC_WEIGHTS=fp16|int8 (fp32, empty or unset: FP32, the default). Any other value throws.
[[nodiscard]] inline FcWeights fc_weights_from_env() {
  const char* v = std::getenv("BT_VK_FC_WEIGHTS");
  if (v == nullptr || std::string_view(v).empty() || std::string_view(v) == "fp32") {
    return FcWeights::kFp32;
  }
  if (std::string_view(v) == "fp16") return FcWeights::kFp16;
  if (std::string_view(v) == "int8") return FcWeights::kInt8;
  throw std::runtime_error(std::string("BT_VK_FC_WEIGHTS must be fp32, fp16 or int8, got ") + v);
}

// IEEE binary16 bits of f, rounded to nearest even; overflow gives infinity, NaN stays NaN.
[[nodiscard]] inline std::uint16_t float_to_half(const float f) {
  const auto x = std::bit_cast<std::uint32_t>(f);
  const auto sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
  const std::uint32_t abs = x & 0x7fffffffu;
  if (abs >= 0x7f800000u) return sign | (abs > 0x7f800000u ? 0x7e00u : 0x7c00u);  // NaN, inf
  if (abs >= 0x477ff000u) return sign | 0x7c00u;  // rounds to >= 65520: inf
  if (abs < 0x38800000u) {                        // below 2^-14: a subnormal half (or zero)
    if (abs < 0x33000000u) return sign;           // below 2^-25: rounds to zero
    const std::uint32_t mant = (abs & 0x7fffffu) | 0x800000u;
    const int shift = 126 - static_cast<int>(abs >> 23);  // 14..24
    const std::uint32_t half = mant >> shift;
    const std::uint32_t rem = mant & ((1u << shift) - 1u);
    const std::uint32_t mid = 1u << (shift - 1);
    return sign | static_cast<std::uint16_t>(half + (rem > mid || (rem == mid && (half & 1u))));
  }
  // Normal: rebias the exponent (127 -> 15) and round the 13 dropped mantissa bits.
  const std::uint32_t r = abs - 0x38000000u;
  const std::uint32_t half = r >> 13;
  const std::uint32_t rem = r & 0x1fffu;
  return sign |
         static_cast<std::uint16_t>(half + (rem > 0x1000u || (rem == 0x1000u && (half & 1u))));
}

[[nodiscard]] inline float half_to_float(const std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1fu;
  const std::uint32_t mant = h & 0x3ffu;
  if (exp == 0) {  // zero or subnormal: mant * 2^-24
    const float v = std::ldexp(static_cast<float>(mant), -24);
    return sign != 0 ? -v : v;
  }
  if (exp == 31) return std::bit_cast<float>(sign | 0x7f800000u | (mant << 13));
  return std::bit_cast<float>(sign | ((exp + 112) << 23) | (mant << 13));
}

// The reduced-precision copy of one [rows, cols] FC weight matrix, as the 32-bit words the
// Vulkan kernel reads (new_cifar_dense_linear_bt.comp, WEIGHTS): row-major, the lowest bits first.
//   fp16: rows * cols / 2 words, two binary16 values each (unpackHalf2x16 order).
//   int8: rows * cols / 4 words, four two's-complement bytes each, then rows words of FP32 scales.
// Sized for its mode at construction (so it allocates when its buffer is placed; empty in FP32
// mode), filled by quantize(). cols must be a multiple of 4.
struct FcWeightCopy {
  FcWeightCopy(const FcWeights mode, const int rows, const int cols, std::pmr::memory_resource* mr)
      : mode(mode), rows(rows), cols(cols), words(size_for(mode, rows, cols), mr) {}

  // From the FP32 weights w ([rows, cols], row-major).
  void quantize(const float* w) {
    const size_t n = static_cast<size_t>(rows) * cols;
    if (mode == FcWeights::kFp16) {
      for (size_t i = 0; i < n; i += 2) {
        words[i / 2] = float_to_half(w[i]) | (std::uint32_t{float_to_half(w[i + 1])} << 16);
      }
    } else if (mode == FcWeights::kInt8) {
      for (int r = 0; r < rows; ++r) {
        const float* row = w + static_cast<size_t>(r) * cols;
        float m = 0.0f;
        for (int j = 0; j < cols; ++j) m = std::max(m, std::abs(row[j]));
        const float scale = m / 127.0f;
        words[n / 4 + r] = std::bit_cast<std::uint32_t>(scale);
        for (int j = 0; j < cols; ++j) {
          const float q =
              scale > 0.0f ? std::clamp(std::nearbyint(row[j] / scale), -127.0f, 127.0f) : 0.0f;
          const auto byte = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
          const size_t i = static_cast<size_t>(r) * cols + j;
          words[i / 4] =
              (words[i / 4] & ~(0xffu << (8 * (i % 4)))) | (std::uint32_t{byte} << (8 * (i % 4)));
        }
      }
    }
  }

  // INT8: the scale of row r.
  [[nodiscard]] float scale(const int r) const {
    return std::bit_cast<float>(words[static_cast<size_t>(rows) * cols / 4 + r]);
  }

  // The weight [r, j] as the kernel reads it.
  [[nodiscard]] float dequantized(const int r, const int j) const {
    const size_t i = static_cast<size_t>(r) * cols + j;
    if (mode == FcWeights::kFp16) {
      return half_to_float(static_cast<std::uint16_t>(words[i / 2] >> (16 * (i % 2))));
    }
    const auto q =
        static_cast<std::int8_t>(static_cast<std::uint8_t>(words[i / 4] >> (8 * (i % 4))));
    return scale(r) * static_cast<float>(q);
  }

  FcWeights mode;
  int rows;
  int cols;
  std::pmr::vector<std::uint32_t> words;

 private:
  static size_t size_for(const FcWeights mode, const int rows, const int cols) {
    if (mode != FcWeights::kFp32 && cols % 4 != 0) {
      throw std::runtime_error("FcWeightCopy: cols must be a multiple of 4, got " +
                               std::to_string(cols));
    }
    const size_t n = static_cast<size_t>(rows) * cols;
    return mode == FcWeights::kFp16 ? n / 2 : mode == FcWeights::kInt8 ? n / 4 + rows : 0;
  }
};

}  // namespace bt
