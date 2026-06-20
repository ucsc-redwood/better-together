#pragma once
// ----------------------------------------------------------------------------
// Differential-oracle comparison helpers for cross-backend correctness tests.
//
// The CPU/OpenMP path is the reference ("oracle"): a backend kernel is correct
// iff, on the same deterministically seeded input, its stage-output buffer
// matches the OMP output. Integer / structural stages (morton, sort, unique,
// radix tree, edge-count, prefix-sum, octree) compare for *exact* equality;
// float stages (conv / linear) compare within a relative+absolute tolerance.
//
// These return ::testing::AssertionResult so call sites read naturally:
//
//     EXPECT_TRUE(bt::testing::ExactEqual(ref, out, "tree stage1 morton"));
//     EXPECT_TRUE(bt::testing::NearEqual(ref, out, 1e-4f, 1e-5f, "conv1"));
//
// On failure they report the first mismatch (exact) or the worst element and
// max-abs-diff (tolerant), so a wrong-but-nonzero kernel is debuggable instead
// of merely "buffer changed".
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>

namespace bt::testing {

// Exact element-wise equality for integer / structural stages. Only the first
// `count` elements are compared (stages such as unique/radix-tree fill a valid
// prefix of an over-allocated buffer); pass count = ref.size() for full buffers.
template <class T>
[[nodiscard]] ::testing::AssertionResult ExactEqual(
    std::span<const T> ref,
    std::span<const T> out,
    std::string_view label = "",
    std::size_t count = static_cast<std::size_t>(-1)) {
  const std::size_t n = (count == static_cast<std::size_t>(-1)) ? ref.size() : count;
  if (ref.size() < n || out.size() < n) {
    return ::testing::AssertionFailure()
           << label << ": comparison length " << n << " exceeds buffer (ref=" << ref.size()
           << ", out=" << out.size() << ")";
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!(ref[i] == out[i])) {
      return ::testing::AssertionFailure()
             << label << ": element mismatch at index " << i << " of "
             << n
             // unary + promotes (u)int8_t to a printable integer instead of a char
             << " (ref=" << +ref[i] << ", out=" << +out[i] << ")";
    }
  }
  return ::testing::AssertionSuccess();
}

// Tolerant comparison for float stages: passes iff for every element
// |out - ref| <= atol + rtol * |ref|. NaN/Inf mismatches always fail. Reports
// the worst element (largest absolute difference) for debuggability.
[[nodiscard]] inline ::testing::AssertionResult NearEqual(
    std::span<const float> ref,
    std::span<const float> out,
    float rtol = 1e-4f,
    float atol = 1e-5f,
    std::string_view label = "",
    std::size_t count = static_cast<std::size_t>(-1)) {
  const std::size_t n = (count == static_cast<std::size_t>(-1)) ? ref.size() : count;
  if (ref.size() < n || out.size() < n) {
    return ::testing::AssertionFailure()
           << label << ": comparison length " << n << " exceeds buffer (ref=" << ref.size()
           << ", out=" << out.size() << ")";
  }
  double max_abs_diff = 0.0;
  std::size_t worst = 0;
  bool ok = true;
  for (std::size_t i = 0; i < n; ++i) {
    const double a = static_cast<double>(out[i]);
    const double b = static_cast<double>(ref[i]);
    const double diff = std::abs(a - b);
    const bool bad = std::isnan(diff) || std::isinf(diff) || diff > atol + rtol * std::abs(b);
    // Treat any NaN/Inf as maximally bad so it surfaces as the worst element.
    if (std::isnan(diff) || std::isinf(diff) || diff > max_abs_diff) {
      max_abs_diff = std::isnan(diff) ? std::numeric_limits<double>::infinity() : diff;
      worst = i;
    }
    if (bad) ok = false;
  }
  if (ok) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << label << ": exceeds tolerance (rtol=" << rtol << ", atol=" << atol
         << "); max|diff|=" << max_abs_diff << " at index " << worst << " of " << n
         << " (ref=" << ref[worst] << ", out=" << out[worst] << ")";
}

// ---- contiguous-range overloads --------------------------------------------
// Two range types so a golden std::pmr::vector can be compared against a plain
// std::vector (e.g. a sorted copy) without converting either side.

template <std::ranges::contiguous_range R1, std::ranges::contiguous_range R2>
[[nodiscard]] auto ExactEqual(const R1& ref,
                              const R2& out,
                              std::string_view label = "",
                              std::size_t count = static_cast<std::size_t>(-1)) {
  using T = std::ranges::range_value_t<R1>;
  static_assert(std::is_same_v<T, std::ranges::range_value_t<R2>>,
                "ExactEqual: ref and out must hold the same element type");
  return ExactEqual<T>(std::span<const T>(std::ranges::data(ref), std::ranges::size(ref)),
                       std::span<const T>(std::ranges::data(out), std::ranges::size(out)),
                       label,
                       count);
}

template <std::ranges::contiguous_range R1, std::ranges::contiguous_range R2>
[[nodiscard]] auto NearEqual(const R1& ref,
                             const R2& out,
                             float rtol = 1e-4f,
                             float atol = 1e-5f,
                             std::string_view label = "",
                             std::size_t count = static_cast<std::size_t>(-1)) {
  return NearEqual(std::span<const float>(std::ranges::data(ref), std::ranges::size(ref)),
                   std::span<const float>(std::ranges::data(out), std::ranges::size(out)),
                   rtol,
                   atol,
                   label,
                   count);
}

}  // namespace bt::testing
