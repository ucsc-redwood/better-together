#pragma once

#include <cstdio>
#include <memory_resource>
#include <stdexcept>
#include <vector>

namespace cifar_sparse {

/* ========= 1D NDArray ========= */
class Ndarray1D {
 private:
  const int size_;
  std::pmr::vector<float> data_;

 public:
  // Constructor with default memory resource
  explicit Ndarray1D(const int size,
                     std::pmr::memory_resource* mr = std::pmr::get_default_resource())
      : size_(size), data_(size, 0.0f, mr) {}

  // Get an element from the 1D NDArray
  [[nodiscard]] float get(const int i) const { return data_[i]; }

  // Set an element in the 1D NDArray
  void set(const int i, const float value) { data_[i] = value; }

  // Accessor for size
  [[nodiscard]] int size() const { return size_; }

  // Allow direct access to data for efficiency
  [[nodiscard]] const float* data() const { return data_.data(); }
  [[nodiscard]] float* data() { return data_.data(); }
};

/* ========= 2D NDArray ========= */
class Ndarray2D {
 private:
  const int rows_;
  const int cols_;
  std::pmr::vector<float> data_;

 public:
  // Constructor with default memory resource
  Ndarray2D(int rows, int cols, std::pmr::memory_resource* mr = std::pmr::get_default_resource())
      : rows_(rows), cols_(cols), data_(rows * cols, 0.0f, mr) {}

  // Get an element from the 2D NDArray at (i, j)
  [[nodiscard]] float get(const int i, const int j) const { return data_[i * cols_ + j]; }

  // Set an element in the 2D NDArray at (i, j)
  void set(const int i, const int j, const float value) { data_[i * cols_ + j] = value; }

  // Accessor methods
  [[nodiscard]] int rows() const { return rows_; }
  [[nodiscard]] int cols() const { return cols_; }
  [[nodiscard]] const float* data() const { return data_.data(); }
  [[nodiscard]] float* data() { return data_.data(); }
};

/* ========= 4D NDArray ========= */
class Ndarray4D {
 private:
  const int d0_;
  const int d1_;
  const int d2_;
  const int d3_;
  std::pmr::vector<float> data_;

 public:
  // Constructor with default memory resource
  Ndarray4D(const int d0,
            const int d1,
            const int d2,
            const int d3,
            std::pmr::memory_resource* mr = std::pmr::get_default_resource())
      : d0_(d0), d1_(d1), d2_(d2), d3_(d3), data_(d0 * d1 * d2 * d3, 0.0f, mr) {}

  // Get an element from the 4D NDArray at (i, j, k, l)
  [[nodiscard]] float get(const int i, const int j, const int k, const int l) const {
    const int offset = i * (d1_ * d2_ * d3_) + j * (d2_ * d3_) + k * d3_ + l;
    return data_[offset];
  }

  // Set an element in the 4D NDArray at (i, j, k, l)
  void set(const int i, const int j, const int k, const int l, const float value) {
    const int offset = i * (d1_ * d2_ * d3_) + j * (d2_ * d3_) + k * d3_ + l;
    data_[offset] = value;
  }

  // Accessor methods
  [[nodiscard]] int d0() const { return d0_; }
  [[nodiscard]] int d1() const { return d1_; }
  [[nodiscard]] int d2() const { return d2_; }
  [[nodiscard]] int d3() const { return d3_; }
  [[nodiscard]] const float* data() const { return data_.data(); }
  [[nodiscard]] float* data() { return data_.data(); }
};

}  // namespace cifar_sparse
