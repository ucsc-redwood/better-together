#pragma once

#include <memory_resource>
#include <vector>

/* ========= 1D NDArray ========= */
class Ndarray1D {
 private:
  const int d0_;
  std::pmr::vector<float> data_;

 public:
  // Constructor with default memory resource
  explicit Ndarray1D(const int size,
                     std::pmr::memory_resource* mr = std::pmr::get_default_resource())
      : d0_(size), data_(size, 0.0f, mr) {}

  // Accessor for methods
  [[nodiscard]] int d0() const { return d0_; }
  [[nodiscard]] int total_size() const { return d0_; }

  [[nodiscard]] const float* data() const { return data_.data(); }
  [[nodiscard]] float* data() { return data_.data(); }
  [[nodiscard]] std::pmr::vector<float>& pmr_vec() { return data_; }
  [[nodiscard]] const std::pmr::vector<float>& pmr_vec() const { return data_; }
};

/* ========= 2D NDArray ========= */
class Ndarray2D {
 private:
  const int d0_;  // rows
  const int d1_;  // cols
  std::pmr::vector<float> data_;

 public:
  // Constructor with default memory resource
  explicit Ndarray2D(const int rows,
                     const int cols,
                     std::pmr::memory_resource* mr = std::pmr::get_default_resource())
      : d0_(rows), d1_(cols), data_(rows * cols, 0.0f, mr) {}

  // Accessor methods
  [[nodiscard]] int d0() const { return d0_; }
  [[nodiscard]] int d1() const { return d1_; }
  [[nodiscard]] int total_size() const { return d0_ * d1_; }

  [[nodiscard]] const float* data() const { return data_.data(); }
  [[nodiscard]] float* data() { return data_.data(); }
  [[nodiscard]] std::pmr::vector<float>& pmr_vec() { return data_; }
  [[nodiscard]] const std::pmr::vector<float>& pmr_vec() const { return data_; }
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
  explicit Ndarray4D(const int d0,
                     const int d1,
                     const int d2,
                     const int d3,
                     std::pmr::memory_resource* mr = std::pmr::get_default_resource())
      : d0_(d0), d1_(d1), d2_(d2), d3_(d3), data_(d0 * d1 * d2 * d3, 0.0f, mr) {}

  // Accessor methods
  [[nodiscard]] int d0() const { return d0_; }
  [[nodiscard]] int d1() const { return d1_; }
  [[nodiscard]] int d2() const { return d2_; }
  [[nodiscard]] int d3() const { return d3_; }
  [[nodiscard]] int total_size() const { return d0_ * d1_ * d2_ * d3_; }

  [[nodiscard]] const float* data() const { return data_.data(); }
  [[nodiscard]] float* data() { return data_.data(); }
  [[nodiscard]] std::pmr::vector<float>& pmr_vec() { return data_; }
  [[nodiscard]] const std::pmr::vector<float>& pmr_vec() const { return data_; }
};
