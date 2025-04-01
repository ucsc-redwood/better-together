#pragma once

#include <array>
#include <iostream>
#include <memory_resource>
#include <numeric>
#include <string>
#include <vector>

template <size_t ND>
class NDArray {
 public:
  using Shape = std::array<size_t, ND>;

  // Construct the array given its shape.
  explicit NDArray(const Shape& shape,
                   std::pmr::memory_resource* mr = std::pmr::new_delete_resource())
      : shape_(shape), data_(mr) {
    compute_strides();
    total_size_ = std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<>());
    data_.resize(total_size_, 0.0f);
  }

  // Overloaded operator() for element access.
  template <typename... Indices>
    requires(sizeof...(Indices) == ND)
  float& operator()(Indices... indices) {
    size_t idx = compute_index({static_cast<size_t>(indices)...});
    return data_[idx];
  }

  template <typename... Indices>
    requires(sizeof...(Indices) == ND)
  const float& operator()(Indices... indices) const {
    size_t idx = compute_index({static_cast<size_t>(indices)...});
    return data_[idx];
  }

  NDArray<1> flatten() const {
    NDArray<1> flat({total_size_});
    std::copy(data_.begin(), data_.end(), flat.raw());
    return flat;
  }

  [[nodiscard]] const Shape& shape() const { return shape_; }

  [[nodiscard]] float* raw() { return data_.data(); }
  [[nodiscard]] const float* raw() const { return data_.data(); }

  // Returns the memory usage in bytes
  [[nodiscard]] size_t size() const { return total_size_; }
  [[nodiscard]] size_t memory_usage_bytes() const { return size() * sizeof(float); }

  // Utility to print the shape.
  void print_shape(const std::string& name) const {
    std::cout << name << ": (";
    for (size_t i = 0; i < ND; ++i) {
      std::cout << shape_[i];
      if (i + 1 < ND) std::cout << " × ";
    }
    std::cout << ")";

    std::cout << " = " << total_size_ << " elements";

    std::cout << "\n";
  }

  // Print the contents of the array
  void print(const std::string& name = "") const {
    if (!name.empty()) {
      print_shape(name);
    }

    if constexpr (ND == 1) {
      // 1D array
      std::cout << "[";
      for (size_t i = 0; i < shape_[0]; ++i) {
        std::cout << (*this)(i);
        if (i < shape_[0] - 1) std::cout << ", ";
      }
      std::cout << "]\n";
    } else if constexpr (ND == 2) {
      // 2D array
      std::cout << "[\n";
      for (size_t i = 0; i < shape_[0]; ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < shape_[1]; ++j) {
          std::cout << (*this)(i, j);
          if (j < shape_[1] - 1) std::cout << ", ";
        }
        std::cout << "]";
        if (i < shape_[0] - 1) std::cout << ",";
        std::cout << "\n";
      }
      std::cout << "]\n";
    } else {
      // Higher dimensional arrays - print first few elements
      std::cout << "Array contents : [";
      for (size_t i = 0; i < std::min(total_size_, size_t(10)); ++i) {
        std::cout << data_[i];
        if (i < std::min(total_size_, size_t(10)) - 1) std::cout << ", ";
      }
      if (total_size_ > 10) std::cout << ", ...";

      // and last 10 elements

      for (size_t i = total_size_ - 10; i < total_size_; ++i) {
        std::cout << data_[i];
        if (i < total_size_ - 1) std::cout << ", ";
      }

      std::cout << "]";
    }

    std::cout << "\n";
  }

 private:
  const Shape shape_;
  Shape strides_;
  size_t total_size_;

  // TODO: use pmr::vector
  // actual dense data holder
  // -------------------------------------------------
  std::pmr::vector<float> data_;
  // -------------------------------------------------

  // Compute strides for row-major order.
  void compute_strides() {
    strides_[ND - 1] = 1;
    for (size_t i = ND - 1; i > 0; --i) {
      strides_[i - 1] = strides_[i] * shape_[i];
    }
  }

  // Compute the flat index from multi-dimensional indices.
  // TODO: make this work in both CPU and GPU
  size_t compute_index(const Shape& indices) const {
    size_t idx = 0;
    for (size_t i = 0; i < ND; ++i) {
      idx += indices[i] * strides_[i];
    }
    return idx;
  }
};
