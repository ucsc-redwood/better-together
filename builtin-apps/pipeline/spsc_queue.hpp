#pragma once

#include <atomic>
#include <cstddef>

// ----------------------------------------------------------------------------
// SPSCQueue is a single-producer single-consumer queue.
// ----------------------------------------------------------------------------

template <typename T, size_t Size = 1024>
requires std::is_move_constructible_v<T>
class SPSCQueue {
  static_assert((Size & (Size - 1)) == 0, "Size must be a power of 2");

 public:
  SPSCQueue() = default;
  ~SPSCQueue() = default;

  // Add a move version of enqueue
  bool enqueue(T&& item) {
    const size_t head = head_.load(std::memory_order_relaxed);
    const size_t next_head = (head + 1) & mask_;

    if (next_head == tail_.load(std::memory_order_acquire)) {
      return false;  // full
    }

    buffer_[head] = std::move(item);
    head_.store(next_head, std::memory_order_release);
    return true;
  }

  bool enqueue(const T& item) {
    const size_t head = head_.load(std::memory_order_relaxed);
    const size_t next_head = (head + 1) & mask_;

    if (next_head == tail_.load(std::memory_order_acquire)) {
      return false;  // full
    }

    buffer_[head] = item;
    head_.store(next_head, std::memory_order_release);
    return true;
  }

  bool dequeue(T& item) {
    const size_t tail = tail_.load(std::memory_order_relaxed);

    if (tail == head_.load(std::memory_order_acquire)) {
      return false;  // empty
    }

    item = std::move(buffer_[tail]);
    tail_.store((tail + 1) & mask_, std::memory_order_release);
    return true;
  }

  [[nodiscard]] bool empty() const {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool full() const {
    const size_t next_head = (head_.load(std::memory_order_relaxed) + 1) & mask_;
    return next_head == tail_.load(std::memory_order_acquire);
  }

 private:
  static constexpr size_t mask_ = Size - 1;
  T buffer_[Size];

  alignas(64) std::atomic<size_t> head_{0};
  alignas(64) std::atomic<size_t> tail_{0};
};
