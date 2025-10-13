#include <omp.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

// High-precision mathematical stress operations
double stress_math(double x, double y, int iterations) {
  double result = x;
  double temp = y;

  // Perform intensive mathematical operations
  for (int i = 0; i < iterations; i++) {
    // Trigonometric operations (expensive on CPU)
    result = std::sin(result) * std::cos(temp) + std::tan(result * 0.1);
    temp = std::sqrt(std::abs(result)) + std::pow(std::abs(temp), 1.1);

    // Logarithmic operations
    result = std::log(std::abs(result) + 1.0) + std::exp(temp * 0.01);
    temp = std::atan(result) + std::asin(std::clamp(temp * 0.1, -1.0, 1.0));

    // Complex polynomial calculations
    result = result * result * 0.1 + temp * temp * 0.1;
    temp = result * temp + std::sin(result + temp);

    // High-precision calculations
    result = std::fmod(result * 1.618033988749, 1.0) + temp * 0.618033988749;
    temp = std::fmod(temp * 2.718281828459, 1.0) + result * 0.2718281828459;
  }

  return result;
}

// Memory-intensive operations
void stress_memory_access(std::vector<double>& data, int iterations) {
  const size_t n = data.size();
  double accumulator = 0.0;

  for (int i = 0; i < iterations; i++) {
    // Strided memory access pattern
    size_t stride = (i * 7) % n;
    accumulator += data[stride] * (i + 1);

    // Reverse stride access
    size_t rev_stride = (n - 1 - stride) % n;
    accumulator -= data[rev_stride] * (i + 1) * 0.5;

    // Random-like access pattern
    size_t rand_stride = (stride * 1103515245u + 12345u) % n;
    accumulator += std::sin(data[rand_stride]) * std::cos(i);
  }

  // Write back to prevent optimization
  data[0] = accumulator;
}

// Prime number calculation (CPU intensive)
bool is_prime(long long n) {
  if (n <= 1) return false;
  if (n <= 3) return true;
  if (n % 2 == 0 || n % 3 == 0) return false;

  for (long long i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) {
      return false;
    }
  }
  return true;
}

// Count primes in range (very CPU intensive)
long long count_primes(long long start, long long end) {
  long long count = 0;
  for (long long i = start; i <= end; i++) {
    if (is_prime(i)) {
      count++;
    }
  }
  return count;
}

int main() {
  // Get number of threads
  int num_threads = omp_get_max_threads();
  spdlog::info("Starting OpenMP stress test with {} threads", num_threads);

  // Set number of threads
  omp_set_num_threads(num_threads);

  // Configuration
  constexpr double target_duration = 10.0;  // 10 seconds
  constexpr size_t data_size = 1000000;     // 1M elements
  constexpr int math_iterations = 10000;    // Mathematical stress iterations
  constexpr int memory_iterations = 5000;   // Memory access stress iterations

  // Initialize data
  std::vector<double> data(data_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-10.0, 10.0);

  // Fill with random data
  for (size_t i = 0; i < data_size; i++) {
    data[i] = dis(gen);
  }

  // Start timing
  auto start_time = std::chrono::high_resolution_clock::now();
  double elapsed = 0.0;
  int iteration_count = 0;

  spdlog::info("Starting CPU stress test for {} seconds...", target_duration);

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    double thread_result = 0.0;

    while (elapsed < target_duration) {
#pragma omp single
      {
        auto current_time = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double>(current_time - start_time).count();
        iteration_count++;
      }

      if (elapsed >= target_duration) break;

      // Mathematical stress
      double math_result = stress_math(thread_id + iteration_count, thread_result, math_iterations);

      // Memory bandwidth stress
      stress_memory_access(data, memory_iterations);

      // Prime number calculation (very CPU intensive)
      long long prime_start = 1000000 + thread_id * 10000 + iteration_count * 100;
      long long prime_count = count_primes(prime_start, prime_start + 1000);

      // Additional computational stress
      double temp = math_result;
      for (int j = 0; j < 1000; j++) {
        temp = std::sin(temp) + std::cos(temp * 1.1) + std::tan(temp * 0.9);
        temp = std::sqrt(std::abs(temp)) + std::pow(std::abs(temp), 1.05);
        temp = std::fmod(temp * 1.414213562373, 1.0) + 0.1;
      }

      thread_result = temp + prime_count * 0.001;

      // Progress logging (only from thread 0)
      if (thread_id == 0 && iteration_count % 100 == 0) {
        spdlog::info(
            "Stress test running... {:.1f}s elapsed, {} iterations", elapsed, iteration_count);
      }
    }

    // Final intensive computation
    double final_result = 0.0;
    for (int k = 0; k < 10000; k++) {
      final_result += std::sin(k + thread_id) * std::cos(k * thread_id) +
                      std::sqrt(std::abs(k)) * std::pow(std::abs(thread_id), 0.1);
    }

    // Store result to prevent optimization
    data[thread_id % data_size] = final_result + thread_result;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();

  spdlog::info("OpenMP stress test completed!");
  spdlog::info("Total time: {:.2f} seconds", total_elapsed);
  spdlog::info("Total iterations: {}", iteration_count);
  spdlog::info("Threads used: {}", num_threads);
  spdlog::info("Average iteration time: {:.3f} ms", (total_elapsed * 1000.0) / iteration_count);

  // Calculate some final statistics to prevent optimization
  double sum = 0.0;
  for (size_t i = 0; i < std::min(data_size, size_t(1000)); i++) {
    sum += data[i];
  }
  spdlog::info("Final computation result: {:.6f}", sum);

  return 0;
}