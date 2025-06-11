#include <omp.h>

#include <chrono>
#include <iostream>
#include <vector>

// Function to perform matrix multiplication
void matrix_multiply(const std::vector<std::vector<double>>& A,
                     const std::vector<std::vector<double>>& B,
                     std::vector<std::vector<double>>& C,
                     bool parallel) {
  int n = A.size();

  if (parallel) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        C[i][j] = 0;
        for (int k = 0; k < n; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        C[i][j] = 0;
        for (int k = 0; k < n; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
  }
}

int main() {
  // Print OpenMP information
  std::cout << "OpenMP version: " << _OPENMP << std::endl;
  std::cout << "Number of available processors: " << omp_get_num_procs() << std::endl;
  std::cout << "Max threads: " << omp_get_max_threads() << std::endl;

  // Matrix size
  const int n = 1000;

  // Initialize matrices
  std::vector<std::vector<double>> A(n, std::vector<double>(n, 1.0));
  std::vector<std::vector<double>> B(n, std::vector<double>(n, 1.0));
  std::vector<std::vector<double>> C(n, std::vector<double>(n, 0.0));

  // Test serial execution
  auto start = std::chrono::high_resolution_clock::now();
  matrix_multiply(A, B, C, false);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> serial_time = end - start;
  std::cout << "Serial execution time: " << serial_time.count() << " seconds" << std::endl;

  // Test parallel execution
  start = std::chrono::high_resolution_clock::now();
  matrix_multiply(A, B, C, true);
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> parallel_time = end - start;
  std::cout << "Parallel execution time: " << parallel_time.count() << " seconds" << std::endl;

  // Calculate speedup
  double speedup = serial_time.count() / parallel_time.count();
  std::cout << "Speedup: " << speedup << "x" << std::endl;

  return 0;
}
