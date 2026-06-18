#include <iostream>
#include <numeric>  // for std::accumulate
#include <thread>
#include <vector>

#include "../builtin-apps/affinity.hpp"

/*
 * This improved program:
 * 1. Retrieves total hardware threads (cores) via std::thread::hardware_concurrency().
 * 2. For each core i, launches one thread that tries to bind itself to core i multiple times.
 * 3. Collects success counts, failure counts, and at least one error message (if any).
 * 4. Prints a thorough report, including per-core success rates.
 */

static const int NUM_ATTEMPTS_PER_CORE = 10;

static void run_default_test() {
  // Detect how many CPU cores/threads are available
  unsigned int num_cores = std::thread::hardware_concurrency();
  if (num_cores == 0) {
    // Fallback: If hardware_concurrency() returns 0, assume 1 core.
    num_cores = 1;
  }

  // Data structures for collecting results.
  // success_counts[i] = number of times binding to core i succeeded
  // error_messages[i] = first error message (if any) from attempts on core i
  std::vector<int> success_counts(num_cores, 0);
  std::vector<std::string> error_messages(num_cores);
  std::vector<std::thread> threads;
  threads.reserve(num_cores);

  // Launch one thread per core, each trying to bind multiple times.
  for (unsigned int i = 0; i < num_cores; i++) {
    threads.emplace_back([i, &success_counts, &error_messages]() {
      for (int attempt = 1; attempt <= NUM_ATTEMPTS_PER_CORE; ++attempt) {
        try {
          // Attempt to bind this thread to core i
          bind_thread_to_cores({static_cast<int>(i)});
          success_counts[i]++;
        } catch (const std::exception& e) {
          // Store the first error message we encounter
          if (error_messages[i].empty()) {
            error_messages[i] = e.what();
          }
        }
      }
    });
  }

  // Join all threads
  for (auto& t : threads) {
    t.join();
  }

  // Print report
  std::cout << "\n=== Thread Pinning Reliability Report ===\n\n";
  for (unsigned int i = 0; i < num_cores; i++) {
    std::cout << "Core " << i << ":\n";
    std::cout << "  Attempts: " << NUM_ATTEMPTS_PER_CORE << "\n";
    std::cout << "  Successes: " << success_counts[i] << "\n";
    if (success_counts[i] < NUM_ATTEMPTS_PER_CORE) {
      // At least one failure occurred
      std::cout << "  Failures: " << (NUM_ATTEMPTS_PER_CORE - success_counts[i]) << "\n";
      std::cout << "  First error: "
                << (error_messages[i].empty() ? "<No message>" : error_messages[i]) << "\n";
    }
    std::cout << std::endl;
  }

  // Calculate totals
  int total_attempts = NUM_ATTEMPTS_PER_CORE * static_cast<int>(num_cores);
  int total_successes = std::accumulate(success_counts.begin(), success_counts.end(), 0);

  std::cout << "=== Summary ===\n";
  std::cout << "Total cores tested       : " << num_cores << "\n";
  std::cout << "Total attempts (overall) : " << total_attempts << "\n";
  std::cout << "Total successes          : " << total_successes << "\n";
  std::cout << "Total failures           : " << (total_attempts - total_successes) << "\n\n";
}

int main() {
  run_default_test();

  // // Parse command line arguments if provided
  // std::vector<int> target_cores;
  // if (argc > 1) {

  //   // Convert command line arguments to integers
  //   for (int i = 1; i < argc; i++) {
  //     try {
  //       target_cores.push_back(std::atoi(argv[i]));
  //     } catch (const std::exception& e) {
  //       std::cerr << "Error parsing argument '" << argv[i] << "': " << e.what() << std::endl;
  //       return 1;
  //     }
  //   }

  //   std::vector<std::thread> threads;
  //   threads.reserve(target_cores.size());

  //   for (int core : target_cores) {
  //     threads.emplace_back([core]() {
  //       bind_txmakhread_to_cores({core});
  //       while (true);
  //     });
  //   }

  //   for (auto& t : threads) {
  //     t.join();
  //   }

  // } else {
  // }

  return 0;
}
