#pragma once

#include <iostream>
#include <string>
#include <vector>

[[nodiscard]]
inline std::pair<int, std::vector<char*>> sanitize_argc_argv_for_benchmark(
    const int argc, char** argv, const std::string& benchmark_out_filename) {
  // print original args
  for (int i = 0; i < argc; ++i) {
    std::cout << "Original arg " << i << ": " << argv[i] << std::endl;
  }

  // Store strings in a vector that will live as long as the returned pointers
  static std::vector<std::string> stored_strings;
  stored_strings.clear();  // Clear any previous strings

  // Copy original arguments (excluding --device and its value)
  for (int i = 0; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.find("--device=") != std::string::npos) {
      // Skip --device=XXX format
      continue;
    }
    if (arg == "--device") {
      // Skip both --device and its value
      ++i;
      continue;
    }
    stored_strings.push_back(std::move(arg));
  }

  // Add additional arguments
  stored_strings.push_back("--benchmark_repetitions=5");
  stored_strings.push_back("--benchmark_out=" + benchmark_out_filename);
  stored_strings.push_back("--benchmark_out_format=json");

  // Print the full argument list
  std::cout << "\nFull argument list:\n";
  for (size_t i = 0; i < stored_strings.size(); ++i) {
    std::cout << "new_argv[" << i << "]: " << stored_strings[i] << "\n";
  }

  // Create the mutable argv array from the stored strings
  std::vector<char*> mutable_argv;
  for (auto& str : stored_strings) {
    mutable_argv.push_back(str.data());
  }

  return {static_cast<int>(mutable_argv.size()), mutable_argv};
}

// ---------------------------------------------------------------------
// No outfile
// ---------------------------------------------------------------------

[[nodiscard]]
inline std::pair<int, std::vector<char*>> sanitize_argc_argv_for_pipe_benchmark(
    const int argc, char** argv) {
  // print original args
  for (int i = 0; i < argc; ++i) {
    std::cout << "Original arg " << i << ": " << argv[i] << std::endl;
  }

  // Store strings in a vector that will live as long as the returned pointers
  static std::vector<std::string> stored_strings;
  stored_strings.clear();  // Clear any previous strings

  // Copy original arguments (excluding --device and its value)
  for (int i = 0; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.find("--device=") != std::string::npos) {
      // Skip --device=XXX format
      continue;
    }
    if (arg == "--device") {
      // Skip both --device and its value
      ++i;
      continue;
    }

    if (arg == "-s") {
      ++i;
      continue;
    }

    if (arg == "-i") {
      ++i;
      continue;
    }

    stored_strings.push_back(std::move(arg));
  }

  // Add additional arguments
  // stored_strings.push_back("--benchmark_repetitions=5");

  // stored_strings.push_back("--benchmark_out=" + benchmark_out_filename);
  // stored_strings.push_back("--benchmark_out_format=json");

  // Print the full argument list
  std::cout << "\nFull argument list:\n";
  for (size_t i = 0; i < stored_strings.size(); ++i) {
    std::cout << "new_argv[" << i << "]: " << stored_strings[i] << "\n";
  }

  // Create the mutable argv array from the stored strings
  std::vector<char*> mutable_argv;
  for (auto& str : stored_strings) {
    mutable_argv.push_back(str.data());
  }

  return {static_cast<int>(mutable_argv.size()), mutable_argv};
}