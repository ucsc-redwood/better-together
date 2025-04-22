#include <CLI/CLI.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct MyData {
  std::vector<int> ids;
  std::vector<float> scores;
  std::vector<std::string> names;  // we'll handle this separately since strings are tricky
};

bool save(const MyData& data, const std::string& filename) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return false;
  }

  try {
    // Save vector<int>
    size_t size_ids = data.ids.size();
    out.write(reinterpret_cast<const char*>(&size_ids), sizeof(size_ids));
    out.write(reinterpret_cast<const char*>(data.ids.data()), sizeof(int) * size_ids);

    // Save vector<float>
    size_t size_scores = data.scores.size();
    out.write(reinterpret_cast<const char*>(&size_scores), sizeof(size_scores));
    out.write(reinterpret_cast<const char*>(data.scores.data()), sizeof(float) * size_scores);

    // Save vector<string>
    size_t size_names = data.names.size();
    out.write(reinterpret_cast<const char*>(&size_names), sizeof(size_names));
    for (const auto& name : data.names) {
      size_t len = name.size();
      out.write(reinterpret_cast<const char*>(&len), sizeof(len));
      out.write(name.data(), len);
    }
  } catch (const std::exception& e) {
    std::cerr << "Error while saving: " << e.what() << std::endl;
    return false;
  }

  out.close();
  return true;
}

bool load(const std::string& filename, MyData& data) {
  if (!std::filesystem::exists(filename)) {
    std::cerr << "File does not exist: " << filename << std::endl;
    return false;
  }

  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    std::cerr << "Failed to open file for reading: " << filename << std::endl;
    return false;
  }

  try {
    // Load vector<int>
    size_t size_ids;
    in.read(reinterpret_cast<char*>(&size_ids), sizeof(size_ids));
    if (size_ids > 1000000) {  // Sanity check to prevent excessive allocation
      std::cerr << "Invalid ids size: " << size_ids << std::endl;
      return false;
    }
    data.ids.resize(size_ids);
    in.read(reinterpret_cast<char*>(data.ids.data()), sizeof(int) * size_ids);

    // Load vector<float>
    size_t size_scores;
    in.read(reinterpret_cast<char*>(&size_scores), sizeof(size_scores));
    if (size_scores > 1000000) {  // Sanity check to prevent excessive allocation
      std::cerr << "Invalid scores size: " << size_scores << std::endl;
      return false;
    }
    data.scores.resize(size_scores);
    in.read(reinterpret_cast<char*>(data.scores.data()), sizeof(float) * size_scores);

    // Load vector<string>
    size_t size_names;
    in.read(reinterpret_cast<char*>(&size_names), sizeof(size_names));
    if (size_names > 1000000) {  // Sanity check to prevent excessive allocation
      std::cerr << "Invalid names size: " << size_names << std::endl;
      return false;
    }
    data.names.resize(size_names);
    for (size_t i = 0; i < size_names; ++i) {
      size_t len;
      in.read(reinterpret_cast<char*>(&len), sizeof(len));
      if (len > 1000000) {  // Sanity check to prevent excessive allocation
        std::cerr << "Invalid string length: " << len << std::endl;
        return false;
      }
      std::string name(len, '\0');
      in.read(name.data(), len);
      data.names[i] = std::move(name);
    }
  } catch (const std::exception& e) {
    std::cerr << "Error while loading: " << e.what() << std::endl;
    return false;
  }

  in.close();
  return true;
}

std::string getAppDataPath() {
#ifdef __ANDROID__
  return "/data/local/tmp/";
#else
  return "./";
#endif
}

int main(int argc, char** argv) {
  CLI::App app{"My PMR Data App"};

  bool save_flag = false;
  app.add_flag("-s, --save", save_flag);
  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  std::string filepath = getAppDataPath() + "data.bin";
  std::cout << "Using file path: " << filepath << std::endl;

  if (save_flag) {
    std::cout << "Saving data..." << std::endl;
    MyData data;
    data.ids = {1, 2, 3};
    data.scores = {4.0f, 5.0f, 6.0f};
    data.names = {"Alice", "Bob", "Charlie"};

    if (save(data, filepath)) {
      std::cout << "Data saved successfully." << std::endl;
    } else {
      std::cerr << "Failed to save data." << std::endl;
      return 1;
    }
  } else {
    std::cout << "Loading data..." << std::endl;
    MyData data;
    if (load(filepath, data)) {
      std::cout << "Data loaded successfully." << std::endl;
      std::cout << "IDs: ";
      for (const auto& id : data.ids) {
        std::cout << id << " ";
      }
      std::cout << std::endl;
    } else {
      std::cerr << "Failed to load data. Try running with -s to save sample data first."
                << std::endl;
      return 1;
    }
  }

  return 0;
}