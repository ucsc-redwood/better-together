#pragma once

#include <curl/curl.h>
#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

// ----------------------------------------------------------------------------
// The Server Address is fixed for now.
// ----------------------------------------------------------------------------

// <base_url>/<base_dir>/<device_id>/<app_name>/schedule_<schedule_id>.json
//
// e.g., http://192.168.1.95/schedule_files_v2/3A021JEHN02756/CifarSparse/schedule_001.json

constexpr const char* kScheduleBaseUrl = "http://192.168.1.95/";
constexpr const char* kDefaultScheduleBaseDir = "schedule_files_v2/";

[[nodiscard]]
static std::string make_full_url(const std::string& base_dir,
                                 const std::string& device_id,
                                 const std::string& app_name,
                                 int schedule_id) {
  return kScheduleBaseUrl + base_dir + "/" + device_id + "/" + app_name + "/schedule_" +
         std::to_string(schedule_id) + ".json";
}

[[nodiscard]]
static std::string make_full_url(const std::string& device_id,
                                 const std::string& app_name,
                                 int schedule_id) {
  return make_full_url(kDefaultScheduleBaseDir, device_id, app_name, schedule_id);
}

// ----------------------------------------------------------------------------
// Curl JSON Functions
// ----------------------------------------------------------------------------

// Callback function for libcurl: appends received data to a std::string.
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
  const size_t totalSize = size * nmemb;
  auto* readBuffer = static_cast<std::string*>(userp);
  readBuffer->append(static_cast<char*>(contents), totalSize);
  return totalSize;
}

// fetch_json_from_url: downloads the JSON text from the given URL, parses it, and returns the JSON
// object.
[[nodiscard]]
static nlohmann::json fetch_json_from_url(const std::string& url) {
  spdlog::info("Fetching JSON from URL: {}", url);

  // Initialize variables for libcurl.
  CURL* curl_handle = nullptr;
  CURLcode res;
  std::string readBuffer;

  // Global initialization for libcurl.
  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl_handle = curl_easy_init();
  if (!curl_handle) {
    curl_global_cleanup();
    throw std::runtime_error("Failed to create CURL connection");
  }

  curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, WriteCallback);
  curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, &readBuffer);
  curl_easy_setopt(curl_handle, CURLOPT_USERAGENT, "libcurl-agent/1.0");

  // Perform the request, res will get the return code.
  res = curl_easy_perform(curl_handle);
  if (res != CURLE_OK) {
    std::string error_message =
        "curl_easy_perform() failed: " + std::string(curl_easy_strerror(res));
    curl_easy_cleanup(curl_handle);
    curl_global_cleanup();
    throw std::runtime_error(error_message);
  }

  // Cleanup libcurl resources.
  curl_easy_cleanup(curl_handle);
  curl_global_cleanup();

  // Parse the JSON from the fetched string.
  nlohmann::json j = nlohmann::json::parse(readBuffer);
  return j;
}

// // Example main function to test the above function.
// int main() {
//   // Replace with your actual remote JSON URL.
//   std::string url = "https://api.example.com/data.json";

//   try {
//     nlohmann::json data = fetch_json_from_url(url);
//     // Print the fetched and parsed JSON with indentation.
//     std::cout << data.dump(4) << std::endl;
//   } catch (const std::exception& ex) {
//     std::cerr << "Error: " << ex.what() << std::endl;
//     return EXIT_FAILURE;
//   }

//   return EXIT_SUCCESS;
// }
