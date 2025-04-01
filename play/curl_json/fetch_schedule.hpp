#include <curl/curl.h>

#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

// Callback function for curl to write response data
inline size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
  size_t totalSize = size * nmemb;
  output->append((char*)contents, totalSize);
  return totalSize;
}

/**
 * Fetches a schedule JSON file using curl from a server.
 *
 * @param device_id Optional device ID (e.g., "jetson")
 * @param application Optional application name (e.g., "CifarDense")
 * @param schedule_id Optional schedule ID number (e.g., 1)
 * @param filepath Optional direct filepath to the JSON file
 * @return Parsed JSON data
 * @throws std::runtime_error if the request fails or parsing fails
 * @throws std::invalid_argument if neither filepath nor all three parameters are provided
 */
inline nlohmann::json fetch_schedule(const std::string& device_id = "",
                                     const std::string& application = "",
                                     int schedule_id = -1,
                                     const std::string& filepath = "") {
  const std::string SERVER_ADDRESS = "192.168.1.95";
  std::string url;

  if (!filepath.empty()) {
    // Use the direct filepath if provided
    url = "http://" + SERVER_ADDRESS + "/" + filepath;
  } else if (!device_id.empty() && !application.empty() && schedule_id >= 0) {
    // Format schedule_id to ensure it's a 3-digit number with leading zeros
    std::ostringstream ss;
    ss << std::setw(3) << std::setfill('0') << schedule_id;
    std::string schedule_num = ss.str();

    url = "http://" + SERVER_ADDRESS + "/schedule_files_v2/" + device_id + "/" + application +
          "/schedule_" + schedule_num + ".json";
  } else {
    throw std::invalid_argument(
        "Either provide a filepath or all three parameters: device_id, application, and "
        "schedule_id");
  }

  // Initialize curl
  CURL* curl = curl_easy_init();
  if (!curl) {
    throw std::runtime_error("Failed to initialize curl");
  }

  std::string response;

  try {
    // Set curl options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);

    std::cout << "Connecting to: " << url << std::endl;

    // Perform the request
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      std::string error_msg = "Curl error: ";
      error_msg += curl_easy_strerror(res);

      long http_code = 0;
      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
      if (http_code > 0) {
        error_msg += " (HTTP response code: " + std::to_string(http_code) + ")";
      }

      throw std::runtime_error(error_msg);
    }

    // Clean up curl
    curl_easy_cleanup(curl);

    // Parse JSON response
    try {
      return nlohmann::json::parse(response);
    } catch (const nlohmann::json::parse_error& e) {
      throw std::runtime_error("Failed to parse JSON data: " + std::string(e.what()));
    }
  } catch (const std::exception& e) {
    curl_easy_cleanup(curl);
    throw;
  }
}
