#include <sys/sysinfo.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Function to get the CPU model name from /proc/cpuinfo
std::string getCPUModel() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.find("Hardware") != std::string::npos || line.find("Processor") != std::string::npos) {
      return line.substr(line.find(":") + 2);  // Extract the value after ':'
    }
  }
  return "Unknown";
}

// Function to get the number of CPU cores
int getCPUCores() {
  return sysconf(_SC_NPROCESSORS_ONLN);  // Get online processors count
}

// Function to get CPU architecture
std::string getCPUArchitecture() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.find("CPU architecture") != std::string::npos) {
      return line.substr(line.find(":") + 2);
    }
  }
  return "Unknown";
}

// Function to read CPU frequency
std::vector<std::string> getCPUFrequencies() {
  std::vector<std::string> frequencies;
  int coreCount = getCPUCores();

  for (int i = 0; i < coreCount; ++i) {
    std::ostringstream path;
    path << "/sys/devices/system/cpu/cpu" << i << "/cpufreq/scaling_cur_freq";
    std::ifstream freqFile(path.str());
    if (freqFile) {
      std::string freq;
      std::getline(freqFile, freq);
      frequencies.push_back(freq + " kHz");
    } else {
      frequencies.push_back("Unavailable");
    }
  }
  return frequencies;
}

// Function to get CPU features
std::string getCPUFeatures() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.find("Features") != std::string::npos) {
      return line.substr(line.find(":") + 2);
    }
  }
  return "Unknown";
}

int main() {
  std::cout << "Android CPU Information:\n";
  std::cout << "-------------------------\n";
  std::cout << "CPU Model: " << getCPUModel() << "\n";
  std::cout << "CPU Architecture: " << getCPUArchitecture() << "\n";
  std::cout << "Number of CPU Cores: " << getCPUCores() << "\n";
  std::cout << "CPU Features: " << getCPUFeatures() << "\n";

  std::vector<std::string> frequencies = getCPUFrequencies();
  std::cout << "CPU Frequencies per Core:\n";
  for (size_t i = 0; i < frequencies.size(); ++i) {
    std::cout << "  Core " << i << ": " << frequencies[i] << "\n";
  }

  return 0;
}
