#include "fetch_schedule.hpp"

int main() {
  auto schedule1 = fetch_schedule("3A021JEHN02756", "CifarDense", 1);
  std::cout << "Method 1: " << schedule1["schedule"]["schedule_id"] << std::endl;

  return 0;
}
