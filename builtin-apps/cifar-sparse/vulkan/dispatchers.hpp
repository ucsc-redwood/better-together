
#include "../../common/kiss-vk/engine.hpp"
#include "../sparse_appdata.hpp"

namespace cifar_sparse::vulkan {

// How many images to process per iteration together
constexpr auto kNumBatches = 1;

class Singleton {
 public:
  // Delete copy constructor and assignment operator to prevent copies
  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

  static Singleton &getInstance() {
    static Singleton instance;
    return instance;
  }

  kiss_vk::VulkanMemoryResource::memory_resource *get_mr() { return engine.get_mr(); }

  void process_stage_1(cifar_sparse::AppData &app_data);
  void process_stage_2(cifar_sparse::AppData &app_data);
  void process_stage_3(cifar_sparse::AppData &app_data);
  void process_stage_4(cifar_sparse::AppData &app_data);
  void process_stage_5(cifar_sparse::AppData &app_data);
  void process_stage_6(cifar_sparse::AppData &app_data);
  void process_stage_7(cifar_sparse::AppData &app_data);
  void process_stage_8(cifar_sparse::AppData &app_data);
  void process_stage_9(cifar_sparse::AppData &app_data);

  template <int stage>
    requires(stage >= 1 && stage <= 9)
  void run_stage(cifar_sparse::AppData &app_data) {
    if constexpr (stage == 1) {
      process_stage_1(app_data);
    } else if constexpr (stage == 2) {
      process_stage_2(app_data);
    } else if constexpr (stage == 3) {
      process_stage_3(app_data);
    } else if constexpr (stage == 4) {
      process_stage_4(app_data);
    } else if constexpr (stage == 5) {
      process_stage_5(app_data);
    } else if constexpr (stage == 6) {
      process_stage_6(app_data);
    } else if constexpr (stage == 7) {
      process_stage_7(app_data);
    } else if constexpr (stage == 8) {
      process_stage_8(app_data);
    } else if constexpr (stage == 9) {
      process_stage_9(app_data);
    }
  }

 private:
  Singleton();
  ~Singleton() { spdlog::info("Singleton instance destroyed."); }

  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> algorithms;
};

}  // namespace cifar_sparse::vulkan
