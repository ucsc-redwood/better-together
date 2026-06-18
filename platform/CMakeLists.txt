# platform/ -- the app-agnostic foundation: bt::core (registry + util) and the two
# GPU engine libraries bt::cuda / bt::vulkan. App kernels are NOT listed here; each
# apps/<app>/CMakeLists.txt contributes its own sources via target_sources(), so a
# component owns its files while the link structure stays backend-bundled.

# ---- bt::core -- shared CPU/OpenMP runtime (device registry + utilities) -----
add_library(bt_core STATIC
  registry/conf.cpp
  registry/device_registry.cpp
  util/hex_dump.cpp
)
add_library(bt::core ALIAS bt_core)
target_compile_features(bt_core PUBLIC cxx_std_20)
# Repo root stays the PUBLIC include dir (includes are repo-root-prefixed, e.g.
# "platform/registry/conf.hpp"); runtime/<->app decoupling is enforced by the
# guard-runtime-agnostic CTest, not by per-dir include scoping.
target_include_directories(bt_core PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
target_link_libraries(bt_core PUBLIC
  spdlog::spdlog CLI11::CLI11 glm::glm libmorton bt::openmp nlohmann_json::nlohmann_json)

# device_specs_embedded.hpp (conf.cpp includes it) is generated from devices/*.json by
# scripts/embed_device_specs.py. Regenerate it at build time whenever a device JSON or
# the generator changes, so editing devices/*.json is picked up without a manual step.
# The script writes identical bytes when nothing changed, so a clean tree stays clean;
# if no python is found, the committed header is used as-is.
if(BT_PYTHON)
  file(GLOB BT_DEVICE_JSONS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/devices/*.json)
  add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/platform/registry/generated/device_specs_embedded.hpp
    COMMAND ${BT_PYTHON} ${CMAKE_SOURCE_DIR}/scripts/embed_device_specs.py
    DEPENDS ${BT_DEVICE_JSONS} ${CMAKE_SOURCE_DIR}/scripts/embed_device_specs.py
    COMMENT "Embedding devices/*.json -> device_specs_embedded.hpp"
    VERBATIM)
  add_custom_target(bt_device_specs
    DEPENDS ${CMAKE_SOURCE_DIR}/platform/registry/generated/device_specs_embedded.hpp)
  add_dependencies(bt_core bt_device_specs)
endif()

# Vocabulary codegen: vocab.json -> platform/vocab/generated/bt_vocab.hpp (the
# ProcessorType enum etc., included by conf.hpp) + optimizer/smt/bt_vocab.py. Same
# committed-fallback pattern as the device specs (bt_codegen depends bt_core on it).
include(${CMAKE_SOURCE_DIR}/cmake/bt_codegen.cmake)
bt_codegen(TARGET  bt_vocab
           SCRIPT  ${CMAKE_SOURCE_DIR}/scripts/embed_vocab.py
           DEPENDS ${CMAKE_SOURCE_DIR}/vocab.json
           OUTPUTS ${CMAKE_SOURCE_DIR}/platform/vocab/generated/bt_vocab.hpp
                   ${CMAKE_SOURCE_DIR}/optimizer/smt/bt_vocab.py)

# ---- bt::cuda -- CUDA backend engine (app kernels added by apps/<app>) --------
# enable_language(CUDA) is called once at the root (covers every subdir's CUDA targets).
if(BT_ENABLE_CUDA)
  add_library(bt_cuda STATIC engine/cuda/cu_mem_resource.cu)
  add_library(bt::cuda ALIAS bt_cuda)
  target_compile_features(bt_cuda PUBLIC cxx_std_20 cuda_std_20)
  target_include_directories(bt_cuda PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
  set_target_properties(bt_cuda PROPERTIES CUDA_ARCHITECTURES "${BT_CUDA_ARCH}")
  target_compile_options(bt_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012 -Xcompiler=-fopenmp>)
  target_link_libraries(bt_cuda PUBLIC bt::core)
endif()

# ---- bt::vulkan -- Vulkan backend engine (kiss-vk; app dispatchers added by apps) -
# Uses vulkan-hpp's dynamic dispatch + vk::DynamicLoader (dlopen libvulkan at runtime;
# no volk, no link-time -lvulkan). VulkanHeaders/VMA are fetched at the root so their
# SOURCE_DIR vars are visible here. SPIR-V shaders are pre-baked committed headers.
if(BT_ENABLE_VULKAN)
  add_library(bt_vulkan STATIC
    engine/vulkan/algorithm.cpp
    engine/vulkan/base_engine.cpp
    engine/vulkan/sequence.cpp
    engine/vulkan/vma_pmr.cpp
  )
  add_library(bt::vulkan ALIAS bt_vulkan)
  target_compile_features(bt_vulkan PUBLIC cxx_std_20)
  target_include_directories(bt_vulkan PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
  target_include_directories(bt_vulkan SYSTEM PUBLIC
    $<BUILD_INTERFACE:${VulkanHeaders_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${VulkanMemoryAllocator_SOURCE_DIR}/include>)
  target_link_libraries(bt_vulkan PUBLIC bt::core ${CMAKE_DL_LIBS})

  # Shaders: bt::vulkan compiles the committed *_spv.h directly. `bake-shaders`
  # (opt-in target; needs glslc+xxd) re-bakes them -- it replaces the old root
  # Makefile. See cmake/bt_shaders.cmake for why .comp->.spv is not on ALL.
  include(${CMAKE_SOURCE_DIR}/cmake/bt_shaders.cmake)
  bt_bake_shaders()

  # kiss-vk engine self-test (no app, no --device).
  if(BT_BUILD_TESTS)
    add_executable(test-kiss-vk engine/vulkan/test_main.cpp)
    target_link_libraries(test-kiss-vk PRIVATE bt::vulkan bt::core GTest::gtest GTest::gtest_main bt::openmp)
    add_test(NAME test-kiss-vk COMMAND test-kiss-vk)
    set_tests_properties(test-kiss-vk PROPERTIES LABELS "vulkan;engine")
  endif()
endif()
