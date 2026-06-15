# BetterTogether — CMake build (WORK IN PROGRESS).
#
# A separate, modern-CMake build that lives alongside the xmake build during the
# migration (see docs/CMAKE-MIGRATION-RFC.md). xmake stays the source of truth
# until each target passes a parity gate.
#
# Backends are orthogonal, combinable options (hybrid = enable two at once):
#   BT_ENABLE_OPENMP  always on (the bt::core CPU runtime)
#   BT_ENABLE_CUDA    cross-compiled to Jetson Orin (sm_87), run there
#   BT_ENABLE_VULKAN  x86 -> run on an integrated-GPU box (rocky-ryzen) or Jetson
#
# Modern-CMake conventions: target-scoped settings only, per-target compile
# features, namespaced ALIAS targets (bt::core/cuda/vulkan), imported targets
# for deps, generator expressions, explicit source lists (no file(GLOB)).
cmake_minimum_required(VERSION 3.25)

project(better_together
  VERSION 0.1.0
  DESCRIPTION "Profile-guided software pipelining for heterogeneous edge SoCs"
  LANGUAGES CXX)

# ---- options --------------------------------------------------------------
option(BT_ENABLE_CUDA   "Build the CUDA backend"   ON)
option(BT_ENABLE_VULKAN "Build the Vulkan backend" OFF)
option(BT_BUILD_TESTS   "Build unit tests"         ON)
set(BT_CUDA_ARCH   "87" CACHE STRING "CUDA architecture(s): 87=Orin, 89=Ada")
set(BT_TEST_DEVICE "pc" CACHE STRING "Device spec passed to tests via --device")

# ---- project-wide niceties (kept minimal; prefer per-target settings) ------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(cmake/CPM.cmake)

# OpenMP — abstracted as bt::openmp so Android (NDK) can use -fopenmp
# -static-openmp while desktop/Jetson use the imported bt::openmp target.
if(ANDROID)
  add_library(bt_openmp INTERFACE)
  target_compile_options(bt_openmp INTERFACE -fopenmp)
  target_link_options(bt_openmp INTERFACE -fopenmp -static-openmp)
else()
  find_package(OpenMP REQUIRED)
  add_library(bt_openmp INTERFACE)
  target_link_libraries(bt_openmp INTERFACE OpenMP::OpenMP_CXX)
endif()
add_library(bt::openmp ALIAS bt_openmp)

# ---- third-party deps (versions mirror xmake's) ---------------------------
# spdlog 1.15.x bundles a newer fmt that compiles cleanly on modern clang
# (Android NDK r29); 1.14.1's bundled fmt hit a consteval error there.
CPMAddPackage("gh:gabime/spdlog@1.15.3")
CPMAddPackage("gh:CLIUtils/CLI11@2.4.2")
CPMAddPackage(NAME glm GITHUB_REPOSITORY g-truc/glm GIT_TAG 1.0.1)
CPMAddPackage(NAME libmorton GITHUB_REPOSITORY Forceflow/libmorton GIT_TAG v0.2.12
              OPTIONS "BUILD_TESTING OFF")
if(BT_BUILD_TESTS)
  CPMAddPackage(NAME googletest GITHUB_REPOSITORY google/googletest GIT_TAG v1.17.0
                OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON")
endif()

# ---- bt::core — shared CPU/OpenMP runtime (tree + cifar OMP dispatchers) ---
add_library(bt_core STATIC
  builtin-apps/conf.cpp
  builtin-apps/app.cpp
  builtin-apps/hex_dump.cpp
  builtin-apps/tree/tree_appdata.cpp
  builtin-apps/tree/safe_tree_appdata.cpp
  builtin-apps/tree/omp/dispatchers.cpp
  builtin-apps/cifar-dense/omp/dispatchers.cpp
  builtin-apps/cifar-sparse/omp/dispatchers.cpp
)
add_library(bt::core ALIAS bt_core)
target_compile_features(bt_core PUBLIC cxx_std_20)
target_include_directories(bt_core PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(bt_core PUBLIC
  spdlog::spdlog CLI11::CLI11 glm::glm libmorton bt::openmp)

# ---- bt::cuda — CUDA backend (tree + cifar) -------------------------------
if(BT_ENABLE_CUDA)
  enable_language(CUDA)
  add_library(bt_cuda STATIC
    builtin-apps/common/cuda/cu_mem_resource.cu
    builtin-apps/tree/cuda/01_morton.cu
    builtin-apps/tree/cuda/02_sort.cu
    builtin-apps/tree/cuda/03_unique.cu
    builtin-apps/tree/cuda/04_radix_tree.cu
    builtin-apps/tree/cuda/05_edge_count.cu
    builtin-apps/tree/cuda/06_prefix_sum.cu
    builtin-apps/tree/cuda/07_octree.cu
    builtin-apps/tree/cuda/dispatchers.cu
    builtin-apps/cifar-dense/cuda/all_kernels.cu
    builtin-apps/cifar-dense/cuda/dispatchers.cu
    builtin-apps/cifar-sparse/cuda/all_kernels.cu
    builtin-apps/cifar-sparse/cuda/dispatchers.cu
  )
  add_library(bt::cuda ALIAS bt_cuda)
  target_compile_features(bt_cuda PUBLIC cxx_std_20 cuda_std_20)
  target_include_directories(bt_cuda PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
  set_target_properties(bt_cuda PROPERTIES CUDA_ARCHITECTURES "${BT_CUDA_ARCH}")
  target_compile_options(bt_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012 -Xcompiler=-fopenmp>)
  target_link_libraries(bt_cuda PUBLIC bt::core)
endif()

# ---- bt::vulkan — Vulkan backend (kiss-vk + tree/cifar dispatchers) --------
# Uses vulkan-hpp's dynamic dispatch + vk::DynamicLoader (dlopen libvulkan at
# runtime; no volk, no link-time -lvulkan). Pinned to 1.3.290 because newer
# vulkan-hpp removed vk::DynamicLoader. SPIR-V shaders are pre-baked headers.
if(BT_ENABLE_VULKAN)
  CPMAddPackage(NAME VulkanHeaders
    GITHUB_REPOSITORY KhronosGroup/Vulkan-Headers GIT_TAG v1.3.290 DOWNLOAD_ONLY YES)
  CPMAddPackage(NAME VulkanMemoryAllocator
    GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator GIT_TAG v3.1.0 DOWNLOAD_ONLY YES)

  add_library(bt_vulkan STATIC
    builtin-apps/common/kiss-vk/algorithm.cpp
    builtin-apps/common/kiss-vk/base_engine.cpp
    builtin-apps/common/kiss-vk/sequence.cpp
    builtin-apps/common/kiss-vk/vma_pmr.cpp
    builtin-apps/tree/vulkan/dispatchers.cpp
    builtin-apps/cifar-dense/vulkan/dispatchers.cpp
    builtin-apps/cifar-sparse/vulkan/dispatchers.cpp
  )
  add_library(bt::vulkan ALIAS bt_vulkan)
  target_compile_features(bt_vulkan PUBLIC cxx_std_20)
  target_include_directories(bt_vulkan PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
  target_include_directories(bt_vulkan SYSTEM PUBLIC
    $<BUILD_INTERFACE:${VulkanHeaders_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${VulkanMemoryAllocator_SOURCE_DIR}/include>)
  target_link_libraries(bt_vulkan PUBLIC bt::core ${CMAKE_DL_LIBS})
endif()

# ---- tests ----------------------------------------------------------------
if(BT_BUILD_TESTS)
  enable_testing()

  # CPU/OpenMP unit tests.
  function(bt_add_omp_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE bt::core GTest::gtest GTest::gtest_main bt::openmp)
    add_test(NAME ${name} COMMAND ${name} --device ${BT_TEST_DEVICE})
  endfunction()

  bt_add_omp_test(test-tree-omp         builtin-apps/tree/omp/test_main.cpp)
  bt_add_omp_test(test-cifar-dense-omp  builtin-apps/cifar-dense/omp/test_main.cpp)
  bt_add_omp_test(test-cifar-sparse-omp builtin-apps/cifar-sparse/omp/test_main.cpp)

  # CUDA unit tests — built on x86, run on the Jetson.
  if(BT_ENABLE_CUDA)
    function(bt_add_cuda_test name src)
      add_executable(${name} ${src})
      set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "${BT_CUDA_ARCH}")
      target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012 -Xcompiler=-fopenmp>)
      target_link_libraries(${name} PRIVATE bt::cuda bt::core GTest::gtest GTest::gtest_main bt::openmp)
      add_test(NAME ${name} COMMAND ${name} --device ${BT_TEST_DEVICE})
    endfunction()

    bt_add_cuda_test(test-tree-cu         builtin-apps/tree/cuda/test_main.cu)
    bt_add_cuda_test(test-cifar-dense-cu  builtin-apps/cifar-dense/cuda/test_main.cu)
    bt_add_cuda_test(test-cifar-sparse-cu builtin-apps/cifar-sparse/cuda/test_main.cu)
  endif()

  # Vulkan unit tests — run on an integrated-GPU box (rocky-ryzen) or the Jetson.
  if(BT_ENABLE_VULKAN)
    function(bt_add_vk_test name src)
      add_executable(${name} ${src})
      target_link_libraries(${name} PRIVATE bt::vulkan bt::core GTest::gtest GTest::gtest_main bt::openmp)
      add_test(NAME ${name} COMMAND ${name} --device ${BT_TEST_DEVICE})
    endfunction()

    # kiss-vk engine self-test takes no --device.
    add_executable(test-kiss-vk builtin-apps/common/kiss-vk/test_main.cpp)
    target_link_libraries(test-kiss-vk PRIVATE bt::vulkan bt::core GTest::gtest GTest::gtest_main bt::openmp)
    add_test(NAME test-kiss-vk COMMAND test-kiss-vk)

    bt_add_vk_test(test-tree-vk         builtin-apps/tree/vulkan/test_main.cpp)
    bt_add_vk_test(test-cifar-dense-vk  builtin-apps/cifar-dense/vulkan/test_main.cpp)
    bt_add_vk_test(test-cifar-sparse-vk builtin-apps/cifar-sparse/vulkan/test_main.cpp)
  endif()
endif()
