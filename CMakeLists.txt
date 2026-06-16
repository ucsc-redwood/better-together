# BetterTogether — CMake build (WORK IN PROGRESS).
#
# A separate, modern-CMake build that lives alongside the xmake build during the
# migration (see docs/reports-for-human/cmake-migration-rfc.md). xmake stays the source of truth
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
option(BT_ENABLE_CUDA      "Build the CUDA backend"        ON)
option(BT_ENABLE_VULKAN    "Build the Vulkan backend"      OFF)
option(BT_BUILD_TESTS      "Build unit tests"              ON)
option(BT_BUILD_BENCHMARKS "Build benchmark (bm-*) targets" OFF)
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
if(BT_BUILD_BENCHMARKS)
  CPMAddPackage(NAME benchmark GITHUB_REPOSITORY google/benchmark GIT_TAG v1.9.5
                OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_INSTALL OFF"
                        "BENCHMARK_ENABLE_GTEST_TESTS OFF")
  # Header-only JSON, for bm-gen-logs reading schedule files (replaces the old
  # curl/HTTP fetch -- schedules are now plain local files, see schedule_source.hpp).
  CPMAddPackage("gh:nlohmann/json@3.11.3")
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

# ---- OMP single-app runners (executor; BT-Implementer single-app smoke) -----
# Always built (not gated behind BT_BUILD_BENCHMARKS). ${ARGN} lets a runner
# pull in its app's dispatcher .cpp when it isn't part of bt::core (e.g. octree).
function(bt_add_omp_run name src)
  add_executable(${name} ${src} ${ARGN})
  target_link_libraries(${name} PRIVATE bt::core bt::openmp)
endfunction()

bt_add_omp_run(run-cifar-dense-omp builtin-apps/cifar-dense/omp/main.cpp)

# ---- utility / diagnostics (standalone CPU tools) -------------------------
add_executable(test-affinity utility/test_affinity.cpp)   # core-pinning probe
target_link_libraries(test-affinity PRIVATE bt::core bt::openmp)
add_executable(query-cpuinfo utility/query_cpuinfo.cpp)   # CPU topology (Linux/Android)
add_executable(stressor-omp utility/stress/stressor-omp.cpp)  # OMP CPU interference load
target_link_libraries(stressor-omp PRIVATE bt::core bt::openmp)

# ---- OMP per-stage benchmarks (profiler; the paper's per-stage table) ------
# bm_main.cpp registers one Google-benchmark per stage (BM_Stage1..N). Isolated
# per-stage latency on the CPU/OMP backend. (Interference-aware bm-table-* is C3.)
if(BT_BUILD_BENCHMARKS)
  function(bt_add_omp_app name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE bt::core bt::openmp ${ARGN})
  endfunction()
  bt_add_omp_app(bm-tree-omp         builtin-apps/tree/omp/bm_main.cpp         benchmark::benchmark)
  bt_add_omp_app(bm-cifar-dense-omp  builtin-apps/cifar-dense/omp/bm_main.cpp  benchmark::benchmark)
  bt_add_omp_app(bm-cifar-sparse-omp builtin-apps/cifar-sparse/omp/bm_main.cpp benchmark::benchmark)
  # core-type classifier (probe each core, label big/medium/little/super)
  bt_add_omp_app(bm-check-core-types utility/bm_core_type.cpp benchmark::benchmark)
endif()

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

  # ---- CUDA runners (executor) + benchmarks (profiler) ----
  # Sources currently under pipe/; they relocate out of pipe/ when the duplication
  # farm is deleted (RFC Phase C3). curl-based bm-gen-logs is still TBD.
  function(bt_add_cuda_app name src)
    add_executable(${name} ${src})
    set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "${BT_CUDA_ARCH}")
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012 -Xcompiler=-fopenmp>)
    target_link_libraries(${name} PRIVATE bt::cuda bt::core bt::openmp ${ARGN})
  endfunction()

  bt_add_cuda_app(run-pipe-tree-cu         pipe/tree-cu/main.cu)
  bt_add_cuda_app(run-pipe-cifar-dense-cu  pipe/cifar-dense-cu/main.cu)
  bt_add_cuda_app(run-pipe-cifar-sparse-cu pipe/cifar-sparse-cu/main.cu)

  # Single-app CUDA runners (BT-Implementer per-app smoke; sources in builtin-apps).
  bt_add_cuda_app(run-tree-cu          builtin-apps/tree/cuda/main.cu)
  bt_add_cuda_app(run-cifar-dense-cu   builtin-apps/cifar-dense/cuda/main.cu)
  bt_add_cuda_app(run-cifar-sparse-cu  builtin-apps/cifar-sparse/cuda/main.cu)

  # utility: OMP-inside-a-CUDA-TU sanity (test_omp.cu)
  add_executable(test-omp utility/test_omp.cu)
  set_target_properties(test-omp PROPERTIES CUDA_ARCHITECTURES "${BT_CUDA_ARCH}")
  target_compile_options(test-omp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
  target_link_libraries(test-omp PRIVATE bt::openmp)

  if(BT_BUILD_BENCHMARKS)
    bt_add_cuda_app(bm-baseline-tree-cu         pipe/tree-cu/bm_baseline.cu         benchmark::benchmark)
    bt_add_cuda_app(bm-baseline-cifar-dense-cu  pipe/cifar-dense-cu/bm_baseline.cu  benchmark::benchmark)
    bt_add_cuda_app(bm-baseline-cifar-sparse-cu pipe/cifar-sparse-cu/bm_baseline.cu benchmark::benchmark)
    # per-stage (BM_Stage1..N) — tree/cuda has no bm_main.cu upstream, only cifar
    bt_add_cuda_app(bm-cifar-dense-cu  builtin-apps/cifar-dense/cuda/bm_main.cu  benchmark::benchmark)
    bt_add_cuda_app(bm-cifar-sparse-cu builtin-apps/cifar-sparse/cuda/bm_main.cu benchmark::benchmark)
    # fully-offload-vs-normal baseline
    bt_add_cuda_app(bm-fully-tree-cu         pipe/tree-cu/bm_fully_vs_normal.cu         benchmark::benchmark)
    bt_add_cuda_app(bm-fully-cifar-dense-cu  pipe/cifar-dense-cu/bm_fully_vs_normal.cu  benchmark::benchmark)
    bt_add_cuda_app(bm-fully-cifar-sparse-cu pipe/cifar-sparse-cu/bm_fully_vs_normal.cu benchmark::benchmark)
    # interference-aware schedule profiler (the paper's BTPM table). Reads the
    # schedule from a LOCAL file (adb push / scp) -- no libcurl, no HTTP server.
    bt_add_cuda_app(bm-gen-logs-tree-cu         pipe/tree-cu/bm_gen_log.cu         nlohmann_json::nlohmann_json)
    bt_add_cuda_app(bm-gen-logs-cifar-dense-cu  pipe/cifar-dense-cu/bm_gen_log.cu  nlohmann_json::nlohmann_json)
    bt_add_cuda_app(bm-gen-logs-cifar-sparse-cu pipe/cifar-sparse-cu/bm_gen_log.cu nlohmann_json::nlohmann_json)
  endif()
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

  # utility: print the compiled vulkan-hpp header version (compile-time constant)
  add_executable(query-vulkan-hpp-version utility/query_vulkan_hpp_version.cpp)
  target_compile_features(query-vulkan-hpp-version PRIVATE cxx_std_20)
  target_include_directories(query-vulkan-hpp-version SYSTEM PRIVATE
    $<BUILD_INTERFACE:${VulkanHeaders_SOURCE_DIR}/include>)

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

  # ---- Vulkan single-app runners (executor) ----
  # Always built when Vulkan is enabled (not gated behind BT_BUILD_BENCHMARKS).
  function(bt_add_vk_run name src)
    add_executable(${name} ${src} ${ARGN})
    target_link_libraries(${name} PRIVATE bt::vulkan bt::core bt::openmp)
  endfunction()

  bt_add_vk_run(run-tree-vk builtin-apps/tree/vulkan/run_main.cpp)

  # Vulkan GPU interference load (kiss-vk engine; no benchmark, no volk)
  add_executable(stressor-vk utility/stress/stressor-vk.cpp)
  target_link_libraries(stressor-vk PRIVATE bt::vulkan bt::core bt::openmp)

  # ---- Vulkan benchmarks (profiler) ----
  if(BT_BUILD_BENCHMARKS)
    function(bt_add_vk_app name src)
      add_executable(${name} ${src})
      target_link_libraries(${name} PRIVATE bt::vulkan bt::core bt::openmp ${ARGN})
    endfunction()
    bt_add_vk_app(bm-baseline-tree-vk         pipe/tree-vk/bm_baseline.cpp         benchmark::benchmark)
    bt_add_vk_app(bm-baseline-cifar-dense-vk  pipe/cifar-dense-vk/bm_baseline.cpp  benchmark::benchmark)
    bt_add_vk_app(bm-baseline-cifar-sparse-vk pipe/cifar-sparse-vk/bm_baseline.cpp benchmark::benchmark)
    # per-stage (BM_Stage1..N)
    bt_add_vk_app(bm-tree-vk         builtin-apps/tree/vulkan/bm_main.cpp         benchmark::benchmark)
    bt_add_vk_app(bm-cifar-dense-vk  builtin-apps/cifar-dense/vulkan/bm_main.cpp  benchmark::benchmark)
    bt_add_vk_app(bm-cifar-sparse-vk builtin-apps/cifar-sparse/vulkan/bm_main.cpp benchmark::benchmark)
    # fully-offload-vs-normal baseline
    bt_add_vk_app(bm-fully-tree-vk         pipe/tree-vk/bm_fully_vs_normal.cpp         benchmark::benchmark)
    bt_add_vk_app(bm-fully-cifar-dense-vk  pipe/cifar-dense-vk/bm_fully_vs_normal.cpp  benchmark::benchmark)
    bt_add_vk_app(bm-fully-cifar-sparse-vk pipe/cifar-sparse-vk/bm_fully_vs_normal.cpp benchmark::benchmark)
    # interference-aware schedule profiler (the paper's BTPM table). Reads the
    # schedule from a LOCAL file (adb push / scp) -- no libcurl, no HTTP server.
    bt_add_vk_app(bm-gen-logs-tree-vk         pipe/tree-vk/bm_gen_log.cpp         nlohmann_json::nlohmann_json)
    bt_add_vk_app(bm-gen-logs-cifar-dense-vk  pipe/cifar-dense-vk/bm_gen_log.cpp  nlohmann_json::nlohmann_json)
    bt_add_vk_app(bm-gen-logs-cifar-sparse-vk pipe/cifar-sparse-vk/bm_gen_log.cpp nlohmann_json::nlohmann_json)
  endif()
endif()

# ---- tests ----------------------------------------------------------------
if(BT_BUILD_TESTS)
  enable_testing()

  # Every test is tagged with a backend LABEL so CI can gate per-runner:
  #   ctest -L omp     (every PR, desktop, no GPU)
  #   ctest -L cuda    (Jetson self-hosted runner)
  #   ctest -L vulkan  (integrated-GPU box / Android via adb)
  # Hardware-gated tests GTEST_SKIP() when the device/cores are absent; gtest
  # still exits 0 on a skip, so a skipped test is a pass (not a CI failure) —
  # no SKIP_RETURN_CODE needed. Extra source args (${ARGN}) let a test pull in
  # its app's dispatcher .cpp (e.g. octree, whose stages aren't in bt::core).

  # CPU/OpenMP unit tests.
  function(bt_add_omp_test name src)
    add_executable(${name} ${src} ${ARGN})
    target_link_libraries(${name} PRIVATE bt::core GTest::gtest GTest::gtest_main bt::openmp)
    add_test(NAME ${name} COMMAND ${name} --device ${BT_TEST_DEVICE})
    set_tests_properties(${name} PROPERTIES LABELS "omp")
  endfunction()

  bt_add_omp_test(test-tree-omp         builtin-apps/tree/omp/test_main.cpp)
  bt_add_omp_test(test-cifar-dense-omp  builtin-apps/cifar-dense/omp/test_main.cpp)
  bt_add_omp_test(test-cifar-sparse-omp builtin-apps/cifar-sparse/omp/test_main.cpp)
  # test_sort.cpp is the one suite with real correctness checks (is_sorted +
  # permutation oracle). Kept after removing the dead standalone octree app
  # (the live octree pipeline is the `tree` app; see git history).
  bt_add_omp_test(test-sort-omp         builtin-apps/octree/omp/test_sort.cpp)
  # Pipeline schedule contract: chunks must contiguously cover [1, n_stages].
  # The correctness gate the pipeline path was missing (catches the dropped-stage class).
  bt_add_omp_test(test-schedule-omp     builtin-apps/pipeline/test_schedule.cpp)

  # CUDA unit tests — built on x86, run on the Jetson.
  if(BT_ENABLE_CUDA)
    function(bt_add_cuda_test name src)
      add_executable(${name} ${src} ${ARGN})
      set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "${BT_CUDA_ARCH}")
      target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012 -Xcompiler=-fopenmp>)
      target_link_libraries(${name} PRIVATE bt::cuda bt::core GTest::gtest GTest::gtest_main bt::openmp)
      add_test(NAME ${name} COMMAND ${name} --device ${BT_TEST_DEVICE})
      set_tests_properties(${name} PROPERTIES LABELS "cuda")
    endfunction()

    bt_add_cuda_test(test-tree-cu         builtin-apps/tree/cuda/test_main.cu)
    bt_add_cuda_test(test-cifar-dense-cu  builtin-apps/cifar-dense/cuda/test_main.cu)
    bt_add_cuda_test(test-cifar-sparse-cu builtin-apps/cifar-sparse/cuda/test_main.cu)
  endif()

  # Vulkan unit tests — run on an integrated-GPU box (rocky-ryzen) or the Jetson.
  if(BT_ENABLE_VULKAN)
    function(bt_add_vk_test name src)
      add_executable(${name} ${src} ${ARGN})
      target_link_libraries(${name} PRIVATE bt::vulkan bt::core GTest::gtest GTest::gtest_main bt::openmp)
      add_test(NAME ${name} COMMAND ${name} --device ${BT_TEST_DEVICE})
      set_tests_properties(${name} PROPERTIES LABELS "vulkan")
    endfunction()

    # kiss-vk engine self-test takes no --device.
    add_executable(test-kiss-vk builtin-apps/common/kiss-vk/test_main.cpp)
    target_link_libraries(test-kiss-vk PRIVATE bt::vulkan bt::core GTest::gtest GTest::gtest_main bt::openmp)
    add_test(NAME test-kiss-vk COMMAND test-kiss-vk)
    set_tests_properties(test-kiss-vk PROPERTIES LABELS "vulkan")

    bt_add_vk_test(test-tree-vk         builtin-apps/tree/vulkan/test_main.cpp)
    bt_add_vk_test(test-cifar-dense-vk  builtin-apps/cifar-dense/vulkan/test_main.cpp)
    bt_add_vk_test(test-cifar-sparse-vk builtin-apps/cifar-sparse/vulkan/test_main.cpp)
  endif()
endif()
