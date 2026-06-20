# BetterTogether — CMake build (the build system).
#
# CMake is the sole build system; xmake was retired 2026-06-16 (history:
# docs/reports-for-human/cmake-migration-rfc.md). Not (yet) ported from xmake: the
# volk diagnostics (check-vulkan / query-warpsize) and the NNAPI/EdgeTPU targets --
# their sources still live under utility/ and can be re-added when needed.
#
# Structure: this root is thin -- options, third-party deps, the shared target
# helpers, and the language/header setup -- then one add_subdirectory per component:
#   platform/  bt::core + bt::cuda / bt::vulkan engine libs (+ codegen, shaders)
#   apps/      per-app kernels (contributed to the backend libs via target_sources)
#              and the per-app runner / test / benchmark targets
#   runtime/   the app-agnostic pipeline runtime's schedule-contract test
#   profiler/  BT-Profiler pipeline runners + per-cell benchmark drivers
#   tools/     standalone probes and interference loads
# The backend libs stay bundled (every target links the whole bt::core/cuda/vulkan);
# runtime<->app decoupling is enforced by the guard-runtime-agnostic CTest, not by
# per-dir include scoping (includes remain repo-root-prefixed).
#
# Backends are orthogonal, combinable options (hybrid = enable two at once):
#   BT_ENABLE_OPENMP  always on (the bt::core CPU runtime)
#   BT_ENABLE_CUDA    cross-compiled to Jetson Orin (sm_87), run there
#   BT_ENABLE_VULKAN  x86 -> run on an integrated-GPU box (rocky-ryzen) or Jetson
cmake_minimum_required(VERSION 3.25)

project(
    better_together
    VERSION 0.1.0
    DESCRIPTION "Profile-guided software pipelining for heterogeneous edge SoCs"
    LANGUAGES CXX
)

# ---- options --------------------------------------------------------------
option(BT_ENABLE_CUDA "Build the CUDA backend" ON)
option(BT_ENABLE_VULKAN "Build the Vulkan backend" OFF)
option(BT_BUILD_TESTS "Build unit tests" ON)
option(BT_BUILD_BENCHMARKS "Build benchmark (bm-*) targets" OFF)
set(BT_CUDA_ARCH "87" CACHE STRING "CUDA architecture(s): 87=Orin, 89=Ada")
set(BT_TEST_DEVICE "pc" CACHE STRING "Device spec passed to tests via --device")

# ---- project-wide niceties (kept minimal; prefer per-target settings) ------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# All executables land flat in build/<preset>/ regardless of which component
# subdirectory defines them, so the run-on-*.sh deploy scripts find them at the
# same paths as before the per-component add_subdirectory split.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
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
cpmaddpackage("gh:gabime/spdlog@1.15.3")
cpmaddpackage("gh:CLIUtils/CLI11@2.4.2")
cpmaddpackage(NAME glm GITHUB_REPOSITORY g-truc/glm GIT_TAG 1.0.1)
cpmaddpackage(NAME libmorton GITHUB_REPOSITORY Forceflow/libmorton GIT_TAG v0.2.12
              OPTIONS "BUILD_TESTING OFF"
)
# Header-only JSON: device specs parsed by bt::core (conf.cpp) + schedule files
# read by bm-gen-logs. Always fetched (bt::core needs it).
cpmaddpackage("gh:nlohmann/json@3.11.3")
if(BT_BUILD_TESTS)
    cpmaddpackage(NAME googletest GITHUB_REPOSITORY google/googletest GIT_TAG v1.17.0
                  OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON"
    )
endif()
if(BT_BUILD_BENCHMARKS)
    cpmaddpackage(NAME benchmark GITHUB_REPOSITORY google/benchmark GIT_TAG v1.9.5
                  OPTIONS "BENCHMARK_ENABLE_TESTING OFF" "BENCHMARK_ENABLE_INSTALL OFF"
                          "BENCHMARK_ENABLE_GTEST_TESTS OFF"
    )
    # bm-prof stamps the git sha into its provenance JSON. Capture it at BUILD time
    # (not configure time) via an always-run custom target, so a commit+rebuild
    # refreshes the sha; bm_prof_common.hpp picks up the generated header. The
    # script only rewrites the file when the sha changes, so it won't force rebuilds.
    set(BT_GIT_SHA_HEADER ${CMAKE_BINARY_DIR}/generated/bt_git_sha.h)
    add_custom_target(
        bt_git_sha
        BYPRODUCTS ${BT_GIT_SHA_HEADER}
        COMMAND
            ${CMAKE_COMMAND} -DGIT_SRC_DIR=${CMAKE_SOURCE_DIR}
            -DGIT_OUT=${BT_GIT_SHA_HEADER} -P
            ${CMAKE_SOURCE_DIR}/cmake/write_git_sha.cmake
        COMMENT "Capturing git sha for bm-prof provenance"
        VERBATIM
    )
endif()

# Python for the codegen (devices/*.json + vocab.json) and the optimizer/guard tests;
# optional, mirroring the committed-fallback pattern (no python -> committed headers).
find_program(BT_PYTHON NAMES python3 python)

# Shared target helpers: bt_add_{omp,cuda,vk}_{run,app,test}; used by every component.
include(cmake/bt_targets.cmake)

# CUDA is enabled once here so every component's CUDA targets compile (a subdir
# enable_language would not reach sibling dirs). PC preset leaves BT_ENABLE_CUDA OFF,
# so nvcc is never invoked there.
if(BT_ENABLE_CUDA)
    enable_language(CUDA)
endif()

# Vulkan headers fetched here so the SOURCE_DIR vars are visible in platform/apps/tools.
# vulkan-hpp pinned to 1.3.290 (newer removed vk::DynamicLoader); VMA header-only.
if(BT_ENABLE_VULKAN)
    cpmaddpackage(NAME VulkanHeaders
      GITHUB_REPOSITORY KhronosGroup/Vulkan-Headers GIT_TAG v1.3.290 DOWNLOAD_ONLY YES
    )
    cpmaddpackage(NAME VulkanMemoryAllocator
      GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator GIT_TAG v3.1.0 DOWNLOAD_ONLY YES
    )
endif()

# ---- tests: infra + the cross-cutting (non-app) suites --------------------
if(BT_BUILD_TESTS)
    # Hardware-gated tests GTEST_SKIP() when the device is absent; gtest exits 0 on a
    # skip, so a skip is a pass. Backend LABELs gate per-runner: -L omp (every PR) /
    # -L cuda (Jetson) / -L vulkan (iGPU box or Android via adb). Per-app tests are
    # registered in apps/<app>/CMakeLists.txt; the two cross-cutting suites are here.
    enable_testing()

    if(BT_PYTHON)
        # BT-Optimizer (Python/z3) suite -- own label "optimizer", deliberately NOT in any
        # backend gate (`ctest -L omp` stays C++-only). pytest over optimizer/tests.
        add_test(
            NAME test-optimizer
            COMMAND
                ${BT_PYTHON} -m pytest ${CMAKE_SOURCE_DIR}/optimizer/tests -q
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/optimizer
        )
        set_tests_properties(test-optimizer PROPERTIES LABELS "optimizer")

        # guard-runtime-agnostic: fail if runtime/ references an app by name in CODE
        # (comments excluded). Build-enforces the P3 decoupling -- a header-only leak
        # link-scoping cannot catch. LABEL "guard"; hardware-free, rides the omp gate.
        add_test(
            NAME guard-runtime-agnostic
            COMMAND
                ${BT_PYTHON}
                ${CMAKE_SOURCE_DIR}/scripts/guard_runtime_agnostic.py
        )
        set_tests_properties(
            guard-runtime-agnostic
            PROPERTIES LABELS "omp;guard"
        )

        # validate-devices: structural schema check of devices/*.json + a golden core-topology
        # characterization that locks each device's affinity map against silent drift. The
        # embed step bakes these JSONs into the runtime verbatim, so this is the guard that a
        # structurally-valid-but-wrong value can't slip in unnoticed. Hardware-free -> omp gate.
        add_test(
            NAME validate-devices
            COMMAND ${BT_PYTHON} ${CMAKE_SOURCE_DIR}/scripts/validate_devices.py
        )
        set_tests_properties(validate-devices PROPERTIES LABELS "omp;guard")
    endif()
endif()

# ---- components -----------------------------------------------------------
add_subdirectory(platform) # bt::core / bt::cuda / bt::vulkan (must precede apps/)
add_subdirectory(apps) # app kernels (target_sources -> backend libs) + targets
add_subdirectory(runtime) # schedule-contract test
add_subdirectory(profiler) # pipeline runners + benchmark drivers
add_subdirectory(tools) # probes + interference loads
