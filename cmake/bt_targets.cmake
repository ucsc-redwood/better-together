# Shared target helpers, include()d once from the root CMakeLists so every per-component
# CMakeLists (platform/ runtime/ apps/ profiler/ tools/) can use them. Definitions are
# identical to the pre-split inline helpers; only their home moved.

# ---- executables that link the runtime/core (no test registration) ----
function(bt_add_omp_run name src)
  add_executable(${name} ${src} ${ARGN})
  target_link_libraries(${name} PRIVATE bt::core bt::openmp)
endfunction()

function(bt_add_omp_app name src)        # google-benchmark OMP app (extra libs via ARGN)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE bt::core bt::openmp ${ARGN})
endfunction()

function(bt_add_cuda_app name src)
  add_executable(${name} ${src})
  set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "${BT_CUDA_ARCH}")
  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012 -Xcompiler=-fopenmp>)
  target_link_libraries(${name} PRIVATE bt::cuda bt::core bt::openmp ${ARGN})
endfunction()

function(bt_add_vk_run name src)
  add_executable(${name} ${src} ${ARGN})
  target_link_libraries(${name} PRIVATE bt::vulkan bt::core bt::openmp)
endfunction()

function(bt_add_vk_app name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE bt::vulkan bt::core bt::openmp ${ARGN})
endfunction()

# ---- gtest executables + CTest registration (backend LABEL set; kind label added by caller) ----
function(bt_add_omp_test name src)
  add_executable(${name} ${src} ${ARGN})
  target_link_libraries(${name} PRIVATE bt::core GTest::gtest GTest::gtest_main bt::openmp)
  add_test(NAME ${name} COMMAND ${name} --device ${BT_TEST_DEVICE})
  set_tests_properties(${name} PROPERTIES LABELS "omp")
endfunction()

function(bt_add_cuda_test name src)
  add_executable(${name} ${src} ${ARGN})
  set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "${BT_CUDA_ARCH}")
  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20012 -Xcompiler=-fopenmp>)
  target_link_libraries(${name} PRIVATE bt::cuda bt::core GTest::gtest GTest::gtest_main bt::openmp)
  add_test(NAME ${name} COMMAND ${name} --device ${BT_TEST_DEVICE})
  set_tests_properties(${name} PROPERTIES LABELS "cuda")
endfunction()

function(bt_add_vk_test name src)
  add_executable(${name} ${src} ${ARGN})
  target_link_libraries(${name} PRIVATE bt::vulkan bt::core GTest::gtest GTest::gtest_main bt::openmp)
  add_test(NAME ${name} COMMAND ${name} --device ${BT_TEST_DEVICE})
  set_tests_properties(${name} PROPERTIES LABELS "vulkan")
endfunction()
