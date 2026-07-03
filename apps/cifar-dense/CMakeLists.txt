# cifar-dense -- AlexNet inference on CIFAR-10 (dense). Float (NearEqual) compares.

target_sources(bt_core PRIVATE omp/dispatchers.cpp)

# The OMP kernels in this TU are both the numerical oracle and a scheduled CPU
# target. Default Release flags target baseline x86-64 (SSE2, no FMA); every
# x86 box in play (rocky-ryzen Zen4, GitHub ubuntu runners) is x86-64-v3
# capable (AVX2+FMA, Haswell 2013+), which doubles the vector width and fuses
# the MACs. Scoped to this one source file so the rest of the build stays
# baseline. (aarch64 targets are unaffected: NEON+FMA is baseline there.)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set_source_files_properties(
        omp/dispatchers.cpp
        TARGET_DIRECTORY bt_core
        PROPERTIES COMPILE_OPTIONS "-march=x86-64-v3"
    )
endif()

if(BT_ENABLE_CUDA)
    target_sources(bt_cuda PRIVATE cuda/all_kernels.cu cuda/dispatchers.cu)
endif()
if(BT_ENABLE_VULKAN)
    target_sources(bt_vulkan PRIVATE vulkan/dispatchers.cpp)
endif()

# ---- runners ----
bt_add_omp_run(run-cifar-dense-omp omp/main.cpp)
if(BT_ENABLE_CUDA)
    bt_add_cuda_app(run-cifar-dense-cu cuda/main.cu)
endif()

# ---- benchmarks (per-stage BM_Stage1..N) ----
if(BT_BUILD_BENCHMARKS)
    bt_add_omp_app(bm-cifar-dense-omp omp/bm_main.cpp benchmark::benchmark)
    if(BT_ENABLE_CUDA)
        bt_add_cuda_app(bm-cifar-dense-cu cuda/bm_main.cu benchmark::benchmark)
    endif()
    if(BT_ENABLE_VULKAN)
        bt_add_vk_app(bm-cifar-dense-vk vulkan/bm_main.cpp benchmark::benchmark)
    endif()
endif()

# ---- tests ----
if(BT_BUILD_TESTS)
    bt_add_omp_test(test-cifar-dense-omp omp/test_main.cpp)
    set_tests_properties(
        test-cifar-dense-omp
        PROPERTIES LABELS "omp;differential"
    )
    bt_add_omp_test(test-pipeline-e2e-cifar-dense-omp omp/test_pipeline_main.cpp)
    set_tests_properties(
        test-pipeline-e2e-cifar-dense-omp
        PROPERTIES LABELS "omp;runtime"
    )

    if(BT_ENABLE_CUDA)
        bt_add_cuda_test(test-cifar-dense-cu cuda/test_main.cu)
        set_tests_properties(
            test-cifar-dense-cu
            PROPERTIES LABELS "cuda;differential"
        )
        bt_add_cuda_test(test-pipeline-e2e-cifar-dense-cu cuda/test_pipeline_main_cu.cu)
        set_tests_properties(
            test-pipeline-e2e-cifar-dense-cu
            PROPERTIES LABELS "cuda;runtime"
        )
    endif()
    if(BT_ENABLE_VULKAN)
        bt_add_vk_test(test-cifar-dense-vk vulkan/test_main.cpp)
        set_tests_properties(
            test-cifar-dense-vk
            PROPERTIES LABELS "vulkan;differential"
        )
        bt_add_vk_test(test-pipeline-e2e-cifar-dense-vk vulkan/test_pipeline_main_vk.cpp)
        set_tests_properties(
            test-pipeline-e2e-cifar-dense-vk
            PROPERTIES LABELS "vulkan;runtime"
        )
    endif()
endif()
