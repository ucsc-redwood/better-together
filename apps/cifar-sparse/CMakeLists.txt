# cifar-sparse -- pruned/sparse AlexNet (irregular memory). Float (NearEqual) compares.

target_sources(bt_core PRIVATE omp/dispatchers.cpp)
if(BT_ENABLE_CUDA)
    target_sources(bt_cuda PRIVATE cuda/all_kernels.cu cuda/dispatchers.cu)
endif()
if(BT_ENABLE_VULKAN)
    target_sources(bt_vulkan PRIVATE vulkan/dispatchers.cpp)
endif()

# ---- runners ----
if(BT_ENABLE_CUDA)
    bt_add_cuda_app(run-cifar-sparse-cu cuda/main.cu)
endif()

# ---- benchmarks (per-stage BM_Stage1..N) ----
if(BT_BUILD_BENCHMARKS)
    bt_add_omp_app(bm-cifar-sparse-omp omp/bm_main.cpp benchmark::benchmark)
    if(BT_ENABLE_CUDA)
        bt_add_cuda_app(bm-cifar-sparse-cu cuda/bm_main.cu benchmark::benchmark)
    endif()
    if(BT_ENABLE_VULKAN)
        bt_add_vk_app(bm-cifar-sparse-vk vulkan/bm_main.cpp benchmark::benchmark)
    endif()
endif()

# ---- tests ----
if(BT_BUILD_TESTS)
    bt_add_omp_test(test-cifar-sparse-omp omp/test_main.cpp)
    set_tests_properties(
        test-cifar-sparse-omp
        PROPERTIES LABELS "omp;differential"
    )
    bt_add_omp_test(test-pipeline-e2e-cifar-sparse-omp omp/test_pipeline_main.cpp)
    set_tests_properties(
        test-pipeline-e2e-cifar-sparse-omp
        PROPERTIES LABELS "omp;runtime"
    )

    if(BT_ENABLE_CUDA)
        bt_add_cuda_test(test-cifar-sparse-cu cuda/test_main.cu)
        set_tests_properties(
            test-cifar-sparse-cu
            PROPERTIES LABELS "cuda;differential"
        )
        bt_add_cuda_test(test-pipeline-e2e-cifar-sparse-cu cuda/test_pipeline_main_cu.cu)
        set_tests_properties(
            test-pipeline-e2e-cifar-sparse-cu
            PROPERTIES LABELS "cuda;runtime"
        )
    endif()
    if(BT_ENABLE_VULKAN)
        bt_add_vk_test(test-cifar-sparse-vk vulkan/test_main.cpp)
        set_tests_properties(
            test-cifar-sparse-vk
            PROPERTIES LABELS "vulkan;differential"
        )
        bt_add_vk_test(test-pipeline-e2e-cifar-sparse-vk vulkan/test_pipeline_main_vk.cpp)
        set_tests_properties(
            test-pipeline-e2e-cifar-sparse-vk
            PROPERTIES LABELS "vulkan;runtime"
        )
    endif()
endif()
