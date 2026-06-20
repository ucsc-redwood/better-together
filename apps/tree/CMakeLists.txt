# tree -- 3D octree construction (morton -> sort -> unique -> radix-tree ->
# edge-count -> prefix-sum -> octree-build). Exact (integer/structural) compares.
# tree_appdata is shared by all three backends, so it lives in bt::core.

target_sources(
    bt_core
    PRIVATE tree_appdata.cpp safe_tree_appdata.cpp omp/dispatchers.cpp
)
if(BT_ENABLE_CUDA)
    target_sources(
        bt_cuda
        PRIVATE
            cuda/01_morton.cu
            cuda/02_sort.cu
            cuda/03_unique.cu
            cuda/04_radix_tree.cu
            cuda/05_edge_count.cu
            cuda/06_prefix_sum.cu
            cuda/07_octree.cu
            cuda/dispatchers.cu
    )
endif()
if(BT_ENABLE_VULKAN)
    target_sources(bt_vulkan PRIVATE vulkan/dispatchers.cpp)
endif()

# ---- runners (BT-Implementer per-app smoke) ----
if(BT_ENABLE_CUDA)
    bt_add_cuda_app(run-tree-cu cuda/main.cu)
endif()
if(BT_ENABLE_VULKAN)
    bt_add_vk_run(run-tree-vk vulkan/run_main.cpp)
endif()

# ---- benchmarks (per-stage BM_Stage1..N; tree/cuda has no bm_main upstream) ----
if(BT_BUILD_BENCHMARKS)
    bt_add_omp_app(bm-tree-omp omp/bm_main.cpp benchmark::benchmark)
    if(BT_ENABLE_VULKAN)
        bt_add_vk_app(bm-tree-vk vulkan/bm_main.cpp benchmark::benchmark)
    endif()
endif()

# ---- tests ----
if(BT_BUILD_TESTS)
    bt_add_omp_test(test-tree-omp omp/test_main.cpp)
    set_tests_properties(test-tree-omp PROPERTIES LABELS "omp;differential")
    # tree through the REAL concurrent worker/SPSC ring with a multi-tier OMP schedule.
    bt_add_omp_test(test-pipeline-e2e-omp omp/test_pipeline_main.cpp)
    set_tests_properties(test-pipeline-e2e-omp PROPERTIES LABELS "omp;runtime")

    if(BT_ENABLE_CUDA)
        bt_add_cuda_test(test-tree-cu cuda/test_main.cu)
        set_tests_properties(test-tree-cu PROPERTIES LABELS "cuda;differential")
        # hybrid OMP|CUDA schedule: concurrent CPU+GPU on zero-copy pinned UMA.
        bt_add_cuda_test(test-pipeline-e2e-cu cuda/test_pipeline_main_cu.cu)
        set_tests_properties(
            test-pipeline-e2e-cu
            PROPERTIES LABELS "cuda;runtime"
        )
    endif()
    if(BT_ENABLE_VULKAN)
        bt_add_vk_test(test-tree-vk vulkan/test_main.cpp)
        set_tests_properties(
            test-tree-vk
            PROPERTIES LABELS "vulkan;differential"
        )
        # hybrid OMP|Vulkan schedule: concurrent CPU+GPU on shared UMA.
        bt_add_vk_test(test-pipeline-e2e-vk vulkan/test_pipeline_main_vk.cpp)
        set_tests_properties(
            test-pipeline-e2e-vk
            PROPERTIES LABELS "vulkan;runtime"
        )
    endif()
endif()
