# tools/ -- standalone diagnostics & interference loads (not part of any app).

add_executable(test-affinity probes/probe_affinity.cpp) # core-pinning probe
target_link_libraries(test-affinity PRIVATE bt::core bt::openmp)
add_executable(query-cpuinfo probes/query_cpuinfo.cpp) # CPU topology (Linux/Android)
add_executable(stressor-omp probes/stress/stressor-omp.cpp) # OMP CPU interference load
target_link_libraries(stressor-omp PRIVATE bt::core bt::openmp)

if(BT_BUILD_BENCHMARKS)
    # core-type classifier (probe each core, label big/medium/little/super)
    bt_add_omp_app(bm-check-core-types probes/bm_core_type.cpp benchmark::benchmark)
endif()

if(BT_ENABLE_CUDA)
    # OMP-inside-a-CUDA-TU sanity (probe_omp.cu)
    add_executable(test-omp probes/probe_omp.cu)
    set_target_properties(
        test-omp
        PROPERTIES CUDA_ARCHITECTURES "${BT_CUDA_ARCH}"
    )
    target_compile_options(
        test-omp
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    )
    target_link_libraries(test-omp PRIVATE bt::openmp)
endif()

if(BT_ENABLE_VULKAN)
    # print the compiled vulkan-hpp header version (compile-time constant)
    add_executable(query-vulkan-hpp-version probes/query_vulkan_hpp_version.cpp)
    target_compile_features(query-vulkan-hpp-version PRIVATE cxx_std_20)
    target_include_directories(
        query-vulkan-hpp-version
        SYSTEM
        PRIVATE $<BUILD_INTERFACE:${VulkanHeaders_SOURCE_DIR}/include>
    )
    # Vulkan GPU interference load (kiss-vk engine; no benchmark, no volk)
    add_executable(stressor-vk probes/stress/stressor-vk.cpp)
    target_link_libraries(stressor-vk PRIVATE bt::vulkan bt::core bt::openmp)
endif()
