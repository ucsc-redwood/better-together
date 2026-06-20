# profiler/ -- BT-Profiler drivers: the per-app pipeline runners and the per-cell
# benchmark drivers (baseline / fully-vs-normal / gen-logs / prof) that feed the
# optimizer. One target per app x driver, generated to avoid hand-list drift (a
# dropped line builds fine and only fails at deploy).

# The set of profiled apps -- drives both the pipeline runners and the benchmark drivers,
# so a new app is added in exactly one place (no runner/foreach drift).
set(BT_PROFILED_APPS tree cifar-dense cifar-sparse)

if(BT_ENABLE_CUDA)
    foreach(app ${BT_PROFILED_APPS})
        bt_add_cuda_app(run-pipe-${app}-cu ${app}-cu/main.cu)
    endforeach()

    if(BT_BUILD_BENCHMARKS)
        foreach(app ${BT_PROFILED_APPS})
            bt_add_cuda_app(bm-baseline-${app}-cu ${app}-cu/bm_baseline.cu        benchmark::benchmark)
            bt_add_cuda_app(bm-fully-${app}-cu    ${app}-cu/bm_fully_vs_normal.cu benchmark::benchmark)
            bt_add_cuda_app(bm-gen-logs-${app}-cu ${app}-cu/bm_gen_log.cu         nlohmann_json::nlohmann_json)
            bt_add_cuda_app(bm-prof-${app}-cu     ${app}-cu/bm_prof.cu            benchmark::benchmark nlohmann_json::nlohmann_json)
            add_dependencies(bm-prof-${app}-cu bt_git_sha)
            target_include_directories(
                bm-prof-${app}-cu
                PRIVATE ${CMAKE_BINARY_DIR}/generated
            )
        endforeach()
    endif()
endif()

if(BT_ENABLE_VULKAN AND BT_BUILD_BENCHMARKS)
    foreach(app ${BT_PROFILED_APPS})
        bt_add_vk_app(bm-baseline-${app}-vk ${app}-vk/bm_baseline.cpp        benchmark::benchmark)
        bt_add_vk_app(bm-fully-${app}-vk    ${app}-vk/bm_fully_vs_normal.cpp benchmark::benchmark)
        bt_add_vk_app(bm-gen-logs-${app}-vk ${app}-vk/bm_gen_log.cpp         nlohmann_json::nlohmann_json)
        bt_add_vk_app(bm-prof-${app}-vk     ${app}-vk/bm_prof.cpp            benchmark::benchmark nlohmann_json::nlohmann_json)
        add_dependencies(bm-prof-${app}-vk bt_git_sha)
        target_include_directories(
            bm-prof-${app}-vk
            PRIVATE ${CMAKE_BINARY_DIR}/generated
        )
    endforeach()
endif()
