# profiler/ -- BT-Profiler drivers: the per-app pipeline runners and the per-cell
# benchmark drivers (baseline / fully-vs-normal / gen-logs / prof) that feed the
# optimizer. One target per app x driver, generated to avoid hand-list drift (a
# dropped line builds fine and only fails at deploy).

if(BT_ENABLE_CUDA)
  bt_add_cuda_app(run-pipe-tree-cu         tree-cu/main.cu)
  bt_add_cuda_app(run-pipe-cifar-dense-cu  cifar-dense-cu/main.cu)
  bt_add_cuda_app(run-pipe-cifar-sparse-cu cifar-sparse-cu/main.cu)

  if(BT_BUILD_BENCHMARKS)
    foreach(app tree cifar-dense cifar-sparse)
      bt_add_cuda_app(bm-baseline-${app}-cu ${app}-cu/bm_baseline.cu        benchmark::benchmark)
      bt_add_cuda_app(bm-fully-${app}-cu    ${app}-cu/bm_fully_vs_normal.cu benchmark::benchmark)
      bt_add_cuda_app(bm-gen-logs-${app}-cu ${app}-cu/bm_gen_log.cu         nlohmann_json::nlohmann_json)
      bt_add_cuda_app(bm-prof-${app}-cu     ${app}-cu/bm_prof.cu            benchmark::benchmark nlohmann_json::nlohmann_json)
      add_dependencies(bm-prof-${app}-cu bt_git_sha)
      target_include_directories(bm-prof-${app}-cu PRIVATE ${CMAKE_BINARY_DIR}/generated)
    endforeach()
  endif()
endif()

if(BT_ENABLE_VULKAN AND BT_BUILD_BENCHMARKS)
  foreach(app tree cifar-dense cifar-sparse)
    bt_add_vk_app(bm-baseline-${app}-vk ${app}-vk/bm_baseline.cpp        benchmark::benchmark)
    bt_add_vk_app(bm-fully-${app}-vk    ${app}-vk/bm_fully_vs_normal.cpp benchmark::benchmark)
    bt_add_vk_app(bm-gen-logs-${app}-vk ${app}-vk/bm_gen_log.cpp         nlohmann_json::nlohmann_json)
    bt_add_vk_app(bm-prof-${app}-vk     ${app}-vk/bm_prof.cpp            benchmark::benchmark nlohmann_json::nlohmann_json)
    add_dependencies(bm-prof-${app}-vk bt_git_sha)
    target_include_directories(bm-prof-${app}-vk PRIVATE ${CMAKE_BINARY_DIR}/generated)
  endforeach()
endif()
