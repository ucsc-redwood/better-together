-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- ----------------------------------------------------------------
-- Builtin Apps (Tree, CIFAR-Sparse, CIFAR-Dense) Static Library
-- ----------------------------------------------------------------

target("builtin-apps")
do
	set_kind("static")
	add_rules("common_flags")
	set_group("static-libs")

	add_headerfiles({
		-- common headers
		"affinity.hpp",
		"app.hpp",
		"conf.hpp",
		"resources_path.hpp",

		-- cifar-sparse headers
		"cifar-sparse/arg_max.hpp",
		"cifar-sparse/sparse_appdata.hpp",
		"cifar-sparse/omp/dispatchers.hpp",
		"cifar-sparse/omp/all_kernels.hpp",

		-- cifar-dense headers
		"cifar-dense/arg_max.hpp",
		"cifar-dense/dense_appdata.hpp",
		"cifar-dense/omp/dispatchers.hpp",
		"cifar-dense/omp/all_kernels.hpp",
		
		-- tree headers
		"tree/tree_appdata.hpp",
		"tree/safe_tree_appdata.hpp",
		"tree/omp/dispatchers.hpp",
		"tree/omp/func_brt.hpp",
		"tree/omp/func_edge.hpp",
		"tree/omp/func_morton.hpp",
		"tree/omp/func_octree.hpp",
		"tree/omp/func_sort.hpp",
		"tree/omp/temp_storage.hpp",

		-- Add missing common header
		"debug_logger.hpp",
	})

	add_files({
		"conf.cpp",
		"app.cpp",

		-- cifar-dense implementations
		"cifar-dense/dense_appdata.cpp",
		"cifar-dense/omp/dispatchers.cpp",
		"cifar-dense/omp/all_kernels.cpp",

		-- cifar-sparse implementations
		"cifar-sparse/sparse_appdata.cpp",
		"cifar-sparse/omp/dispatchers.cpp",
		"cifar-sparse/omp/all_kernels.cpp",

		-- tree implementations
		"tree/tree_appdata.cpp",
		"tree/safe_tree_appdata.cpp",
		"tree/omp/dispatchers.cpp",
	})
end

target("test-cifar-sparse-omp")
do
	add_rules("common_flags", "run_on_android")
	set_kind("binary")
	add_files({
		"cifar-sparse/omp/test_main.cpp",
	})

	add_deps("builtin-apps")

	add_packages("gtest")
end

target("bm-mini-cifar-sparse-omp")
do
	add_rules("common_flags", "run_on_android")
	set_kind("binary")
	add_files({
		"cifar-sparse/omp/bm_main.cpp",
	})

	add_deps("builtin-apps")

	add_packages("benchmark")
end

-- ----------------------------------------------------------------------------
-- Vulkan Static Library
-- ----------------------------------------------------------------------------

if has_config("use_vulkan") then
	target("builtin-apps-vulkan")
	do
		set_kind("static")
		set_group("static-libs")
		add_rules("common_flags", "vulkan_config")

		add_deps("kiss-vk")

		add_headerfiles({
			-- App specific headers
			"cifar-sparse/vulkan/dispatchers.hpp",
			"cifar-dense/vulkan/dispatchers.hpp",
			"tree/vulkan/dispatchers.hpp",
			"tree/vulkan/tmp_storage.hpp",
			"tree/vulkan/vk_appdata.hpp",
		})

		add_files({
			-- App specific implementations
			"cifar-sparse/vulkan/dispatchers.cpp",
			"cifar-dense/vulkan/dispatchers.cpp",
			"tree/vulkan/dispatchers.cpp",
		})
	end


	target("test-cifar-sparse-vk")
	do
		add_rules("common_flags", "vulkan_config", "run_on_android")
		set_kind("binary")
		add_files({
			"cifar-sparse/vulkan/test_main.cpp",
		})

		add_deps("builtin-apps-vulkan")
		add_deps("builtin-apps")
		
		add_packages("gtest")
	end


	target("bm-mini-cifar-sparse-vk")
	do
		add_rules("common_flags", "vulkan_config", "run_on_android")
		set_kind("binary")
		add_files({
			"cifar-sparse/vulkan/bm_main.cpp",
		})

		add_deps("builtin-apps-vulkan")
		add_deps("builtin-apps")
		
		add_packages("benchmark")
	end

	target("bm-pipe-cifar-sparse-vk")
	do
		add_rules("common_flags", "vulkan_config", "run_on_android")
		set_kind("binary")
		add_files({
			"cifar-sparse/vulkan/bm_main_pipe.cpp",
		})

		add_deps("builtin-apps-vulkan")
		add_deps("builtin-apps")
		
		add_packages("benchmark")
	end

end

-- ----------------------------------------------------------------------------
-- CUDA Static Library
-- ----------------------------------------------------------------------------

if has_config("use_cuda") then
	target("builtin-apps-cuda")
	do
		set_kind("static")
		set_group("static-libs")
		add_rules("common_flags", "cuda_config")

		add_headerfiles({
			-- Common CUDA headers
			"common/cuda/cu_mem_resource.cuh",
			"common/cuda/helpers.cuh",
			"common/cuda/manager.cuh",

			-- CIFAR sparse CUDA headers
			"cifar-sparse/cuda/all_kernels.cuh",
			"cifar-sparse/cuda/dispatchers.cuh",

			-- CIFAR dense CUDA headers
			"cifar-dense/cuda/all_kernels.cuh",
			"cifar-dense/cuda/dispatchers.cuh",

			-- Tree CUDA headers
			"tree/cuda/01_morton.cuh",
			"tree/cuda/02_sort.cuh",
			"tree/cuda/03_unique.cuh",
			"tree/cuda/04_radix_tree.cuh",
			"tree/cuda/05_edge_count.cuh",
			"tree/cuda/06_prefix_sum.cuh",
			"tree/cuda/07_octree.cuh",
			"tree/cuda/agents/prefix_sum_agent.cuh",
			"tree/cuda/agents/unique_agent.cuh",
			"tree/cuda/common/helper_cuda.hpp",
			"tree/cuda/common/helper_functions.hpp",
			"tree/cuda/common/helper_math.hpp",
			"tree/cuda/common/helper_string.hpp",
			"tree/cuda/common/helper_timer.hpp",
			"tree/cuda/common.cuh",
			"tree/cuda/func_morton.cuh",
			"tree/cuda/dispatchers.cuh",
			"tree/cuda/temp_storage.hpp",
		})

		add_files({
			-- Common CUDA implementations
			"common/cuda/cu_mem_resource.cu",

			-- CIFAR dense CUDA implementations
			"cifar-dense/cuda/all_kernels.cu",
			"cifar-dense/cuda/dispatchers.cu",

			-- CIFAR sparse CUDA implementations
			"cifar-sparse/cuda/all_kernels.cu",
			"cifar-sparse/cuda/dispatchers.cu",

			-- Tree CUDA implementations
			"tree/cuda/01_morton.cu",
			"tree/cuda/02_sort.cu",
			"tree/cuda/03_unique.cu",
			"tree/cuda/04_radix_tree.cu",
			"tree/cuda/05_edge_count.cu",
			"tree/cuda/06_prefix_sum.cu",
			"tree/cuda/07_octree.cu",
			"tree/cuda/dispatchers.cu",
		})

		-- Best CUDA library
		add_packages("cub")
	end
end
