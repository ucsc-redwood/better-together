-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("bm-new-pipe-cifar-sparse-vk")
do
	add_rules("pipe_config", "common_flags", "vulkan_config", "run_on_android")

	add_headerfiles({
		"task.hpp",
		"run_stages.hpp",
		"../templates.hpp",
		"../templates_vk.hpp",
		"generated_code.hpp",
	})

	add_files({
		"bm_main.cpp",
		"task.cpp",
	})

	add_deps("builtin-apps-vulkan", "builtin-apps")

	add_packages("benchmark")
end
