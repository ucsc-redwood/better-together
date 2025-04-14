-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- target("bm-new-pipe-cifar-sparse-cu")
-- do
-- 	add_rules("pipe_config", "common_flags", "cuda_config", "run_on_android")

-- 	add_headerfiles({
-- 		"task.hpp",
-- 		"../templates.hpp",
-- 		"../templates_cu.cuh",
-- 	})

-- 	add_files({
-- 		"bm_main.cu",
-- 		"task.cpp",
-- 	})

-- 	add_deps("builtin-apps", "builtin-apps-cuda")

-- 	add_packages("benchmark")
-- end

target("new-pipe-cifar-sparse-cu")
do
	add_rules("pipe_config", "common_flags", "cuda_config")

	add_headerfiles({
		"task.hpp",
		"../templates.hpp",
		"../templates_cu.cuh",
		"../config_reader.hpp",
	})

	add_files({
		"main.cu",
		"task.cpp",
	})

	add_deps("builtin-apps", "builtin-apps-cuda")

	add_packages("nlohmann_json")
end
