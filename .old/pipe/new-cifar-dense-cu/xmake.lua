-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- Benchmark for the new pipeline

-- target("bm-new-pipe-cifar-dense-cu")
-- do
-- 	add_rules("pipe_config", "common_flags", "cuda_config")

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

-- Single execution of the pipeline for debugging or testing purposes

target("new-pipe-cifar-dense-cu")
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
