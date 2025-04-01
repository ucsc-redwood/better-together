-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- target("bm-new-pipe-tree-cu")
-- do
-- 	add_rules("pipe_config", "common_flags", "run_on_android")

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
-- 	add_cugencodes("native")

-- 	add_packages("benchmark")
-- end


target("new-pipe-tree-cu")
do
	add_rules("pipe_config", "common_flags")

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
	add_cugencodes("native")

	add_links("nvToolsExt")

	add_packages("nlohmann_json")
end
