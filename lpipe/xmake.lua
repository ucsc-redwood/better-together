-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- ----------------------------------------------------------------
-- Pipeline
-- ----------------------------------------------------------------

add_requires("concurrentqueue")

rule("lpipe_config")
on_load(function(target)
	target:set("kind", "binary")
	target:set("group", "lpipe")

	target:add("includedirs", "$(projectdir)")

	target:add("packages", "concurrentqueue")
end)
rule_end()

-- -- ----------------------------------------------------------------
-- -- Pipeline Targets
-- -- ----------------------------------------------------------------


target("lpipe")
do
	add_rules("lpipe_config", "common_flags", "run_on_android")

	add_files({
		"main.cpp",
	})

	add_deps("builtin-apps")
end

target("bm_kernels")
do
	add_rules("lpipe_config", "common_flags", "run_on_android")

	add_files({
		"bm_kernels.cpp",
	})

	add_deps("builtin-apps")
	add_packages("benchmark")
end

-- if has_config("use_vulkan") then
-- 	includes("new-cifar-dense-vk")
-- 	includes("new-cifar-sparse-vk")
-- 	includes("new-tree-vk")
-- end

-- if has_config("use_cuda") then
-- 	includes("new-cifar-dense-cu")	
-- 	includes("new-cifar-sparse-cu")
-- 	includes("new-tree-cu")
-- end