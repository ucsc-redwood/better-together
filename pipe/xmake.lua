-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- ----------------------------------------------------------------
-- Pipeline
-- ----------------------------------------------------------------

add_requires("concurrentqueue")

rule("pipe_config")
on_load(function(target)
	target:set("kind", "binary")
	target:set("group", "pipe")

	target:add("includedirs", "$(projectdir)")

	target:add("packages", "concurrentqueue")
end)
rule_end()

-- ----------------------------------------------------------------
-- Pipeline Targets
-- ----------------------------------------------------------------

if has_config("use_vulkan") then
	includes("new-cifar-dense-vk")
	includes("new-cifar-sparse-vk")
	includes("new-tree-vk")
end

if has_config("use_cuda") then
	includes("new-cifar-dense-cu")	
	includes("new-cifar-sparse-cu")
	includes("new-tree-cu")
end