-- ----------------------------------------------------------------------------
-- Run stages with interference (100 tasks)
-- ----------------------------------------------------------------------------

target("bm-baseline-tree-vk")
do
	add_rules("common_flags", "vulkan_config", "run_on_android")
	set_kind("binary")
	add_files({
		"bm_baseline.cpp",
	})

	add_deps("builtin-apps-vulkan")
	add_deps("builtin-apps")

	add_packages("benchmark")
end

target("bm-fully-tree-vk")
do
	add_rules("common_flags", "vulkan_config", "run_on_android")
	set_kind("binary")
	add_files({
		"bm_fully_vs_normal.cpp",
	})

	add_deps("builtin-apps-vulkan")
	add_deps("builtin-apps")
end

