-- ----------------------------------------------------------------------------
-- Run stages with interference (100 tasks)
-- ----------------------------------------------------------------------------

target("bm-baseline-cifar-dense-vk")
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

-- ----------------------------------------------------------------------------
-- Run stages with interference (100 tasks)
-- Use this to generate Table, which will feed into optimizer
-- ----------------------------------------------------------------------------

target("bm-table-cifar-dense-vk")
do
	add_rules("common_flags", "vulkan_config", "run_on_android")
	set_kind("binary")
	add_files({
		"bm_for_table.cpp",
	})

	add_deps("builtin-apps-vulkan")
	add_deps("builtin-apps")
end

target("bm-fully-cifar-dense-vk")
do
	add_rules("common_flags", "vulkan_config", "run_on_android")
	set_kind("binary")
	add_files({
		"bm_fully_vs_normal.cpp",
	})

	add_deps("builtin-apps-vulkan")
	add_deps("builtin-apps")
end


-- -- ----------------------------------------------------------------------------
-- -- Measure the real time performance of "schedules" on 100 tasks
-- -- The Result. 
-- -- ----------------------------------------------------------------------------

-- target("bm-schedule-cifar-sparse-vk")
-- do
-- 	add_rules("common_flags", "vulkan_config", "run_on_android")
-- 	set_kind("binary")
-- 	add_files({
-- 		"bm_real_schedule.cpp",
-- 	})

-- 	add_deps("builtin-apps-vulkan")
-- 	add_deps("builtin-apps")
		
-- 	add_packages("benchmark")
-- end

-- -- ----------------------------------------------------------------------------
-- -- Generate the Log/Graph of "schedules" (100 tasks)
-- -- ----------------------------------------------------------------------------

-- target("bm-gen-logs-cifar-sparse-vk")
-- do
-- 	add_rules("common_flags", "vulkan_config", "run_on_android")
-- 	set_kind("binary")
-- 	add_files({
-- 		"bm_gen_log.cpp",
-- 	})

-- 	add_deps("builtin-apps-vulkan")
-- 	add_deps("builtin-apps")
-- end

