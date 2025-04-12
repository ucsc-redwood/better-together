-- ----------------------------------------------------------------------------
-- Generate the measurement of "schedules" instance on 100 tasks
-- ----------------------------------------------------------------------------

target("gen-records-cifar-sparse-vk")
do
	add_rules("common_flags", "vulkan_config", "run_on_android")
	set_kind("binary")
	add_files({
		"gen_record.cpp",
	})

	add_deps("builtin-apps-vulkan")
	add_deps("builtin-apps")
end

-- ----------------------------------------------------------------------------
-- Measure the real time performance of "schedules" on 100 tasks
-- ----------------------------------------------------------------------------

target("bm-schedule-cifar-sparse-vk")
do
	add_rules("common_flags", "vulkan_config", "run_on_android")
	set_kind("binary")
	add_files({
		"bm_schedule.cpp",
	})

	add_deps("builtin-apps-vulkan")
	add_deps("builtin-apps")
		
	add_packages("benchmark")
end
