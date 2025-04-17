
-- ----------------------------------------------------------------------------
-- Test, just test if the dispatchers are working
-- ----------------------------------------------------------------------------

target("test-octree-vk")
do
	add_rules("common_flags", "vulkan_config", "run_on_android")
	set_group("test")
	set_kind("binary")
	add_files({
		"./test_main.cpp",
	})

	add_deps("builtin-apps-vulkan")
	add_deps("builtin-apps")
		
	add_packages("gtest")
end
