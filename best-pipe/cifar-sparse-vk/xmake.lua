target("pipe-cifar-sparse-vk")
do
	add_rules("common_flags", "vulkan_config", "run_on_android")
	set_kind("binary")
	add_files({
		"main.cpp",
	})

	add_deps("builtin-apps-vulkan")
	add_deps("builtin-apps")
		
	add_packages("benchmark")
end