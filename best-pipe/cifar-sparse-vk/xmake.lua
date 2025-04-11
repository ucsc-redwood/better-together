target("gen-record-pipe-cifar-sparse-vk")
do
	add_rules("common_flags", "vulkan_config", "run_on_android")
	set_kind("binary")
	add_files({
		"gen_record.cpp",
	})

	add_deps("builtin-apps-vulkan")
	add_deps("builtin-apps")
end

target("bm-best-cifar-sparse-vk")
do
	add_rules("common_flags", "vulkan_config", "run_on_android")
	set_kind("binary")
	add_files({
		"bm_main.cpp",
	})

	add_deps("builtin-apps-vulkan")
	add_deps("builtin-apps")
		
	add_packages("benchmark")
end
