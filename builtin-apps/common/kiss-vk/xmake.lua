-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- currently using:
-- - vulkan-headers-1.3.290
-- - vulkan-hpp-v1.3.290
-- - vulkan-memory-allocator-v3.1.0

-- ----------------------------------------------------------------
-- kiss-vk (Keep-It-Simple-Stupid Vulkan)
-- ----------------------------------------------------------------

add_requires("vulkan-headers")
add_requires("vulkan-hpp")
add_requires("vulkan-memory-allocator")

if has_config("use_vulkan") then
	target("kiss-vk")
	do
		set_kind("static")
		add_rules("common_flags", "vulkan_config")
		set_group("static-libs")

		add_headerfiles({
			"algorithm.hpp",
			"base_engine.hpp",
			"engine.hpp",
			"sequence.hpp",
			"vk.hpp",
			"vma_pmr.hpp",
		})

		add_files({
			"algorithm.cpp",
			"base_engine.cpp",
			"sequence.cpp",
			"vma_pmr.cpp",
		})
	end
end