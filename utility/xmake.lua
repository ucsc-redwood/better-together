-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.


-- ----------------------------------------------------------------
-- Utility Target: Check Core Types
-- ----------------------------------------------------------------

target("bm-check-core-types")
do
	set_kind("binary")
	add_rules("common_flags", "run_on_android")
	add_files({
		"bm_core_type.cpp",
	})

	add_packages("benchmark")
end

if has_config("use_vulkan") then
	target("check-vulkan")
	do
		set_kind("binary")
		add_rules("vulkan_config", "run_on_android")
		add_files({
			"check_vulkan.cpp",
		})

		add_packages("volk")
	end
end


-- ----------------------------------------------------------------
-- Utility Target: try pinning threads to all cores and verify if it works
-- ----------------------------------------------------------------

target("test-affinity")
do
	set_kind("binary")
	add_rules("run_on_android")
	add_files({
		"test_affinity.cpp",
	})
end


-- ----------------------------------------------------------------
-- Utility (Vulkan)
-- ----------------------------------------------------------------
-- Utility Target: Find the current GPU's Warp Size
-- Utility Target: Query the current Vulkan-Hpp version
-- Vulkan-Hpp version: 309
-- And 
-- Vulkan-Hpp version: 290
-- ----------------------------------------------------------------

if has_config("use_vulkan") then

	target("query-warpsize")
	do
		set_kind("binary")
		add_rules("vulkan_config", "run_on_android")
		add_files({
			"query_warpsize.cpp",
		})
		add_packages("volk")
	end

	target("query-vulkan-hpp-version")
	do
		set_kind("binary")
		add_rules("vulkan_config", "run_on_android")
		add_files({
			"query_vulkan_hpp_version.cpp",
		})
	end

	if is_plat("linux") or is_plat("android") then
		target("query-cpuinfo")
		do
			set_kind("binary")
			add_rules("run_on_android")
			add_files({
				"query_cpuinfo.cpp",
			})
		end
	end

end
