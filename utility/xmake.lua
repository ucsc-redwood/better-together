-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- ----------------------------------------------------------------
-- Utility (Vulkan)
-- ----------------------------------------------------------------

if has_config("use_vulkan") then
	add_requires("volk")
end

rule("utility_config")
on_load(function(target)
	target:set("kind", "binary")
	target:set("group", "utility")
end)
rule_end()

-- ----------------------------------------------------------------
-- Utility Target: Find the current GPU's Warp Size
-- ----------------------------------------------------------------
if has_config("use_vulkan") then
	target("query-warpsize")
	do
		add_rules("utility_config", "vulkan_config", "run_on_android")
		add_files({
			"query_warpsize.cpp",
		})
		add_packages("volk")
	end
end

-- ----------------------------------------------------------------
-- Utility Target: Query the current CPU Information
-- ----------------------------------------------------------------

if has_config("use_vulkan") then
	if is_plat("linux") or is_plat("android") then
		target("query-cpuinfo")
		do
			add_rules("utility_config", "run_on_android")
			add_files({
				"query_cpuinfo.cpp",
			})
		end
	end
end

-- ----------------------------------------------------------------
-- Utility Target: try pinning threads to all cores and verify if it works
-- ----------------------------------------------------------------

target("test-affinity")
do
	add_rules("utility_config", "run_on_android")
	add_files({
		"test_affinity.cpp",
	})
end

-- ----------------------------------------------------------------
-- Utility Target: Query the current Vulkan-Hpp version
-- Vulkan-Hpp version: 309
-- And 
-- Vulkan-Hpp version: 290
-- ----------------------------------------------------------------

target("query-vulkan-hpp-version")
do
	add_rules("utility_config", "vulkan_config", "run_on_android")
	add_files({
		"query_vulkan_hpp_version.cpp",
	})
end

