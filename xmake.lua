-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

add_rules("mode.debug", "mode.release", "mode.releasedbg")

set_languages("c++20")
set_warnings("all", "extra")

if is_mode("release") or is_mode("releasedbg") then
	set_optimize("faster") -- O2
end

if not is_plat("android") then
	-- Clang is better for cross-platform consistency (all platforms can use Clang)
	set_toolchains("clang")
end

-- add_cxxflags("-stdlib=libc++")

-- ----------------------------------------------------------------
-- Common packages used in the project
-- ----------------------------------------------------------------

-- currently using:
-- - cli11-v2.4.2
-- - spdlog-v1.14.1
-- - glm-1.0.1
add_requires("spdlog") -- everything
add_requires("cli11") -- all binaries
add_requires("glm") -- tree applications

add_requires("nlohmann_json")
add_requires("libcurl")
add_requires("cnpy")
add_requires("benchmark")

-- OpenMP is handled differently on Android
if not is_plat("android") then
	add_requires("openmp")
end

-- Common configurations
rule("common_flags")
on_load(function(target)
	-- OpenMP flags for Android (special case)
	if is_plat("android") then
		target:add("cxxflags", "-fopenmp -static-openmp")
		target:add("ldflags", "-fopenmp -static-openmp")
	else
		target:add("packages", "openmp")
	end

	-- Add common packages to the target
	target:add("packages", "cli11")
	target:add("packages", "spdlog")
	target:add("packages", "glm")

	-- if has cuda 
	if has_config("cuda") then
	    target:add("cuflags", "-Xcompiler", "-fopenmp", {force = true})
	    target:add("ldflags", "-fopenmp", {force = true})
	end

	-- -- for adding debugging
	-- target:add("cxxflags", "-pg")
end)
rule_end()

-- ----------------------------------------------------------------
-- CUDA configuration
-- ----------------------------------------------------------------

option("use_cuda")
    set_description("CUDA backend")
    set_showmenu(true)
    set_values("yes", "no")
option_end()

-- ----------------------------------------------------------------
-- Vulkan configuration
-- ----------------------------------------------------------------

option("use_vulkan")
    set_description("Vulkan backend")
    set_showmenu(true)
    set_values("yes", "no")
option_end()

rule("vulkan_config")
on_load(function(target)
	target:add("packages", "vulkan-headers")
	target:add("packages", "vulkan-hpp")
	target:add("packages", "vulkan-memory-allocator")
end)
rule_end()

-- ----------------------------------------------------------------
-- Android configuration
-- ----------------------------------------------------------------

includes("android.lua")

rule("run_on_android")
if is_plat("android") then
	on_run(run_on_android)
end
rule_end()

-- ----------------------------------------------------------------
-- Projects
-- ----------------------------------------------------------------

includes("builtin-apps/common/kiss-vk") -- Keep-It-Simple-Stupid Vulkan library
includes("builtin-apps") -- the three applications
-- includes("benchmarks")
includes("pipe")
includes("lpipe")
includes("play")
includes("utility")

