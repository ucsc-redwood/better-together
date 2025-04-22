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

-- ----------------------------------------------------------------
-- Common packages used in the project
-- ----------------------------------------------------------------

-- currently using:
-- - cli11-v2.4.2
-- - spdlog-v1.14.1
-- - glm-1.0.1
add_requires("spdlog") -- everything
add_requires("cli11") -- all binaries

add_requires("libmorton") -- octree applications
add_requires("glm") -- tree applications

add_requires("nlohmann_json")
add_requires("libcurl")
add_requires("cnpy")
add_requires("benchmark")
add_requires("gtest")

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
	target:add("packages", "nlohmann_json")
	target:add("packages", "libmorton")
	target:add("packages", "libcurl")
	
	-- -- for adding debugging
	-- target:add("cxxflags", "-pg")
	target:add("includedirs", "$(projectdir)")
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

rule("cuda_config")
on_load(function(target)
    -- Avoid JIT compilation by targeting specific GPU architecture (SM87)
    -- This improves runtime performance and ensures deterministic behavior
    -- JIT compilation is not supported on Tegra devices in safe context
    target:add("cuflags", "--generate-code arch=compute_87,code=sm_87", {force = true})

	-- Add NVTX library for Nsight Systems to visualize regions of interest
	target:add("ldflags", "-lnvToolsExt", {force = true})

	-- Add OpenMP support for parallel execution on CPU
	target:add("cuflags", "-Xcompiler", "-fopenmp", {force = true})
	target:add("ldflags", "-fopenmp", {force = true})
end)
rule_end()

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

		-- if target:is_plat("macosx") then
		-- 	target:add("links", "vulkan")
		-- 	target:add("linkdirs", "/opt/homebrew/lib")
		-- 	target:add("includedirs", "/opt/homebrew/include")
		-- 	target:add("rpathdirs", "/opt/homebrew/lib")
		-- end
	end)
rule_end()

if has_config("use_vulkan") then
	add_requires("vulkan-headers")
	add_requires("vulkan-hpp")
	add_requires("vulkan-memory-allocator")
	add_requires("volk")
end

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
includes("pipe")
includes("playground")
includes("utility")

package("cli11")
    set_kind("library", {headeronly = true})
    set_homepage("https://github.com/CLIUtils/CLI11")
    set_description("CLI11 is a command line parser for C++11 and beyond that provides a rich feature set with a simple and intuitive interface.")
    set_license("BSD")

    add_urls("https://github.com/CLIUtils/CLI11/archive/refs/tags/$(version).tar.gz",
             "https://github.com/CLIUtils/CLI11.git")
    add_versions("v2.5.0", "17e02b4cddc2fa348e5dbdbb582c59a3486fa2b2433e70a0c3bacb871334fd55")
    add_versions("v2.4.2", "f2d893a65c3b1324c50d4e682c0cdc021dd0477ae2c048544f39eed6654b699a")
    add_versions("v2.4.1", "73b7ec52261ce8fe980a29df6b4ceb66243bb0b779451dbd3d014cfec9fdbb58")
    add_versions("v2.3.2", "aac0ab42108131ac5d3344a9db0fdf25c4db652296641955720a4fbe52334e22")
    add_versions("v2.2.0", "d60440dc4d43255f872d174e416705f56ba40589f6eb07727f76376fb8378fd6")

    if not is_host("windows") then
        add_extsources("pkgconfig::CLI11")
    end

    if is_plat("windows") then
        add_syslinks("shell32")
    end
    on_install("windows", "linux", "macosx", "android", function (package)
        os.cp("include", package:installdir())
    end)

    on_test(function (package)
        assert(package:check_cxxsnippets({test = [[
            CLI::App app{"Test", "test"};
        ]]}, {configs = {languages = "cxx11"}, includes = "CLI/CLI.hpp"}))
    end)
