-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

local ANDROID_CONFIG = {
	ignored_devices = { "ZY22FLDDK7", "ce0717178d7758b00b7e" },
	remote_base_path = "/data/local/tmp", -- Base directory for all executables
}

-- ----------------------------------------------------------------------------
-- Special packages that are not available on Android
--  'note: the following packages are unsupported on android/arm64-v8a:'
-- ----------------------------------------------------------------------------

if is_plat("windows") then
	package("concurrentqueue")
	set_homepage("https://github.com/cameron314/concurrentqueue")
	set_description("A fast multi-producer, multi-consumer lock-free concurrent queue for C++11")
	set_license("BSD")

	add_urls(
		"https://github.com/cameron314/concurrentqueue/archive/refs/tags/$(version).tar.gz",
		"https://github.com/cameron314/concurrentqueue.git"
	)

	add_versions("v1.0.4", "87fbc9884d60d0d4bf3462c18f4c0ee0a9311d0519341cac7cbd361c885e5281")

	add_configs("c_api", { description = "Build C API", default = false, type = "boolean" })

	add_deps("cmake")

	add_includedirs("include", "include/concurrentqueue/moodycamel")

	on_load(function(package)
		if not package:config("c_api") then
			package:set("kind", "library", { headeronly = true })
		end
	end)

	on_install(function(package)
		local configs = {}
		table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:is_debug() and "Debug" or "Release"))
		table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
		import("package.tools.cmake").install(package, configs)

		io.writefile(
			path.join(package:installdir("include"), "concurrentqueue", "concurrentqueue.h"),
			[[
    #pragma once
    
    #pragma message please update include <concurrentqueue/concurrentqueue.h> to <concurrentqueue.h>
    #include "moodycamel/concurrentqueue.h"
            ]]
		)

		if package:config("c_api") then
			io.writefile(
				"xmake.lua",
				[[
                    add_rules("mode.debug", "mode.release")
                    target("concurrentqueue-c")
                        set_kind("$(kind)")
                        add_files("c_api/*.cpp")
                        add_headerfiles("(c_api/concurrentqueue.h)")
                        if is_plat("windows") and is_kind("shared") then
                            add_defines("DLL_EXPORT")
                        end
                ]]
			)
			import("package.tools.xmake").install(package)

			if package:is_plat("windows") and (not package:config("shared")) then
				package:add("defines", "MOODYCAMEL_STATIC")
			end
		end
	end)

	-- on_test(function (package)
	--     assert(package:check_cxxsnippets({test = [[
	--         #include <concurrentqueue.h>
	--         void test() {
	--             moodycamel::ConcurrentQueue<int> q;
	--             q.enqueue(25);
	--         }
	--     ]]}, {configs = {languages = "c++11"}}))
	--     if package:config("c_api") then
	--         assert(package:has_cfuncs("moodycamel_cq_create", {includes = "c_api/concurrentqueue.h"}))
	--     end
	-- end)

	package_end()
end

if is_plat("android") then
	package("benchmark")
	set_kind("library")
	add_deps("cmake")
	set_urls("https://github.com/google/benchmark.git")
	add_versions("v1.9.0", "12235e24652fc7f809373e7c11a5f73c5763fc4c")
	add_versions("v1.9.1", "c58e6d0710581e3a08d65c349664128a8d9a2461")

	-- Add description and homepage for better package management
	set_description("A microbenchmark support library")
	set_homepage("https://github.com/google/benchmark")

	on_install(function(package)
		local configs = {
			"-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"),
			"-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"),
			"-DBENCHMARK_DOWNLOAD_DEPENDENCIES=on",
			"-DHAVE_THREAD_SAFETY_ATTRIBUTES=0",
		}
		import("package.tools.cmake").install(package, configs)
	end)
	package_end()

	-- package("cli11")
	--     set_kind("library", {headeronly = true})
	--     set_homepage("https://github.com/CLIUtils/CLI11")
	--     set_description("CLI11 is a command line parser for C++11 and beyond that provides a rich feature set with a simple and intuitive interface.")
	--     set_license("BSD")
	--     add_deps("cmake")

	--     add_urls("https://github.com/CLIUtils/CLI11/archive/refs/tags/$(version).tar.gz",
	--             "https://github.com/CLIUtils/CLI11.git")
	--     add_versions("v2.4.2", "f2d893a65c3b1324c50d4e682c0cdc021dd0477ae2c048544f39eed6654b699a")
	--     add_versions("v2.4.1", "73b7ec52261ce8fe980a29df6b4ceb66243bb0b779451dbd3d014cfec9fdbb58")
	--     add_versions("v2.3.2", "aac0ab42108131ac5d3344a9db0fdf25c4db652296641955720a4fbe52334e22")
	--     add_versions("v2.2.0", "d60440dc4d43255f872d174e416705f56ba40589f6eb07727f76376fb8378fd6")

	--     if not is_host("windows") then
	--         add_extsources("pkgconfig::CLI11")
	--     end

	--     -- on_install("windows", "linux", "android", "macosx", function (package)
	--     --     os.cp("include", package:installdir())
	--     -- end)

	--     on_install(function(package)
	--         import("package.tools.cmake").install(package)
	--     end)

	--     -- on_test(function (package)
	--     --     assert(package:check_cxxsnippets({test = [[
	--     --         CLI::App app{"Test", "test"};
	--     --     ]]}, {configs = {languages = "cxx11"}, includes = "CLI/CLI.hpp"}))
	--     -- end)
	-- package_end()
end

-- ----------------------------------------------------------------------------
-- Common helper functions
-- ----------------------------------------------------------------------------

-- function parse_device_list()
--     -- Get connected devices
--     local devices_output = try { function()
--         return os.iorun("adb devices")
--     end}

--     if not devices_output then
--         raise("Failed to get device list from adb")
--     end

--     -- Parse device list
--     local devices = {}
--     for line in devices_output:gmatch("[^\r\n]+") do
--         if line:find("%s*device$") then
--             local device_id = line:match("(%S+)%s+device")
--             if device_id and not table.contains(ANDROID_CONFIG.ignored_devices, device_id) then
--                 table.insert(devices, device_id)
--             end
--         end
--     end

--     if #devices == 0 then
--         raise("No connected devices found!")
--     end

--     return devices
-- end

-- ----------------------------------------------------------------------------
-- Android deployment helper function
-- ----------------------------------------------------------------------------

-- function run_on_android(target)
--     local exec_path = target:targetfile()
--     local target_name = target:name()
--     local remote_path = ANDROID_CONFIG.remote_base_path  -- .. "/" .. target_name

--     if not os.isfile(exec_path) then
--         raise("Executable not found at: " .. exec_path)
--     end

--     -- Get connected devices
--     local devices_output = try { function()
--         return os.iorun("adb devices")
--     end}

--     if not devices_output then
--         raise("Failed to get device list from adb")
--     end

--     -- Parse device list
--     local devices = {}
--     for line in devices_output:gmatch("[^\r\n]+") do
--         if line:find("%s*device$") then
--             local device_id = line:match("(%S+)%s+device")
--             if device_id and not table.contains(ANDROID_CONFIG.ignored_devices, device_id) then
--                 table.insert(devices, device_id)
--             end
--         end
--     end

--     if #devices == 0 then
--         raise("No connected devices found!")
--     end

--     -- Run on each device
--     import("core.base.option")
--     local args = option.get("arguments") or {}

--     for i, device_id in ipairs(devices) do
--         print(string.format("[%d/%d] Running %s on device: %s", i, #devices, target_name, device_id))

--         -- Deploy and execute
--         local adb_commands = {
--             {"-s", device_id, "push", exec_path, remote_path .. "/" .. target_name},
--             {"-s", device_id, "shell", "chmod", "+x", remote_path .. "/" .. target_name},
--         }

--         -- Execute commands

--         for _, cmd in ipairs(adb_commands) do
--             if os.execv("adb", cmd) ~= 0 then
--                 print(string.format("Warning: Failed to execute adb command on device %s", device_id))
--             end
--         end

--         -- Run the binary with arguments
--         local run_command = {"-s", device_id, "shell", remote_path .. "/" .. target_name}

--         table.join2(run_command, args, {"--device=" .. device_id})
--         if os.execv("adb", run_command) ~= 0 then
--             print(string.format("Warning: Failed to run %s on device %s", target_name, device_id))
--         end

--         print()
--     end
-- end

function run_on_android(target)
	local function get_connected_devices()
		local devices = {}
		local devices_output = try({
			function()
				return os.iorun("adb devices")
			end,
		})
		if not devices_output then
			raise("Failed to get device list from adb")
		end

		for line in devices_output:gmatch("[^\r\n]+") do
			local device_id = line:match("^(%S+)%s+device$")
			if device_id and not table.contains(ANDROID_CONFIG.ignored_devices, device_id) then
				table.insert(devices, device_id)
			end
		end
		return devices
	end

	local function deploy_and_run(device_id, exec_path, remote_path, target_name, args)
		print(string.format("Deploying and running '%s' on device: %s", target_name, device_id))

		local binary_path = string.format("%s/%s", remote_path, target_name)

		-- Deploy binary
		local adb_commands = {
			{ "-s", device_id, "push", exec_path, binary_path },
			{ "-s", device_id, "shell", "chmod", "+x", binary_path },
		}

		for _, cmd in ipairs(adb_commands) do
			if os.execv("adb", cmd) ~= 0 then
				print(string.format("[Error] Failed to execute adb command on device %s", device_id))
				return
			end
		end

		-- Run binary
		local run_command = { "-s", device_id, "shell", binary_path }
		table.join2(run_command, args, { "--device=" .. device_id })
		if os.execv("adb", run_command) ~= 0 then
			print(string.format("[Error] Failed to run %s on device %s", target_name, device_id))
		end
	end

	-- Validate target file
	local exec_path = target:targetfile()
	local target_name = target:name()
	local remote_path = ANDROID_CONFIG.remote_base_path
	if not os.isfile(exec_path) then
		raise("Executable not found at: " .. exec_path)
	end

	-- Get connected devices
	local devices = get_connected_devices()
	if #devices == 0 then
		raise("No connected devices found!")
	end

	-- Run on each device
	import("core.base.option")
	local args = option.get("arguments") or {}
	for i, device_id in ipairs(devices) do
		print(string.format("\n[%d/%d] Processing device: %s", i, #devices, device_id))
		deploy_and_run(device_id, exec_path, remote_path, target_name, args)
	end
end

-- ----------------------------------------------------------------------------
-- Push All resources to the device ('tar' version)
-- ----------------------------------------------------------------------------

task("push-all-resources")
set_menu({
	usage = "$project push-all-resources",
	description = "Push All resources to the device.",
})
on_run(function()
	local remote_path = ANDROID_CONFIG.remote_base_path
	local data_dir = "./resources/" -- Local resources directory

	-- 1) Get connected devices
	local devices_output = try({
		function()
			return os.iorun("adb devices")
		end,
	})

	if not devices_output then
		raise("Failed to get device list from adb")
	end

	-- 2) Parse device list
	local devices = {}
	for line in devices_output:gmatch("[^\r\n]+") do
		if line:find("%s*device$") then
			local device_id = line:match("(%S+)%s+device")
			if device_id and not table.contains(ANDROID_CONFIG.ignored_devices, device_id) then
				table.insert(devices, device_id)
			end
		end
	end

	if #devices == 0 then
		raise("No connected devices found!")
	end

	-- 3) Pack resources into a tar.gz file (avoid writing inside `resources/`)
	os.execv("tar", { "-czvf", "resources.tar.gz", "-C", data_dir, "." })

	-- 4) Push and extract on each device
	for i, device_id in ipairs(devices) do
		print(string.format("[%d/%d] Pushing all resources to device: %s", i, #devices, device_id))

		-- Push the tar.gz archive to the remote path
		os.execv("adb", { "-s", device_id, "push", "resources.tar.gz", remote_path })

		-- -- Extract the archive on the device
		-- os.execv("adb", {
		--     "-s", device_id,
		--     "shell",
		--     "tar",
		--     "--no-same-owner",
		--     "-xzvf", remote_path .. "/resources.tar.gz",
		--     "-C", remote_path
		-- })

		os.execv("adb", {
			"-s",
			device_id,
			"shell",
			"mkdir -p "
				.. remote_path
				.. "/resources && tar --no-same-owner -xzvf "
				.. remote_path
				.. "/resources.tar.gz -C "
				.. remote_path
				.. "/resources",
		})
	end

	-- 5) Cleanup local archive
	os.rm("resources.tar.gz")
end)
task_end()
