-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

local ANDROID_CONFIG = {
	-- ignored_devices = { "ZY22FLDDK7" },
	ignored_devices = { "ZY22FLDDK7", "ce0717178d7758b00b7e" },
	-- ignored_devices = { "3A021JEHN02756", "9b034f1b" },
	remote_base_path = "/data/local/tmp", -- Base directory for all executables
}

-- ----------------------------------------------------------------------------
-- Special packages that are not available on Android
--  'note: the following packages are unsupported on android/arm64-v8a:'
-- ----------------------------------------------------------------------------

if is_plat("android") then
	package("benchmark")
	set_kind("library")
	add_deps("cmake")
	set_urls("https://github.com/google/benchmark.git")
	add_versions("v1.9.0", "12235e24652fc7f809373e7c11a5f73c5763fc4c")
	add_versions("v1.9.1", "c58e6d0710581e3a08d65c349664128a8d9a2461")
	add_versions("v1.9.2", "afa23b7699c17f1e26c88cbf95257b20d78d6247")

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

end

-- ----------------------------------------------------------------------------
-- Run on Android
-- ----------------------------------------------------------------------------

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
