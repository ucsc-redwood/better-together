-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("play-ndarray")
do
	add_rules("common_flags", "run_on_android")

    add_includedirs("$(projectdir)")

	add_files({
		"main.cpp",
		"omp/dispatchers.cpp",
	})

	add_deps("builtin-apps")

	add_packages("cnpy")
end


target("bm-play-ndarray")
do
	add_rules("common_flags", "run_on_android")

    add_includedirs("$(projectdir)")

	add_files({
		"bm_main.cpp",
		"omp/dispatchers.cpp",
	})

	add_deps("builtin-apps")

	add_packages("cnpy")
	add_packages("benchmark")
end

if has_config("use_cuda") then

	-- CUDA test program

	target("play-ndarray-cu")
	do
		add_rules("common_flags")

		add_includedirs("$(projectdir)")

		add_files({
			"cuda_main.cu",
			"cuda/kernels.cu",
			"omp/dispatchers.cpp",
		})

		add_deps("builtin-apps", "builtin-apps-cuda")

		add_cugencodes("native", {force = true})

		add_packages("cnpy")
	end

	-- Benchmark program


	target("bm-play-ndarray-cu")
	do
		add_rules("common_flags")

		add_includedirs("$(projectdir)")

		add_files({
			"bm_cuda_main.cu",
			"cuda/kernels.cu",
			"omp/dispatchers.cpp",
		})

		add_deps("builtin-apps", "builtin-apps-cuda")

		add_cugencodes("native", {force = true})

		add_packages("cnpy")
		add_packages("benchmark")
	end



	-- target("pipe-ndarray-cu")
	-- do
	-- 	add_rules("common_flags")

	-- 	add_includedirs("$(projectdir)")

	-- 	add_files({
	-- 		"pipe_cu_main.cu",
	-- 		"cuda/dispatchers.cu",
	-- 		"cuda/kernels.cu",
	-- 		"omp/dispatchers.cpp",
	-- 	})

	-- 	add_links("nvToolsExt")

	-- 	add_deps("builtin-apps", "builtin-apps-cuda")

	-- 	add_cugencodes("native", {force = true})

	-- 	add_packages("cnpy")
	-- end


end
