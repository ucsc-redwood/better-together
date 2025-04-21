-- ----------------------------------------------------------------------------
-- Test, just test if the dispatchers are working
-- ----------------------------------------------------------------------------

target("run-cifar-dense-cu")
do
	add_rules("common_flags", "cuda_config", "run_on_android")
	set_group("test")
	set_kind("binary")
	add_files({
		"./main.cu",
	})

	add_deps("builtin-apps-cuda")
	add_deps("builtin-apps")
		
	add_packages("gtest")
end

target("test-cifar-dense-cu")
do
	add_rules("common_flags", "cuda_config", "run_on_android")
	set_group("test")
	set_kind("binary")
	add_files({
		"./test_main.cu",
	})

	add_deps("builtin-apps-cuda")
	add_deps("builtin-apps")
		
	add_packages("gtest")
end

-- ----------------------------------------------------------------------------
-- Benchmark for individual stages
-- ----------------------------------------------------------------------------

target("bm-cifar-dense-cu")
do
	add_rules("common_flags", "cuda_config", "run_on_android")
	set_group("micro-benchmark")
	set_kind("binary")
	add_files({
		"./bm_main.cu",
	})

	add_deps("builtin-apps-cuda")
	add_deps("builtin-apps")
		
	add_packages("benchmark")
end

