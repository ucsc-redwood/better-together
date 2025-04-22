-- ----------------------------------------------------------------------------
-- Test, just test if the dispatchers are working
-- ----------------------------------------------------------------------------

target("run-cifar-dense-omp")
do
	add_rules("common_flags", "run_on_android")
	set_group("test")
	set_kind("binary")
	add_files({
		"./main.cpp",
	})

	add_deps("builtin-apps")
end


target("test-cifar-dense-omp")
do
	add_rules("common_flags", "run_on_android")
	set_group("test")
	set_kind("binary")
	add_files({
		"./test_main.cpp",
	})

	add_deps("builtin-apps")

	add_packages("gtest")
end

-- ----------------------------------------------------------------------------
-- Benchmark for individual stages
-- ----------------------------------------------------------------------------

target("bm-cifar-dense-omp")
do
	add_rules("common_flags", "run_on_android")
	set_group("micro-benchmark")
	set_kind("binary")
	add_files({
		"./bm_main.cpp",
	})

	add_deps("builtin-apps")

	add_packages("benchmark")
end
