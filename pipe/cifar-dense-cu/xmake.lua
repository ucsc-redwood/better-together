-- ----------------------------------------------------------------------------
-- Run stages with interference (100 tasks)
-- ----------------------------------------------------------------------------

target("run-pipe-cifar-dense-cu")
do
	add_rules("common_flags", "cuda_config", "run_on_android")
	set_kind("binary")
	add_files({
		"main.cu",
	})

	add_deps("builtin-apps-cuda")
	add_deps("builtin-apps")
end

target("bm-baseline-cifar-dense-cu")
do
	add_rules("common_flags", "cuda_config", "run_on_android")
	set_kind("binary")
	add_files({
		"bm_baseline.cu",
	})

	add_deps("builtin-apps-cuda")
	add_deps("builtin-apps")

	add_packages("benchmark")
end

-- -- ----------------------------------------------------------------------------
-- -- Run stages with interference (100 tasks)
-- -- Use this to generate Table, which will feed into optimizer
-- -- ----------------------------------------------------------------------------

-- target("bm-fully-cifar-dense-cu")
-- do
-- 	add_rules("common_flags", "cuda_config", "run_on_android")
-- 	set_kind("binary")
-- 	add_files({
-- 		"bm_fully_vs_normal.cpp",
-- 	})

-- 	add_deps("builtin-apps-cuda")
-- 	add_deps("builtin-apps")
-- end

-- -- ----------------------------------------------------------------------------
-- -- Generate the Log/Graph of "schedules" (100 tasks)
-- -- ----------------------------------------------------------------------------

-- target("bm-gen-logs-cifar-dense-cu")
-- do
-- 	add_rules("common_flags", "cuda_config", "run_on_android")
-- 	set_kind("binary")
-- 	add_files({
-- 		"bm_gen_log.cpp",
-- 	})

-- 	add_deps("builtin-apps-cuda")
-- 	add_deps("builtin-apps")
-- end
