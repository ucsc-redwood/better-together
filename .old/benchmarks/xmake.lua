-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

-- ----------------------------------------------------------------
-- Common benchmark configuration
-- ----------------------------------------------------------------

add_requires("benchmark")

rule("benchmark_config")
on_load(function(target)
	target:set("kind", "binary")
	target:set("group", "benchmarks")
	target:add("includedirs", "$(projectdir)")
	target:add("packages", "benchmark")
end)
rule_end()

-- ----------------------------------------------------------------
-- Benchmarks
-- ----------------------------------------------------------------

includes("bm-tree")
includes("bm-cifar-sparse")
includes("bm-cifar-dense")
includes("bm-misc")
