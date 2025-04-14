-- Copyright (c) 2025 Yanwen Xu (yxu83@ucsc.edu). MIT License.

target("play-curl-json")
do
	add_rules("common_flags", "run_on_android")

    add_includedirs("$(projectdir)")

	add_headerfiles({
		"fetch_schedule.hpp",
	})
    
	add_files({
		"main.cpp",
	})

	add_deps("builtin-apps")
    add_packages("libcurl")
	add_packages("nlohmann_json")
end
