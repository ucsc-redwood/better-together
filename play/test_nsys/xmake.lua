if has_config("use_cuda") then
	target("test-nsys")
	do
		add_rules("common_flags")

		add_includedirs("$(projectdir)")

		add_files({
			"main.cu",
		})

		add_links("nvToolsExt")

		add_cugencodes("native", {force = true})

		add_packages("cnpy")
	end
end
