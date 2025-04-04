if has_config("use_cuda") then
	target("test-nsys")
	do
		add_rules("common_flags", "cuda_config")

		add_includedirs("$(projectdir)")

		add_files({
			"main.cu",
		})

		add_packages("cnpy")
	end
end
