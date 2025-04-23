if has_config("use_vulkan") then
	includes("cifar-dense-vk")
	includes("cifar-sparse-vk")
	includes("tree-vk")
end

if has_config("use_cuda") then
	includes("cifar-dense-cu")
	includes("cifar-sparse-cu")
	-- includes("tree-cu")
end
