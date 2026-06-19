# apps/ -- one subdirectory per application. Each contributes its kernels to the
# backend libraries (bt_core/cuda/vulkan, defined in platform/) via target_sources(),
# and defines its own runner / test / benchmark executables.
add_subdirectory(tree)
add_subdirectory(cifar-dense)
add_subdirectory(cifar-sparse)
