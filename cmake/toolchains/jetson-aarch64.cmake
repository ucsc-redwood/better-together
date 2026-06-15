# Cross-compile toolchain for NVIDIA Jetson Orin (aarch64, sm_87).
# Use INSIDE the NVIDIA cross container:
#   nvcr.io/nvidia/jetpack-linux-aarch64-crosscompile-x86:6.1  (CUDA 12.6, matches JetPack 6.2)
# Configure: cmake -S . -B build-jetson \
#   -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/jetson-aarch64.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# CUDA: x86 nvcc emitting aarch64 host code via the cross g++; statically links cudart.
set(CMAKE_CUDA_COMPILER       /usr/local/cuda-12.6/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER  aarch64-linux-gnu-g++)
set(CMAKE_CUDA_ARCHITECTURES  87)

# Resolve find_package() libs from the cross sysroot, not the x86 host.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
