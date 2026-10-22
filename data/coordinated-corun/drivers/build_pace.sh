#!/usr/bin/env bash
# Build lpa_pace (design D1) for arm64 Android, linking the static libraries of the llm-phase-affinity
# Samsung build of llama.cpp 8345f33 (GGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm+fp16, Vulkan on, OpenMP and
# KleidiAI off), with that build's compile flags. Output: build/ccr/lpa_pace (gitignored).
set -euo pipefail
NDK=/home/doremy/android-ndk-r30
B=/home/doremy/Desktop/ucsc/better-together/build/lpa/llamacpp-android-s24
S=/home/doremy/llamacpp-compare/llama.cpp
CXX=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang++
SRC=/home/doremy/Desktop/ucsc/bt-ccr/data/coordinated-corun/drivers/lpa_pace.cpp
OUT=/home/doremy/Desktop/ucsc/bt-ccr/build/ccr/lpa_pace
$CXX -std=c++17 -g -DANDROID -fdata-sections -ffunction-sections -funwind-tables -fstack-protector-strong \
  -D_FORTIFY_SOURCE=2 -O3 -DNDEBUG -Wall -Wextra -pthread \
  -I$S/include -I$S/ggml/include \
  $SRC -o $OUT \
  -static-libstdc++ -Wl,--gc-sections \
  $B/src/libllama.a $B/ggml/src/libggml.a $B/ggml/src/libggml-cpu.a $B/ggml/src/ggml-vulkan/libggml-vulkan.a \
  $B/ggml/src/libggml-base.a -lm -ldl \
  $NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/28/libvulkan.so -pthread -latomic
ls -la $OUT
