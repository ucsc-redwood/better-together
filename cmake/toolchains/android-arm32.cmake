# Android armeabi-v7a (32-bit) cross toolchain — for legacy 32-bit-only devices
# (e.g. the Moto g pure / MT6762G ships a 32-bit userspace). Mirrors
# android-arm64.cmake but selects the armeabi-v7a ABI.
set(BT_ANDROID_NDK_VERSION
    "29.0.14206865"
    CACHE STRING
    "NDK version under $ANDROID_HOME/ndk (matches Gradle ndkVersion)"
)

if(DEFINED ENV{ANDROID_NDK_HOME})
    set(_bt_ndk "$ENV{ANDROID_NDK_HOME}")
elseif(
    DEFINED ENV{ANDROID_HOME}
    AND EXISTS "$ENV{ANDROID_HOME}/ndk/${BT_ANDROID_NDK_VERSION}"
)
    set(_bt_ndk "$ENV{ANDROID_HOME}/ndk/${BT_ANDROID_NDK_VERSION}")
elseif(DEFINED ENV{ANDROID_NDK_ROOT})
    set(_bt_ndk "$ENV{ANDROID_NDK_ROOT}")
else()
    message(
        FATAL_ERROR
        "No Android NDK found. Set ANDROID_NDK_HOME, or install NDK ${BT_ANDROID_NDK_VERSION} "
        "under $ANDROID_HOME/ndk:  sdkmanager \"ndk;${BT_ANDROID_NDK_VERSION}\""
    )
endif()

set(ANDROID_ABI armeabi-v7a)
set(ANDROID_ARM_NEON ON)
set(ANDROID_PLATFORM android-28)
set(ANDROID_STL c++_shared)

include("${_bt_ndk}/build/cmake/android.toolchain.cmake")

# On 32-bit ABIs vulkan.hpp leaves VULKAN_HPP_TYPESAFE_CONVERSION undefined (handles are
# plain uint64_t, so the vk::Handle(VkHandle) ctors become `explicit`), which breaks our
# brace-initialized handle returns (e.g. kiss-vk/vma_pmr.cpp). Force the 64-bit-style
# type-safe handle semantics so the same source compiles on armeabi-v7a.
add_compile_definitions(VULKAN_HPP_TYPESAFE_CONVERSION=1)
