# Android arm64-v8a cross toolchain — wraps the NDK's android.toolchain.cmake.
# Requires ANDROID_NDK_HOME (or ANDROID_NDK_ROOT) to point at an NDK (r26+).
#
#   export ANDROID_NDK_HOME=~/android-ndk-r27c
#   cmake --preset android && cmake --build --preset android
if(DEFINED ENV{ANDROID_NDK_HOME})
  set(_bt_ndk "$ENV{ANDROID_NDK_HOME}")
elseif(DEFINED ENV{ANDROID_NDK_ROOT})
  set(_bt_ndk "$ENV{ANDROID_NDK_ROOT}")
else()
  message(FATAL_ERROR "Set ANDROID_NDK_HOME to your Android NDK path (r26+).")
endif()

set(ANDROID_ABI arm64-v8a)
set(ANDROID_PLATFORM android-28)
set(ANDROID_STL c++_shared)

include("${_bt_ndk}/build/cmake/android.toolchain.cmake")
