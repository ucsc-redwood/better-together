# Android arm64-v8a cross toolchain — wraps the NDK's android.toolchain.cmake.
#
# NDK resolution order:
#   1. $ANDROID_NDK_HOME                              (explicit override)
#   2. $ANDROID_HOME/ndk/${BT_ANDROID_NDK_VERSION}    (SDK-managed; default)
#   3. $ANDROID_NDK_ROOT
#
# BT_ANDROID_NDK_VERSION mirrors Gradle's `android { ndkVersion "..." }`.
#
#   cmake --preset android && cmake --build --preset android
set(BT_ANDROID_NDK_VERSION "29.0.14206865" CACHE STRING
    "NDK version under $ANDROID_HOME/ndk (matches Gradle ndkVersion)")

if(DEFINED ENV{ANDROID_NDK_HOME})
  set(_bt_ndk "$ENV{ANDROID_NDK_HOME}")
elseif(DEFINED ENV{ANDROID_HOME} AND EXISTS "$ENV{ANDROID_HOME}/ndk/${BT_ANDROID_NDK_VERSION}")
  set(_bt_ndk "$ENV{ANDROID_HOME}/ndk/${BT_ANDROID_NDK_VERSION}")
elseif(DEFINED ENV{ANDROID_NDK_ROOT})
  set(_bt_ndk "$ENV{ANDROID_NDK_ROOT}")
else()
  message(FATAL_ERROR
    "No Android NDK found. Set ANDROID_NDK_HOME, or install NDK ${BT_ANDROID_NDK_VERSION} "
    "under $ANDROID_HOME/ndk:  sdkmanager \"ndk;${BT_ANDROID_NDK_VERSION}\"")
endif()

set(ANDROID_ABI arm64-v8a)
set(ANDROID_PLATFORM android-28)
set(ANDROID_STL c++_shared)

include("${_bt_ndk}/build/cmake/android.toolchain.cmake")
