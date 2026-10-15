# bt_shaders.cmake -- compile the Vulkan compute shaders at build time.
#
# Every platform/engine/vulkan/shaders/comp/*.comp is compiled with glslc to SPIR-V and
# embedded as a C header (<name>_spv / <name>_spv_len) under the build tree; bt::vulkan
# includes them through shaders/all_shaders.hpp as "h/<name>_spv.h". Nothing generated is
# committed: glslc output is not byte-reproducible across versions, so committed .spv
# files only produced noise diffs.
#
# glslc is required whenever BT_ENABLE_VULKAN is ON. Where to get it:
#   * Linux      apt install glslc  (or the LunarG Vulkan SDK)
#   * Android    the NDK ships it under shader-tools/<host>/ (found below)
#   * cross image  Dockerfile.cross* install it
function(bt_build_shaders)
    set(sdir ${CMAKE_SOURCE_DIR}/platform/engine/vulkan/shaders)
    set(gen ${CMAKE_BINARY_DIR}/generated/vk_shaders)

    # glslc is a build-host tool: look it up on the host even when cross-compiling.
    set(ndk_hints)
    if(DEFINED ANDROID_NDK)
        file(GLOB ndk_hints ${ANDROID_NDK}/shader-tools/*)
    endif()
    find_program(BT_GLSLC glslc HINTS ${ndk_hints} NO_CMAKE_FIND_ROOT_PATH)
    if(NOT BT_GLSLC)
        message(
            FATAL_ERROR
            "BT_ENABLE_VULKAN needs glslc to compile the shaders. Install it "
            "(Ubuntu: `apt install glslc`; Android: it ships in the NDK) or pass "
            "-DBT_GLSLC=/path/to/glslc."
        )
    endif()

    file(GLOB comps CONFIGURE_DEPENDS ${sdir}/comp/*.comp)
    set(headers)
    foreach(comp ${comps})
        get_filename_component(name ${comp} NAME_WE)
        set(spv ${gen}/spv/${name}.spv)
        set(hdr ${gen}/h/${name}_spv.h)
        add_custom_command(
            OUTPUT ${hdr}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${gen}/spv ${gen}/h
            COMMAND
                ${BT_GLSLC} --target-env=vulkan1.3 -O -fshader-stage=compute -o
                ${spv} ${comp}
            COMMAND
                ${CMAKE_COMMAND} -DIN=${spv} -DOUT=${hdr} -DNAME=${name} -P
                ${CMAKE_SOURCE_DIR}/cmake/bt_embed_spv.cmake
            DEPENDS ${comp} ${CMAKE_SOURCE_DIR}/cmake/bt_embed_spv.cmake
            COMMENT "Compiling shader ${name}.comp"
            VERBATIM
        )
        list(APPEND headers ${hdr})
    endforeach()

    add_custom_target(bt_shaders DEPENDS ${headers})
    set(BT_SHADER_GEN_DIR ${gen} PARENT_SCOPE)
endfunction()
