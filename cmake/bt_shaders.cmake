# bt_shaders.cmake -- bake the Vulkan compute shaders into committed C headers.
#
# Replaces the standalone root Makefile (which CMake never invoked and which P5a
# silently broke: it moved the shaders to platform/engine/vulkan/shaders/ but the
# committed *_spv.h kept their old builtin-apps path-derived variable names, so a
# re-bake from the new tree would have emitted different names and broken
# all_shaders.hpp). The bake is now `xxd -i` run *from the spv/ dir*, so the
# variable name is the bare basename (<name>_spv) -- path-independent.
#
# Two deliberate properties:
#  * Committed-header fallback: bt::vulkan compiles the committed *_spv.h directly,
#    so a tree without glslc/xxd (e.g. the Android/NDK preset) builds fine. The bake
#    is an explicit, opt-in `bake-shaders` target -- NOT part of ALL.
#  * glslc is not byte-reproducible across versions (measured: 5 of 29 .spv differ
#    when re-baked locally), so .comp->.spv must never run on a default build or it
#    would dirty committed artifacts. Re-bake explicitly when a .comp changes:
#        cmake --build <dir> --target bake-shaders
function(bt_bake_shaders)
    set(sdir ${CMAKE_SOURCE_DIR}/platform/engine/vulkan/shaders)
    find_program(BT_GLSLC glslc)
    find_program(BT_XXD xxd)
    if(NOT BT_GLSLC OR NOT BT_XXD)
        message(
            STATUS
            "Shader bake: glslc/xxd not both found -> using committed *_spv.h as-is "
            "(no `bake-shaders` target)"
        )
        return()
    endif()

    file(GLOB comps CONFIGURE_DEPENDS ${sdir}/comp/*.comp)
    set(headers)
    foreach(comp ${comps})
        get_filename_component(name ${comp} NAME_WE)
        set(spv ${sdir}/spv/${name}.spv)
        set(hdr ${sdir}/h/${name}_spv.h)
        add_custom_command(
            OUTPUT ${hdr}
            # 1) compile glsl -> spir-v;  2) `xxd -i` from spv/ so the C array is named
            #    <name>_spv (path-independent), prefixed with #pragma once.
            COMMAND
                ${BT_GLSLC} --target-env=vulkan1.3 -O -fshader-stage=compute -o
                ${spv} ${comp}
            COMMAND
                ${CMAKE_COMMAND} -E chdir ${sdir}/spv sh -c
                "{ echo '#pragma once'; ${BT_XXD} -i ${name}.spv; } > ${hdr}"
            DEPENDS ${comp}
            COMMENT "Baking shader ${name}.comp -> .spv + ${name}_spv.h"
            VERBATIM
        )
        list(APPEND headers ${hdr})
    endforeach()

    add_custom_target(bake-shaders DEPENDS ${headers}) # opt-in; not built by ALL
endfunction()
