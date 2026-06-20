# bt_codegen.cmake -- generic "regenerate a committed file from a JSON source at build
# time" helper, generalizing the device-spec codegen pattern (a python script writes
# identical bytes when nothing changed, so a clean tree stays clean; if no python is
# found the committed output is used as-is). Requires BT_PYTHON (find_program) in scope.
#
#   bt_codegen(TARGET bt_vocab
#              SCRIPT  ${CMAKE_SOURCE_DIR}/scripts/embed_vocab.py
#              DEPENDS ${CMAKE_SOURCE_DIR}/vocab.json
#              OUTPUTS ${CMAKE_SOURCE_DIR}/builtin-apps/generated/bt_vocab.hpp ...)
# Adds an add_custom_target(${TARGET}) and makes bt_core depend on it.
function(bt_codegen)
    cmake_parse_arguments(ARG "" "TARGET;SCRIPT" "DEPENDS;OUTPUTS" ${ARGN})
    if(NOT BT_PYTHON)
        return() # no python -> use the committed generated files as-is
    endif()
    add_custom_command(
        OUTPUT ${ARG_OUTPUTS}
        COMMAND ${BT_PYTHON} ${ARG_SCRIPT}
        DEPENDS ${ARG_DEPENDS} ${ARG_SCRIPT}
        COMMENT "Codegen (${ARG_TARGET}): ${ARG_SCRIPT}"
        VERBATIM
    )
    add_custom_target(${ARG_TARGET} DEPENDS ${ARG_OUTPUTS})
    add_dependencies(bt_core ${ARG_TARGET})
endfunction()
