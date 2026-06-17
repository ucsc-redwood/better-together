# Run at BUILD time (cmake -P) to capture the current short git sha into a header.
# Re-runs every build (driven by an always-out-of-date custom target) so a
# commit+rebuild refreshes bm-prof provenance without reconfiguring. Only rewrites
# the file when the sha changed, to avoid needless recompiles of its includers.
#   -DGIT_SRC_DIR=<repo> -DGIT_OUT=<header path>
execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${GIT_SRC_DIR}
                OUTPUT_VARIABLE GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT GIT_SHA)
  set(GIT_SHA "unknown")
endif()
set(CONTENT "#define BT_GIT_SHA \"${GIT_SHA}\"\n")
set(OLD "")
if(EXISTS ${GIT_OUT})
  file(READ ${GIT_OUT} OLD)
endif()
if(NOT OLD STREQUAL CONTENT)
  file(WRITE ${GIT_OUT} "${CONTENT}")
endif()
