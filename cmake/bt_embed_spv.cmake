# bt_embed_spv.cmake -- run with `cmake -P`: turn a SPIR-V binary into a C header.
#   -DIN=<file.spv> -DOUT=<name_spv.h> -DNAME=<name>
# Emits `unsigned char <NAME>_spv[]` and `unsigned int <NAME>_spv_len` (the names
# `xxd -i` produced), so shaders/all_shaders.hpp's SHADER_ENTRY needs no change.
file(READ "${IN}" hex HEX)
string(LENGTH "${hex}" hex_len)
math(EXPR size "${hex_len} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
file(
    WRITE "${OUT}"
    "#pragma once\nunsigned char ${NAME}_spv[] = {${bytes}};\nunsigned int ${NAME}_spv_len = ${size};\n"
)
