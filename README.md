# Redwood All-in-One

```
===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 C Header               22         6390         6390            0            0
 C++                    68        14853         9219         2525         3109
 C++ Header             61         8858         5967         1387         1504
 Fish                    1            3            3            0            0
 GLSL                   22         1775         1212          251          312
 JSON                   54       113267       113267            0            0
 Lua                    23         1712         1087          337          288
 Makefile                1           38           20            9            9
 Python                 33         5641         4020          773          848
 Shell                   2           96           65           22            9
 SVG                    10        24296        24019          277            0
 TOML                    1           12           12            0            0
-------------------------------------------------------------------------------
 Markdown                5          182            0          131           51
 |- BASH                 3            4            4            0            0
 |- Python               1           18           10            4            4
 (Total)                            204           14          135           55
===============================================================================
 Total                 303       177123       165281         5712         6130
===============================================================================
```

## Requirements

Need 

* `uv` -> to install `curl -LsSf https://astral.sh/uv/install.sh | sh`
* `just` -> to install `cargo install just`

## Collect Data


To collect the _profiling table_

```bash
    just collect-all-android
```

then the raw log will be stored at 

`data/bm_logs` folder


