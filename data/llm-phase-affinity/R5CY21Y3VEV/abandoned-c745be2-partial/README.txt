Abandoned: first 6 runs of round 1 of task 2.1 with the owner's llama-bench (c745be2,
/data/local/tmp/lcpp). Stopped because that build lacks i8mm (no smmla in its disassembly)
and was 22% slower at CPU prefill than llama.cpp 8345f33 built with
-march=armv8.6-a+dotprod+i8mm+fp16 (diag-build-comparison.txt). Not used in any table.
