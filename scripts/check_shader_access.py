#!/usr/bin/env python3
"""Check that every buffer binding of the Vulkan compute shaders is declared the way the
shader uses it.

The engine decides which buffers to flush (host -> GPU) and invalidate (GPU -> host) from the
`readonly` / `writeonly` qualifiers it reads out of the SPIR-V (platform/engine/vulkan/
algorithm.cpp). glslang rejects a write to a `readonly` buffer but accepts a read of a
`writeonly` one, and a missing qualifier is silently read-write, so this script compiles each
shader and compares the declared access with what the optimized SPIR-V does with the buffer.

  declared read-write, used read-only   -> add `readonly`    (the engine would invalidate it for nothing)
  declared read-write, used write-only  -> add `writeonly`
  declared write-only, used read        -> wrong: the engine would skip the flush it needs
  declared read-only,  used write       -> wrong (glslang already rejects this)

Needs glslc and spirv-dis (Ubuntu: `apt install glslc spirv-tools`). Exit 1 on any mismatch.
Usage: scripts/check_shader_access.py [shader.comp ...]   (default: every shader)
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

SHADERS = Path(__file__).resolve().parent.parent / "platform/engine/vulkan/shaders/comp"
ATOMICS_RW = (
    "OpAtomicIAdd OpAtomicISub OpAtomicAnd OpAtomicOr OpAtomicXor OpAtomicSMin OpAtomicUMin "
    "OpAtomicSMax OpAtomicUMax OpAtomicExchange OpAtomicCompareExchange OpAtomicIIncrement "
    "OpAtomicIDecrement"
).split()


def analyze(spv: str):
    """binding -> (declared, used), each one of read-only / write-only / read-write / unused."""
    dis = subprocess.run(
        ["spirv-dis", "--raw-id", "--no-header", spv], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    binding, non_writable, non_readable, member_nw, member_nr, members = (
        {},
        set(),
        set(),
        {},
        {},
        {},
    )
    var_type, pointee = {}, {}
    for line in dis:
        if m := re.match(r"\s*OpDecorate %(\d+) Binding (\d+)", line):
            binding[int(m[1])] = int(m[2])
        elif m := re.match(r"\s*OpDecorate %(\d+) NonWritable", line):
            non_writable.add(int(m[1]))
        elif m := re.match(r"\s*OpDecorate %(\d+) NonReadable", line):
            non_readable.add(int(m[1]))
        elif m := re.match(r"\s*OpMemberDecorate %(\d+) (\d+) NonWritable", line):
            member_nw.setdefault(int(m[1]), set()).add(int(m[2]))
        elif m := re.match(r"\s*OpMemberDecorate %(\d+) (\d+) NonReadable", line):
            member_nr.setdefault(int(m[1]), set()).add(int(m[2]))
        elif m := re.match(r"\s*%(\d+) = OpTypeStruct(.*)", line):
            members[int(m[1])] = len(m[2].split())
        elif m := re.match(r"\s*%(\d+) = OpTypePointer StorageBuffer %(\d+)", line):
            pointee[int(m[1])] = int(m[2])
        elif m := re.match(r"\s*%(\d+) = OpVariable %(\d+) StorageBuffer", line):
            var_type[int(m[1])] = int(m[2])
    variables = [v for v in var_type if v in binding]
    derived = {v: v for v in variables}
    reads, writes = dict.fromkeys(variables, False), dict.fromkeys(variables, False)
    for line in dis:
        if m := re.match(
            r"\s*%(\d+) = (?:OpAccessChain|OpInBoundsAccessChain|OpPtrAccessChain|OpCopyObject) %\d+ %(\d+)",
            line,
        ):
            if int(m[2]) in derived:
                derived[int(m[1])] = derived[int(m[2])]
        elif m := re.match(r"\s*%\d+ = (Op\w+) %\d+ %(\d+)", line):
            op, ptr = m[1], int(m[2])
            if ptr in derived:
                v = derived[ptr]
                if op == "OpLoad" or op == "OpAtomicLoad":
                    reads[v] = True
                elif op in ATOMICS_RW:
                    reads[v] = writes[v] = True
        elif m := re.match(r"\s*OpStore %(\d+)", line):
            if int(m[1]) in derived:
                writes[derived[int(m[1])]] = True
        elif re.search(r"OpFunctionCall|OpCopyMemory", line):
            for ref in re.findall(r"%(\d+)", line):
                if int(ref) in derived:  # a pointer escapes: cannot tell, assume both
                    reads[derived[int(ref)]] = writes[derived[int(ref)]] = True
    out = {}
    for v in variables:
        block = pointee.get(var_type[v])
        nw = v in non_writable or (
            block is not None
            and members.get(block, 0) > 0
            and len(member_nw.get(block, ())) == members[block]
        )
        nr = v in non_readable or (
            block is not None
            and members.get(block, 0) > 0
            and len(member_nr.get(block, ())) == members[block]
        )
        declared = "read-only" if nw else "write-only" if nr else "read-write"
        used = (
            "read-write"
            if reads[v] and writes[v]
            else "read-only"
            if reads[v]
            else "write-only"
            if writes[v]
            else "unused"
        )
        out[binding[v]] = (declared, used)
    return out


def main() -> int:
    files = [Path(a) for a in sys.argv[1:]] or sorted(SHADERS.glob("*.comp"))
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            spv = f"{tmp}/{f.stem}.spv"
            subprocess.run(
                [
                    "glslc",
                    "--target-env=vulkan1.3",
                    "-O",
                    "-fshader-stage=compute",
                    "-o",
                    spv,
                    str(f),
                ],
                check=True,
            )
            for b, (declared, used) in sorted(analyze(spv).items()):
                if used == "unused" or declared == used:
                    continue
                bad += 1
                print(f"{f.name}: binding {b} is declared {declared} but used {used}")
    print("shader buffer qualifiers: OK" if not bad else f"{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
