#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare raw SSBO output from spirv_software_matrix's single-call fixtures.

Run one workgroup (64 threads), descriptor set 0 binding 0 = 512-byte storage
buffer. Use the actual requested Vulkan subgroup size, not the fixture filename.
"""
import argparse
import ctypes
import math
import struct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", nargs="?", type=Path, help="raw 512-byte SSBO downloaded after dispatch")
parser.add_argument("--depth", type=int, choices=(8, 16), required=True)
parser.add_argument("--subgroup-size", type=int, choices=(32, 64), default=32)
parser.add_argument("--calls", type=int, default=1)
parser.add_argument("--rounding-case", type=int, choices=range(1, 11))
parser.add_argument("--reference", type=Path, help="write the expected binary output")
args = parser.parse_args()
if args.calls < 1 or (args.output is None and args.reference is None):
    parser.error("provide an output to check or --reference; --calls must be positive")
fma = ctypes.CDLL(None).fmaf
fma.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
fma.restype = ctypes.c_float

def unpack(word):
    return struct.unpack("<ee", struct.pack("<I", word))

def packed(values):
    try:
        return struct.unpack("<I", struct.pack("<ee", *values))[0]
    except OverflowError:
        out = bytearray()
        for v in values:
            try:
                out.extend(struct.pack("<e", v))
            except OverflowError:
                out.extend(struct.pack("<e", math.copysign(math.inf, v)))
        return struct.unpack("<I", out)[0]

def round_half(value):
    return unpack(packed((value, 0.0)))[0]

# The fixture uses IR LaneId without the profile's larger-than-guest masking.
inputs = [[(0x3c003c00 + i * 0x01000100 + ((thread % args.subgroup_size) << 3)) & 0xffffffff
           for i in range(8)] for thread in range(64)]
if args.rounding_case is not None:
    cases = [
        (0x1000, 0x3c00, 0x3c00, 0), (0x1000, 0x3c00, 0x3c01, 0),
        (0x0001, 0x3800, 0, 0), (0x0003, 0x3800, 0, 0),
        (0x8001, 0x3800, 0x8000, 0), (0x7bff, 0x3c00, 0x4c00, 0),
        (0xfbff, 0x3c00, 0xcc00, 0), (0x0400, 0x3800, 0, 0),
        (0x1000, 0x3c00, 0x3c00, 0x1000), (0xbc003c00, 0x3c00, 0x0001, 0),
    ]
    a, b, c, upper_a = cases[args.rounding_case - 1]
    inputs = [[a if thread % 4 == 0 else 0] * 2 +
              [upper_a if thread % 4 == 0 else 0] * 2 +
              [b | (b << 16)] * 2 + [c | (c << 16)] * 2 for thread in range(64)]
for _ in range(args.calls):
    outputs = []
    for thread in range(64):
        lane = thread % args.subgroup_size
        warp = (thread // args.subgroup_size) * args.subgroup_size + (lane & ~31)
        acc = [list(unpack(inputs[thread][6 + row])) for row in range(2)]
        for k in range(args.depth):
            component = (k % 8) // 2
            a_lane = warp + (lane & 28) + component
            for row in range(2):
                a = unpack(inputs[a_lane][row + (k // 8) * 2])[k % 2]
                for col in range(2):
                    b_lane = warp + (lane & 3) * 8 + col * 4 + component
                    b = unpack(inputs[b_lane][4 + k // 8])[k % 2]
                    acc[row][col] = fma(a, b, acc[row][col])
            if k % 8 == 7:
                acc = [[round_half(v) for v in row] for row in acc]
        outputs.append([packed(row) for row in acc])
    for thread in range(64):
        inputs[thread][6:] = outputs[thread]
expected = b"".join(struct.pack("<II", *row) for row in outputs)
if args.reference is not None:
    args.reference.write_bytes(expected)
    print(f"Wrote reference: {args.reference} ({len(expected)} bytes)")
if args.output is None:
    raise SystemExit(0)
actual = args.output.read_bytes()
if actual != expected:
    for offset in range(0, min(len(actual), len(expected)), 4):
        if actual[offset:offset+4] != expected[offset:offset+4]:
            print(f"thread={offset // 8} word={(offset % 8) // 4} "
                  f"actual={actual[offset:offset+4].hex()} expected={expected[offset:offset+4].hex()}")
            break
    raise SystemExit(f"MISMATCH: actual {len(actual)} bytes; expected {len(expected)}")
print(f"PASS: 64 lanes, K={args.depth}, subgroup={args.subgroup_size}, calls={args.calls}, bit exact")
