#!/usr/bin/env python3
"""Prepare identical logical inputs across FP16/BF16 and four layouts."""
import argparse
import hashlib
import json
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-case", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    source = args.source_case.resolve(strict=True)
    batch, m, n, k, dtype, ta, tb = map(int, (source / "input/case.txt").read_text().split())
    if (dtype, ta, tb) != (1, 0, 0):
        parser.error("source must be FP16 t00")
    a = np.fromfile(source / "input/x1.bin", dtype="<f2").reshape(batch, m, k).astype(np.float32)
    b = np.fromfile(source / "input/x2.bin", dtype="<f2").reshape(batch, k, n).astype(np.float32)
    logical_hash = hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for target_dtype in (1, 2):
        for target_ta in (0, 1):
            for target_tb in (0, 1):
                name = f"d{target_dtype}_t{target_ta}{target_tb}"
                case = args.output / name
                (case / "input").mkdir(parents=True, exist_ok=True)
                (case / "output").mkdir(exist_ok=True)
                for filename, values, transpose in (("x1.bin", a, target_ta), ("x2.bin", b, target_tb)):
                    if target_dtype == 1:
                        words = values.astype("<f2").view("<u2")
                        decoded = words.view("<f2").astype(np.float32)
                    else:
                        bits = values.view(np.uint32)
                        words = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
                        decoded = (words.astype(np.uint32) << 16).view(np.float32)
                    if not np.array_equal(values, decoded):
                        raise RuntimeError("Source values must be exactly representable in both dtypes")
                    if transpose:
                        words = words.swapaxes(-1, -2)
                    words.tofile(case / "input" / filename)
                fields = (batch, m, n, k, target_dtype, target_ta, target_tb)
                (case / "input/case.txt").write_text(" ".join(map(str, fields)) + "\n")
                manifest.append(name + "\t" + "\t".join(map(str, fields)))
    (args.output / "cases.tsv").write_text("\n".join(manifest) + "\n")
    (args.output / "logical-input.json").write_text(json.dumps({
        "source_case": str(source), "logical_fp32_sha256": logical_hash,
        "shape": [batch, m, n, k], "cases": 8,
        "exactly_representable_in_both_dtypes": True,
    }, indent=2) + "\n")
    print(f"Prepared 8 layout probes; identical logical input SHA-256={logical_hash}")


if __name__ == "__main__":
    main()
