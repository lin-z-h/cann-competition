#!/usr/bin/env python3
"""Repeat device runs and verify existing plus independent random cases."""
import argparse
import hashlib
import pathlib
import subprocess

import numpy as np


def check(case_dir):
    fields = list(map(int, (case_dir / "input/case.txt").read_text().split()))
    batch, m, n, k, dtype, ta, tb = fields

    def decode(path, shape):
        words = np.fromfile(path, dtype="<u2")
        values = (words.view("<f2").astype(np.float32) if dtype == 1 else
                  (words.astype(np.uint32) << 16).view(np.float32))
        return values.reshape(shape).astype(np.float64)

    a = decode(case_dir / "input/x1.bin", (batch, k, m) if ta else (batch, m, k))
    b = decode(case_dir / "input/x2.bin", (batch, n, k) if tb else (batch, k, n))
    if ta:
        a = a.swapaxes(-1, -2)
    if tb:
        b = b.swapaxes(-1, -2)
    expected = (a @ b).max(axis=-1).sum(axis=-1)
    actual = np.fromfile(case_dir / "output/y.bin", dtype="<f4")
    if actual.shape != expected.shape:
        raise RuntimeError(f"{case_dir.name}: output size {actual.shape}, expected {expected.shape}")
    error = np.abs(actual - expected)
    tolerance = 1e-4 + 1e-4 * np.abs(expected)
    if not np.all(np.isfinite(actual) & (error <= tolerance)):
        raise RuntimeError(f"{case_dir.name}: actual={actual.tolist()} expected={expected.tolist()} error={error.tolist()}")
    print(f"PASS {case_dir.name}: max_abs_error={error.max():.6g}", flush=True)


def random_cases(root, extra_shapes=()):
    shapes = [(1, 128, 257, 64), (1, 256, 256, 64), (1, 256, 257, 64),
              (1, 35, 37, 29), (2, 7, 19, 32), (1, 128, 1025, 64),
              (1, 4096, 256, 64), (1, 2048, 513, 64),
              (1, 2048, 512, 256), (1, 2048, 257, 64)]
    shapes = list(dict.fromkeys([*shapes, *extra_shapes]))
    for dtype in (1, 2):
        for ta in (0, 1):
            for tb in (0, 1):
                for batch, m, n, k in shapes:
                    name = f"random_b{batch}_m{m}_n{n}_k{k}_d{dtype}_t{ta}{tb}"
                    # Keep each case unchanged when other shapes are added or
                    # removed from the suite, so repeat runs are comparable.
                    seed = int.from_bytes(hashlib.sha256(name.encode("ascii")).digest()[:8], "little")
                    rng = np.random.default_rng(seed)
                    case = root / name
                    (case / "input").mkdir(parents=True, exist_ok=True)
                    (case / "output").mkdir(exist_ok=True)
                    (case / "input/case.txt").write_text(f"{batch} {m} {n} {k} {dtype} {ta} {tb}\n")
                    for filename, shape, transpose in (("x1.bin", (batch, m, k), ta),
                                                        ("x2.bin", (batch, k, n), tb)):
                        values = rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)
                        if dtype == 1:
                            words = values.astype("<f2").view("<u2")
                        else:
                            bits = values.view(np.uint32)
                            words = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
                        if transpose:
                            words = words.swapaxes(-1, -2)
                        words.tofile(case / "input" / filename)
                    yield case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=pathlib.Path, required=True)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--random-shape", action="append", default=[], metavar="B,M,N,K",
                        help="append a positive random shape across both dtypes and four layouts; requires --random")
    parser.add_argument("--case-filter", default="")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    extra_shapes = []
    for value in args.random_shape:
        try:
            shape = tuple(map(int, value.split(",")))
        except ValueError:
            parser.error("--random-shape must contain four positive integers B,M,N,K")
        if len(shape) != 4 or min(shape) < 1:
            parser.error("--random-shape must contain four positive integers B,M,N,K")
        extra_shapes.append(shape)
    if extra_shapes and not args.random:
        parser.error("--random-shape requires --random")
    exe = args.exe.resolve(strict=True)
    root = args.run_dir.resolve(strict=True)
    cases = [root / line.split("\t")[0] for line in (root / "cases.tsv").read_text().splitlines()]
    if args.random:
        cases.extend(random_cases(root / "random", extra_shapes))
    if args.case_filter:
        cases = [case for case in cases if args.case_filter in case.name]
    if not cases:
        parser.error("no cases selected")
    for iteration in range(args.repeat):
        for case in cases:
            (case / "output/y.bin").unlink(missing_ok=True)
            subprocess.run([str(exe)], cwd=case, check=True, timeout=120)
            check(case)
        print(f"Iteration {iteration + 1}: {len(cases)}/{len(cases)} cases passed", flush=True)


if __name__ == "__main__":
    main()
