#!/usr/bin/env python3
"""Collect msprof task metrics and validate each single-kernel application run."""
import argparse
import csv
import hashlib
import json
import pathlib
import statistics
import subprocess

from cannlab_verify import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=pathlib.Path, required=True)
    parser.add_argument("--case-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--metrics", default="PipeUtilization")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    exe = args.exe.resolve(strict=True)
    case = args.case_dir.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for sample in range(args.samples):
        sample_dir = output / f"sample-{sample + 1}"
        if sample_dir.exists():
            raise RuntimeError(f"Refusing to mix existing profiler data: {sample_dir}")
        (case / "output/y.bin").unlink(missing_ok=True)
        with (output / f"sample-{sample + 1}.log").open("w") as log:
            subprocess.run(["msprof", f"--application={exe}", f"--output={sample_dir}",
                            "--runtime-api=on", f"--aic-metrics={args.metrics}"],
                           cwd=case, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
        check(case)
        rows = []
        for path in sample_dir.glob("PROF_*/mindstudio_profiler_output/op_summary*.csv"):
            with path.open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
        compute = [row for row in rows if row["Task Type"] in ("MIX_AIC", "AI_CORE", "AIV", "MIX_AIV")]
        if len(compute) != 1 or len(rows) != 1:
            raise RuntimeError(f"Expected exactly one compute kernel: {len(compute)} compute rows, {len(rows)} total op rows")
        row = compute[0]
        record = {"sample": sample + 1, "kernel_count": len(compute), **row}
        records.append(record)
        print(f"SAMPLE {sample + 1}: kernel_count=1 duration_us={row['Task Duration(us)']}", flush=True)
    durations = [float(row["Task Duration(us)"]) for row in records]
    summary = {"exe": str(exe), "exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
               "case": str(case), "metrics": args.metrics, "samples": records,
               "median_us": statistics.median(durations), "min_us": min(durations), "max_us": max(durations)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"RESULT median_us={summary['median_us']:.3f} range_us={min(durations):.3f}..{max(durations):.3f}", flush=True)


if __name__ == "__main__":
    main()
