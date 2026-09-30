#!/usr/bin/env bash
set -Eeuo pipefail

stage="startup"
trap 'rc=$?; printf "FAILED at stage: %s (exit %s)\n" "$stage" "$rc" >&2; exit "$rc"' ERR

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$script_dir/CMakeLists.txt" ]]; then
    repo_root="$script_dir"
elif [[ -f "$script_dir/../CMakeLists.txt" ]]; then
    repo_root="$(cd "$script_dir/.." && pwd)"
else
    printf 'Cannot locate CMakeLists.txt beside the script or one directory above it.\n' >&2
    exit 2
fi
cd "$repo_root"

stage="environment setup"
if [[ -z "${CANN_ENV_SCRIPT:-}" ]]; then
    for candidate in /home/developer/Ascend/cann-9.0.0/set_env.sh \
                     /home/developer/Ascend/ascend-toolkit/set_env.sh; do
        if [[ -f "$candidate" ]]; then
            CANN_ENV_SCRIPT="$candidate"
            break
        fi
    done
fi
if [[ -n "${CANN_ENV_SCRIPT:-}" ]]; then
    if [[ ! -f "$CANN_ENV_SCRIPT" ]]; then
        printf 'CANN_ENV_SCRIPT does not exist: %s\n' "$CANN_ENV_SCRIPT" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "$CANN_ENV_SCRIPT"
fi

# Non-interactive SSH sessions do not inherit the CANNLab terminal's driver
# library path. Only add paths that actually exist on this machine.
for driver_lib in /usr/local/Ascend/driver/lib64/common \
                  /usr/local/Ascend/driver/lib64/driver; do
    if [[ -d "$driver_lib" ]]; then
        export LD_LIBRARY_PATH="$driver_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done

for required in cmake python3; do
    if ! command -v "$required" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$required" >&2
        exit 2
    fi
done

if command -v npu-smi >/dev/null 2>&1; then
    stage="NPU visibility check"
    npu-smi info
else
    printf 'Note: npu-smi is not on PATH; runtime will provide the device availability check.\n'
fi

for variant in new_design reliable; do
    build_dir="$repo_root/build/cannlab-$variant"
    stage="$variant CMake configure"
    cmake -S "$repo_root" -B "$build_dir" \
        -DNPU_ARCH="${NPU_ARCH:-dav-2201}" \
        -DCANN_KERNEL="$variant"

    stage="$variant device compilation"
    cmake --build "$build_dir" --parallel "${BUILD_JOBS:-4}"

    run_dir="$build_dir/run"
    manifest="$run_dir/cases.tsv"
    stage="$variant test case preparation"
    SMOKE_RUN_DIR="$run_dir" python3 - <<'PY'
import os
import pathlib
import struct

root = pathlib.Path(os.environ["SMOKE_RUN_DIR"])

def encode(value, dtype):
    if dtype == 1:
        raw = struct.pack("<e", value)
        return struct.unpack("<H", raw)[0], struct.unpack("<e", raw)[0]
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    word = (rounded >> 16) & 0xFFFF
    return word, struct.unpack("<f", struct.pack("<I", word << 16))[0]

cases = [("smoke_1x1", 1, 1, 1, 32, 1, 0, 0)]
for dtype, dtype_name in ((1, "f16"), (2, "bf16")):
    for tx1 in (0, 1):
        for tx2 in (0, 1):
            cases.append((f"small_{dtype_name}_t{tx1}{tx2}", 2, 7, 19, 32, dtype, tx1, tx2))
            cases.append((f"tails_{dtype_name}_t{tx1}{tx2}", 1, 35, 37, 29, dtype, tx1, tx2))
cases.extend([
    ("mode_c_single_m_tile_f16_t00", 1, 128, 257, 64, 1, 0, 0),
    ("mode_c_aligned_f16_t00", 1, 256, 256, 64, 1, 0, 0),
    ("mode_c_f16_t00", 1, 256, 257, 64, 1, 0, 0),
    ("mode_c_bf16_t11", 1, 256, 257, 64, 2, 1, 1),
    ("mode_b_f16_t00", 1, 4096, 256, 64, 1, 0, 0),
    ("mode_b_bf16_t10", 1, 4096, 256, 64, 2, 1, 0),
])

manifest = []
for name, batch, m, n, k, dtype, tx1, tx2 in cases:
    case_root = root / name
    input_root = case_root / "input"
    output_root = case_root / "output"
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    a_words = [0] * (batch * m * k)
    b_words = [0] * (batch * k * n)
    logical_a = []
    logical_base = []
    for b in range(batch):
        batch_a = []
        for i in range(m):
            row = []
            for kk in range(k):
                value = (((b * 7 + i * 3 + kk * 5) % 13) - 6) / 16.0
                word, value = encode(value, dtype)
                row.append(value)
                offset = (b * k + kk) * m + i if tx1 else (b * m + i) * k + kk
                a_words[offset] = word
            batch_a.append(row)
        logical_a.append(batch_a)

    for b in range(batch):
        bases = []
        for kk in range(k):
            _, base = encode(((kk % 9) + 1) / 8.0, dtype)
            bases.append(base)
            for j in range(n):
                scale = 0.5 + (j % 5) * 0.125
                value = base * scale
                word, decoded = encode(value, dtype)
                if decoded != value:
                    raise RuntimeError("factorized reference values must be exactly representable")
                offset = (b * n + j) * k + kk if tx2 else (b * k + kk) * n + j
                b_words[offset] = word
        logical_base.append(bases)

    case_dir = case_root / "input" / "case.txt"
    case_dir.write_text(
        f"{batch} {m} {n} {k} {dtype} {tx1} {tx2}\n", encoding="ascii"
    )
    (input_root / "x1.bin").write_bytes(struct.pack(f"<{len(a_words)}H", *a_words))
    (input_root / "x2.bin").write_bytes(struct.pack(f"<{len(b_words)}H", *b_words))

    expected = []
    scales = [0.5 + (j % 5) * 0.125 for j in range(n)]
    min_scale = min(scales)
    max_scale = max(scales)
    for b in range(batch):
        score = 0.0
        for i in range(m):
            dot = sum(logical_a[b][i][kk] * logical_base[b][kk] for kk in range(k))
            score += dot * (max_scale if dot >= 0.0 else min_scale)
        expected.append(score)
    (input_root / "expected.txt").write_text(
        "".join(format(value, ".17g") + "\n" for value in expected),
        encoding="ascii",
    )
    manifest.append("\t".join(map(str, (name, batch, m, n, k, dtype, tx1, tx2))))

root.mkdir(parents=True, exist_ok=True)
(root / "cases.tsv").write_text("\n".join(manifest) + "\n", encoding="ascii")
print(f"Prepared {len(cases)} cases under {root}")
PY
case_count="$(wc -l < "$manifest")"
    passed=0
    while IFS=$'\t' read -r case_name batch m n k dtype tx1 tx2; do
        case_dir="$run_dir/$case_name"
        rm -f "$case_dir/output/y.bin"
        stage="$variant device run: $case_name"
        (cd "$case_dir" && "$build_dir/batch_mat_mul_max_sum_custom")

        stage="$variant precision check: $case_name"
        KERNEL_VARIANT="$variant" CASE_NAME="$case_name" CASE_DIR="$case_dir" python3 - <<'PY'
import math
import os
import pathlib
import struct

root = pathlib.Path(os.environ["CASE_DIR"])
actual_path = root / "output" / "y.bin"
expected_path = root / "input" / "expected.txt"
expected = [float(line) for line in expected_path.read_text(encoding="ascii").splitlines()]
if not actual_path.is_file() or actual_path.stat().st_size != 4 * len(expected):
    raise SystemExit(f"Expected {len(expected)} FP32 outputs: {actual_path}")
actual = struct.unpack(f"<{len(expected)}f", actual_path.read_bytes())
for index, (got, want) in enumerate(zip(actual, expected)):
    error = abs(got - want)
    tolerance = 1e-4 + 1e-4 * abs(want)
    print(f"{os.environ['KERNEL_VARIANT']}/{os.environ['CASE_NAME']}[{index}]: actual={got:.9g}, expected={want:.9g}, abs_error={error:.3g}, tolerance={tolerance:.3g}")
    if not math.isfinite(got) or error > tolerance:
        raise SystemExit("FP32 output is outside the smoke-test tolerance")
PY
        passed=$((passed + 1))
    done < "$manifest"
    printf '%s: %d/%s device cases passed.\n' "$variant" "$passed" "$case_count"
done

stage="complete"
printf 'CANNLab device suite passed for reliable and new_design kernels.\n'
