#!/usr/bin/env bash
set -Eeuo pipefail
stage="startup"
trap 'rc=$?; printf "FAILED at stage: %s (exit %s)\n" "$stage" "$rc" >&2; exit "$rc"' ERR
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
stage="environment"
env_script="${CANN_ENV_SCRIPT:-/home/developer/Ascend/cann-9.0.0/set_env.sh}"
[[ -f "$env_script" ]] || { printf 'Missing CANN environment: %s\n' "$env_script" >&2; exit 2; }
source "$env_script"
for lib_dir in /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64/driver; do
    if [[ -d "$lib_dir" ]]; then
        export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done
for command_name in cmake python3 msprof npu-smi; do
    command -v "$command_name" >/dev/null || { printf 'Missing command: %s\n' "$command_name" >&2; exit 2; }
done
python3 -c 'import numpy, pytest'
npu-smi info
run_dir="${CANNLAB_RUN_DIR:-$repo_root/build/cannlab-new_design/run}"
[[ -f "$run_dir/cases.tsv" ]] || {
    printf 'Missing prepared cases: %s; run cannlab_smoke.sh to prepare the device cases first.\n' "$run_dir" >&2
    exit 2
}
build_dir="${CANNLAB_BUILD_DIR:-$repo_root/build/cannlab-tuned}"
kernel_file="${CANNLAB_KERNEL_FILE:-experiments/cannlab_tuning/kernel.asc}"
stage="CPU model"
python3 -m pytest -q -p no:cacheprovider tests/new_design_test.py
stage="configure"
cmake -S . -B "$build_dir" -DNPU_ARCH="${NPU_ARCH:-dav-2201}" \
    -DCANN_KERNEL_FILE="$kernel_file" \
    -DCANNLAB_MAX_CALL_TILES="${CANNLAB_MAX_CALL_TILES:-4}" \
    -DCANNLAB_BASE_M="${CANNLAB_BASE_M:-0}" \
    -DCANNLAB_ALIGNED_FAST_PATH="${CANNLAB_ALIGNED_FAST_PATH:-ON}" \
    -DCANNLAB_TRACE=OFF
stage="device compilation"
cmake --build "$build_dir" --parallel "${BUILD_JOBS:-4}"
stage="device precision and repeatability"
python3 scripts/cannlab_verify.py --exe "$build_dir/batch_mat_mul_max_sum_custom" \
    --run-dir "$run_dir" --random --repeat "${CANNLAB_REPEAT:-1}"
profile_root="${CANNLAB_PROFILE_DIR:-$build_dir/profile-$(date +%Y%m%d_%H%M%S)}"
stage="msprof, precision and single-kernel checks"
profile_run_dir="${CANNLAB_PROFILE_RUN_DIR:-$run_dir}"
read -r -a profile_cases <<< "${CANNLAB_PROFILE_CASES:-mode_b_f16_t00 mode_b_bf16_t10}"
for case_name in "${profile_cases[@]}"; do
    [[ -f "$profile_run_dir/$case_name/input/case.txt" ]] || {
        printf 'Missing profiler case: %s\n' "$profile_run_dir/$case_name" >&2
        exit 2
    }
    python3 scripts/cannlab_profile.py --exe "$build_dir/batch_mat_mul_max_sum_custom" \
        --case-dir "$profile_run_dir/$case_name" --output "$profile_root/$case_name" \
        --samples "${CANNLAB_SAMPLES:-3}"
done
stage="complete"
printf 'Device suite and profiler checks passed. Profiles: %s\n' "$profile_root"
