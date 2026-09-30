"""独立实验版的 CPU 数学与调度检查；不验证 CANN 设备布局或同步。"""

from itertools import product
from pathlib import Path

import numpy as np
import pytest


def quantize(values, dtype):
    values = np.asarray(values, dtype=np.float32)
    if dtype == "fp16":
        return values.astype(np.float16).astype(np.float32)
    # Round float32 to bfloat16 (round to nearest, ties to even).
    bits = values.view(np.uint32).copy()
    bits += 0x7FFF + ((bits >> 16) & 1)
    bits &= 0xFFFF0000
    return bits.view(np.float32)


def encode(logical, transposed):
    return np.swapaxes(logical, -2, -1).copy() if transposed else logical.copy()


def decode(storage, transposed):
    return np.swapaxes(storage, -2, -1) if transposed else storage


def plan(batch, m, n, k, mc, nc, cores):
    tm, tn = (m + mc - 1) // mc, (n + nc - 1) // nc
    qm = batch * tm
    if (m <= 32 and n <= 128) or batch >= cores or cores == 1 or tm == tn == 1:
        mode = "A"
    elif tm > 1 and qm >= (cores + 1) // 2:
        mode = "B"
    elif tn > 1:
        mode = "C"
    elif tm > 1:
        mode = "B"
    else:
        mode = "A"
    partitions = min(tn, max(2, (cores + qm - 1) // qm)) if mode == "C" else 1
    jobs = batch if mode == "A" else batch * tm * partitions
    blocks = min(cores, jobs)
    call_tiles = min(4, (tn + partitions - 1) // partitions) if k % 16 == 0 else 1
    return mode, tm, tn, partitions, blocks, call_tiles


def simulated_result(a_storage, b_storage, ta, tb, cores, mc=8, nc=8):
    a = decode(a_storage, ta).astype(np.float64)
    b = decode(b_storage, tb).astype(np.float64)
    batch, m, k = a.shape
    n = b.shape[-1]
    mode, tm, tn, partitions, blocks, call_tiles = plan(
        batch, m, n, k, mc, nc, cores
    )
    # Logical tasks are assigned in block-index strides. Each slot has one writer.
    slots = {}
    expected = set(product(range(batch), range(tm), range(partitions)))
    jobs = [
        (bi, mi, pi)
        for bi in range(batch)
        for mi in range(tm)
        for pi in range(partitions)
    ]
    if mode == "A":
        jobs = [(bi, mi, 0) for bi in range(batch) for mi in range(tm)]
        owners = {bi: bi % blocks for bi in range(batch)}
    else:
        owners = {bi: bi % blocks for bi in range(batch)}
    segment_tiles = {}
    for index, (bi, mi, pi) in enumerate(jobs):
        block = bi % blocks if mode == "A" else index % blocks
        assert block < blocks
        start, end = pi * tn // partitions, (pi + 1) * tn // partitions
        seen = []
        row_max = np.full(min(mc, m - mi * mc), -np.inf)
        for segment in range(start, end, call_tiles):
            for tile in range(segment, min(segment + call_tiles, end)):
                seen.append(tile)
                n0, n1 = tile * nc, min((tile + 1) * nc, n)
                m0, m1 = mi * mc, min((mi + 1) * mc, m)
                dots = a[bi, m0:m1] @ b[bi, :, n0:n1]
                row_max = np.maximum(row_max, dots.max(axis=1))
        segment_tiles[bi, mi, pi] = seen
        slots[bi, mi, pi] = row_max
    assert set(slots) == expected
    for bi, mi in product(range(batch), range(tm)):
        assert sorted(
            tile for pi in range(partitions) for tile in segment_tiles[bi, mi, pi]
        ) == list(range(tn))
    result = []
    for bi in range(batch):
        assert owners[bi] < blocks
        total = 0.0
        for mi in range(tm):
            merged = np.maximum.reduce([slots[bi, mi, pi] for pi in range(partitions)])
            total += float(merged.sum())
        result.append(total)
    return np.array(result), mode, blocks


@pytest.mark.parametrize("dtype,ta,tb", product(("fp16", "bf16"), (False, True), (False, True)))
@pytest.mark.parametrize(
    "shape,cores,expected_mode",
    [
        ((7, 9, 17, 40), 4, "A"),  # more batches than blocks, K tail
        ((2, 17, 9, 32), 8, "A"),  # small shape uses the validated batch owner
        ((2, 23, 57, 128), 8, "A"),  # both M and N have tails
        ((2, 33, 9, 32), 8, "B"),  # M stripes outside the small-shape gate
        ((1, 7, 33, 40), 16, "A"),  # small-shape N tail stays batch-owned
        ((1, 7, 129, 32), 4, "C"),  # several bounded segments per partition
        ((1, 1, 1, 32), 16, "A"),
    ],
)
def test_modes_and_layouts(dtype, ta, tb, shape, cores, expected_mode):
    batch, m, n, k = shape
    rng = np.random.default_rng(71)
    a = quantize(rng.uniform(-1, 1, size=(batch, m, k)), dtype)
    b = quantize(rng.uniform(-1, 1, size=(batch, k, n)), dtype)
    a_storage, b_storage = encode(a, ta), encode(b, tb)
    actual, mode, _ = simulated_result(a_storage, b_storage, ta, tb, cores)
    expected = np.array([
        sum(max(float(np.dot(a[bi, mi].astype(np.float64),
                             b[bi, :, ni].astype(np.float64)))
                for ni in range(n))
            for mi in range(m))
        for bi in range(batch)
    ])
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert mode == expected_mode
    np.testing.assert_array_equal(a_storage, encode(a, ta))
    np.testing.assert_array_equal(b_storage, encode(b, tb))


def test_negative_values_and_n_partition_counterexample():
    a = np.zeros((1, 2, 32), dtype=np.float32)
    b = np.zeros((1, 32, 129), dtype=np.float32)
    a[0, :, 0] = 1
    b[0, 0] = -11
    b[0, 0, 0], b[0, 0, 128] = -1, -1
    result, mode, _ = simulated_result(a, b, False, False, 16)
    assert mode == "C" and result[0] == -2

    # Different partitions win for different rows. A scalar per partition
    # would incorrectly return 10 instead of 20.
    b[0, 0, :] = 0
    a[0, 1, 0] = 0
    a[0, 1, 1] = 1
    b[0, 0, 0], b[0, 1, 128] = 10, 10
    result, _, _ = simulated_result(a, b, False, False, 16)
    assert result[0] == 20


def test_k_partials_must_merge_before_max():
    first, second = np.array([10, 0]), np.array([-10, 9])
    assert max(first + second) == 9
    assert max(first) + max(second) == 19


def test_single_kernel_candidate_is_separate_from_submission():
    root = Path(__file__).parents[1]
    candidate = (root / "experiments/new_design/kernel.asc").read_text()
    baseline = (root / "kernel.asc").read_text()
    assert candidate != baseline
    assert candidate.count("__global__") == 1
    assert "SetAtomicAdd" not in candidate
    assert "callTiles" in candidate and "nGroups" in candidate
    assert "for (uint32_t batchIdx = blockIdx; batchIdx < params_.batch;" in candidate
