"""FP16/BF16 小面板直接点积实验的数值检查；不运行 Ascend C。"""

import numpy as np
import pytest


def quantize(values, dtype):
    values = np.asarray(values, dtype=np.float32)
    if dtype == "fp16":
        return values.astype(np.float16).astype(np.float32)
    bits = values.view(np.uint32).copy()
    bits += 0x7FFF + ((bits >> 16) & 1)
    bits &= 0xFFFF0000
    return bits.view(np.float32)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "batch,m,n,k",
    [(1, 1, 16, 32), (1, 8, 32, 2048), (2, 7, 31, 48),
     (2, 32, 128, 128), (64, 2, 9, 128)],
)
def test_k_contiguous_direct_dot_matches_fp64_golden(batch, m, n, k, dtype):
    rng = np.random.default_rng(batch * 100000 + m * 10000 + n * 100 + k)
    # Physical X1=[B,M,K] and X2=[B,N,K].
    x1 = quantize(rng.normal(0, 0.2, (batch, m, k)), dtype)
    x2 = quantize(rng.normal(0, 0.2, (batch, n, k)), dtype)
    a32 = x1.astype(np.float32)
    b32 = x2.astype(np.float32)
    products = a32[:, :, None, :] * b32[:, None, :, :]
    partial = products.sum(axis=-1, dtype=np.float32)
    actual = partial.max(axis=2).sum(axis=1, dtype=np.float32)
    golden_dots = np.einsum(
        "bmk,bnk->bmn", x1.astype(np.float64), x2.astype(np.float64)
    )
    golden = golden_dots.max(axis=2).sum(axis=1).astype(np.float32)
    assert np.allclose(actual, golden, rtol=1e-4, atol=1e-4)


def test_all_negative_similarities_do_not_clamp_to_zero():
    x1 = np.full((1, 3, 32), 1.0, dtype=np.float16)
    x2 = np.full((1, 17, 32), -0.5, dtype=np.float16)
    dots = (
        x1.astype(np.float32)[:, :, None, :]
        * x2.astype(np.float32)[:, None, :, :]
    ).sum(axis=-1, dtype=np.float32)
    actual = dots.max(axis=2).sum(axis=1, dtype=np.float32)
    assert actual[0] == -48.0


@pytest.mark.parametrize("group_size", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("n", [9, 16, 31, 32])
def test_column_groups_cover_each_column_once(n, group_size):
    rng = np.random.default_rng(n)
    dots = rng.normal(-0.5, 0.25, (3, 7, n)).astype(np.float32)
    groups = (n + group_size - 1) // group_size
    partial = np.empty((3, 7, groups), dtype=np.float32)
    covered = []
    for group in range(groups):
        cols = list(range(group * group_size, min(n, (group + 1) * group_size)))
        covered.extend(cols)
        partial[:, :, group] = dots[:, :, cols].max(axis=2)
    assert covered == list(range(n))
    assert np.array_equal(
        partial.max(axis=2).sum(axis=1, dtype=np.float32),
        dots.max(axis=2).sum(axis=1, dtype=np.float32),
    )
