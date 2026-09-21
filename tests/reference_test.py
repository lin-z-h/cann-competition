"""CPU checks for the shape/layout and fixed reduction semantics used by kernel.asc."""

from itertools import product
from pathlib import Path
import re

import numpy as np


def check_source_contract():
    """Guard compile fixes, the layout contract, and the aligned reductions."""
    source = (Path(__file__).parents[1] / "kernel.asc").read_text(encoding="utf-8")
    assert "using namespace matmul_tiling" not in source
    assert "AscendC::CubeFormat" not in source
    assert "constexpr uint32_t kFp32BlockElements = 8;" in source
    assert "constexpr uint32_t kMaxMTiles = 512;" in source
    assert "constexpr uint32_t kTailScratchFloats = 64;" in source
    assert "constexpr uint32_t kReduceWorkspaceBytes = 32;" in source
    assert "constexpr int32_t kJudgeBfloat16 = 2;" in source
    assert source.count("dtype == kJudgeBfloat16") == 2
    assert "transposeX1 != transposeX2 && m > 512 && n > 512" not in source
    assert "BMMS_PROBE_MODE" not in source
    assert "probeRepeats" not in source
    assert "jobIdx % blockCount == blockIdx" not in source
    assert "for (uint32_t jobIdx = blockIdx; jobIdx < totalJobs;" in source
    assert "if (params_.mTiles <= params_.baseM)" not in source
    assert "tileMaxFloats * sizeof(float)" in source
    assert "reduceWorkspaceBuffer_, kReduceWorkspaceBytes" in source
    assert "if (m <= 16) {\n        return 16;" in source
    assert "if (m <= 32) {\n        return 32;" in source
    assert source.count("::CubeFormat::ND, T, Transpose>;") == 1
    assert "RowGemv ? ::CubeFormat::VECTOR : ::CubeFormat::ND" in source
    assert "tilingApi.SetCType(\n            matmul_tiling::TPosition::VECIN," in source
    assert re.search(r"(?<![:\w])TPosition::", source) is None
    assert re.search(r"(?<![:\w])CubeFormat::", source) is None

    # The normal path reduces full baseM x baseN tiles.  When both dimensions
    # have a tail, the aligned prefix of the last M tile remains a matrix
    # operation and only its final 1-7 rows use one-row matmuls.
    assert "uint32_t reduceShape[] = {params_.baseM, params_.baseN};" in source
    assert "uint32_t reduceShape[] = {1, params_.baseN};" in source
    assert "ReduceMax<float, Pattern::Reduce::AR, true>(" in source
    assert "tileMax, source, reduceShape, true);" in source
    assert "Max(rowMax, rowMax, tileMax, params_.baseM);" in source
    assert "ReduceSum<float>(tileMax, rowMax, reduceWorkspace, params_.baseM);" in source
    assert "Add(batchSum, batchSum, tileMax, kFp32BlockElements);" in source
    assert "if (currentN < params_.baseN || currentM < params_.baseM) {" in source
    assert "currentM / kFp32BlockElements * kFp32BlockElements" in source
    assert "mStart + alignedM" in source
    assert "currentM - alignedM" in source
    assert "params_.m % kFp32BlockElements != 0" in source
    assert "currentM % kFp32BlockElements != 0" in source
    assert "ProcessAlignedMTileWithNTails(" in source
    assert "params_.m, params_.n, params_.k, params_.k, params_.baseN" in source
    assert "GetTensorC<true>(mmOutput, false, false);" in source
    assert "params_.n % kFp32BlockElements != 0" in source
    assert "SetMatmulInputs(batchIdx, mStart + row, 1);" in source
    assert "Duplicate(padded, kPaddingMin, params_.baseN);" in source
    # The tail copy must be a whole number of 32-byte blocks; the sub-block
    # columns are repaired with scalar stores.
    assert "const uint32_t alignedN =" in source
    assert "mmReady[row * params_.baseN],\n                        0.0F, alignedN);" in source
    assert "padded.SetValue(row * params_.baseN + col, kPaddingMin);" in source
    assert "0.0F,\n                        currentN);" not in source
    assert "mmReady.GetValue(" not in source
    assert "padded[row * params_.baseN]" in source
    # The row fallback uses scalar stores only to replace invalid columns in
    # the final 32-byte block; valid Matmul output is never read scalarly.
    assert "rowMax.SetValue(row, 0.0F);" in source
    assert "HardEvent::V_S" in source
    assert "HardEvent::S_V" in source
    assert "GetTensorC<true>(scratch_" not in source
    assert "useGmTailPath" not in source
    # Superseded tail handling: partial-width reductions and per-lane fixes.
    assert "Adds(rowMax, tileMax, 0.0F, currentM);" not in source
    assert "ReduceSum<float>(tileMax, rowMax, reduceWorkspace, currentM);" not in source
    assert "uint32_t reduceShape[] = {currentM, params_.baseN};" not in source
    assert "mmReady.SetValue(row * params_.baseN + col, kPaddingMin);" not in source
    assert "mmReady[row * rowStride + currentN]" not in source

    assert "mmOutputQueue_.EnQue(mmOutput);" in source
    assert "mmOutputQueue_.DeQue<float>();" in source
    assert "if (params_.m == 1 && params_.n == 1)" in source
    assert "Duplicate(rowMax, kPaddingMin, params_.baseM);" in source
    assert "GM_ADDR workspace = workspaceDevice" not in source
    assert "GM_ADDR partial =" not in source
    assert "batch_matmul_max_sum_final_sum" not in source
    assert source.count("__global__") == 1
    assert "SyncAll<false>();" not in source
    assert "SyncAll<true>(sync_, syncLocal, blockCount);" in source
    assert "aclrtMemsetAsync(syncDevice, syncBytes, 0, syncBytes, stream)" in source
    assert "offset + mTileIdx" in source
    assert "SetAtomicAdd<float>();" not in source
    # Multi-tile Matmul calls use the documented asynchronous producer path;
    # single-tile and tail calls retain the synchronous API pairing.
    assert "matmul_.SetWorkspace(asyncWorkspace_);" in source
    assert "Iterate<false>();" in source
    assert "GetTensorC<false>" in source
    assert "Iterate<true>();" in source
    assert "GetTensorC<true>" in source
    assert "using MatmulObject = typename std::conditional<" in source
    assert "CFG_NORM>," in source
    assert "CFG_MDL>>::type;" in source
    assert "__schedmode__(1) __global__ __mix__(1, 1)" in source
    assert "ProcessBatch(batchIdx);" in source
    assert "blockCount >= params_.batch * 2" in source
    assert "ProcessMTile(batchIdx, mTileIdx, batchSum);" in source
    assert "const uint64_t x2TileOffset" in source
    assert "SetSingleShape(currentM, params_.n, params_.k);" in source
    assert "SetMatmulInputs(batchIdx, mStart, currentM);" in source
    # N-parallel only uses the already-verified asynchronous Matmul pairing
    # for aligned, contiguous groups containing at least two N tiles.  K-tail
    # segments and single-tile groups remain on the synchronous path.
    assert "const uint32_t groupTiles = endNTile - firstNTile;" in source
    assert "const bool asyncGroup = groupTiles > 1 && !RowGemv;" in source
    assert "for (uint32_t nTileOffset = 0;" in source
    assert "const uint32_t nTileIdx = firstNTile + nTileOffset;" in source
    assert source.count("matmul_.template GetTensorC<false>(") == 2
    assert "BatchMatmulMaxSumKernel<T, TransposeX1, TransposeX2, RowGemv> op;" in source
    assert "bool useRowGemv = m <= 8" in source
    assert "!transposeX1 && transposeX2 && k % 16 == 0;" in source
    assert "const uint32_t tilingM = useRowGemv ? 1U : m;" in source
    assert "matmul_tiling::CubeFormat::VECTOR" in source
    assert "if (!useRowGemv && tilingApi.SetFixSplit" in source
    assert "batch_matmul_max_sum_kernel<half, false, true, true>" in source
    assert "SyncAll<true>(sync_, syncLocal, blockCount);" in source
    assert "batch_matmul_max_sum_kernel<bfloat16_t, true, true>" in source
    assert "batch_matmul_max_sum_kernel<half, false, false>" in source
    assert source.index("REGIST_MATMUL_OBJ") < source.index(
        "op.Init(x1, x2, partial, asyncWorkspace, y, params, tiling);"
    )
    assert "baseM = 16;" in source
    assert "baseN = 16;" in source
    assert "baseM = static_cast<uint32_t>(tilingApi.GetBaseM());" in source
    assert "baseN = static_cast<uint32_t>(tilingApi.GetBaseN());" in source
    assert source.index("if (!BuildTiling(") < source.index(
        "const uint32_t mTiles = (m + baseM - 1) / baseM;"
    )
    assert "const uint32_t blockCount = std::min(coreCount, totalJobs);" in source
    assert "currentN % kFp32BlockElements == 0" in source
    assert "mmReady[row * params_.baseN + currentN]" in source
    assert "return 128;" in source
    assert "const size_t partialBytes = partialValueBytes + syncBytes;" in source


def golden(x1_storage, x2_storage, transpose_x1, transpose_x2):
    x1 = np.swapaxes(x1_storage, -1, -2) if transpose_x1 else x1_storage
    x2 = np.swapaxes(x2_storage, -1, -2) if transpose_x2 else x2_storage
    similarity = np.matmul(x1.astype(np.float64), x2.astype(np.float64))
    return np.sum(np.max(similarity, axis=-1), axis=-1).astype(np.float32)


def tiled_model(x1_storage, x2_storage, transpose_x1, transpose_x2, base_m, base_n):
    """Mirror of the kernel: per-M-tile MaxSim over N, summed in M order."""
    x1 = np.swapaxes(x1_storage, -1, -2) if transpose_x1 else x1_storage
    x2 = np.swapaxes(x2_storage, -1, -2) if transpose_x2 else x2_storage
    batch, m, _ = x1.shape
    n = x2.shape[-1]
    output = np.empty(batch, dtype=np.float32)
    for batch_idx in range(batch):
        partials = []
        for m_start in range(0, m, base_m):
            current_m = min(base_m, m - m_start)
            row_max = None
            for n_start in range(0, n, base_n):
                current_n = min(base_n, n - n_start)
                tile = np.matmul(
                    x1[batch_idx, m_start : m_start + current_m].astype(np.float32),
                    x2[batch_idx, :, n_start : n_start + current_n].astype(np.float32),
                )
                tile_max = np.max(tile, axis=-1)
                row_max = tile_max.copy() if row_max is None else np.maximum(row_max, tile_max)
            partials.append(np.sum(row_max, dtype=np.float32))
        output[batch_idx] = np.sum(np.asarray(partials, dtype=np.float32), dtype=np.float32)
    return output


def physical(logical, transpose):
    return np.swapaxes(logical, -1, -2).copy() if transpose else logical.copy()


def quantize_bfloat16(values):
    """Round float32 values to bfloat16, retaining them in a NumPy float32 array."""
    bits = values.astype(np.float32).view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return ((bits + rounding_bias) & np.uint32(0xFFFF0000)).view(np.float32)


def run_case(batch, m, n, k, dtype, seed, all_negative=False):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.05 if all_negative else -1.0, 1.0, (batch, m, k)).astype(np.float32)
    x2 = rng.uniform(-1.0, -0.05 if all_negative else 1.0, (batch, k, n)).astype(np.float32)
    if dtype == "bfloat16":
        x1 = quantize_bfloat16(x1)
        x2 = quantize_bfloat16(x2)
    else:
        x1 = x1.astype(dtype)
        x2 = x2.astype(dtype)
    for transpose_x1, transpose_x2 in product((False, True), repeat=2):
        x1_storage = physical(x1, transpose_x1)
        x2_storage = physical(x2, transpose_x2)
        expected = golden(x1_storage, x2_storage, transpose_x1, transpose_x2)
        actual = tiled_model(
            x1_storage,
            x2_storage,
            transpose_x1,
            transpose_x2,
            base_m=16 if m <= 16 else (32 if m <= 32 else (64 if m <= 64 else 128)),
            base_n=16 if n <= 16 else (32 if n <= 32 else (64 if n <= 64 else 128)),
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
        if all_negative:
            assert np.all(actual < 0), actual


def main():
    check_source_contract()
    # Full M tiles and 8-lane-aligned N tails are the only in-place C route.
    # Model its baseN-strided physical storage, including all-negative rows.
    for base_m, base_n, valid_n in ((16, 64, 40), (32, 128, 72), (64, 128, 120)):
        valid = -np.arange(1, base_m * valid_n + 1, dtype=np.float32).reshape(base_m, valid_n)
        physical_c = np.empty((base_m, base_n), dtype=np.float32)
        physical_c[:, :valid_n] = valid
        physical_c[:, valid_n:] = -np.finfo(np.float32).max
        np.testing.assert_array_equal(physical_c.max(axis=1), valid.max(axis=1))
    cases = [
        (1, 1, 1, 32, np.float16, 1, False),
        (2, 7, 13, 40, np.float16, 2, False),
        (3, 17, 33, 64, np.float16, 3, False),
        (2, 65, 129, 128, np.float16, 4, False),
        (2, 19, 23, 56, np.float16, 5, True),
        (2, 33, 65, 72, "bfloat16", 6, False),
        (1, 129, 257, 512, np.float16, 7, False),
        # M tails that are not a multiple of the 8-element fp32 block.
        (2, 5, 17, 48, np.float16, 8, False),
        (1, 3, 129, 64, np.float16, 9, False),
        (2, 129, 33, 40, "bfloat16", 10, True),
    ]
    for case in cases:
        run_case(*case)
    print(f"PASS: {len(cases)} shapes x 4 storage layouts")


if __name__ == "__main__":
    main()
