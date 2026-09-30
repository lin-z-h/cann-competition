# CANNLab 转置 A 整理实验

本目录是独立设备调优实验，根目录可靠 `kernel.asc`、原新设计及 `experiments/cannlab_tuning/kernel.asc` 均保持分开。

同一逻辑输入 `B=1,M=4096,N=256,K=64` 的两类型、四布局受控采样显示：t00/t01 约 44–46 us，t10/t11 约 122–126 us。差距主要跟随 A 的物理布局。详见[设备记录](../../docs/history/cannlab_20260930_session.md)。

## 候选与诊断

- `kernel.asc`：在同一个全局 kernel 内，用 Vector 把本核当前 M tile 的 `[K,M]` 数据整理成连续 `[M,K]`，写入独占 GM workspace，再由非转置 Matmul 读取。Transpose 指令沿 K 重复。
- `repeat_m.asc`：相同数据流，Transpose 指令改为沿 M 重复，减少地址数组的设置次数。该改动须分别验证精度和性能。
- `batch_fix.asc`：精确4B写出模式B最终结果，避免不同核标量写相邻输出；只通过部分新增边界，尚未完成回归。
- `batch_diag.asc`：回读packed A逐位比较和partial诊断；编译完成，未执行。

启用条件为原 A 转置、M 至少 2048 且整除 128、K 为 16 的倍数且不超过 64、batch 少于可用核数且有足够 M tile 并行度；tiler 返回的 baseM 也必须整除 M 且为 16 的倍数。条件保证使用模式 B，完整 M tile 由一个核独占写入，无需增加 kernel 或跨核整理屏障。其它路径保留通用 N 尾块修复。

追加的输入/输出 UB buffer 合计 `2*baseM*K*2` 字节，最大 32 KiB；GM workspace 增加 `B*M*K*2` 字节。完成本核存储后等待 MTE3，再交给 Matmul。

截至2026-09-30停止调优：`kernel.asc` 的103/103设备用例及八布局24次profiler精度/单kernel检查通过，线上531572为15/15、重算约20.01分。`repeat_m.asc`八布局精度及四转置布局采样完成，没有明确收益；扩展回归在新增多batch边界失败。`batch_fix.asc`新增t00/t01通过、t10仍为NaN；K16的Transpose参数疑点未修复验证。CPU模型不能替代设备检查，全部候选仍与可靠版分开。详细状态见[HANDOFF](../../HANDOFF.md)。

## 复现

远端初始化已确认的 CANN 环境后显式选文件：

```bash
source /home/developer/Ascend/cann-9.0.0/set_env.sh
cmake -S . -B build/cannlab-pack-a -DNPU_ARCH=dav-2201 \
  -DCANN_KERNEL_FILE=experiments/cannlab_pack_a/kernel.asc \
  -DCANNLAB_ALIGNED_FAST_PATH=OFF -DCANNLAB_TRACE=OFF
cmake --build build/cannlab-pack-a -j4
python3 scripts/cannlab_verify.py \
  --exe build/cannlab-pack-a/batch_mat_mul_max_sum_custom \
  --run-dir build/cannlab-new_design/run --random
```

准备同一逻辑输入的八布局，单独采集一个布局：

```bash
python3 scripts/cannlab_layout_probe.py \
  --source-case build/cannlab-new_design/run/mode_b_f16_t00 \
  --output build/layout-probes
python3 scripts/cannlab_profile.py \
  --exe build/cannlab-pack-a/batch_mat_mul_max_sum_custom \
  --case-dir build/layout-probes/d1_t10 \
  --output build/profile-pack-a-fresh/d1_t10 --samples 3
```

验证器 `--random-shape B,M,N,K` 可重复提供，为默认 103 用例增加指定形状的两类型、四布局随机输入；重复形状自动去重。每个用例的随机种子由名字固定，不改变已有输入。
