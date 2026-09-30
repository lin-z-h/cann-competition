# CANNLab 设备诊断与调优候选

本目录从 `experiments/new_design/kernel.asc` 的线上通过版独立复制，根目录可靠版本与原新设计版本均保留。这里的修改面向 CANNLab CANN 9.0.0 实测，不代表已经通过线上 CANNJudge 评测。

最终候选在设备上完成 95 个统一套件用例，另 8 个分段尾块用例连续两轮通过，合计覆盖 103 个用例；最终六次 profiler 采集均通过精度和单次计算 kernel 检查。源码规范化 LF 后 SHA-256 为 `6260c031f28bbaba6119a8f16edf4985b59a507e83260ee1d6e79fe91341f686`。

停止时补充：固定输入103用例也已覆盖，剥离未启用TRACE的上传副本530365为线上15/15，重算约20.10分，未超过根可靠版最近记录约20.85分。新增多batch/K16/N尾边界复现NaN，因此本候选不是全面验证的替代基线；最终状态与下一步见[HANDOFF](../../HANDOFF.md)。

## N 尾块修复

设备诊断发现 `M=128,N=257,K=64` 的模式 C，前两个 N 分组行最大值正确，但最后一个 N 尾块只有首行正确、后续行出现零值。原消费者将顺序写出的 ND 尾块按完整 `baseN` 行宽读取。安装 SDK 的 `copy_cube_out_fixpipe.h` 中，顺序写出的 stride 使用实际 `baseWidth`，与消费者假定不一致。

修复把最后的 N 尾块从多 tile 调用中分离，单独同步调用并关闭 sequential C，让输出使用 `SetOrgShape` 指定的 `orgNc=baseN`。完整 N tile 保留分段和异步路径。单次运行仍只启动一个计算 kernel。

`CANNLAB_TRACE` 只为本目录启用 host tiling 和首 batch partial 输出，关闭时不增加诊断拷贝。`CANNLAB_MAX_CALL_TILES` 可取 1–4，默认 4，用于同调度下分段长度的受控性能对比。

`CANNLAB_BASE_M=0` 保留原 tile 选择，64 为本轮已测但变慢的实验。`CANNLAB_ALIGNED_FAST_PATH` 控制对齐模式 B 的 FP16 是否复用已验证的原消费者；源文件与 CMake 默认启用。条件为 M/N/K 对齐、整段 N 能在一次 MatMul 调用内完成。五组交替采样的 FP16 中位数快约 4.7%，BF16 收益未超出波动，因此 BF16 保留修复基线。

## 可重复验证

在远端项目根目录初始化 `/home/developer/Ascend/cann-9.0.0/set_env.sh` 后：

```bash
cmake -S . -B build/cannlab-perf-4 \
  -DCANN_KERNEL_FILE=experiments/cannlab_tuning/kernel.asc \
  -DCANNLAB_MAX_CALL_TILES=4 -DCANNLAB_TRACE=OFF
cmake --build build/cannlab-perf-4 -j4
python3 scripts/cannlab_verify.py \
  --exe build/cannlab-perf-4/batch_mat_mul_max_sum_custom \
  --run-dir build/cannlab-new_design/run --random
python3 scripts/cannlab_profile.py \
  --exe build/cannlab-perf-4/batch_mat_mul_max_sum_custom \
  --case-dir build/cannlab-new_design/run/mode_b_f16_t00 \
  --output build/profile-unique-run --samples 3
```

`run` 目录的原始 23 个用例由 `scripts/cannlab_smoke.sh` 生成。该脚本测试原始版本时可能因已知精度错误提前停止，但在运行前已生成全部用例。独立验证器读取量化后的实际输入，用 NumPy FP64 MatMul 计算参考值，检查 FP32 输出；`--random` 添加两种输入类型、四布局的随机边界用例。`--repeat` 可在一次准备数据后重复运行同一套输入。

Profiler 脚本对每次采集独立验证精度，要求 `op_summary` 只有一个计算 kernel，并保存三次原始指标及中位数、最小值、最大值。输出目录必须是新目录，避免混入旧采集数据。采集与其它设备测试串行执行。

统一入口 `bash scripts/cannlab_tune.sh` 集中检查环境、CPU 模型、配置、设备编译、设备精度与 msprof。可通过 `CANNLAB_ALIGNED_FAST_PATH=OFF` 选择修复基线，通过 `CANNLAB_REPEAT=2` 重复相同数据。该入口要求已准备 `build/cannlab-new_design/run/cases.tsv`，缺失时会在前置检查阶段停止并说明原因。

结果与限制见 [本轮记录](../../docs/history/cannlab_20260930_session.md)。
