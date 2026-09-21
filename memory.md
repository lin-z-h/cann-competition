# BatchMatmulMaxSum 优化记忆与实验方法

更新时间：2026-09-21

目标：15 个测试点全部正确，单次迭代严格只启动 1 个 kernel，最终平均分至少 30。

## 1. 当前可靠基线

### 已验证提交

- `372764`：15/15 Pass；旧记录的 `19.8436` 使用了错误的线性估分，不能当官方分数。
- `372921`：15/15 Pass，只在主 `ProcessMTile` 路径启用异步 Matmul；按题面公式和现存 15 个耗时重算平均约 `20.288`，旧记录 `20.3304` 不再使用。
- `372963`：在 `372921` 上增加“大尺寸、BF16、跨布局”调度切换；全 0、无设备用时，已回退，不能当作基线。
- `373483`：与 `372921` 同一算法、仍带关闭状态的探针框架，15/15 Pass；耗时与 `372921` 基本一致，可作为远端可恢复基线。
- `378513`：删除探针框架后再次加入“大尺寸、BF16、跨布局时令 `nGroups=1`”；15/15 Pass，但 case 11 从 `224.58 us` 退化到 `330.97 us`，证明该调度切换无效，已回退。

`372921` 的 15 个耗时（μs）：

```text
11.88, 22.93, 29.51, 15.44, 12.60,
98.70, 16.58, 103.32, 135.84, 146.63,
230.49, 226.42, 32.75, 33.89, 36.52
```

其中主路径异步化的已测收益：

- case 9：`141.73 -> 135.84`
- case 10：`217.52 -> 146.63`
- case 11：`253.65 -> 230.49`

因此异步流水是已被评测证实的方向，不是假设。

## 2. 能稳定跑通，关键是规避了什么

### 2.1 严格遵守题意，而不是把它当普通 GEMM

- `transposeX1/transposeX2` 只描述 storage shape，不能在设备侧再执行一次逻辑转置。
- batch 必须一一配对，绝不能 broadcast。
- 归约顺序必须固定为 `Max(N)` 后 `Sum(M)`，不能利用代数形式改写。
- FP16/BF16 输入都使用 FP32 累加与归约，最终输出 FP32。

这些约束直接决定地址计算、Matmul 模板参数和归约结构；任何一处“看起来等价”的改写都可能 Wrong Answer。

### 2.2 只保留一个全局 kernel 入口

评测要求每次迭代恰好 1 次 kernel launch。早期多 kernel 拆分即使逻辑正确，也会触发：

```text
Profiling rule violated: each iteration must launch exactly 1 kernel.
```

因此当前实现把 BatchMatmul、N 维 Max、M 维 Sum、跨核 partial merge 全部放在同一个 MIX kernel 中。Host 侧的内存申请、清零和 stream 同步不能变成额外计算 kernel。

### 2.3 尾块不能按“自然宽度”想当然

Matmul 输出和 Vector API 受 32 字节对齐、`baseM/baseN` stride 和 L0C/UB 搬运规则约束。稳定方案是：

- 正常 tile 按 `baseM × baseN` 归约。
- N 尾块先复制到 padded UB，非法列填 `-FLT_MAX`，再做整 tile ReduceMax。
- M/N 同时有尾数时，先处理 8 行对齐前缀，最后 1–7 行走逐行 Matmul。
- 不能用未对齐的 `currentN` 直接做块搬运，也不能标量读取尚未正确落入 UB 的 Matmul 结果。

这避开了此前尾块 stride、越界 lane 和全负输入下错误初始化的问题。

### 2.4 K 尾数不能直接复用连续 N-group Matmul

连续 N 分组能显著降低重复的 Matmul 设置/消息开销，但实测 K 非 16 对齐的 case 8 会错误。最终采用同一份归约代码、两种调度：

- `K % 16 == 0`：每个 N group 使用连续 N 区间，一次 Matmul 产生多个 N tile。
- `K % 16 != 0`：group 内按单 N tile 调用 Matmul，并使用非 sequential `GetTensorC`。

这样 case 8 恢复到约 `102 μs`，同时 case 12 从约 `285 μs` 降到约 `223 μs`。

### 2.5 跨核合并必须显式同步且空间不重叠

- N 并行时，每个 `(batch, mTile, nGroup)` 写独立 FP32 行最大值。
- 所有工作块完成后执行一次 `SyncAll<true>`。
- owner block 再沿 nGroup 做 Max、沿 M 做 Sum。
- M 并行时同理，每个 mTile 写独立 partial sum，barrier 后由 owner 汇总。
- partial、sync、Matmul system workspace 和异步 C workspace 必须分区且满足对齐。

错误的 barrier、复用同一 GM 区域或提前释放 workspace，都会表现为随机错、全 0 或死锁。

### 2.6 编译成功不等于资源可承载

异步 Matmul 的方向正确，但把异步分支复制到多个大函数后，提交 `372846` 全 15 例都是 0 用时/100% 错误；缩到仅主路径一处异步后，`372921` 立即 15/15 Pass 并提速。随后只增加少量 Host 调度逻辑的 `372963` 也出现全 0，说明当前源码/编译产物已接近敏感资源边界；继续加功能前必须先删掉探针、死代码和重复模板分支。

结论：dav-2201 上不仅要关心源码长度，还要关心模板展开、设备指令体积、寄存器、UB/L1/L0 和 MIX kernel 资源压力。以后禁止为了“覆盖所有路径”复制整套异步归约代码。

后续 `373483` 证明同类源码能够正常编译运行，因此“全 0”也可能包含评测环境的瞬态失败，不能只凭一次全 0 就归因。判断规则改为：先看是否有编译/设备耗时及全部 case 的一致模式；若代码变化很小且没有明确编译日志，应回到同一哈希或等价代码复验一次，但不得连续盲投。

## 3. 已证伪或风险很高的方向

- 多 global kernel / 两阶段 kernel：违反 1-launch 规则。
- 把 `baseM` 或 tiling 强行改成 64，而设备侧仍依赖原 stride：全 0。
- 全面切换 `CFG_NORM`：全 0。
- 未满足完整 tiling 条件时盲开 MDL preload：全 0。
- 在已有大 UB 占用上再堆多个 direct-vector buffer：容易触发资源失败。
- 同时复制多套同步/异步 Matmul 大函数：设备代码/资源膨胀，全 0。
- 用硬 barrier 代替当前 soft/global sync：曾导致超时。
- 仅改 Host malloc、memset 等：评测统计 device kernel，通常不会改善 case 时间。
- 无证据地扫 tile 参数：提交次数昂贵，且容易破坏 stride/尾块契约。
- 对“大尺寸、BF16、跨布局”统一强制 `nGroups=1`：`378513` 虽然全通过，但 case 11 由约 `225 us` 退化到约 `331 us`。降低 N 并行度造成的 Cube 并行损失大于异步流水收益。

失败实验不是“没用”，但同一机制没有新证据时不得重复提交。

## 4. 后续优化方法：先有机制和证据，再写代码

采用固定闭环：

```text
Measure -> Classify -> Hypothesize -> Change one mechanism
        -> Verify 15/15 -> Compare per-case -> Keep/Revert -> Record
```

每次实验必须先写清楚：

1. 目标 case 与当前耗时。
2. 判断瓶颈属于 Cube、Vector、MTE2/GM、同步还是调度不足。
3. 依据哪条硬件机制会改善。
4. 预期哪些 case 变快、哪些保持不变、允许的最大回退。
5. 失败时如何恢复到最近的 15/15 Pass。

没有上述五项，不提交。

### 4.1 CANN / AscendC 方法

#### A. Cube 与 Vector 流水重叠

官方说明 MIX 场景中同步 `Iterate<true>` 会反复发生 AIV/AIC 消息与等待；异步 `Iterate<false>` 可减少核间交互。使用时必须：

- `Iterate<false>` 与 `GetTensorC<false>` 成对。
- 为每个 block 提供不重叠的 GM workspace。
- workspace 至少覆盖该调用的 `singleCoreM × singleCoreN × sizeof(float)`，并考虑 base tile padding。
- 优先用于一次 Matmul 能连续产生多个 N tile 的路径；单 tile 不值得付出 GM workspace 往返。
- 只保留一份异步消费循环，通过调度把适合的 shape 引到该路径，避免模板代码膨胀。

当前 `372921` 已验证该方法有效。

#### B. Double Buffer 只在资源预算允许时使用

官方流水优化建议队列深度 2，使搬入、计算、搬出重叠。但本题当前 UB 已同时容纳：

- `baseM × baseN` FP32 Matmul 输出；
- 同尺寸 tail padded buffer；
- 行最大值、归约 workspace、partial queues、sync buffer。

因此不能直接把所有队列改成 2。正确做法是先缩减或复用 buffer，再只对瓶颈队列做双缓冲，并检查是否导致资源/occupancy 下降。

#### C. 减少循环头开销，扩大“单次有效工作”

CANN 的 Matmul/FlashAttention 优化案例都强调：在片上空间允许时，让一次 Matmul 处理多个 base tile，减少 SetTensor、Iterate 消息和循环头开销。当前连续 N-group 的 case 12 提速已经证明这一点。

下一步优先通过调度复用已有多-tile 异步主路径，而不是再复制新函数。

#### D. MTE2 与 K 轴优化必须由 case 证据触发

官方 Matmul 调优案例建议，当 K 很大、K 轴数据无法全载 L1 时，再考虑 MDL、K 轴错峰访问或调整 `baseK/stepK/depthA1/depthB1`。本题不能再次“无条件开 preload”。应先用 shape 分类或 profiling 证明目标 case 是大 K/MTE2 瓶颈，再单独实验。

#### E. Tiling 联合调优

`baseM/baseN/baseK`、pipeline depth、L1/L0 buffer 数和 core 分配必须联合考虑。更大 tile 减少循环，但会增加 UB/L1/L0 压力；更多 N group 提高并行度，却增加 partial 写回、barrier 和 merge。调度阈值应来自 per-case A/B 实测，而不是固定偏好。

### 4.2 从 CUDA 方法迁移的通用原则

CUDA 与 Ascend 架构不同，API 不能照搬，但以下机制相同：

- **合并且对齐访存**：让连续线程/向量访问连续地址；对应本题要保持 ND stride、32 字节对齐和整块搬运。
- **数据复用优先**：CUDA 用 shared memory/register 减少重复 global load；Ascend 对应 L1/L0/UB 复用。应优先复用 x1 tile、避免同一 A tile 因 N 切分被重复从 GM 搬入。
- **计算与搬运重叠**：CUDA 的多 stage pipeline 对应 Ascend 的异步 Matmul、队列和 double buffer。
- **occupancy/资源平衡**：更深流水、更大 tile 和更多临时 buffer 不一定更快，可能因寄存器或片上内存降低并发，甚至 launch/编译失败。
- **减少全局同步**：仅在跨 block 数据依赖确实需要时 barrier；能在 block 内完成的归约不要写 GM 再读回。
- **融合 epilogue**：CUDA GEMM 常把后处理融合进 epilogue；本题对应目标是 C tile 一到 UB 就立即 Max，不落完整 `B×M×N` 中间矩阵。

## 5. 接下来按优先级执行的代码路线

### P0：保护基线与结果账本

- 任何新版本先跑 `python tests/reference_test.py`。
- 上传前核对编辑器内容 SHA-256。
- 必须等待上一提交结束再投下一版。
- 新版本若不是 15/15 Pass，立即回到最近 Pass 代码，不在错误版本上叠改。
- 每次记录 submission ID、15 个耗时、估算分、唯一改动和结论。

### P1：通过调度复用唯一异步主路径

不增加第二份异步设备代码。对目前走 N-parallel、但每组 tile 太少而无法形成流水的 shape，尝试切换到 M-parallel/主路径，让已有异步循环工作。

准入条件：

- shape 类别必须可由 dtype/layout/M/N/K 的一般性质描述，不能硬编码测试点编号或精确隐藏 shape。
- 一次只切换一个类别。
- 目标 case 至少应有 10% 理论收益，否则不值得消耗提交。

### P2：降低 N-parallel 的 GM merge 成本

对必须 N-parallel 的 case：

- 检查 group 数是否过多，是否出现“每组仅一个 tile却仍付出 barrier+GM merge”。
- 让每组至少承担两个 N tile，再与减少 core 并行度的损失做 A/B。
- 尝试 owner block 合并时用更连续的 partial layout，减少小块 DataCopy。

这比复制异步 N-group 函数风险低。

### 当前单变量实验：M-parallel 直接 job 分配

历史 Pass 结果显示，引入 M 并行时 case 12 从约 `227 us` 降到约 `33 us`，因此 `ProcessParallel` 是已证实热点。旧实现中每个核都会扫描全部 `batch * mTiles`，并对每个 job 执行 `% blockCount` 判断；新实现改为从 `blockIdx` 开始、按 `blockCount` 递增的 grid-stride 分配。该改动只减少设备侧无效标量控制流，不改变 Matmul 输入、partial 布局、同步点或归约顺序。

验收标准：15/15 Pass；主要观察 case 12，若收益落在正常评测噪声内则不继续围绕该微优化叠改。

结果：提交 `378782` 15/15 Pass，case 12 为 `33.26 us`，与既有 `31.9–33.9 us` 波动区间一致。结论是无效 job 扫描不占主导；保留 grid-stride 写法作为等价简化，但停止在此方向继续投入。

### 下一单变量实验：向量化 M-partial 汇总

M-parallel owner 原先逐项 `GetValue` 从 GM 读取 partial 并做标量累加。新路径在 `mTiles <= baseM`（现有 `partialRowInQueue_` 容量足够）时，一次连续搬入全部 partial，再用 FP32 `ReduceSum` 汇总；容量不足时保留原标量 fallback。该改动不新增 UB buffer，不改变 Max(N) 后 Sum(M) 的计算阶段，只替换 Sum(M) 内部的确定性归约实现。

结果：提交 `378864` 全 Wrong Answer，已完整回退到 `378782` 的标量汇总。由于所有 case 一致失败而该分支只应影响 M-parallel case，优先判断为设备代码生成、队列位置/API 契约或资源问题，而不是普通数值误差。禁止复用 `partialRowInQueue_ + tileMaxQueue_` 的这套写法再次提交。

### `kernel_try.asc` 可迁移结论

参考版本最重要的价值是做 UB 活跃区间分析，而不是照搬 partial API：

- `tileMaxQueue_` 的真实峰值为 `max(baseM, 64)` 个 FP32；64 来自最多 7 个尾行各占 8 个 float，再加一个 8-float scratch block。原先的 `baseM * 8` 明显过量。
- 当前调用的是基础 `ReduceSum(dst, src, work, count)`，最大 count 为 `kMaxMTiles=512`。FP32 每个 repeat 处理 64 个元素，首轮最多产生 8 个 FP32 中间量，32 字节 workspace 足够。
- 没有显式流水同步时，不直接采用 `GM -> VECCALC TBuf -> ReduceSum`；这与官方 TQue/事件同步模式不一致，可能读取尚未完成的 MTE2 数据。
- 条件化 tail/sync/async buffer 分配有价值，但必须分别做 liveness 证明和单变量评测，不能与 partial 归约一起提交。

当前实验只缩小 `tileMaxQueue_` 和基础 ReduceSum workspace，不改变任何 DataCopy、Matmul、同步点或数值顺序。目标是降低 UB 占用并观察是否改善 occupancy/稳定性。

结果：提交 `379068` 15/15 Pass，case 12 为 `32.65 us`，总体处于原基线波动范围。说明缩容正确，但仅降低 UB 占用没有直接性能收益。

下一实验使用参考版的“连续 partial 搬运 + Vector ReduceSum”机制，但修正其同步缺口：GM 搬入专用 `VECCALC TBuf` 后显式执行 `MTE2_V` SetFlag/WaitFlag，再开始 ReduceSum；输出继续通过 `VECOUT TQue` 同步到 GM。该 buffer 只在与 `Process()` 完全相同的 M-parallel 条件下初始化。相较失败的 `378864`，不复用 N-parallel 队列，也不把结果放入尺寸语义不同的 `tileMaxQueue_`。

结果：提交 `379144` 15/15 Pass，case 12 为 `33.01 us`，仍无可测收益，已回退该 partial 汇总；结论是 case 12 的耗时不由 owner 的标量 GM 汇总主导。

### 探针恢复出的 shape 桶

`372097`（N 桶）与 `372146`（M 桶）逐行 diff 证明，除了 probe mode 和对应表达式外代码完全相同。结合无探针基线 `372041`，可得到：case 0 的 M/N 均为 33–128；case 1 的 M/N 均为 9–32；case 2/3 为 M<=8、N 为 9–32；case 5/6 为 M 9–32、N 33–128；case 12 为 M>512；case 13 为 M 129–512、N>512。并行路径 case 7–11 的 probe 被提前 return 绕过，不能据此分类。

因此不再投入 `M=N=1` 专用路径。下一单变量实验把 `17<=M<=32` 的 `baseM` 从 32 改为 16，使其产生两个 M tile，在小 Batch/大 K 场景增加 M 维并行；`M<=16` 及其他 M 桶不变。

结果：提交 `379213` 全 0、无设备耗时。计划中的同哈希复验因提交 API 连续网络失败而没有创建新 ID、没有完成判定。本地已恢复到最近 Pass `379068`；除非提交链路稳定且仍有必要，否则不重复该实验。

用户要求复验后，提交 `379331` 使用与 `379213` 完全相同的 SHA-256，结果仍为 15 case 全 0、零设备耗时。由此确认该 tiling 改动是确定性失败，正式证伪；禁止再次提交。原因应优先从 `SetFixSplit(16, baseN)` 与当前 Matmul/stride/资源契约不兼容处排查，而不是解释成性能波动。

### P3：为大 K case 做 MTE2/L1 定向优化

仅在确认大 K 类别后尝试：

- 调整 MDL tiling 的 `baseK/stepKa/stepKb/depthA1/depthB1`；
- 检查 K 轴错峰是否适用于 dav-2201 和当前 Matmul API；
- 以 MTE2 等待下降和目标 case 实测为验收，不以“能编译”为验收。

### P4：小 shape 专用路径

小 case 的固定开销占比高，但 direct-vector 路径风险也最高。只有在先确定 storage layout、M/N/K 范围和 UB 预算后，才实现一个窄条件专用路径；不得新增多 kernel，不得为未知 shape 硬编码答案或兜底。

## 6. 参考资料

- 华为官方：[Ascend C 算子性能优化实用技巧——流水优化](https://www.hiascend.com/developer/techArticles/20240819-1)
- 华为官方：[基于 Ascend C 的 Matmul 算子性能优化最佳实践](https://www.hiascend.com/developer/techArticles/20240816-1)
- 华为官方：[基于 Ascend C 的 FlashAttention 算子性能优化最佳实践](https://www.hiascend.com/developer/techArticles/20240607-1)
- 华为官方：[Matmul 高阶 API 与 Tiling 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/opdevg/Ascendcopdevg/atlas_ascendc_10_0038.html)
- Ascend 官方示例：[Ascend/samples](https://gitee.com/ascend/samples)
- NVIDIA 官方：[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- NVIDIA 官方：[cuBLASDx Achieving High Performance](https://docs.nvidia.com/cuda/cublasdx/performance.html)

## 7. 不可违反的提交纪律

- 正确性第一；性能优化不能改变 batch 配对、storage layout 或归约顺序。
- 每次迭代仍必须恰好 1 个 kernel launch。
- 不凭感觉同时改多个维度。
- 不把 0 用时全错解释成普通数值误差；先查编译、启动和资源。
- 不用“提交一次看看”代替本地契约测试和代码审查。
- 每日提交次数有限，只有能明确回答一个技术问题的实验才允许提交。

## 8. 2026-09-21：20 次架构计划开始执行

### 8.1 零提交审计

- 开始执行 `plan.md` 时，本地 `kernel.asc` SHA-256 仍为可靠基线 `0d26f7b3e0582888db430d0a2a07b1f691fa0d6b60844932ac8dd250ed6f1922`。
- `python tests/reference_test.py` 继续通过 `10 shapes × 4 storage layouts`。
- 远端审计确认 `379068` 是同一 hash 的 15/15 Pass，不需要为“确认基线”再消耗一次提交。
- 历史 probe 的倍率不能直接按 `2x/3x/4x/5x` 判定 shape 桶，因为 kernel 启动、Matmul 注册和同步等固定开销不会随 `probeRepeats` 成比例增长。以前仅凭总耗时倍率得出的部分 M/N 桶结论置信度过高；以后必须使用同源代码的 control，并用 `t(r)=fixed+r×work` 模型或正交 probe 联合判断。
- 对归一化源码相同的旧 probe 做对照后，可较可靠地判断：case 2、5、8、13 的 storage layout 属于 `transposeX1=false, transposeX2=true`；这意味着两侧逻辑 K 向量在 storage 中连续。case 0 可确定 `transposeX2=false`，但仅凭已有两个 probe 无法区分 `transposeX1=false/true`。
- 旧 K 对齐 probe 与 control 的强信号集中在 case 7；但该代代码本身不是 15/15 Pass，因此只把它当候选证据，不能覆盖当前正确版本已经验证的 K-tail 路由事实。

### 8.2 尝试 01A：官方 GEMV 专用路径（已证伪并回退）

目的：历史 probe 显示 case 0 是 M=1 家族，官方 CANN `matmul_gemv` 样例规定 M=1 时可把 A format 设为 `CubeFormat::VECTOR`。实验只对一般条件 `M=1 && FP16 && transposeX2=false` 启用 GEMV tiling/Matmul specialization，其余 shape 完整保留 ND 基线路径；仍然只有一个 global kernel 和一次 launch。

提交信息：

- submission：`384363`
- object ID：`6ab087cab0477ec41ee7a27e`
- 候选 SHA-256：`c6cfe9ce1ddb2b092935af8c72159ce336debb13242de0cc3f93c6f74680d803`
- 结果：15/15 Pass
- 耗时：`11.68, 22.25, 29.10, 15.83, 12.62, 99.78, 19.43, 101.51, 139.24, 149.79, 230.70, 225.10, 32.11, 33.50, 37.60 us`

结论：

- 唯一明确命中的 case 0 仅从 `11.93` 变为 `11.68 us`，提升约 `2.1%`，低于计划规定的 10% 目标门槛，属于评测波动范围。
- 按官方对数公式与该次 `best_time` 重算，平均分相对 `379068` 约 `-0.093`；case 6 的 `17.17 -> 19.43 us` 并非 GEMV 路由命中，更像单次噪声，但即使忽略它也没有足够 GEMV 收益。
- 这是完整 Pass 且目标 case 有非零正常耗时，不属于失败抖动，不复验、不继续扫描 GEMV split/config。
- 已完整回退；当前 `kernel.asc` 再次精确恢复为 `379068` 的 SHA-256 `0d26f7b3...f1922`，参考测试通过。

### 8.3 如何区分抖动与确定性失败

- **15/15 Pass 但性能变化小**：先换算实际积分贡献，并区分命中与未命中 case。8–10% 是强证据线，不是一票否决线；窄路由在目标 case 有可解释的 5% 级收益、预计积分为正且不改变非目标路径时可以暂留，由后续不同实验自然复验，避免专门重复提交。
- **Compile Error**：同源码不重投；先读取编译日志并修正根因。
- **部分 case Wrong Answer 且有正常设备耗时**：优先视为确定性算法/layout/tail 错误；保留错误分布证据并回退，不靠重投解决。
- **全部 case 为 0、无设备耗时**：可能是编译产物、launch/resource 问题，也可能是评测环境抖动。先检查改动规模、模板实例数、远端日志和同期平台状态；只有代码变化很小、无可解释资源风险且存在外部故障证据时，才允许同 hash 最多复验一次。
- **Running 时间较长但尚无 result**：只是排队/评测中，不能当失败，更不能创建第二个 submission。
- **同 hash 已出现两次相同全 0**：视为确定性失败，例如 `379213/379331`，禁止第三次重投。

### 8.4 尝试 01B：小 M 行级 GEMV（保留为当前候选）

目的：在 01A 已证明官方 `VECTOR` A-format GEMV 能正确运行的基础上，把满足一般条件的 FP16 小 M 输入拆为逐 query-row GEMV，并由现有 M-parallel 路径把不同行分配给多个核。首版仅路由 `M<=8 && !transposeX1 && transposeX2 && K%16==0`，其余输入仍走可靠基线；仍只有一个 global kernel、每次迭代只 launch 一次。

提交信息：

- submission：`384408`
- object ID：`6ab08949b0477ec41ee85239`
- 候选 SHA-256：`71119a384993ce48057ef22edc809f5b2c9c5a6f4ce11dc0376f020fc2d915d7`
- 结果：15/15 Pass
- 耗时：`11.80, 21.83, 27.75, 15.85, 12.71, 99.36, 17.13, 101.98, 136.71, 147.69, 231.38, 225.93, 29.69, 32.91, 36.40 us`

结论：

- 明确命中的 case 2 从 `29.39` 降到 `27.75 us`，提升 `5.58%`；按官方题面的对数公式和 `best_time=2.16 us`，该 case 约从 `13.444` 增至 `13.705`，单 case 增加 `0.261`，折合 15 case 平均仅约 `+0.017`。
- 本次 15 case 重算平均分约从 `20.124` 增至 `20.295`，即 `+0.171`；其中 case 12/13 等未命中路由的明显变快只能视为有利抖动，不能归功于行级 GEMV。保留依据是目标 case 的可解释正收益、下一次提交自然复验，以及新路径没有改变其他 shape 的运行语义；不能再把单次总分增量归因于该机制。
- 原计划“基座实验必须快 10%”过于机械。以后以实际积分贡献和风险共同判断：窄路由若 15/15、目标 case 有可解释的正收益、非目标路径代码语义不变且没有稳定回退，可以保留 5% 级收益；10% 继续作为强证据标准，而不是一票否决线。
- 当前保留该候选，不为确认 5.58% 立即重复提交。同一代码的稳定性由下一项建立在该候选上的实验自然复验；若后续样本中 case 2 回到基线波动范围，再回退该分支。
- `Running` 持续约 7 分钟后正常完成，说明这次长等待是队列/评测时延，不是失败；期间没有重复提交，节省了一次机会。

### 8.5 plan 尝试 01：调度代价模型暂缓

- 按 `plan.md` 的准入条件，必须先列出目标 case 的当前路由，并解释至少两个低分 case 的资源浪费。
- 已有可靠 M/N 桶 probe 只覆盖未提前进入并行路径的 case；case 7–11 在旧版本中被 `ProcessNParallel/ProcessParallel` 提前 return，probe 循环没有执行。因此不能用这些耗时反推它们的 shape 或当前 `mTiles/nTiles/nGroups`。
- 历史单变量仅能证明“把特定大 BF16 跨 layout 输入统一强制 `nGroups=1`”会让 case 11 从约 `225 us` 退化到 `331 us`，不足以拟合 Batch/M/N 三路完整代价模型。
- 结论：尝试 01 暂缓，不提交猜测性阈值；同样依赖该模型的尝试 02 暂缓。额度保留给已有直接测量依据的独立分支。

### 8.6 plan 尝试 05：N-group 连续段异步 Matmul（保留）

目的：当前主 `ProcessMTile` 的异步 `Iterate<false>/GetTensorC<false>` 已有历史正收益，但 N-parallel 的连续多 tile segment 仍逐 tile 同步。实验只对 `K%16==0` 隐含保证的连续 segment、`groupTiles>1` 且非 Row-GEMV 的矩阵前缀启用同一套已验证异步配对；K-tail、单 tile、M 行尾和 Row-GEMV 保持同步路径。调度公式、partial 布局、归约次序与 kernel launch 数均不变。

提交信息：

- submission：`384543`
- object ID：`6ab08cffb0477ec41eecd026`
- 候选 SHA-256：`da2f4f2be9c95c06e98c406f19674f0276cf7f4ac69372d1192c8f44a1ed312b`
- 结果：15/15 Pass
- 耗时：`11.45, 21.71, 27.74, 15.09, 12.97, 98.48, 17.38, 102.96, 138.01, 149.63, 231.48, 185.12, 33.14, 29.16, 37.11 us`

结论：

- 相对直接父版本 `384408`，case 11（0-based）从 `225.93` 降至 `185.12 us`，提升 `18.06%`；后续提交 `384622` 的 case 11 为 `182.46 us`，即使 case 2 已切换到 AIV，case 11 仍稳定快于旧版，构成两次独立正向证据。
- case 13 从 `32.91` 降至 `29.16 us`，但后续不改此路径的 `384622` 又为 `27.85 us`；不能仅根据时间推断 case 13 是否命中 N-group 异步。把这项 11.4% 的单次变化列为待归因信号，不能当第二个已确认的路径收益。
- 按官方对数公式和公开 best time 重算，15 case 平均分约 `20.295 -> 20.471`，增加 `+0.176`；case 11 单 case 约 `17.548 -> 19.204`，折合平均贡献约 `+0.110`。其它未明确命中路径的变化不归因于本机制；主要保留证据是 case 11 的两次稳定约 18% 耗时收益。
- case 2 为 `27.74 us`，与父版本 `27.75 us` 几乎完全一致，已经自然复验行级 GEMV 的 `29.39 -> 27.7x us` 收益，因此 01B 正式保留，无需专门复验。
- case 12 从父版本异常偏快的 `29.69` 回到 `33.14 us`，接近长期 `31.9–33.9 us` 区间；这是父版本的有利抖动消失，不是本次 N-group 异步回退。
- 当前可靠候选升级为 `384543`。下一步只考虑计划 06/07 的依赖检查；若无法证明 Vector 或 partial merge 是剩余关键路径，则取消对应槽位，不在已经获胜的异步循环上盲目叠加。

### 8.7 plan 尝试 06–08：依赖审计

- 尝试 07 所述“一个 worker 连续处理相邻多个 N segment、在 UB 内合并后只写一次 partial”已经是当前 `ComputeNGroupRowMax` 的既有机制；384543 只是把其中连续多 tile segment 异步化。再次实现同一机制不会回答新问题，因此不占用提交槽位。
- 尝试 06 的 C 输出双缓冲要求先证明 Vector epilogue 仍位于关键路径；尝试 08 要求先证明 owner partial merge 成本显著。384543 只证明同步 Matmul 等待是瓶颈，没有提供这两项证据。
- 结论：06/08 暂缓，不在已获胜路径上盲目叠队列和 UB 占用；07 视为已由现有架构覆盖，不重复提交。

### 8.8 plan 尝试 12：连续 layout 的 FP16 AIV 直算（已证伪并回退）

目的：case 2 已可靠识别为 `M<=8、N=9–32、FP16、transposeX1=false、transposeX2=true、K%16==0`，两侧逻辑 K 向量在 storage 中连续。原型让每个 block 负责一个 query row，K 以最多 512 元素分块；x1 chunk 搬入一次，对每个 document 执行 FP16→FP32 Cast、Mul、ReduceSum，按固定 K-chunk 顺序累加，完成 Max(N) 后跨核同步并 Sum(M)。仍为一个 global kernel 和一次 launch。

提交信息：

- submission：`384622`
- object ID：`6ab08fa9b0477ec41eee10b3`
- 候选 SHA-256：`70e914e9152394d2b5d6ba2fd966aed13ab4fab63524ee6a101ec211d566110c`
- 结果：15/15 Pass
- 耗时：`12.00, 22.90, 32.81, 15.93, 12.20, 98.45, 16.46, 102.03, 136.87, 146.85, 229.39, 182.46, 30.50, 27.85, 37.63 us`

结论：

- 唯一明确命中的 case 2 从父版本 `27.74` 退化到 `32.81 us`，慢 `18.28%`；按官方公式该 case 约从 `13.706` 降到 `12.970`，折合平均贡献约 `-0.049`。改动路径和回退方向完全对应，不能用其它未命中 case 的有利波动掩盖。
- 虽然本次按全部单次耗时和官方公式重算的平均分反而约增加 `+0.135`，增量来自未命中路径的评测波动，不是 AIV 机制收益，因此不保留。
- 根因判断：对这一 K/shape，逐 document 的 MTE2、两次 Cast、Mul、ReduceSum 和串行循环成本高于 Cube/GEMV 固定开销；单纯调 K chunk 或加双缓冲不能消除逐 document 指令与归约成本。
- 按计划分支止损，依赖 AIV 原型获胜的尝试 13–16 全部取消；不扩 BF16、不做 layout 重排、不为该路径扫描 chunk 大小。
- 已精确恢复 `384543`：`kernel.asc` SHA-256 为 `da2f4f2be9c95c06e98c406f19674f0276cf7f4ac69372d1192c8f44a1ed312b`，与远端源码逐行 diff 为 0，参考测试通过。

### 8.9 浏览器提交守门修复

- 384622 提交前，系统剪贴板粘贴三次未通过编辑器回读守门；三次均在点击“提交代码”前终止，没有创建 submission、没有消耗评测次数。
- 原因是提交页文件选择存在两个 `kernel.asc` 控件，且 Windows 系统剪贴板在该会话中返回空内容。
- `.mcp_tools/paste_and_submit.js` 现会明确选择第一个 `kernel.asc` 控件，使用页面内 `ClipboardEvent` 注入源码，并拦截“复制当前文件”回调逐字回读。只有规范化内容完全一致才点击提交。

### 8.10 尝试 17/18 的零提交准入审计（暂缓）

- 当前可靠候选 `384543` 已在普通 Matmul 模板中使用 `CFG_MDL`，Host 调用 `MatmulApiTiling::GetTiling` 后读取实际 `baseM/baseN`，但没有对目标 case 的 `baseK/stepKa/stepKb/depthA1/depthB1`、MTE2 等待或 L1/L0 占用的可观测记录。
- 当前 Windows 环境没有可调用的 CANN 编译器、`npu-smi` 或 `msprof`。本地附带的华为 `matmul_preload` 样例支持 A2，但该样例要求 K 全载（`singleK <= baseK * stepK`）且 A1/B1 深度对应双缓冲；仅修改 preload flag 或照搬示例的某组深度无法保证当前动态 tiling 符合契约。
- [华为 Matmul tiling 的 SetMatmulConfigParams 文档](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0704.html) 明确要求 Host tiling 配置与 device 的 MatmulConfig 匹配；不能只切设备模板。文档所列 L1 缓存 UB 特性也不支持 A2，不应把它当作本题优化点。
- 结论：计划 17 未通过“实际 tiling + 资源 + 瓶颈证据”的准入，暂缓且不提交；依赖 17 胜出的 18 一并暂缓。尝试 19 同样缺少“低并行度且大 K”的具体 case 证据，暂缓 Split-K；不根据仅有耗时猜 K 值。

### 8.11 尝试 03/04 的零提交准入审计（暂缓）

- [华为 Matmul tiling 文档](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0704.html) 提供 `FIRSTM/FIRSTN` 的跨 tile 遍历顺序，说明多 M tile 的输出顺序在配置一致时理论上可以定义。
- 但当前大 M 高并行度路径的工作单元已经是单个 M tile，超块会减少并行 job；现有结果证明 M 并行让历史 case 12 从约 227 us 下降到约 33 us。若不能先证明其它目标 case 仍有多个相邻 M tile 分配给同一 block，并同时记录 tile 次序、C stride 与失去的并行度，则 03 的“复用 B”可能得不偿失。
- 结论：03 暂缓，不靠官方有遍历顺序这一事实就贸然拼接 M-supercall；04 依赖 03 的正确与收益，随之暂缓。没有提交，不影响 384543。

### 8.12 尝试 10 的 UB/异步契约审计（零提交，暂缓）

- 当前 `Init` 同时初始化单槽 C 输出队列（`4×baseM×baseN` 字节）及完整尾块 padded buffer（相同大小），还有 rowMax、tileMax、partial 输入/输出队列、归约与 `GetBlockNum()×32` 同步缓冲。以实际 tiling 返回 `baseM=baseN=128` 为例，每份 FP32 C tile 为 65,536 字节：把 C 队列改成双槽后，仅这两槽和原样保留的 padded buffer 就需要 196,608 字节，尚未计入其它缓冲。
- [华为 A2/A3 UB bank 最佳实践](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0025.html) 给出 192 KiB UB。**这只是 A2/A3 参照，不代表已确认评测设备型号**。原样双槽不能作为通用方案；必须先获得实际设备 UB 容量。只有严格证明某窄路由不会访问 `tailPaddedBuffer_`，才有条件跳过此 buffer 的初始化并重新核算所有队列和 bank 冲突。
- [华为 GetTensorC 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0639.html) 说明异步 `Iterate` 必须配 `SetWorkspace`，且异步结果消费、输出连续写的布局有各自契约；已有主路径满足单槽异步配对，但双槽必须另外证明每个槽从 `GetTensorC` 到 `ReduceMax` 再到 `FreeTensor` 期间没有重用，也不能仅靠 `InitBuffer(...,2,...)` 假定实现了 Cube/Vector 重叠。
- 当前没有目标 case 的实际 `baseM/baseN`、UB 容量或 Cube/Vector 等待比例，无法证明窄门控双槽能提升至少两个主路径 case。尝试 10 继续暂缓；不改 `kernel.asc`，不为不完整的双缓冲占用提交次数。由 10 派生的 11 是独立的尾块语义问题，仍需官方 mask/stride 契约和逐 lane 证明，不因为 10 暂缓就直接视作已证伪。

### 8.13 实际可获取的评测信息与评分公式纠错

- 旧独立浏览器脚本的保存登录态已失效，但 Playwright CLI 的已打开会话仍处于 `2040lin` 登录状态；已直接核对线上题面、本人 `384543` 详情及排行榜。详情只含 15 个 case 的耗时、精度与 `best_time`，不含 `M/N/K`、实际 tiling、芯片型号、UB 容量或 Cube/Vector/MTE profiler 时间；排行榜测试点元数据也仅有 ID、baseline、tbest、type。公开题面和本地模板未提供这些隐藏配置，不能把缺失数据编造成“已测量”。
- 本地 `CMakeLists.txt` 指定 `--npu-arch=dav-2201`。[华为毕昇编译器文档](https://www.hiascend.com/document/detail/zh/canncommercial/900/compiler/BishengCompiler/atlas_bisheng_10_0017.html) 说明 Atlas A2 与 A3 均对应 `2201`；因此编译目标不能唯一确定评测机器，也不能据它宣称实际 UB 为 192 KiB。此前 8.12 的 192 KiB 仅为官方 A2/A3 最佳实践的参考上限，不是在线环境实测值。
- 更重要的是，线上题面与 `problem_official.md` 一致，比赛得分规则为 `100 / (1 + log_1.5(t/T))`，而旧计划与若干旧实验用 `100×T/t` 线性估分。已据 `384543` 详情里的 `best_time` 重算：`379068=20.124`、`384408=20.295`、`384543=20.471`、`384622=20.606`。线上排行榜显示本人当前为第 67 名、`384622` 实际得分 `20.61`，验证了重算公式；但该版唯一明确命中的 case 2 确定退化，未命中 case 的有利波动不能归因于 AIV，故本地代码仍保持 `384543`。旧线性估分的单 case 收益和以它设计的固定 `+0.35/+0.25` 准入线均作废；以耗时改善、命中证据、官方公式以及自然复验共同判断。
- 在无 NPU、无 msprof、线上详情不含 tiling 的情况下，真正的目标 case 资源瓶颈不能零提交实测。下一次需要评测时应设计一个合规、单提交、能区分多种瓶颈假设的受控实验；在此之前仍不提交“猜目标 case shape”的优化版。
- 已打开本人 `384543` 提交工程的 `run.sh` 与 `scripts/BatchMatmulMaxSum.py`：前者只做 CANN 编译、样例运行与校验，没有设备/tiling/profiler 输出；后者的 `cases` 仅有公开基础样例 `(B=1,M=2,N=3,K=4)`，不是排行榜 15 个隐藏测试点。不能从这些公开样例反推隐藏 case。
- 登录会话的本人提交列表显示上海时间 2026-09-21 目前有 4 次提交：`384363/384408/384543/384622`，均已完成且 Pass；本轮只读审计新增提交 `0` 次。按用户所述每日 50 次额度，当前至少应保留约 46 次，不为缺乏证据的实验占用它们。

### 8.14 plan 尝试 10：窄路由 C 双缓冲（正确但无收益，已回退）

- 实现：Host 用 `PlatformAscendC::GetCoreMemSize(UB)` 查询实际 UB 容量；只在非 Row-GEMV、`nGroups=1`、`nTiles>1`、M/N 均整 `baseM/baseN` 且 8 对齐、完整 UB 预算（含 16 KiB 保留量）允许时，跳过不可能访问的 padded buffer，令 C 输出队列深度和物理槽数均为 2。主路径先排入第 0 个 C tile，之后每次先排入下一 tile，再出队当前 tile 做 `Max(N)`；其它路径保持单槽。官方 TQue 文档要求连续两次 EnQue 的 queue depth 至少为 2，代码遵守此契约；仍只有一个 global kernel。
- 提交 `390882`，object ID `6ab0f77bb0477ec41e30a047`，SHA-256 `573050bcb51c03d2811e98ab6160a02e2975a095801ade363d02f120cc0b5094`。上传前页面“复制当前文件”回读 51,795 字符，SHA-256 与本地一致；远端源码与本地候选逐行 diff 为 0。结果 15/15 Pass，耗时（0-based case 顺序）：`11.88, 22.24, 27.46, 16.32, 13.05, 100.32, 17.58, 102.94, 139.35, 151.46, 231.87, 181.35, 35.36, 29.36, 37.30 us`。
- 按官方对数公式，候选平均约 `20.328`，父版 `384543` 约 `20.471`，本次低 `0.143`。case 11 从 `185.12` 到 `181.35 us`（约 2.0%）不足以和噪声区分；case 12 从 `33.14` 到 `35.36 us`（慢约 6.7%），case 3 从 `15.09` 到 `16.32 us`（慢约 8.2%）。未观察到两个明确命中 case 快 8% 的预定目标，不能保留。
- 结论：窄路由双缓冲的正确性在在线 15 case 得到验证，但收益假设不成立或命中范围太窄；不能仅从耗时判定哪一个。按止损规则回退，不再无证据调 queue depth、UB margin 或对齐门槛。当前 `kernel.asc` 已精确恢复 `384543` 的 SHA-256 `da2f4f2be9c95c06e98c406f19674f0276cf7f4ac69372d1192c8f44a1ed312b`，本地参考检查通过。本日提交计数现为 5/50。

### 8.15 plan 尝试 11：masked 尾块归约的官方 API 审计（零提交）

- [CANN9.0 高阶 ReduceMax 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_10055.html) 规定 A2/A3 对 `srcInnerPad` 仅支持 `true`；内轴不是 32 字节整数倍时，需要实际按 32 字节 padding 后才能用当前 `Pattern::Reduce::AR` 路径。单纯把 `shape[1]` 改成 `currentN` 而直接吃紧凑 Matmul C 输出，不能满足其物理布局契约。
- [基础 ReduceMax 文档](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0076.html) 确有 bitwise/contiguous mask 版本，但它把多个 repeat 一起归约到一个结果，并不直接给每个 query 行一个最大值；若改用逐行调用，必须额外证明 32 字节行起址、`srcRepStride`、共享临时空间和每行输出的布局，且可能引入大量小粒度指令。
- 因此 11 还不是“可直接替换 padded 复制”的已验证方案。没有逐 lane 全负/双尾证明和性能优势依据时暂缓，不为看似简单的 shape 参数修改消耗一次错误答案提交。尝试 10 的无收益也不自动证伪 11，二者瓶颈不同。

### 8.16 小 M 行级 GEMV 扩围实验（已回退）

- 动机：01B 的 `M<=8` FP16 双侧 K 连续行级 GEMV 已两次 15/15 Pass；旧正交探针把低分 case 5/6 的 M/N 桶定位在 `M=9–32、N=33–128`，且 case 5 的 layout 为 `transposeX1=false, transposeX2=true`。单变量候选只把门控扩大为 `M<=8 || (M<=32 && N>32)`，dtype/layout/K 对齐限制、Matmul 模板、归约和单 kernel launch 保持不变。此实验不能预先证明 case 5 的 dtype 与 K 对齐会命中，属于检测这一缺口的受控提交。
- 本地 `python tests/reference_test.py` 通过 10 shapes × 4 layouts；静态检查因旧门控字面量先失败，随后按实际新路由更新，并让 NumPy 分块模型覆盖 `baseM=1`，重新通过。在线提交前编辑器回读与本地源码规范化换行后完全一致，远端源码 diff 为 0。
- 提交 `394284`，object ID `6ab12c570304f72a56d2df68`，候选 SHA-256 `3c6baf3db1f65ee48d64f82a157136c8ee0e78db6ce1d6409f71cd5e51072fda`；15/15 Pass。0-based case 耗时：`11.63, 21.64, 27.64, 15.31, 12.60, 98.85, 16.98, 101.04, 137.61, 149.44, 230.14, 180.63, 32.59, 29.55, 37.28 us`。
- 最关注的 case 5 从可靠父版 `98.48` 到 `98.85 us`，没有收益；case 6 从 `17.38` 到 `16.98 us`，幅度约 2.3%，不足以证明机制。按本次详情里的相同 best times 重算，平均分 `20.4595→20.5147`，即 `+0.0552`，来自多处未确认命中的小波动，不能作为保留依据。现有详情不公开 dtype/K/实际路由，因此不能判定新门控未命中还是命中后无收益；两种情况下都不值得继续扫 GEMV 阈值。
- 已完整回退源码与参考测试，`kernel.asc` 再次为 `384543` 的 SHA-256 `da2f4f2be9c95c06e98c406f19674f0276cf7f4ac69372d1192c8f44a1ed312b`。今日提交计数为 6/50。

### 8.17 plan 尝试 11 的进一步逐行 ReduceMax 成本审计（零提交）

- 官方基础 ReduceMax 支持 `count` 与 `mask/repeatTime/srcRepStride` 两族 API，但单次调用只产生一个全局最大值；其 `src` 起址要求 32 字节对齐，FP32 `dst` 起址要求 8 字节对齐，sharedTmpBuffer 起址要求 32 字节对齐。当前 Matmul C 的 `baseN` 是 8 的倍数，因此逐行 `src[row*baseN]` 对齐可证明；把每行结果放在 `tileMax[row]` 则不能保证奇数 row 的 8 字节起址对齐。需要至少 8-float 行间距或另一个收集阶段。
- 每行调用 ReduceMax 还要解决 8-stride 结果到连续 `rowMax` 的聚集、每行临时空间与事件同步。即使布局正确，M=32 时需要 32 次基础归约和聚集，现有 padded 路径使用按行 Adds 后一次高阶 AR；没有 N-tail 主导耗时的证据，无法预估计划要求的 12% 提升。只把 `count=currentN` 写到高阶 AR 则仍违反其物理 padding 契约。
- 结论：11 的数值可行性尚需新的收集方案，性能假设也不足；目前不提交。参考：[官方基础 ReduceMax 文档](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0076.html)。

### 8.18 用户授权放宽准入后：plan 02 N-group 双 tile 粒度（失败，已回退）

- 父版 `384543`；唯一变化为 `nTiles>=4 && nGroups>nTiles/2` 时把 `nGroups` 限到 `nTiles/2`，让每个连续 N worker 平均至少两个 tile。未改设备代码、Matmul tiling、同步、归约和 launch 次数。候选 SHA-256 `229995e9de8e097606b74c39d4aac9079e27b7f7447259846b402b9274899e94`。
- 本地参考测试 10 shapes × 4 layouts 通过；远端上传源码 diff 为 0。提交 `394474`，object ID `6ab12f120304f72a56d4b612`。
- 结果是 15/15 全部 Wrong Answer，所有 precision=0、time=0；页面/API 均未给出编译/设备日志。此模式无法用于性能比较。源码与通过版本的实际差异只有上述 Host 条件与注释；设备代码相同。因此可能是新 group 分配引起某个隐藏 shape 的运行期资源/同步失败，也可能是构建/评测环境故障，不能仅凭全 0 确定根因；尤其不能把它解释成普通数值误差或 N-group 性能退化。没有外部故障证据，不重复同一 hash。
- 已完整回退到 `384543`，SHA-256 与远端通过版相同，本地参考检查再次通过。此实验无法验证“双 tile 更快”假设，计划 02 暂不保留。今日提交计数为 7/50。
