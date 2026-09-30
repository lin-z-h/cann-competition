# 开发交接：2026-09-30

本文件记录今天停止设备调优时的实际状态，并在准备 GitHub 更新时整理。文档整理不代表恢复测试、继续提交或执行 push。先看这里，再按需要阅读[设备全过程](docs/history/cannlab_20260930_session.md)、[线上评测](docs/history/cannjudge_20260930_tuning.md)和[赛题](docs/spec/problem_official.md)。

## 1. 先确认哪份代码可以当基线

根目录 `kernel.asc` 没有修改，LF SHA-256：

```text
da2f4f2be9c95c06e98c406f19674f0276cf7f4ac69372d1192c8f44a1ed312b
```

它是历史可靠线上源码，不意味着所有新增设备边界都已验证。原新设计和今天的实验分别保留，不把候选直接覆盖到根目录。

| 源码 | 用途与实际状态 |
| --- | --- |
| `experiments/new_design/kernel.asc` | 原新设计，历史线上 15/15；本次设备测试发现 N 尾块错误，保留原版用于追溯 |
| `experiments/cannlab_tuning/kernel.asc` | N 尾块修复，默认最多 4 个 N tile/调用，仅 FP16 启用对齐消费快路径；固定输入 103 用例覆盖，线上 530365 为 15/15 |
| `experiments/cannlab_pack_a/kernel.asc` | 在同一个 kernel 内整理转置 A；设备 103/103 与八布局 24 次 profiler 检查通过，线上 531572 为 15/15；新增 K16/多 batch 边界尚有问题 |
| `experiments/cannlab_pack_a/repeat_m.asc` | 改变 Transpose 重复方向；编译、八布局精度和四转置布局采样完成，没有明确性能收益；扩展回归在首个新增多 batch 边界失败 |
| `experiments/cannlab_pack_a/batch_fix.asc` | 仅把模式 B 最终输出改为精确 4B DMA 写出；编译通过，新增边界 t00/t01 通过、t10 仍为 NaN，不是完成的修复 |
| `experiments/cannlab_pack_a/batch_diag.asc` | 回读 packed A、比较逐元素位模式及输出 partial 的诊断源码；只确认编译完成，没有执行诊断，不作为提交候选 |

停止时扩展回归没有跑完，预定的 151/175 用例与后续交替性能采集不能计为通过。诊断源码保留是为了恢复调查，不是无用临时文件。

## 2. 今天完成了什么

### 设备计算链路

通过平台生成的 SSH 配置，在实际 NPU 上执行构建和计算。历史设备是 A2，系统 aarch64、Ubuntu 20.04，CANN 9.0.0，应用报告 20 个 Cube 核，编译目标 `dav-2201`。这些信息仅描述此次环境，合作者的 A3 要重新确认。

初期 SSH 转发和非交互终端库路径有问题；恢复平台连接、加载 CANN 环境及驱动库后，设备编译与运行可重复执行。环境连接方法已经独立到[远端指南](docs/cannlab_remote.md)，不保留个人固定 Host。

### 正确性修复

原新设计在 `B=1,M=128,N=257,K=64,FP16,t00` 出错：输出约 `18.6211`，参考约 `9.3125`。诊断显示完整 N 分组正常，最后 N 尾组只有首行正确。

原因是 sequential ND C 尾块按实际宽度写出，消费者按完整 `baseN` 行步长读取。修复将最后 N 尾块从多 tile 调用中拆开，同步获取 C 并关闭 sequential，使用显式 `orgNc=baseN`；完整 tile 保留异步消费。

测试逐步扩大到 103 用例：23 个确定性用例 + 10 个形状 × 两类型 × 四布局的 80 个随机用例。随机种子由用例名的 SHA-256 固定，添加形状不会改变已有输入。修复候选的固定输入 103 覆盖由一次中断日志与最后 10 个续跑日志合并确认；A 整理首版有单次完整 103/103 日志。

CPU 模型 59/59 多次通过，只说明数学/调度模型和已有契约成立，不覆盖 SDK 转置指令、缓存一致性或设备精度。每次有效 profiler 采集另做独立精度与单计算 kernel 断言；workspace 的 `MEMCPY_ASYNC` 不算额外计算 kernel。

## 3. 调优方式：如何从报告决定下一步

采用“独立候选 → 精度检查 → 受控采样 → 保留或淘汰”的顺序，不在可靠源码上直接叠加实验。

1. 先记录设备、CANN、目标架构、源码及二进制哈希、构建参数、形状、布局和输入指纹。
2. 用量化后实际输入的 NumPy FP64 Matmul，独立计算 `max(N)` 再 `sum(M)`；输出须有限且满足 `1e-4 + 1e-4*abs(reference)` 容差。
3. 一次只改变一个策略。采集通常为三次独立进程，保存原始 CSV、中位数及范围；小差距改做 A/B、B/A 交替采样。
4. profiler 逐次核对精度和 `op_summary` 中恰好一个计算 kernel，拒绝混用已有采集目录。
5. 设备测试和采样串行，避免同时使用 NPU 干扰时间。线上评测可以单独进行，但提交按顺序执行并用 ID/hash 确认归属。
6. 单个形状获益不推出线上总分提高，也不把管线活动比例相加成耗时分解。正确性失败的运行不计有效性能结果。

### 已测策略

| 策略 | 结果与决定 |
| --- | --- |
| 每次仅 1 个 N tile | FP16 中位数约 51.24 us，对比 4 tile 的 45.38 us 更慢；保留 4 tile |
| baseM 改为 64 | 两类型都更慢；不采用 |
| 对齐模式 B 复用原消费者 | 五组交替采样 FP16 中位数改善约 4.69%，但有反向波动；BF16 约 0.68% 在噪声范围内，只给 FP16 启用 |
| 同 kernel 内整理转置 A | 在受控 K64、大 M 形状上降低耗时约 26%–29%；保留独立实验，未证明全局收益 |
| Transpose 改为沿 M 重复 | 四布局中位数约 88–92 us，和首版相近；没有明确新收益 |

前三项主要对照形状为 `B=1,M=4096,N=256,K=64`；类型/布局比较要看原始记录，不能跨不同输入直接归因。

## 4. 目前的性能瓶颈

早期拿 FP16 t00 与 BF16 t10 对比，类型和布局同时变化，不能区分原因。后来从同一逻辑 A/B 生成八布局，验证数值在 FP16/BF16 均精确表示；逻辑输入 SHA-256：

```text
dc4e90ce66a62613790cdc9b8e8a5d3a5ccff5115c700c84f82d9edb2f10a8c2
```

关闭消费快路径，统一使用通用修复消费者，形状仍为 `B=1,M=4096,N=256,K=64`：

| 类型 | t00 | t01 | t10 | t11 |
| --- | ---: | ---: | ---: | ---: |
| FP16 基线，中位数 us | 44.841 | 45.841 | 122.562 | 122.342 |
| BF16 基线，中位数 us | 44.421 | 43.621 | 125.923 | 122.962 |
| FP16 整理 A，中位数 us | 46.001 | 45.540 | 90.582 | 87.462 |
| BF16 整理 A，中位数 us | 45.801 | 45.401 | 91.281 | 87.982 |

t10/t11 表示 A 转置。差异主要跟随 A 的布局；整理后 Cube 的 MTE2 活动比由约 0.56–0.58 降至 0.044–0.049。支持优先分析转置 A 的碎片搬运和相关等待，但不能由此定位唯一代码行。

Memory 采样曾报告 `aic_main_mem_read_bw=0.083 GB/s`，没有支持 HBM 带宽饱和的证据。高 scalar ratio 也不是“绝大部分时间花在标量代码”的直接证明。整理后仍约 87–91 us，相比非转置的约 44–46 us 有明显额外代价；改变地址设置次数没有稳定收益，剩余代价需进一步测量。

## 5. 任务怎样划分

模式选择跟随形状、可用核数和 tiler 实际返回值，不依赖隐藏 testcase ID。

- **模式 A：整 batch 归属。** 一个 block 负责完整 batch；batch 数多、很小的矩阵或一个 M/N tile 时优先使用，减少跨核归约。
- **模式 B：M tile 并行。** `B*ceil(M/baseM)` 个任务，每个任务负责当前 M tile 的全部 N；先沿 N 求行最大值再求 M 局部和，写对齐槽位，一次同步后合并各 tile。大 M、M 任务足以利用至少约半数核时使用。
- **模式 C：M×N 分组。** A/B 并行度不够且有多个 N tile 时，再增加 N 分组；写逐行 partial max，一次同步后先合并各 N 组的行最大值，再沿 M 求和。不能直接把各组 sum 相加。

Matmul 以 baseM/baseN tile 消费，一次调用最多 4 个 N tile，完整组使用异步生产/消费；N 尾块独立同步处理，K 尾块使用单 tile 兼容路径。单次执行仍只有一个全局 kernel。

A 整理只在原 A 转置、M≥2048 且整除128、K为16倍数且≤64、batch少于核数且有足够M并行度时启用；tiler返回的baseM必须为16倍数并整除M。这些门控保证模式B和每个M tile独占。两块UB总量最多32KiB，GM workspace增加 `B*M*K*2` 字节；本核写出完成后再由非转置Matmul读取。其跨Vector/Cube可见性尚应在新设备重验。

## 6. 线上为什么下降

今天成功新建两次评测，均 15/15 Pass：

| 版本 | ID | 上传副本 LF SHA-256 | 重算分数 |
| --- | ---: | --- | ---: |
| 根可靠源码最近已有记录 | 522426 | `da2f4f2be9c95c06e98c406f19674f0276cf7f4ac69372d1192c8f44a1ed312b` | 20.8508 |
| N 尾块修复/消费快路径 | 530365 | `cfcfa72a3fc20763963dc7373fe858d69796a25fb7a30a477c34f7658da32c73` | 20.1005 |
| A 整理首版 | 531572 | `5c769a384593764a844516640f68f4485834c33c7cf00172b7913b2aca50cc26` | 20.0119 |

分数是详情的 time/best_time 按题面公式逐项计算后平均，服务器没有返回总分字段。根可靠版历史约20.47与这里的20.85是不同次记录，不混为同一次运行。

修复版相对522426第8项从101.23增至135.47（约慢34%），第12项从179.78增至338.92（约慢89%）；第9项135.74降至95.00、第11项229.02降至198.05，仍无法抵消损失。A整理版与修复版整体接近，没有证明局部K64实验覆盖了线上主要路径。具体退化case的底层原因未定位，不能仅凭编号推断隐藏形状。

首次直接提交带未启用TRACE诊断的源码被HTTP400拒绝，没有创建评测；平台未指明具体行。只剥离三段未启用TRACE的独立副本随后被接受，每份均回读核对hash；原候选与可靠源码未修改。上传副本与原源的映射在[线上记录](docs/history/cannjudge_20260930_tuning.md)中。今后不能把实验源码hash与上传副本hash混用。

## 7. 未解决的正确性问题：恢复时先处理

扩展边界 `B=2,M=2176,N=129,K=16,FP16,t00` 在通用修复基线和repeat_m版出现某个batch为NaN，位置不固定；此前103用例未覆盖。该布局不触发整理A。

### 多 batch 最终写出

模式B最后让不同核用 `y_.SetValue` 写相邻FP32输出。GlobalTensor标量写涉及每核DCache/缓存行写回，相邻输出可能互相覆盖；主程序把输出初值设为NaN，这与“一个正确、一个仍NaN”吻合。`batch_fix.asc` 改为队列同步后用DataCopyPad精确写4B，新增t00/t01通过，支持这一定位，但完整回归未完成。参见[官方SetValue说明](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_00028.html)。

### K=16 整理路径

同一修复版t10仍输出 `[NaN,NaN]`。只读API审查发现 `TransDataTo5HD repeatTimes==1` 有特殊地址规则，要从列表地址起始读写时两repeat stride应为零；当前沿K重复的实现K16时仍给非零stride，可能造成偏移/越界。**尚未修改此参数、未运行逐元素packed A诊断，不能称已经设备确认或修好。** 参见[官方Transpose说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/data_layout_conversion/TransDataTo5HD.md)。沿M版也需检查baseM=16的同类边界。

恢复时先执行诊断并分别验证修复，再覆盖K16/32/48/64、多batch B2/B3/B16/B17、N128/129/256、M尾块与K>64回退。原计划175用例任务在目标回归第一轮t10即停止，没有执行175套件或后续交替采样。

## 8. 合作者如何复现

先按[SSH指南](docs/cannlab_remote.md)新建自己的A3环境，不沿用任何个人Host。查明实际CANN脚本、架构和工作区后，在远端初始化：

```bash
export CANN_ENV_SCRIPT='<自己的 CANN set_env.sh>'
export NPU_ARCH='<自己的设备编译目标>'
source "$CANN_ENV_SCRIPT"
python3 -m pytest -q -p no:cacheprovider tests/new_design_test.py
bash scripts/cannlab_smoke.sh
```

smoke会先生成确定性输入；测试原新设计时可能因已知错误提前停止，这是待修代码的结果，不要跳过精度检查并继续宣称通过。

已有用例后，验证通用修复候选：

```bash
export CANNLAB_KERNEL_FILE=experiments/cannlab_tuning/kernel.asc
export CANNLAB_BUILD_DIR="$PWD/build/tuning-current-device"
export CANNLAB_PROFILE_DIR="$PWD/build/profile-current-device-fresh"
bash scripts/cannlab_tune.sh
```

复现A整理首版时选择 `experiments/cannlab_pack_a/kernel.asc`，同时设置 `CANNLAB_ALIGNED_FAST_PATH=OFF`，保持与本次测量相同。统一入口默认会定义ON，不能仅依据文件默认宏推断最终编译参数。

布局对照和新增边界可分别执行：

```bash
python3 scripts/cannlab_layout_probe.py \
  --source-case build/cannlab-new_design/run/mode_b_f16_t00 \
  --output build/layout-probes
python3 scripts/cannlab_verify.py \
  --exe build/tuning-current-device/batch_mat_mul_max_sum_custom \
  --run-dir build/cannlab-new_design/run --random \
  --random-shape 2,2176,129,16 --case-filter b2_m2176_n129_k16 --repeat 3
python3 scripts/cannlab_profile.py \
  --exe build/tuning-current-device/batch_mat_mul_max_sum_custom \
  --case-dir build/layout-probes/d1_t10 \
  --output build/profile-layout-current-device-fresh/d1_t10 --samples 3
```

重复形状去重；筛选前随机输入全部生成，单例种子固定。profile目录必须是新目录。命令中的路径是仓库相对路径，按本机实际构建目录替换。

## 9. 下一步方向与证据保存

正确性修复完成后，优先查当前异步Matmul期间能否预整理本核下一M tile，复用现有pack队列以重叠搬运和Cube工作；这是建议，**未实现、未测量**。不要直接新增整套双缓冲，先核对UB预算、Matmul占用和异步生产条件。继续观察AIV的MTE2/MTE3/scalar及等待，线上退化还需独立诊断。

本地 `build/remote-evidence/session-20260930/` 保留较早设备日志/CSV；`build/online-evidence/session-20260930/` 有完整线上详情、源码映射和上传副本。后期布局/pack/失败回归日志主要保留在原远端 `build/`，不假设全部已下载。这些忽略目录不随Git传递，合作者应在自己的设备重建证据；共享原始文件时排除凭据。

本次仓库整理删除根目录临时ZIP、一次性任务脚本、旧页面、过期登录状态备份和Python缓存，保留当前忽略登录态、依赖、有效证据与诊断候选。原有tools脚本迁移/删除及其它既有改动保留，不在文档整理时回滚。尚未commit或push；之后发布前检查完整差异及忽略规则。
