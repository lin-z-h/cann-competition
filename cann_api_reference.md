# 本次比赛接口整理

本文面向当前 BatchMatmulMaxSum 竞赛工程，整理实际需要使用的 CANN / Ascend C 接口。内容以本项目的 kernel.asc 为准，并参考官方 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) 中的 MatMul、融合算子、归约和 CANNJudge 示例。

> 适用环境：CANN 9.0.0 及以上、Ascend C、NPU kernel 工程模板。

## 1. 先区分两种工程接口

本项目不是标准 op_host/op_kernel 自定义算子工程，而是比赛提供的 npu_kernel_dev 模板。

| 层次 | 本项目需要的接口 | 是否需要实现 |
|---|---|---|
| 比赛入口 | extern "C" void run_kernel(...) | 需要 |
| 设备 Kernel | __global__ Ascend C kernel | 需要 |
| MatMul 高阶 API | matmul::Matmul、REGIST_MATMUL_OBJ | 需要 |
| MatMul Tiling | matmul_tiling::MatmulApiTiling | 需要 |
| 标准算子注册 | OpDef、OP_ADD、TilingContext | 本题不需要 |
| 标准算子目录 | op_host/、op_kernel/ | 本题不需要 |

学习仓库中的标准算子示例适合用来理解 API，但不能把标准 op_host/op_kernel 目录和注册代码直接提交到本题模板。

## 2. 比赛模板边界接口

### 2.1 run_kernel

当前入口位于 kernel.asc 第 1047 行：

~~~cpp
extern "C" void run_kernel(
    GM_ADDR x1,
    const TensorGroupInfo &info_x1,
    GM_ADDR x2,
    const TensorGroupInfo &info_x2,
    GM_ADDR y,
    const TensorGroupInfo &info_y,
    int64_t availableCoreNum,
    aclrtStream stream,
    bool transposeX1,
    bool transposeX2);
~~~

| 参数 | 用途 |
|---|---|
| x1、x2、y | 输入、输出设备地址，类型为 GM_ADDR |
| info_x1、info_x2、info_y | Tensor 的 shape、dtype 等元信息 |
| availableCoreNum | 可用 AI Core 数，用于计算并行策略 |
| stream | 当前 ACL stream；异步操作和 Kernel 启动使用它 |
| transposeX1、transposeX2 | 只描述 storage shape，不改变数学逻辑 |

### 2.2 TensorGroupInfo 和 dtype

当前代码通过 info_x1.tensors[0].shape、info_x1.tensors[0].dtype 读取元信息。本题 dtype 编码来自比赛模板，不能与 ge::DataType 枚举混用：

| 模板 dtype | 含义 | Kernel 类型 |
|---:|---|---|
| 1 | FLOAT16 | half |
| 2 | BFLOAT16 | bfloat16_t |

## 3. Host 侧 MatMul Tiling

头文件：

~~~cpp
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
~~~

### 3.1 平台对象

~~~cpp
auto *platform =
    platform_ascendc::PlatformAscendCManager::GetInstance();
size_t systemWorkspaceBytes =
    platform->GetLibApiWorkSpaceSize();
~~~

常用接口：

| 接口 | 作用 |
|---|---|
| PlatformAscendCManager::GetInstance() | 获取当前芯片平台对象 |
| GetLibApiWorkSpaceSize() | 查询高阶 MatMul API 所需系统 workspace |
| GetCoreMemSize(...) | 标准算子工程中查询 UB/L1 等核内存 |
| GetSocVersion() | 标准算子工程中查询芯片型号 |

### 3.2 MatmulApiTiling 配置顺序

当前工程在 kernel.asc 第 930 行使用：

~~~cpp
matmul_tiling::MatmulApiTiling tilingApi(*platform);

tilingApi.SetAType(TPosition::GM, aFormat, inputType, transposeX1);
tilingApi.SetBType(TPosition::GM, CubeFormat::ND, inputType, transposeX2);
tilingApi.SetCType(TPosition::VECIN, CubeFormat::ND,
                   DataType::DT_FLOAT);
tilingApi.SetBiasType(TPosition::GM, CubeFormat::ND,
                      DataType::DT_FLOAT);

tilingApi.SetShape(M, N, K);
tilingApi.SetOrgShape(M, N, K);
tilingApi.SetFixSplit(baseM, baseN, -1);
tilingApi.SetBufferSpace(-1, -1, -1);
tilingApi.EnableBias(false);
tilingApi.GetTiling(tiling);
~~~

| 接口 | 说明 |
|---|---|
| SetAType / SetBType | 配置位置、格式、dtype 和转置描述 |
| SetCType | 配置输出位置、格式和 dtype；本题为 VECIN + FP32 |
| SetBiasType | 配置 Bias 类型；本题不启用 Bias，但仍需配置 |
| SetShape(M,N,K) | 当前计算 shape |
| SetOrgShape(M,N,K) | 原始逻辑 shape，用于尾块语义 |
| SetFixSplit(baseM,baseN,-1) | 固定 M/N tile，K tile 自动选择 |
| SetBufferSpace(-1,-1,-1) | 使用平台默认缓冲区规划 |
| EnableBias(false) | 关闭 Bias |
| GetTiling(tiling) | 生成设备侧 TCubeTiling |
| GetBaseM / GetBaseN | 读取最终生成的 tile 大小 |

Set* 接口失败时通常返回非零，GetTiling 失败时返回 -1。失败时不能使用未初始化的 Tiling 启动 Kernel。

> 学习仓库中的早期标准示例常用 MultiCoreMatmulTiling，且 SetAType/SetBType 不一定带 transpose 参数；当前比赛模板使用 MatmulApiTiling，应以当前 CANN 头文件和在线模板签名为准。

## 4. Kernel 侧 MatMul 高阶 API

头文件：

~~~cpp
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
~~~

### 4.1 类型描述

~~~cpp
using AType = matmul::MatmulType<
    AscendC::TPosition::GM, CubeFormat::ND, T, Transpose>;

using BType = matmul::MatmulType<
    AscendC::TPosition::GM, CubeFormat::ND, T, Transpose>;

using CType = matmul::MatmulType<
    AscendC::TPosition::VECIN, CubeFormat::ND, float>;

matmul::Matmul<AType, BType, CType, BiasType, CFG_MDL> matmul_;
~~~

本题涉及：

| 类型 | 使用 |
|---|---|
| TPosition | GM、VECIN、VECCALC |
| CubeFormat | ND；小 M 行 GEMV 路径使用 VECTOR |
| 输入 dtype | half、bfloat16_t |
| 输出 dtype | float |
| MatMul 配置 | 普通矩阵路径 CFG_MDL，行 GEMV 路径 CFG_NORM |

### 4.2 注册、设置输入和执行

~~~cpp
REGIST_MATMUL_OBJ(
    &pipe, GetSysWorkSpacePtr(), matmul_, &tiling);

matmul_.SetOrgShape(M, N, K);
matmul_.SetTensorA(aGlobal, transposeA);
matmul_.SetTensorB(bGlobal, transposeB);
matmul_.SetSingleShape(currentM, currentN, K);

matmul_.Iterate<true>();
matmul_.GetTensorC<true>(mmOutput, false, true);
matmul_.End();
~~~

连续多 tile 的异步生产路径：

~~~cpp
matmul_.Iterate<false>();
matmul_.GetTensorC<false>(mmOutput, false, true);
~~~

规则：

- REGIST_MATMUL_OBJ 必须在首次使用 MatMul 前完成。
- 单 tile、尾块使用同步配对 Iterate<true> / GetTensorC<true>。
- 只有确认输出顺序和生命周期正确时才使用异步 Iterate<false>。
- GetTensorC 得到的 Tensor 仍需按 Alloc -> EnQue -> DeQue -> Free 管理。
- SetSingleShape 必须和当前输出 tile 的实际 M/N 一致。

## 5. GM、UB、队列和本地 Buffer

### 5.1 GM Tensor

~~~cpp
GlobalTensor<T> x;
x.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ptr), elementCount);
GlobalTensor<T> tile = x[offset];
~~~

| 接口 | 用途 |
|---|---|
| SetGlobalBuffer | 将设备 GM 地址绑定为 GlobalTensor |
| operator[] | 创建带元素偏移的 GM 视图，不搬运数据 |
| GetValue / SetValue | 标量访问；只适合很小的尾块修补 |

### 5.2 TPipe、TQue、TBuf

~~~cpp
TPipe pipe;
TQue<QuePosition::VECIN, 1> queue;
TBuf<TPosition::VECCALC> scratch;

pipe.InitBuffer(queue, 1, bytes);
pipe.InitBuffer(scratch, bytes);

auto t = queue.AllocTensor<float>();
queue.EnQue(t);
auto ready = queue.DeQue<float>();
queue.FreeTensor(ready);

auto tmp = scratch.Get<float>();
~~~

规则：

- TQue 用于有生产者/消费者顺序的数据流。
- TBuf 用于纯计算 scratch，不走 EnQue/DeQue/FreeTensor。
- InitBuffer(queue, depth, bytes) 的 depth 是队列槽位数，可用于 Double Buffer。
- 每个队列 Tensor 必须完成 Alloc -> EnQue -> DeQue -> Free。
- TPosition 必须与计算单元和 MatMul 类型描述匹配。

## 6. 数据搬运接口

### 6.1 DataCopyPad

~~~cpp
DataCopyExtParams copyParams = {
    blockCount, blockLenBytes, srcStride, dstStride, 0};

DataCopyPad(dst, src, copyParams);
~~~

GM 到 UB 时可增加 padding 参数：

~~~cpp
DataCopyPadExtParams<float> padParams = {false, 0, 0, 0};
DataCopyPad(dst, src, copyParams, padParams);
~~~

本题注意：

- 传给 Reduce 的 tile 必须满足归约所需的对齐尺寸。
- 无效列使用负无穷填充，不能填 0。
- UB 到 GM 的写回不需要 padding 参数。
- 学习仓库建议优先考虑 DataCopyPad，但仍需核对 stride 和 32B 对齐。

### 6.2 DataCopy

规则对齐的连续搬运可以使用：

~~~cpp
DataCopy(dst, src, copyParams);
~~~

MatMul 高阶 API 已经负责主要的 ND/NZ 格式转换，不要在外层重复转换。

## 7. Vector 计算和归约

本题固定顺序是 MatMul -> ReduceMax(N) -> ReduceSum(M)。

### 7.1 基础 Vector 接口

~~~cpp
Duplicate(dst, value, count);
Adds(dst, src, 0.0F, count);
Max(dst, lhs, rhs, count);
Add(dst, lhs, rhs, count);
~~~

用途：初始化负无穷、复制尾块、跨 N tile 合并最大值、累加 M tile 结果。

### 7.2 ReduceMax

~~~cpp
uint32_t reduceShape[] = {baseM, baseN};
ReduceMax<float, Pattern::Reduce::AR, true>(
    tileMax, source, reduceShape, true);
~~~

AR 表示按行归约。输入是 [baseM, baseN]，输出为每行最大值。尾块必须先 padding 到合法归约范围，并使用负无穷作为 padding 值。

### 7.3 ReduceSum

~~~cpp
LocalTensor<float> workspace = reduceWorkspaceBuffer_.Get<float>();
ReduceSum<float>(tileSum, rowMax, workspace, currentM);
~~~

本题用它将一个 M tile 的行最大值求和为一个 FP32 标量。workspace 大小要按当前 CANN 版本和归约长度核对；当前代码使用 32 bytes scratch。

## 8. 核间同步和流水同步

### 8.1 核间同步

~~~cpp
LocalTensor<int32_t> syncLocal = syncBuffer_.Get<int32_t>();
SyncAll<true>(sync_, syncLocal, blockCount);
~~~

同步前所有 partial 必须已经写入 GM，同步区需要提前清零，所有 block 必须使用一致的 blockCount。

### 8.2 Vector/Scalar 同步

~~~cpp
event_t vToS = static_cast<event_t>(
    pipe.FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(vToS);
WaitFlag<HardEvent::V_S>(vToS);

event_t sToV = static_cast<event_t>(
    pipe.FetchEventID(HardEvent::S_V));
SetFlag<HardEvent::S_V>(sToV);
WaitFlag<HardEvent::S_V>(sToV);
~~~

Vector 指令之间还使用：

~~~cpp
PipeBarrier<PIPE_V>();
~~~

## 9. Workspace 和 ACL Runtime

### 9.1 分配和清零

~~~cpp
aclrtMalloc(
    reinterpret_cast<void **>(&workspaceDevice),
    workspaceBytes,
    ACL_MEM_MALLOC_HUGE_FIRST);

aclrtMemsetAsync(
    syncDevice, syncBytes, 0, syncBytes, stream);
~~~

清零失败时要释放已申请的 workspace 并返回。

### 9.2 Kernel 启动、同步和释放

~~~cpp
batch_matmul_max_sum_kernel<half, false, false>
    <<<blockCount, nullptr, stream>>>(
        x1, x2, partialDevice, asyncWorkspaceDevice,
        y, params, workspaceDevice, tiling);

if (aclrtSynchronizeStream(stream) == ACL_SUCCESS) {
    aclrtFree(workspaceDevice);
}
~~~

本题只允许一次 global Kernel launch。workspace 必须存活到 Kernel 完成，因此只能在 stream 同步后释放。

## 10. Kernel 声明和设备运行时接口

当前入口：

~~~cpp
__schedmode__(1) __global__ __mix__(1, 1)
void batch_matmul_max_sum_kernel(...);
~~~

| 接口 | 作用 |
|---|---|
| __global__ | 声明设备 Kernel 入口 |
| __aicore__ | 声明 AI Core 侧函数 |
| __mix__(1,1) | 启用 Cube/Vector 混合执行配置 |
| __schedmode__(1) | 指定调度模式 |
| GM_ADDR | Kernel GM 地址参数类型 |
| GetBlockIdx() | 当前 block 编号 |
| GetBlockNum() | 本次 launch 的 block 数 |
| GetSysWorkSpacePtr() | 获取 MatMul 库系统 workspace |
| GET_TILING_DATA | 标准 Tiling 工程中解析序列化 Tiling 数据 |

## 11. 建议掌握顺序

1. run_kernel、TensorGroupInfo、aclrtStream。
2. MatmulApiTiling 的 Set* -> GetTiling。
3. MatmulType、Matmul、REGIST_MATMUL_OBJ。
4. SetTensorA/B、SetSingleShape、Iterate、GetTensorC、End。
5. TPipe、TQue、TBuf 的生命周期。
6. DataCopyPad 和 32B 对齐。
7. ReduceMax、ReduceSum。
8. SyncAll、HardEvent、PipeBarrier。
9. aclrtMalloc、aclrtMemsetAsync、Kernel launch、stream synchronize/free。

## 12. 官方学习仓库对应资料

| 主题 | 资料 |
|---|---|
| Ascend C 基础 API、Kernel、Tiling | tutorials/ascendc_operator_development/02_AscendC_basic/ |
| MatMul 高阶 API、Host Tiling | tutorials/ascendc_operator_development/04_matmul_basic/ |
| MatMul 后接 Vector 融合 | tutorials/ascendc_operator_development/05_fused_operator_development/ |
| MatMul 流水、队列、Double Buffer | contrib/tutorials/machine_learning_system/05_ascendc_matmul_tiling_double_buffer/ |
| ReduceSum / ReduceMax | contrib/tutorials/data_structures_compute/04_reduce_priority_queue_heap/ |
| API 使用禁忌和最佳实践 | skills/ascendc-ops-project/references/api_best_practices.md |
| 比赛 Kernel 工程入口和提交边界 | skills/cannjudge-submit/references/kernel-project.md |

## 13. 当前项目检查清单

- [ ] run_kernel 签名与在线模板完全一致。
- [ ] 只启动一次 global Kernel。
- [ ] SetAType/SetBType/SetCType/SetBiasType 与 Kernel 的 MatmulType 一致。
- [ ] SetOrgShape、SetSingleShape 与 storage transpose 语义一致。
- [ ] REGIST_MATMUL_OBJ 在首次使用 MatMul 前完成。
- [ ] Iterate / GetTensorC 模板参数配对正确。
- [ ] 每个队列 Tensor 完成 Alloc -> EnQue -> DeQue -> Free。
- [ ] ReduceMax padding 使用负无穷，不使用 0。
- [ ] ReduceSum 使用 FP32 和足够 workspace。
- [ ] SyncAll 前 partial 已写入 GM，且同步区已清零。
- [ ] 尾块搬运满足 32B 对齐规则。
- [ ] Kernel 未完成前不释放 workspace。
- [ ] 不提交标准算子注册代码、main() 或额外工程文件。

