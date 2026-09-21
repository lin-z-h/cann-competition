# BatchMatmulMaxSum

昇腾 CANNJudge 初赛算子工程。输入为 FP16/BF16 的 query、document embedding，按 `BatchMatMul → Max(N) → Sum(M)` 输出每个 batch 的 FP32 分数；四种输入存储布局由 `transposeX1`、`transposeX2` 指定。完整约束见[赛题说明](docs/spec/problem_official.md)。

## 当前可用版本

仓库根目录的 [`kernel.asc`](kernel.asc) 是线上提交源码。已记录的可靠提交为 **384543，15/15 Pass**，按 LF 换行计算的 SHA-256 是 `da2f4f2be9c95c06e98c406f19674f0276cf7f4ac69372d1192c8f44a1ed312b`。按官方评分公式重算约 20.47 分；这是历史验证结果，仓库整理本身未重新进行线上评测。不要将 [`data/key_kernels.json`](data/key_kernels.json) 中的实验候选直接当作可靠版本。

## 目录

| 路径 | 内容 |
| --- | --- |
| `kernel.asc` | CANNJudge 提交的单文件实现；保持根目录和 LF 换行 |
| `main.asc`、`data_utils.h`、`CMakeLists.txt` | 本地 CANN 示例入口和构建配置，不属于线上提交文件 |
| [`docs/`](docs/README.md) | 赛题、API 资料、设计、计划和实验记录 |
| [`data/`](data/README.md) | 历史评测结果和候选源码归档 |
| [`tests/reference_test.py`](tests/reference_test.py) | NumPy 语义模型及当前源码的静态约束检查 |
| [`tools/`](tools/README.md) | 历史 CANNJudge 浏览器/API 辅助脚本 |

## 本地检查与构建

安装 NumPy 后，在仓库根目录运行 `python tests/reference_test.py`。此检查覆盖 10 组形状和四种存储布局，但不能代替 NPU 编译、精度检查或线上 15 个 case。具备 CANN 9.0.0 环境时，可运行 `cmake -S . -B build` 和 `cmake --build build -j4`；Windows 当前工作环境未配备 CANN/NPU，因此这里没有本地编译结论。

线上模板仅替换 `kernel.asc`。每次实验先确认待提交源码哈希和可用提交次数；评测后将提交 ID、15 项结果与结论记录到[实验记录](docs/history/memory.md)。登录态、cookie、浏览器缓存与临时探针不得入库。历史脚本有账号、登录态和运行环境依赖，使用前阅读[工具说明](tools/README.md)。
