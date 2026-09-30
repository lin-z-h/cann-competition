# BatchMatmulMaxSum

CANNJudge 昇腾算子工程：计算 `BatchMatMul → Max(N) → Sum(M)`，支持 FP16/BF16 输入、四种存储布局和每个 batch 的 FP32 输出。单次运行只启动一个全局计算 kernel，完整约束见[赛题说明](docs/spec/problem_official.md)。

## 接手与远端开发

先读 [HANDOFF.md](HANDOFF.md)，了解截至 2026-09-30 的验证结果、调优方法与未解决问题；简短协作规则见 [AGENTS.md](AGENTS.md)。

每位协作者自行创建 CANNLab 环境，例如 A3。按[远端 SSH 上手指南](docs/cannlab_remote.md)连接 VS Code、打开远程工作区、读取自己本机生成的 SSH 配置，并保持 VS Code 开启。仓库不指定个人 Host、临时转发端口或连接凭据。A2 的已有结果不能直接当作 A3 的验证结果。

## 可靠基线与实验状态

根目录 [`kernel.asc`](kernel.asc) 保持历史可靠源码，规范化 LF 后 SHA-256 为：

```text
da2f4f2be9c95c06e98c406f19674f0276cf7f4ac69372d1192c8f44a1ed312b
```

历史提交 `384543` 为 15/15 Pass，按官方公式重算约 20.47 分。今天读取的同源码提交 `522426` 同样 15/15，重算约 20.85 分；它不是本轮新发起的提交。两份新候选结果如下，分数均为详情重算值，并非排行榜实得字段：

| 版本 | 提交 ID | 线上结果 | 重算分数 |
| --- | ---: | --- | ---: |
| N 尾块修复及 FP16 对齐消费快路径 | 530365 | 15/15 Pass | 20.10 |
| 同 kernel 内整理转置 A | 531572 | 15/15 Pass | 20.01 |

候选在受控设备形状上有局部收益，但线上总分未超过可靠基线。扩展测试另发现多 batch NaN 和 K=16 转置参数问题，修复尚未完成验证；因此未替换根目录源码。详见[线上记录](docs/history/cannjudge_20260930_tuning.md)和[设备记录](docs/history/cannlab_20260930_session.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `kernel.asc` | 可靠的单文件 CANNJudge 提交源码 |
| `main.asc`、`data_utils.h`、`CMakeLists.txt` | 设备调试入口与构建配置，不是线上提交文件 |
| [HANDOFF.md](HANDOFF.md) | 最新交接、调优过程和恢复步骤 |
| [docs/](docs/README.md) | 赛题、API、设计、远端指南与历史证据 |
| [experiments/new_design/](experiments/new_design/README.md) | 原始新设计候选，保留历史版本 |
| [experiments/cannlab_tuning/](experiments/cannlab_tuning/README.md) | N 尾块修复与消费路径实验 |
| [experiments/cannlab_pack_a/](experiments/cannlab_pack_a/README.md) | 转置 A 整理、多 batch 修复及诊断实验 |
| [tests/](tests/README.md) | CPU 数学、调度和源码契约检查 |
| `scripts/` | 设备用例、独立参考校验及 msprof 采集 |
| [tools/](tools/README.md) | 显式选文件的线上检查、提交与登录工具 |
| [data/](data/README.md) | 历史结果与候选归档，不代表当前可靠版本 |

## 检查与构建

本地 CPU 检查需要 NumPy，模型测试另需 pytest：

```bash
python tests/reference_test.py
python -m pytest -q -p no:cacheprovider tests/new_design_test.py
node tools/tests/cli.test.js
```

CPU 检查不验证 Ascend C 编译、设备同步或性能。远端先确认自己的 CANN 环境及目标架构，再初始化环境和构建；以下变量须由当前环境确定，不能照搬历史 A2 配置：

```bash
source "$CANN_ENV_SCRIPT"
cmake -S . -B build/reliable -DNPU_ARCH="$NPU_ARCH"
cmake --build build/reliable -j4
```

默认包含可靠 `kernel.asc`。实验构建通过 `-DCANN_KERNEL_FILE=experiments/.../kernel.asc` 显式选文件。

`scripts/cannlab_smoke.sh` 生成 23 个确定性设备用例并先测试原新设计，再测试可靠版；原新设计可能因已知 N 尾块错误提前停止。`scripts/cannlab_verify.py --random` 在已准备的目录追加 80 个随机用例，共 103 个，用量化后的实际输入计算 NumPy FP64 参考值。`--random-shape B,M,N,K` 可追加边界形状。

`scripts/cannlab_tune.sh` 集中环境检查、CPU 模型、构建、设备精度及 profiler 检查。运行前设置 `CANN_ENV_SCRIPT` 和 `NPU_ARCH`，准备用例目录；可用 `CANNLAB_KERNEL_FILE` 显式选择候选。其历史默认路径和架构来自 A2，A3 首次运行须覆盖并验证。每次 profiler 使用新目录，逐次校验精度与单计算 kernel；完整步骤见 [HANDOFF](HANDOFF.md)。

## 线上工具与凭据

工具通过 `--file` 上传指定源码，线上文件名仍为 `kernel.asc`，不会覆盖本地可靠版本：

```powershell
node tools/submit_api.js preview --file experiments/cannlab_tuning/kernel.asc
node tools/submit_api.js check --state .tmp_cannjudge_state.json
# 获得提交授权后再执行 submit；先检查源码是否含平台拒绝的诊断代码。
```

登录状态由 `node tools/save_login_state.js` 在本机保存，默认相对仓库根目录的 `.tmp_cannjudge_state.json`；网页登录不一定刷新此文件。Cookie、令牌、SSH 配置不入库、不贴到聊天。具体登录和提交说明见[工具文档](tools/README.md)。

`build/`、输入输出、依赖、运行日志和临时登录状态由 Git 忽略。整理仓库或获得一次线上通过，不代表新的候选已经适合替换基线。
