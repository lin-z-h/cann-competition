# 项目协作说明

- 本项目实现 BatchMatmulMaxSum：FP16/BF16 输入、四种布局、FP32 输出；单次运行只启动一个全局计算 kernel。
- 根目录 `kernel.asc` 是可靠基线，哈希见 [README](README.md)。实验放在 `experiments/`，未经明确要求不覆盖可靠版本。
- 接手先读 [HANDOFF](HANDOFF.md)；赛题、设计与历史证据见 [docs](docs/README.md)。历史状态以记录日期为准。
- 修改算子后运行对应 CPU 测试；分别记录设备编译、精度、单次 kernel、msprof 和线上评测，模型测试不能替代设备验证。
- 远端环境由每位协作者自行创建。按 [SSH 指南](docs/cannlab_remote.md) 读取本机最新配置并使用自己的完整 Host；不沿用他人的 Host、端口、设备或路径。
- 远端可连接时直接通过 SSH 开发、同步与测试；同步保留远端 `.git/`、构建输出、输入输出及忽略状态。
- 提交工具用 `--file` 显式选择源码；线上提交须有用户授权。凭据、SSH 配置、日志和临时产物不入库。
- 尽量使用中文写说明，保留必要的代码/API 英文标识。
