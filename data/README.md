# 历史数据

本目录保存此前从 CANNJudge 评测中整理的**静态快照**，用于追溯和比较；它们不会随网站成绩自动更新，也不包含登录凭据。

| 文件 | 内容 | 使用方式 |
| --- | --- | --- |
| [`history_results.json`](history_results.json) | 40 条历史提交摘要；每条含提交 ID、时间、状态、通过数，以及按测试点顺序记录的精度和耗时 | 对照[实验记录](../docs/history/memory.md)分析趋势；不要把单次波动当作优化收益 |
| [`key_kernels.json`](key_kernels.json) | 5 份历史候选源码的键值归档；值为源码文本 | 仅供比对失败与成功实现；键名里的 `pass` 是当时记录，不代表当前可提交 |

测试点数组按原评测返回顺序保存；文档中的 case 编号采用从 0 开始的口径。JSON 没有设备配置、实际 tiling 或完整编译日志，不能据此推断隐藏测试点的形状。

当前可靠源码以仓库根目录的 [`kernel.asc`](../kernel.asc) 和[仓库首页](../README.md)中的提交 ID、SHA-256 为准。新的未验证候选在 [`experiments/new_design/`](../experiments/new_design/README.md)，这里的归档不应作为自动提交来源。
