# 文档导航

| 文档 | 用途 |
| --- | --- |
| [开发交接](../HANDOFF.md) | 2026-09-30 最终状态、调优方法、未解决问题与恢复步骤 |
| [CANNLab SSH 上手](cannlab_remote.md) | 每位协作者自行创建环境、连接 VS Code、查配置并使用 SSH |
| [赛题说明](spec/problem_official.md) | 计算语义、输入布局、精度和评分规则；来自比赛题面的整理 |
| [CANN API 参考](research/cann_api_reference.md) | 当前实现涉及的接口与使用约束 |
| [新设计](research/new_design.md) | 架构研究方案；原候选已有线上通过记录，CANNLab 续测另有 N 尾块失败与独立修复 |
| [实验计划](research/plan.md) | 2026-09 的历史优化计划与尝试状态；不能直接当作当前待办清单 |
| [实验记录](history/memory.md) | 提交结果、已知失败和可靠基线的追溯 |
| [CANNLab 2026-09-30 续测](history/cannlab_20260930_session.md) | 实际设备编译、精度、单次 kernel 与 msprof 调优证据 |
| [CANNJudge 2026-09-30 提交](history/cannjudge_20260930_tuning.md) | 两份候选线上 15/15、上传哈希与分数下降 |

接手先读[开发交接](../HANDOFF.md)和[仓库首页](../README.md)，再查赛题、实验记录、设备证据和设计。远端连接按[SSH指南](cannlab_remote.md)使用每位协作者自己的环境。计划与历史记录中的“当前”“今日”以各段日期为准；CPU模型、线上评测和NPU验证不能相互替代。
