# 测试说明

| 文件 | 检查对象 | 运行方式 |
| --- | --- | --- |
| [`reference_test.py`](reference_test.py) | 根目录可靠 `kernel.asc` 的源码契约和 10 组形状 × 四种布局的 NumPy 语义模型 | `python tests/reference_test.py` |
| [`new_design_test.py`](new_design_test.py) | 独立实验版的 A/B/C 调度、分区和数学语义；包括两种输入类型与四布局 | `pytest -q -p no:cacheprovider tests/new_design_test.py` |

两份测试均为 CPU 检查，不会编译 Ascend C，也不能替代目标 NPU 的资源、同步、精度和线上 15 个测试点验证。`reference_test.py` 包含旧源码的字符串断言，不应用它判定新设计候选的正确性。工具路径选择测试位于 [`tools/tests/cli.test.js`](../tools/tests/cli.test.js)，可运行 `node tools/tests/cli.test.js`；它不联网或提交。
