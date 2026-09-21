# BatchMatmulMaxSum 协作说明

`kernel.asc` 是当前可靠的提交源码：对应线上提交 `384543`，15/15 Pass，SHA-256 为
`da2f4f2be9c95c06e98c406f19674f0276cf7f4ac69372d1192c8f44a1ed312b`。
它在一个 global kernel 内完成 FP16/BF16 输入的 FP32 点积、沿 N 的 Max 和沿 M 的 Sum，
支持四种 storage shape，不使用非确定性原子归约。当前得分约 20.46；30 分是阶段目标。

- `problem_official.md`：赛题规则摘录；`plan.md`：优化实验与止损条件；`memory.md`：提交记录和已知陷阱。
- `tests/reference_test.py`：本地 NumPy 语义/源码约束检查，执行 `python tests/reference_test.py`。**它不代替 NPU 编译和线上 15 case。**
- `history_results.json`、`key_kernels.json`：历史提交结果及候选源码归档，用于比较和追溯，不能视为当前可靠版本。
- `.mcp_tools/`：浏览器上传、提交详情读取、源码 diff 和历史审计脚本。脚本里有本机路径和本人用户 ID，队友使用前需修改；涉及提交的脚本务必先核对候选 SHA 和每日剩余次数。
- `cann_api_reference.md`、`new_design.md`：队友在远端新增的 API/新设计资料。

本地 CANN 9.0.0 环境可运行 `cmake -S . -B build && cmake --build build -j4`；
Windows 当前环境没有 NPU/CANN 编译器，因此实际编译、正确率和性能结论均以 CANNJudge 为准。
线上模板只应替换 `kernel.asc`。一次改动只试一个机制，提交后将 ID、15 项结果、结论写入 `memory.md`；
不把 15 项全 0 当成普通性能退化，也不在 Wrong Answer 上叠加优化。

登录态 `.tmp_cannjudge_state.json`、浏览器缓存、下载的官方样例和临时探针故意不入库。
每位队友自行登录 CANNJudge；不要提交 cookie、访问令牌或含敏感信息的运行日志。
