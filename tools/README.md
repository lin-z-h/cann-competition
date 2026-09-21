# CANNJudge 辅助脚本

这些是历史实验时使用的脚本，按用途分为：`audit_*`、`inspect_*`、`compare_*`、`recover_*`、`dump_*` 用于读取和比较；`submit_*`、`run_*`、`paste_and_submit.js`、`reset_editor.js` 会修改编辑器或发起提交；`standalone_browser.js` 是独立 Playwright 入口。部分脚本是供浏览器自动化运行器调用的 `async (page) => ...` 片段，不能直接用 `node` 执行。

使用条件：

1. 从仓库根目录运行需要读取本地文件的脚本；`kernel.asc` 与本地登录态 `.tmp_cannjudge_state.json` 均按该目录解析。
2. 自行登录 CANNJudge，并核对脚本中的用户 ID、题目 ID 和网页/API 地址；这些值是历史实验时的账号与赛题标识。
3. 提交类脚本会消耗线上次数。执行前核对候选源码哈希、编辑器回读内容以及当日剩余次数。
4. 本地登录态、cookie、下载文件和探针只放在被 `.gitignore` 忽略的临时路径中，不提交到 Git。

脚本保留了历史实验流程，其中部分依赖特定浏览器运行器及其页面 API；目录整理仅调整文件位置与固定本机路径，未重新验证线上接口。优先用 [`submit_api.js`](submit_api.js) 的 `check` 模式检查当前状态，再人工复核任何写入动作。
