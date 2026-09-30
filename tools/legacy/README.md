# 历史工具归档

本目录保留早期 CANNJudge 实验脚本，便于复核[实验记录](../../docs/history/memory.md)。它们不是当前推荐的提交入口；请选择上级目录的 [`submit_api.js`](../submit_api.js) 或 [`standalone_browser.js`](../standalone_browser.js)，用 `--file` 明确指定源码。

- [`api/`](api/)：Node 脚本，读取历史提交、比较源码或检查网页接口。默认从仓库根目录寻找 `.tmp_cannjudge_state.json`，部分脚本含历史账号与题目 ID。
- [`browser/`](browser/)：以 `async (page) => ...` 编写的浏览器运行器片段，不能直接运行 `node 文件名.js`。其中 `submit_*`、`run_*`、`paste_and_submit.js` 会改变编辑器或发起线上提交；`audit_*`、`inspect_*`、`poll_*` 等主要用于读取。

归档位置变化没有重新验证旧片段的网页选择器、账号信息或提交行为。详情见[工具总览](../README.md)。
