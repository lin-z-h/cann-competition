# CANNJudge 工具

从仓库根目录运行以下命令。可运行入口留在 `tools/` 顶层；历史脚本已按运行方式放入 `legacy/api/` 和 `legacy/browser/`。后者多数是供浏览器自动化运行器调用的 `async (page) => ...` 片段，不能直接用 `node` 执行。

## 可运行入口

| 脚本 | 用途 | 是否改动线上状态 |
| --- | --- | --- |
| [`submit_api.js`](submit_api.js) | `preview` 本地预览；`check` 查看最新提交；`submit` 用 API 提交指定 `.asc` 文件 | 仅 `submit` 会提交 |
| [`standalone_browser.js`](standalone_browser.js) | `preview` 本地预览；`poll` 查看最新结果；`submit` 通过 Playwright 页面提交指定 `.asc` 文件 | 仅 `submit` 会提交 |
| [`save_login_state.js`](save_login_state.js) | 打开可见浏览器，让用户登录并将状态保存为本地 JSON | 不提交代码 |

Node.js 18 或更新版本可运行 API 入口。浏览器入口及保存登录状态需要 Playwright 和 Chromium；缺少依赖时，可在仓库根目录运行 `npm install --no-save --no-package-lock playwright`，再运行 `npx playwright install chromium`。`node_modules/` 已被 Git 忽略。

`--file` 接受绝对路径，或以仓库根目录为基准的相对路径。`preview` 和 `submit` 都必须写出 `--file`，脚本不会默认读取根目录的可靠版本。文件须为非空 `.asc`；上传给裁判的文件名仍是比赛要求的 `kernel.asc`。`preview` 不需要登录，输出选中文件路径、字节数和 SHA-256；两个提交入口都会在创建提交后回读远端 `kernel.asc`，核对字节数和 SHA-256 后才报告成功。`--state` 同样接受绝对路径或相对仓库根目录的路径，默认是 `.tmp_cannjudge_state.json`。

```powershell
node tools/submit_api.js --help
node tools/submit_api.js preview --file experiments/new_design/kernel.asc
node tools/submit_api.js check --state .tmp_cannjudge_state.json
node tools/submit_api.js submit --file experiments/new_design/kernel.asc --state .tmp_cannjudge_state.json
node tools/standalone_browser.js poll --state .tmp_cannjudge_state.json
node tools/standalone_browser.js submit --file experiments/new_design/kernel.asc --state .tmp_cannjudge_state.json
```

上述 `submit` 命令会消耗一次线上提交次数；先用 `check` 或 `poll` 核对登录状态和最新提交，再确认候选文件与哈希。默认会从登录状态中的 `cannjudge_user` 自动读取当前账号的对象 ID，不会使用历史账号 ID。若状态文件没有该项，可显式传入 `--user-id <当前账号对象ID>`；必要时用 `--problem-id <题目ID>`。

## 请用户提供登录状态文件

登录状态 JSON 包含会话 Cookie，等同于临时登录凭据。**请不要把 JSON 内容、Cookie 或令牌粘贴到聊天中，也不要提交到 Git。** 推荐让用户在自己的电脑上生成文件，然后只告诉协作者本机绝对路径：

1. 在仓库根目录准备 Playwright 和 Chromium（见上方安装命令）。
2. 运行 `node tools/save_login_state.js`。脚本会打开可见的 CANNJudge 页面；用户自行完成登录。
3. 登录完成后回到终端按 Enter。脚本默认保存到仓库根目录的 `.tmp_cannjudge_state.json`，该路径已被 `.gitignore` 忽略。已有文件需输入 `yes` 才会覆盖。
4. 需要在同一台机器协作时，只提供**本机文件路径**，使用自己实际的仓库目录，不沿用其他开发者的路径。若文件保存在别处，运行保存脚本时可指定 `--state <自己的本机路径>`，后续检查和提交也传入同一 `--state`。
5. 先执行 `node tools/submit_api.js check --state .tmp_cannjudge_state.json` 验证状态。脚本会自动识别保存状态时的当前账号；如果状态文件没有账号信息，再在命令中传 `--user-id`。登录过期时重新运行保存脚本。

只共享路径的前提是协作者与用户共用这台机器和工作区；如果不是，应使用双方认可的安全文件传递方式，仍不要在聊天中发送凭据正文。脚本只读取状态文件，不会在正常输出中打印 Cookie。

## 历史脚本归档

- [`legacy/api/`](legacy/api/)：历史评测、源码和提交接口检查脚本，直接用 Node 执行，通常依赖默认登录状态及历史账号 ID。
- [`legacy/browser/`](legacy/browser/)：浏览器运行器片段，包括只读检查、单次提交、批量探针和编辑器操作。`run_files.js` 从 `.tmp_exp/list.txt` 读取文件列表，`submit_one.js` 从 `.tmp_probes/next.txt` 读取文件路径。`paste_and_submit.js`、`submit_direct.js`、`reset_editor.js` 需要运行器环境变量 `CANN_KERNEL_PATH` 指定源码路径，避免意外读取根目录版本。

归档脚本保留历史实验行为，未重新验证当前 CANNJudge 页面与 API。运行批量脚本前，先核对其中的账号、题目、文件列表和次数限制。登录态、浏览器缓存、下载文件和探针只放在被 Git 忽略的临时路径中。
