# CANNLab 远端 SSH 上手

本指南适用于每位协作者自己的环境，包括新建 A3 环境。不要使用其他开发者的环境名、Host、身份文件或临时本机端口；SSH 配置只在各自本机保存。

## 1. 创建环境并连接 VS Code

1. 在 CANNLab 新建自己的运行环境，选择需要的设备，例如 A3。
2. 环境启动后点击“连接”，选择 VS Code，按平台提示完成连接。
3. 在 VS Code 中打开该环境的远程工作区，确认看到远端项目目录和终端。
4. **开发期间保持 VS Code 开启并保持远程连接。** 本机 SSH 转发依赖平台连接服务，关闭 VS Code 或断开远程连接可能使 SSH 不再可用。

## 2. 找到本机生成的配置

在 Windows 本机 PowerShell 执行：

```powershell
type $env:USERPROFILE\.atomgitdevenv\.ssh\config
```

也可以使用 `Get-Content "$env:USERPROFILE\.atomgitdevenv\.ssh\config"`。从输出中找到对应**自己刚创建环境**的完整 `Host` 名称。配置可能带有代理/`forward-bootstrap` 逻辑；它负责确认或建立 SSH 转发，应通过原配置使用。

配置中的 `127.0.0.1` 和端口是临时转发地址，会随连接变化；不要将它们写死在项目、Codex 指令或长期连接命令中。不要把配置全文、密钥或令牌复制进仓库或聊天。

## 3. 使用自己的 Host

以下命令只在当前 PowerShell 会话定义变量。将占位值替换为刚才读取的完整 Host：

```powershell
$cannSshConfig = Join-Path $env:USERPROFILE '.atomgitdevenv\.ssh\config'
$cannRemoteHost = '<自己的完整 Host>'
ssh -F $cannSshConfig $cannRemoteHost
```

进入远端后先检查实际环境，不假设用户名、系统、设备架构和安装路径与历史机器相同：

```bash
pwd
uname -m
command -v npu-smi
# 若驱动库尚未初始化，先按当前环境的说明加载环境，再运行 npu-smi info。
```

确认工作区路径后 clone 项目或同步源码。项目可放在 `/mnt/workspace/cann-competition`，但这是建议目录，不是所有环境已有的目录。

## 4. 同步并通过 SSH 工作

仍在本机 PowerShell，设置自己实际使用的项目路径：

```powershell
$cannRemoteProject = '/mnt/workspace/cann-competition'
scp -F $cannSshConfig .\main.asc "${cannRemoteHost}:${cannRemoteProject}/"
ssh -F $cannSshConfig $cannRemoteHost "cd '$cannRemoteProject' && pwd"
# 下载证据到本地已有的忽略目录，路径按实际采集位置替换。
scp -r -F $cannSshConfig "${cannRemoteHost}:${cannRemoteProject}/build/profile-example" .\build\
```

上传目录时给 `scp` 加 `-r`。整仓同步应排除 `.git/`、`build/`、`input/`、`output/`、`node_modules/`、凭据及运行状态，保留远端已有内容；不要用镜像删除或 ZIP 覆盖工作区。常规增量更新可只上传改动文件。

本机普通命令不会自动在 NPU 上执行。设备构建、运行和采集必须在远端终端中，或通过 `ssh -F ...` 显式执行。远端 Bash 的 `$变量` 与 PowerShell 的变量展开不同；复杂多行任务优先上传不含凭据的脚本，再用 SSH 调用。

## 5. 配置 CANN 与设备验证

先查明当前机器的 `set_env.sh`、编译器与设备对应的 `NPU_ARCH`。本项目历史验证使用 A2/CANN 9.0.0、`dav-2201`，这些不是 A3 默认值。不要仅按设备显示名猜编译目标，应以当前 CANN/平台配置为准。

远端可设置：

```bash
export CANN_ENV_SCRIPT='<当前环境实际 set_env.sh 路径>'
export NPU_ARCH='<当前设备对应的编译目标>'
source "$CANN_ENV_SCRIPT"
```

随后按 [HANDOFF](../HANDOFF.md) 运行 CPU、设备精度及 profiler 验证。已有脚本保留历史 A2 的路径回退；新环境须显式覆盖配置，确认编译、设备运行和输出，再引用性能数据。

## 6. 连接失败时

- 先确认环境仍运行、VS Code 仍连接并已打开远程工作区。
- 再读取本机最新 SSH 配置，确认使用正确完整 Host；环境重建可能改变 Host。
- 继续使用配置的代理逻辑，不绕过它去固定旧端口。
- 只有连接确实不可用时才请求环境所有者恢复连接；连接正常后可由 Codex 直接完成同步、构建和测试，无需手工搬文件。

每位开发者只管理自己的连接。项目文档保留通用方法，不记录个人配置。
