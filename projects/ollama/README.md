# Ollama Ascend 快速入门接入与验证状态

本项目从用户维护的 [lltiaor/ollama](https://github.com/lltiaor/ollama) 获取共享模板解析出的当前源码，再构建 CANN 后端并完成昇腾 NPU 验收。快速入门直接安装这份源码，不再下载或应用补丁。

[用户文档](docs/Quick-start-Ascend.md)、[测试夹具](tests/test_quick_start_ascend.py)和手动 [workflow](../../.github/workflows/ollama-quick-start.yml) 共同使用同一个源码版本。

## 源码版本契约

| 项目 | 当前值 |
|---|---|
| 源码仓库 | `lltiaor/ollama` |
| 源码分支 | `main` |
| 被测 ref | 共享模板通过 `UPSTREAM_REF` 注入的 fork `main` 当前提交 |
| 安装版本 | 使用同一个 `UPSTREAM_REF` 作为 `OLLAMA_VERSION` |
| 当前核对快照 | 提交 `d28e26ba`，Ollama `v0.40.14`，llama.cpp `b11963` |
| Go 最低版本 | `1.26.10` |

`d28e26ba` 只是撰写时的核对快照，不是 quick-start 的固定输入。fork 没有 release/tag 时，共享模板按 `/commits/HEAD` 解析 `main` 当前 SHA；文档、构建版本和测试断言都会使用这个 SHA。

适配提交在 Ollama v0.40.14 `main` 上修改四个生产文件，共新增 25 行：

| 文件 | 接入内容 |
|---|---|
| `cmake/local.cmake` | 将 `ggml-cann` 加入 llama-server 后端构建与安装 |
| `discover/llama_server.go` | 将 `CANN0` 等设备名归类为 `CANN` |
| `discover/runner.go` | 从 `ASCEND_RT_VISIBLE_DEVICES` 读取可见设备列表 |
| `ml/device.go` | 将调度选中的设备写入推理子进程环境 |

CANN 算子实现复用该提交固定的 llama.cpp version，不复制旧 PR 的后端源码。仓库中的 `patches/ollama-cann.patch` 保留为历史资料，快速入门和测试夹具均不再使用。

## 快速入门行为

文档显式执行以下步骤：

1. 读取共享模板注入的 `UPSTREAM_REF`。
2. 从 `lltiaor/ollama` fetch 并 checkout 该 ref。
3. 使用 `OLLAMA_LLAMA_BACKENDS=cann` 构建 CANN 后端。
4. 启动前台服务，导入固定的 Qwen2.5-0.5B-Instruct Q4_0 GGUF 模型。
5. 分别完成 CLI 与 `/api/chat` 对话。

测试夹具会在源码获取后校验：

- origin 必须是 `https://github.com/lltiaor/ollama.git`。
- HEAD 必须等于模板注入的 `UPSTREAM_REF`。
- 构建时 `OLLAMA_VERSION` 必须使用同一个 `UPSTREAM_REF`，服务版本也按它校验。
- `LLAMA_CPP_VERSION` 必须非空并记录在报告中。
- 四个生产文件中必须存在对应的 CANN 接入。

构建后先检查 `libggml-cann.so`、`llama-server` 和 Ollama 可执行文件，再通过 `llama-server --list-devices --offline` 检查 CANN 设备。推理验收要求非空回复、正数生成 token、`/api/ps` 中的正数设备显存、正数 CANN model buffer 和完整模型层卸载；CPU 回退不算通过。

## Workflow

手动 workflow 使用 `upstream_repo: lltiaor/ollama`，不设置 `fixed_ref`。fork 没有 release/tag 时，共享引擎按兜底顺序解析 `main` 当前提交 SHA 并通过 `UPSTREAM_REF` 注入；源码 checkout、`OLLAMA_VERSION` 和结果 artifact 使用同一个值。

workflow 继续使用 `linux-aarch64-a2-2` runner 和 CANN `9.1.0-910b-ubuntu22.04-py3.12` 镜像，总预算 240 分钟。当前仅启用 `workflow_dispatch`。

## 验收记录

目前已完成源码版本、补丁外置取消、Linux 顶层构建配置和测试夹具的静态检查。原生构建、服务加载和 NPU 推理仍以 GH Actions 的实机结果为准；没有上卡证据时，不将源码兼容性检查视为推理成功。

## 参考资料

- [Ollama 昇腾适配 fork](https://github.com/lltiaor/ollama/tree/main)
- [Ollama 开发构建文档](https://github.com/ollama/ollama/blob/v0.40.14/docs/development.md)
- [llama.cpp CANN 后端](https://github.com/ggml-org/llama.cpp/blob/b11963/docs/backend/CANN.md)
- [Ollama 导入 GGUF 模型](https://docs.ollama.com/import)
- [Ollama 本地快速入门](https://docs.ollama.com/quickstart)
