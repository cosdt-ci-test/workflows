# Ollama CANN 实机验证

Ollama 的 CANN 适配维护在 [lltiaor/ollama](https://github.com/lltiaor/ollama)。
本仓的 [ollama-quick-start](../../.github/workflows/ollama-quick-start.yml) 直接构建 fork 默认分支，并记录实际检出的 Ollama SHA 和对应的 llama.cpp 版本。首轮推送到 `test/ollama-cann-coverage` 时自动运行 `a2`；完成首轮验证后移除推送触发，后续手动选择验证场景。

| Profile | Runner | 验证内容 |
| --- | --- | --- |
| `a2`（默认） | `linux-aarch64-a2-2` | Qwen2.5-0.5B Q4_0：先自动发现 CANN 并选第 2 张可见卡，再打开 `OLLAMA_SCHED_SPREAD` 验证同一模型在两张卡上分层 |
| `a2-formats` | `linux-aarch64-a2-2` | 同一模型的 FP16、Q8_0 分别在单卡上推理 |
| `310p` | `OLLAMA_310P_RUNNER` | 310P 上跑 FP16；需配置真实 310P runner，以及带匹配 CANN 和 Python 3.12 的 `OLLAMA_310P_IMAGE` |

各用例都不设置 `OLLAMA_LLM_LIBRARY`。通过条件包括对话生成、`/api/ps` 报告设备显存、推理子进程的实际可见设备、完整的模型层卸载，以及每张预期卡的非零 CANN 权重缓冲区。完整层卸载不等于所有算子都在 NPU 上执行。

日志及 JSON 证据保存在 runner 挂载的 `/ollama-cache/coverage/reports/<run_id>-<attempt>`；Actions 日志也会打印源码版本与各用例的验证数据。310P 使用 FP16 是因为 [llama.cpp 的 CANN 后端文档](https://github.com/ggml-org/llama.cpp/blob/b11963/docs/backend/CANN.md#datatype-supports)说明该平台的 Q4_0、Q8_0 量化矩阵乘法有限制。

源码安装与运行示例见[快速上手](docs/Quick-start-Ascend.md)。
