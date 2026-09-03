# axolotl

本目录是 [axolotl](https://github.com/axolotl-ai-cloud/axolotl) 的看护配套数据，不是上游源码。Quick Start 流水线在 [.github/workflows/axolotl-quick-start.yml](../../.github/workflows/axolotl-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：新兴适配；阶段 A）。

当前只落地 Quick Start 线，没有 example 清单和 `*-examples.yml`。

上游没有单独的昇腾 CI。设备选择在 `choose_device()`：没有 CUDA / MPS 时，`is_torch_npu_available()` 为真就用 `npu:{local_rank}`。配置校验 `check_npu_config` 会拒绝 Flash Attention、`sdpa`、带 `bit` 的优化器、`load_in_8bit` / `load_in_4bit` 和 `tf32`。文档因此用 `attn_implementation: eager` 和 `optimizer: adamw_torch`，并关掉会在 CUDA 上分配内存的 `lora_*_kernel`。

`upstream_repo` 指向 `axolotl-ai-cloud/axolotl`。monitor 跟上游 Release 走。文档用隐藏 `#test-setup` 把 `$UPSTREAM_REF` 去掉前导 `v` 后填进 `pip install axolotl==<ver>`。共享引擎仍会把 `result.json` 的 `target_ref` 写成解析到的最新 Release tag。

## Quick Start 看护范围

文档在 `docs/Quick-start-Ascend.md`。方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。

打了 `#test`、会被看护的步骤：

- 检查 Python 版本
- 安装 `torch==2.11.0` / `torch_npu==2.11.0`，断言 `npu_available True`
- 按 `$UPSTREAM_REF` 从 PyPI 安装 `axolotl==<ver>`（`--no-build-isolation --no-deps`）及训练路径依赖，打印版本
- 从 ModelScope 取 `Qwen/Qwen2.5-0.5B-Instruct`，链到 `/root/axolotl-qs/model`
- `axolotl train --launcher python` 跑 3 步。工作负载块的预期输出含设备锚点（`"device": "npu:0"`）和 `Training completed!`

隐藏 `#test-setup`（页面不渲染，看护会跑）：

- 把 `$UPSTREAM_REF` 去掉前导 `v` 后填进 `pip install axolotl==<ver>`
- 把可见 JSONL 写入 `/root/axolotl-qs/tiny_alpaca.jsonl`（用户按正文用编辑器保存同一份）
- 把可见 YAML 写入 `/root/axolotl-qs/lora-npu.yml`（用户按正文用编辑器保存同一份）

无标签、**不看护**的步骤：

- `source set_env.sh` 和 `export PATH`（测试进程在 `prepare_environment` 里做等价的 CANN 注入，并且覆盖写回 `os.environ`）
- `npu-smi info`（设备表每次不同，正文只要求退出码 0）
- 可见的 `tiny_alpaca.jsonl` 内容（无标签 `json` 块，不当命令执行）
- 可见的 `lora-npu.yml` 内容（无标签 `yaml` 块，不当命令执行）
- 上游 README 的 `axolotl[deepspeed]`、`UV_TORCH_BACKEND=cu130`、Flash Attention、8-bit 优化器、QLoRA、多卡 / DeepSpeed

`schedule` 保持注释。`doc_url` 走 GitHub Contents API（`ref=${{ github.sha }}`），不走 `raw.githubusercontent.com`。
