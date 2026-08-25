# diffsynth-studio

本目录是 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 的看护配套数据，不是 DiffSynth-Studio 源码。Quick Start 流水线在 [.github/workflows/diffsynth-studio-quick-start.yml](../../.github/workflows/diffsynth-studio-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类与支持程度仍是 `TODO`，待与项目表核对；阶段 A）。

本批只落地 Quick Start 线，没有 example 清单和 `*-examples.yml`。

上游默认分支是 `main`。上游完全没有昇腾 CI（`.github/workflows/` 只有 `publish.yaml`）。最新 GitHub Release 是 `v1.1.9`，该 tag 下没有 `diffsynth/core/device/`。文档从 PyPI 安装当前的 `diffsynth`（撰写时 `2.1.3`），不 checkout `v1.1.9`。共享引擎仍会把 `result.json` 的 `target_ref` 写成解析到的最新 Release tag，和实际安装的 PyPI 版本不一致。这是本仓已有项目同样存在的行为。

## Quick Start 看护范围

文档在 `docs/Quick-start-Ascend.md`。方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。

打了 `#test`、会被看护的步骤：

- 检查 Python 版本
- 安装 `torch==2.9.0` / `torch_npu==2.9.0.post2` 并断言 `npu_available True`
- 安装 PyPI 上的 `diffsynth`（不带 `[npu_aarch64]` extra），并打印 `device_type npu` / `device_name npu:0`
- 用 Stable Diffusion 1.5 在 NPU 上生成一张 `512x512` 图（`num_inference_steps=5`）。工作负载块的预期输出含 `unet.device npu:0`

无标签、**不看护**的步骤：

- `source set_env.sh` 和 `export PATH`（测试进程在 `prepare_environment` 里做等价的 CANN 注入）
- `npu-smi info`（设备表每次不同，正文只要求退出码 0）

`schedule` 保持注释。`doc_url` 走 GitHub Contents API（`ref=${{ github.sha }}`），不走 `raw.githubusercontent.com`。
