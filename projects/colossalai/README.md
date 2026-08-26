# colossalai

本目录是 [ColossalAI](https://github.com/hpcaitech/ColossalAI) 的看护配套数据，不是 ColossalAI 源码。Quick Start 流水线在 [.github/workflows/colossalai-quick-start.yml](../../.github/workflows/colossalai-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

当前只落地 Quick Start 线，没有 example 清单和 `*-examples.yml`。

上游默认分支是 `main`。仓库近乎停滞，README 只写 CUDA 要求。`colossalai/accelerator/npu_accelerator.py` 提供 HCCL 后端，但 `requirements.txt` 钉 `torch>=2.2.0,<=2.5.1`，和 CANN 9.1 镜像上的 `torch==2.9.0` 冲突。文档因此用 `pip install colossalai==0.5.0 --no-deps`，再单独安装 Booster 的导入期依赖。共享引擎仍会把 `result.json` 的 `target_ref` 写成解析到的最新 Release tag（撰写时 `v0.5.0`），与 PyPI 版本一致。

## Quick Start 看护范围

文档在 `docs/Quick-start-Ascend.md`。方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。

打了 `#test`、会被看护的步骤：

- 检查 Python 版本
- 安装 `torch==2.9.0` / `torch_npu==2.9.0.post2` 并断言 `npu_available True`
- 安装 `colossalai==0.5.0 --no-deps` 以及 `transformers` / `peft` / `galore_torch` / `bitsandbytes` / `einops` / `modelscope`，并打印 `accel_name npu` / `accel_device npu:0`
- 用 `launch` + `Booster(plugin=TorchDDPPlugin())` 对 Qwen2.5-0.5B 做一步训练。工作负载块的预期输出含 `boosted_param_device npu:0`

无标签、**不看护**的步骤：

- `source set_env.sh` 和 `export PATH`（测试进程在 `prepare_environment` 里做等价的 CANN 注入）
- `npu-smi info`（设备表每次不同，正文只要求退出码 0）

`schedule` 保持注释。`doc_url` 走 GitHub Contents API（`ref=${{ github.sha }}`），不走 `raw.githubusercontent.com`。
