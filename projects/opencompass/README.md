# opencompass

本目录是 [OpenCompass](https://github.com/open-compass/opencompass) 的看护配套数据，不是 OpenCompass 源码。Quick Start 流水线在 [.github/workflows/opencompass-quick-start.yml](../../.github/workflows/opencompass-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

当前只落地 Quick Start 线，没有 example 清单和 `*-examples.yml`。Quick Start 绿灯只表示 `docs/Quick-start-Ascend.md` 这条原生 Hugging Face + TorchNPU 路径通过，不表示上游 `examples/` 已被看护。

上游默认分支是 `main`。仓库没有健康的昇腾 CI。原生 `HuggingFacewithChatTemplate` 在 `mmengine.device.is_npu_available()` 为真时把 `device_map` 设为 `npu`。本文不覆盖 MindIE、LMDeploy 或 vLLM-Ascend。共享引擎把 `result.json` 的 `target_ref` 写成解析到的最新 Release tag（撰写时 `0.5.4`）。文档可见命令用 `OPENCOMPASS_REF`，默认值也是 `0.5.4`；测试类把引擎注入的 `UPSTREAM_REF` 映射为 `OPENCOMPASS_REF`。

## Quick Start 看护范围

文档在 `docs/Quick-start-Ascend.md`。方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。

打了 `#test`、会被看护的步骤：

- 检查 Python 版本
- 安装 `torch==2.9.0` / `torch_npu==2.9.0.post2`，断言 `npu_available True`
- 按 `OPENCOMPASS_REF` 克隆 OpenCompass 并 `pip install -e . --no-deps`，再按过滤后的 `requirements/runtime.txt` 装导入期依赖（跳过 `torch` 和 GitHub `master` 上的 `rouge_chinese`，`transformers` 钉 4.x）；断言 `opencompass` 版本且 `npu_available True`
- 从 ModelScope 下载 `Qwen/Qwen2-0.5B-Instruct`，断言目录里有 `config.json`
- 在源码根目录写入 `npu_chat.py` / `eval_qwen2_gsm8k.py`，用 `demo_gsm8k_chat_gen`（64 条）+ `--debug` 单进程评测。工作负载块的预期输出含 `opencompass_model_device npu:0`，以及 summary 里的 `demo_gsm8k` / `accuracy` / `gen`

`#test-setup`、会被执行但不比对的步骤：

- 确认 `ASCEND_HOME_PATH` 与 `npu-smi` 可用
- 下载模型并 `store` 本地路径，供后面的评测配置替换

无标签、**不看护**的步骤：

- `source set_env.sh` 和 `export PATH`（测试进程在 `prepare_environment` 里做等价的 CANN 注入）
- `npu-smi info`（设备表每次不同，正文只要求退出码 0）
- MindIE / LMDeploy / vLLM-Ascend 以及多卡数据并行

隐藏 `#test-setup` 只做 CI 跨 run 复用：从 `/root/.cache/cosdt-ci-test/opencompass/` 校验并预置源码树，评测结束后原子写回。可见克隆命令仍写官方 GitHub URL。

`schedule` 保持注释。`doc_url` 走 GitHub Contents API（`ref=${{ github.sha }}`），不走 `raw.githubusercontent.com`。
