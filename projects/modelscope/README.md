# modelscope

本目录是 [ModelScope](https://github.com/modelscope/modelscope) 的看护配套数据，不是 modelscope 源码。Quick Start 流水线在 [.github/workflows/modelscope-quick-start.yml](../../.github/workflows/modelscope-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

## Quick Start 看护

文档在 `docs/Quick-start-Ascend.md`。流水线是 `.github/workflows/modelscope-quick-start.yml`。文档方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。无标签的 `shell` 块给用户复制，看护跳过。

### 看护范围

- **看护**：Python 版本检查；安装 `torch==2.9.0` + `torch_npu==2.9.0.post2` 并检查 `torch.npu.is_available() == True`；从源码安装 modelscope（克隆上游最新 release tag + `uv pip install -e '.[framework]'` + `transformers<5.0` 上限）；`modelscope download --model Qwen/Qwen2.5-0.5B-Instruct` 拉取权重；用 `AutoModelForCausalLM` 加载模型并 `.to('npu:0')` 在单卡上做一次 64-token 文本生成。
- **不看护**（无标签块）：`npu-smi info`（设备表数值每次不同，由 CI 前置步骤打印）；二进制 `pip install modelscope` 安装方式（用户自选路径）。
- **NPU 设备放置说明**：modelscope 的 `pipeline(..., device=...)` 目前只接受 `cpu` / `cuda` / `gpu`（`modelscope/utils/device.py` 的 `verify_device` 会拒绝 `npu`），所以 NPU 冒烟走 `AutoModelForCausalLM.from_pretrained(...).to('npu:0')`，不走 pipeline。看护目标：modelscope 的 Hub 下载 + transformers 加载 + torch_npu 推理链路在昇腾上能跑通。

### 触发

`modelscope-quick-start.yml` 复用 [quick-start-template.yml](../../.github/workflows/quick-start-template.yml) 引擎。`schedule` 保持注释（接入阶段只留手动 `workflow_dispatch`），首次 dispatch 跑通后再打开。文档 `doc_url` 走 Contents API + `ref=${{ github.sha }}`，与 mooncake / lightx2v 同款。

模型权重缓存：`modelscope download` 落盘在 `~/.cache/modelscope`（默认位置），workflow 的 `container_options` 把宿主机 `/data/ci-cache/modelscope/modelscope` bind-mount 到容器内该目录，跨 run 复用；损坏的 safetensors 分片由 `prepare_environment` 里的 `purge_corrupt_models` 清理后重新下载。
