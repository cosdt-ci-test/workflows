# Quick Start (Ascend NPU)

10 分钟在单卡昇腾 NPU 上对 Qwen3-4B-Instruct-2507 进行自我认知微调。本文档结构与默认 [Quick Start](./Quick-start.md) **1:1 对齐**（训练 / 推理 / 推送 3 段），只是把 GPU 换成昇腾 NPU；命令/参数照搬 CUDA 版，环境变量与推理后端替换为 NPU 等价项。
## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。本文档示例使用单卡。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

完整 NPU 适配（镜像选择、DDP/DeepSpeed/MindSpeed、`vllm-ascend` 部署等）请参考 [NPU 最佳实践文档](../BestPractices/NPU-support.md)。

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B3，64 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

```
swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.3.rc2-910b-ubuntu22.04-py3.11
```

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11 |
| CANN | 8.3.rc2 |
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| transformers | `<5.0` |
| peft | `<0.19` |
| modelscope | 1.37.0 |
| ms-swift | 本仓库当前 main 分支源码（`pip install -e .`） |
| 模型 | [Qwen/Qwen3-4B-Instruct-2507](https://www.modelscope.cn/Qwen/Qwen3-4B-Instruct-2507) |
| 数据集 | `AI-ModelScope/alpaca-gpt4-data-zh#500` + `AI-ModelScope/alpaca-gpt4-data-en#500` + `swift/self-cognition#500` |
| 推理后端 | vLLM-Ascend（参数 `--infer_backend vllm`，在 NPU 上自动走 vllm-ascend） |
| 部署 | `swift deploy`（OpenAI 兼容接口，**由 NPU 最佳实践文档负责**，本文档不演示） |

### 检查前置是否满足

检查依赖组件版本
```shell
>>> python --version
Python 3.11.x
>>> python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: xxx
count: xxx
```

确认能看到 NPU 设备：

```shell
npu-smi info
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动；如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

---

## 安装 ms-swift

使用pip进行安装：

```shell
pip install ms-swift -U
pip install uv
uv pip install ms-swift -U --torch-backend=auto
```

从源码安装：

```shell
# Clone the upstream ms-swift repo into ./ms-swift
>>> git clone https://github.com/modelscope/ms-swift.git
# Pin the repo to the exact ref/SHA that triggered this CI run
>>> cd ms-swift && git checkout <UPSTREAM_REF>
# Editable install: `swift` CLI is now on PATH, source changes take effect on rerun
>>> cd ms-swift && uv pip install -e . --torch-backend=auto
# Sanity check the install: should print the installed ms-swift version
>>> python -c "import swift; print('ms-swift', swift.__version__)"
ms-swift xxx
```

`<UPSTREAM_REF>`： 改成实际版本。

## 使用样例

10 分钟在单卡昇腾 NPU 上对 Qwen3-4B-Instruct-2507 进行自我认知微调：

```shell
>>> ASCEND_RT_VISIBLE_DEVICES=0 swift sft \
...     --model Qwen/Qwen3-4B-Instruct-2507 \
...     --tuner_type lora \
...     --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
...               'AI-ModelScope/alpaca-gpt4-data-en#500' \
...               'swift/self-cognition#500' \
...     --torch_dtype bfloat16 \
...     --num_train_epochs 1 \
...     --per_device_train_batch_size 1 \
...     --per_device_eval_batch_size 1 \
...     --learning_rate 1e-4 \
...     --lora_rank 8 \
...     --lora_alpha 32 \
...     --target_modules all-linear \
...     --gradient_accumulation_steps 16 \
...     --eval_steps 50 \
...     --save_steps 50 \
...     --save_total_limit 2 \
...     --logging_steps 5 \
...     --max_length 2048 \
...     --output_dir output \
...     --warmup_ratio 0.05 \
...     --dataloader_num_workers 4 \
...     --model_author swift \
...     --model_name swift-robot
[INFO:swift] start training ...
{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}
...
{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}
...
[INFO:swift] training finished, saving checkpoint ...
Saving checkpoint to output/vx-xxx/checkpoint-xxx
>>> echo "train done, checkpoint dir: $(ls -dt output/*/checkpoint-* | head -n 1)"
train done, checkpoint dir: output/vx-xxx/checkpoint-xxx
```

小贴士：

- 如果要使用自定义数据集进行训练，你可以参考[这里](../Customization/Custom-dataset.md)组织数据集格式，并指定 `--dataset <dataset_path>`。
- `--model_author` 和 `--model_name` 参数只有当数据集中包含 `swift/self-cognition` 时才生效。
- 如果要使用其他模型进行训练，你只需要修改 `--model <model_id/model_path>` 即可。
- 默认使用 **ModelScope** 进行模型和数据集的下载。如果要使用 HuggingFace，指定 `--use_hf true` 即可。
- 训练启动前会自动应用 NPU 适配（`enable_npu_model_patch` 默认开启）。如需排查，参考 [NPU 最佳实践文档](../BestPractices/NPU-support.md) §"NPU 模型 Patch 开关"。

## 训练完成后推理

- 这里的 `--adapters` 需要替换成训练生成的 last checkpoint 文件夹。由于 adapters 文件夹中包含了训练的参数文件 `args.json`，因此不需要额外指定 `--model`、`--system`，swift 会自动读取这些参数。如果要关闭此行为，可以设置 `--load_args false`。

### 交互式命令行推理（transformers / torch_npu 后端）

```shell
>>> ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
...     --adapters output/vx-xxx/checkpoint-xxx \
...     --stream true \
...     --temperature 0 \
...     --max_new_tokens 2048
<<< 你好，请介绍一下你自己。
...
>>> echo "infer transformers done"
infer transformers done
```

### merge-lora 并使用 vLLM-Ascend 加速推理

```shell
>>> ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
...     --adapters output/vx-xxx/checkpoint-xxx \
...     --stream true \
...     --merge_lora true \
...     --infer_backend vllm \
...     --vllm_max_model_len 8192 \
...     --temperature 0 \
...     --max_new_tokens 2048
<<< 你好，请介绍一下你自己。
...
>>> echo "infer vllm-ascend done"
infer vllm-ascend done
```

## 推送 ModelScope

```shell
ASCEND_RT_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<your-model-id>' \
    --hub_token '<your-sdk-token>' \
    --use_hf false
```


## 了解更多

- 完整 NPU 适配说明、兼容性表、DDP/DeepSpeed/MindSpeed、`vllm-ascend` 部署：[NPU 最佳实践](../BestPractices/NPU-support.md)
- 推理与部署完整指南（含 vLLM-Ascend、`swift deploy`、多卡 serving）：[推理与部署](../Instruction/Inference-and-deployment.md)
- 训练参数详解：[命令行参数](../Instruction/Command-line-parameters.md)
- 更多 Shell 脚本：<https://github.com/modelscope/ms-swift/tree/main/examples/ascend>
