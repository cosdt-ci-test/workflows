# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上，用极小数据集对 Qwen2.5-0.5B-Instruct 跑通一个最小的 SFT LoRA 后训练示例，并验证输出目录产物。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0.post2 |
| transformers | `>=4.56.2,<5.0` |
| accelerate | `>=1.4.0` |
| datasets | `>=4.7.0` |
| peft | 最新 release |
| modelscope | 1.37.0 |
| trl | 最新 release 的源码/二进制 |
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct) |
| 数据集 | 文档内联的 4 条极小对话样本（不依赖外部数据集下载） |

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出类似：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。

检查 Python 版本：

```shell #test id="check-py"
python --version
```
输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

检查 NPU 设备运行时可用：

```shell #test id="check-npu-runtime"
python -c "import torch, torch_npu; print(f'torch={torch.__version__}'); print(f'torch_npu={torch_npu.__version__}'); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-npu-runtime"
torch=2.9.0+cpu
torch_npu=2.9.0.post2
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

安装 `transformers` / `peft` / `modelscope`（`trl` 自身会在下一节安装，并自动带入 `accelerate` / `datasets` 依赖）：

```shell #test-setup
pip install 'transformers>=4.56.2,<5.0' 'peft' 'modelscope==1.37.0'
```

打印安装版本：
```shell #test id="install-deps"
python -c "import transformers, peft, modelscope; print(f'transformers={transformers.__version__} peft={peft.__version__} modelscope={modelscope.__version__}')"
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
transformers=xxx peft=xxx modelscope=1.37.0
```

## 安装 TRL

### 使用 uv 进行安装

```shell #test id="trl-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple trl
python -c "import trl; print('trl', trl.__version__)"
```

输出结果类似如下：

```shell #test-result id="trl-install-binary" fuzzy='xxx'
trl xxx
```
- xxx 表示最新的版本号
<!--
```shell #test-setup
uv pip uninstall trl -y
```
-->

### 从源码安装
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证。

> 注意：TRL 上游 `examples/` 已改为**目录式**布局（如 `examples/sft_qlora/`、`examples/grpo_wordle/`），每个示例是一个自包含目录，脚本内用 `# /// script` 头声明依赖、在模块 docstring 里写运行命令。昇腾社区旧文档中的 `examples/scripts/sft.py` / `examples/scripts/dpo.py` 这类路径**已过时**，请勿引用。

```shell #test id="trl-install-source" load="upstream_ref>>ref"
git clone https://github.com/huggingface/trl.git /tmp/trl-src
cd /tmp/trl-src && git checkout <ref>
uv pip install -e .
python -c "import trl; print('trl', trl.__version__)"
```
\<ref> 为安装的最新的 release 分支

输出结果类似如下：

```shell #test-result id="trl-install-source" fuzzy='xxx'
trl xxx
```
- xxx 表示最新的版本号

## 下载基础模型

默认使用 **ModelScope** 进行模型下载（runner 无法访问 HuggingFace）。

```shell #test-setup store="model_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-0.5B-Instruct'))" | tail -n 1
```

## 使用样例：最小 SFT LoRA 后训练

用 4 条内联对话样本（极小数据集，不依赖外部数据集下载）对 Qwen2.5-0.5B-Instruct 做 5 步 LoRA SFT。`SFTTrainer` 通过 `peft_config` 注入 LoRA 适配器，底座权重冻结、只训练新注入的低秩矩阵；训练完成后把适配器保存到 `output/trl-sft-lora`。

```shell #test id="sft-lora" load="model_path>>model_path"
ASCEND_RT_VISIBLE_DEVICES=0 python << 'PY'
import os

import torch
import torch_npu
from datasets import Dataset
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer

print("TRL_SFT_BEGIN")

# 极小数据集：会话格式（{"messages": [...]}），SFTTrainer 自动套用 chat template
data = [
    {"messages": [{"role": "user", "content": "What color is the sky?"},
                  {"role": "assistant", "content": "The sky is blue."}]},
    {"messages": [{"role": "user", "content": "What is 2+2?"},
                  {"role": "assistant", "content": "2+2 equals 4."}]},
    {"messages": [{"role": "user", "content": "Name a planet."},
                  {"role": "assistant", "content": "Mars is a planet."}]},
    {"messages": [{"role": "user", "content": "What do bees make?"},
                  {"role": "assistant", "content": "Bees make honey."}]},
]
train_dataset = Dataset.from_list(data)

trainer = SFTTrainer(
    model="<model_path>",
    train_dataset=train_dataset,
    peft_config=LoraConfig(r=8, lora_alpha=32, task_type=TaskType.CAUSAL_LM),
    args=SFTConfig(
        output_dir="output/trl-sft-lora",
        max_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_length=512,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        model_init_kwargs={"torch_dtype": torch.bfloat16},
    ),
)
print("model device:", next(trainer.model.parameters()).device)
trainer.train()
trainer.save_model("output/trl-sft-lora")
print("TRL_SFT_DONE")
PY
```

> `<model_path>` 为上面「下载基础模型」章节对应命令的输出，由 `#test-setup store="model_path"` 捕获并注入，无需手动替换。

输出结果类似如下（训练日志走 stderr，stdout 只保留首尾标记）：

```shell #test-result id="sft-lora"
...
TRL_SFT_DONE
```

## 结果验证

检查输出目录中的 LoRA 适配器产物：`adapter_config.json`（LoRA 配置）与 `adapter_model.safetensors`（适配器权重）。

```shell #test id="verify-output"
ls output/trl-sft-lora/adapter_config.json output/trl-sft-lora/adapter_model.safetensors
```

输出结果如下：

```shell #test-result id="verify-output"
output/trl-sft-lora/adapter_config.json
output/trl-sft-lora/adapter_model.safetensors
```

小贴士：

- 如果要换用其他后训练方法（DPO / GRPO / KTO 等），只需把 `SFTTrainer` / `SFTConfig` 换成对应的 `DPOTrainer` / `GRPOTrainer` 等，入口形态保持一致。
- 如果要切换其他模型，只需修改 `model=<model_id/model_path>`；本文默认经 **ModelScope** 获取模型。
- 想跑完整训练而非 5 步冒烟，去掉 `max_steps` 并调大 `save_steps` / 数据集规模即可。
