# Quick Start (Ascend NPU)

TRL 用同一套 `Trainer` / `Config` API 覆盖 SFT / DPO / GRPO / PPO 等后训练方法。本示例在单卡昇腾 NPU 上，用同一个 Qwen2.5-0.5B-Instruct 模型先跑通最小 SFT LoRA，再换成 `DPOTrainer` 跑通偏好优化 DPO LoRA，并验证两种方法的产物。

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
| trl | 最新 release（PyPI） |
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)，约 1 GB，首次运行自动下载 |
| 数据集 | 文档内联的极小对话样本：4 条 SFT 对话 + 3 条偏好样本（不依赖外部数据集下载） |

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

```shell #test-result id="check-npu-runtime" fuzzy='xxx'
torch=xxx
torch_npu=xxx
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

安装 `transformers` / `peft` / `modelscope`（`trl` 会在下一节安装，并按依赖声明自动带入 `transformers` / `peft` / `accelerate` / `datasets`），装完打印版本验证：

```shell #test id="install-deps"
uv pip install 'transformers>=4.56.2,<5.0' 'peft' 'modelscope==1.37.0'
python -c "import transformers, peft, modelscope; print(f'transformers={transformers.__version__} peft={peft.__version__} modelscope={modelscope.__version__}')"
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
transformers=xxx peft=xxx modelscope=1.37.0
```

## 安装 TRL

用 uv 安装 PyPI 最新 release，装完打印版本验证：

```shell #test id="trl-install-binary"
uv pip install trl
python -c "import trl; print('trl', trl.__version__)"
```

输出结果类似如下：

```shell #test-result id="trl-install-binary" fuzzy='xxx'
trl xxx
```
- xxx 表示最新的版本号

## 使用样例：最小 SFT LoRA 后训练

用 4 条内联对话样本（极小数据集，不依赖外部数据集下载）对 Qwen2.5-0.5B-Instruct 做 5 步 LoRA SFT。模型由脚本内的 `snapshot_download` 首次运行时自动下载到默认缓存（约 1 GB）；`SFTTrainer` 通过 `peft_config` 注入 LoRA 适配器，底座权重冻结、只训练新注入的低秩矩阵；训练完成后把适配器保存到 `output/trl-sft-lora`。

```shell #test id="sft-lora"
python << 'PY'
import torch
import torch_npu
from datasets import Dataset
from modelscope import snapshot_download
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

model = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')

trainer = SFTTrainer(
    model=model,
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

输出结果类似如下（训练日志走 stderr，stdout 只保留首尾标记）：

```shell #test-result id="sft-lora"
...
TRL_SFT_DONE
```

## 切换方法：偏好优化 DPO LoRA

同一个模型与 LoRA 配置，把 `SFTTrainer` / `SFTConfig` 换成 `DPOTrainer` / `DPOConfig` 就是偏好优化：数据集改为 `prompt` / `chosen` / `rejected` 三段对话，训练让模型更倾向 `chosen` 而非 `rejected` 的回答。这里用 3 条内联偏好样本跑 3 步 DPO LoRA，产物保存到 `output/trl-dpo-lora`。

```shell #test id="dpo-lora"
python << 'PY'
import torch
import torch_npu
from datasets import Dataset
from modelscope import snapshot_download
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

print("TRL_DPO_BEGIN")

# 偏好数据集：prompt / chosen / rejected 三段对话，DPOTrainer 自动套用 chat template
data = [
    {"prompt": [{"role": "user", "content": "What is the capital of France?"}],
     "chosen": [{"role": "assistant", "content": "The capital of France is Paris."}],
     "rejected": [{"role": "assistant", "content": "The capital of France is London."}]},
    {"prompt": [{"role": "user", "content": "How many days are there in a week?"}],
     "chosen": [{"role": "assistant", "content": "There are seven days in a week."}],
     "rejected": [{"role": "assistant", "content": "There are ten days in a week."}]},
    {"prompt": [{"role": "user", "content": "What is 2+2?"}],
     "chosen": [{"role": "assistant", "content": "2+2 equals 4."}],
     "rejected": [{"role": "assistant", "content": "2+2 equals 22."}]},
]
train_dataset = Dataset.from_list(data)

model_path = snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    peft_config=LoraConfig(r=8, lora_alpha=32, task_type=TaskType.CAUSAL_LM),
    max_length=512,
    max_prompt_length=256,
    args=DPOConfig(
        output_dir="output/trl-dpo-lora",
        max_steps=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    ),
)
print("model device:", next(trainer.model.parameters()).device)
trainer.train()
trainer.save_model("output/trl-dpo-lora")
print("TRL_DPO_DONE")
PY
```

输出结果类似如下（训练日志走 stderr，stdout 只保留首尾标记）：

```shell #test-result id="dpo-lora"
...
TRL_DPO_DONE
```

## 结果验证

检查两个输出目录中的 LoRA 适配器产物：`adapter_config.json`（LoRA 配置）与 `adapter_model.safetensors`（适配器权重）。

```shell #test id="verify-output"
ls output/trl-sft-lora/adapter_config.json output/trl-sft-lora/adapter_model.safetensors
```

输出结果如下：

```shell #test-result id="verify-output"
output/trl-sft-lora/adapter_config.json
output/trl-sft-lora/adapter_model.safetensors
```

检查 DPO 输出目录的适配器产物：

```shell #test id="verify-dpo"
ls output/trl-dpo-lora/adapter_config.json output/trl-dpo-lora/adapter_model.safetensors
```

输出结果如下：

```shell #test-result id="verify-dpo"
output/trl-dpo-lora/adapter_config.json
output/trl-dpo-lora/adapter_model.safetensors
```

更多方法（GRPO / PPO / Reward / KTO 等）入口形态一致，切换对应的 `Trainer` / `Config` 即可；GRPO 依赖 vLLM 生成，不在本示例运行。更多用法见 [TRL examples](https://github.com/huggingface/trl/tree/main/examples)。
