# 快速开始：在昇腾 NPU 上用 llm-compressor 做一次 GPTQ 量化

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文介绍如何在单卡 NPU 上安装 [llm-compressor](https://github.com/vllm-project/llm-compressor)，对公开小模型做一次 W4A16 GPTQ，再保存、重载并做一次前向。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件


| 类别             | 要求                                                                                     |
| -------------- | -------------------------------------------------------------------------------------- |
| CANN           | toolkit + 驱动固件已安装并可 `source set_env.sh`                                                |
| Python         | 3.12                                                                                   |
| PyTorch        | `torch==2.10.0` 与 `torch_npu==2.10.0.post4`                                            |
| llm-compressor | 从 PyPI 安装发布版                                                                           |
| 模型             | [nm-testing/tinysmokeqwen3](https://huggingface.co/nm-testing/tinysmokeqwen3)（约 10 MB） |


**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

---



## 1. 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

---



## 2. 检查环境是否就绪

```shell
npu-smi info
```

命令退出码应为 0，并打印设备列表。

```shell #test-setup
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy="xxx"
Python 3.12.xxx
```

---



## 3. 安装 PyTorch NPU 栈

`torch_npu` 从华为 PyPI 额外索引安装，并与 CANN 9.1.0 配对。`numpy` 和 `pyyaml` 也要一起装，缺了会在 `import torch_npu` 之前失败。

```shell #test id="install-torch"
python -m pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  torch==2.10.0 torch_npu==2.10.0.post4 numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-torch"
...
torch 2.10.0...
torch_npu 2.10.0.post4
npu_available True
```

`npu_available` 必须是 `True`。`torch` 版本串可能带 `+cpu` 后缀，以 `npu_available True` 为准。

---



## 4. 安装 llm-compressor

将 `<UPSTREAM_REF>` 换成目标 **PyPI 版本号**（撰写时最新正式版是 `0.13.0`）。



```shell #test id="install-llmcompressor" load="upstream_ref>>UPSTREAM_REF"
python -m pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  llmcompressor==<UPSTREAM_REF> torch==2.10.0 torch_npu==2.10.0.post4
python -c "import llmcompressor; print('llmcompressor', llmcompressor.__version__)"
```

输出结果如下：

```shell #test-result id="install-llmcompressor"
...
llmcompressor ...
```

---



## 5. 在 NPU 上做一次单层 W4A16 GPTQ

下面用公开小模型 `nm-testing/tinysmokeqwen3`、8 条本地校准文本，只量化第 3 层的 `q_proj`。这个模型隐藏维度是 64，分组大小要用 32 才能整除；`oneshot` 收尾会读 `torch.accelerator` 显存，当前 torch_npu 还不支持，先换成空实现。

```shell #test id="oneshot-forward"
python - <<'PY'
from pathlib import Path

import torch
import torch_npu
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

model_id = "nm-testing/tinysmokeqwen3"
device = "npu:0"
out_dir = Path.home() / "llm-compressor-work" / "compressed"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
ds = Dataset.from_dict(
    {
        "text": [
            "The quick brown fox jumps over the lazy dog.",
            "Quantization maps weights to fewer bits.",
            "Ascend NPU runs this oneshot calibration.",
            "A short sentence is enough for a smoke test.",
        ]
        * 2
    }
)
recipe = GPTQModifier(
    ignore=["lm_head"],
    config_groups={
        "group_0": QuantizationScheme(
            targets=["re:.*model.layers.2.self_attn.q_proj$"],
            weights=QuantizationArgs(num_bits=4, strategy="group", group_size=32),
        )
    },
)
torch.accelerator.max_memory_allocated = lambda device=None: 0
torch.accelerator.get_memory_info = lambda device=None: (0, 1)
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=8,
    max_seq_length=64,
    output_dir=str(out_dir),
)

reloaded = AutoModelForCausalLM.from_pretrained(out_dir).to(device)
inputs = tokenizer("hello", return_tensors="pt").to(device)
logits = reloaded(**inputs).logits
qc = reloaded.config.quantization_config
inner = getattr(qc, "quantization_config", qc)
group0 = inner.config_groups["group_0"]
print("LLM_COMPRESSOR_WORKLOAD_DEVICE=npu:0")
print("weight_num_bits", group0.weights.num_bits)
print("targeted", hasattr(reloaded.model.layers[2].self_attn.q_proj, "quantization_scheme"))
print("lm_head_quantized", hasattr(reloaded.lm_head, "quantization_scheme"))
print("logits.device", logits.device)
PY
```

输出结果如下：

```shell #test-result id="oneshot-forward"
...
LLM_COMPRESSOR_WORKLOAD_DEVICE=npu:0
weight_num_bits 4
targeted True
lm_head_quantized False
logits.device npu:0
```

---



## 故障排查


| 现象                                                         | 可能原因                                      | 建议                                                                   |
| ---------------------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------------- |
| `import torch` 报缺 `numpy` 或 `yaml`                         | `torch_npu` 未声明这两项依赖                      | 与 torch 栈一起安装 `numpy` `pyyaml`                                       |
| `torch.npu.is_available()` 为 `False`                       | 未 `source set_env.sh`，或设备未挂进容器            | 重做第 1–2 节                                                            |
| `logits.device` 为 `cpu` 但退出码 0                             | 静默 CPU 回退                                 | 检查 `.to("npu:0")` 与可见设备                                              |
| 下载 `tinysmokeqwen3` 超时                                     | 直连 Hugging Face 慢                         | 设置 `HF_ENDPOINT=https://hf-mirror.com` 后重跑第 5 节                      |
| pip 找不到 `torch_npu==2.10.0.post4`                          | 没用华为 extra-index                          | 用第 3 节的 `--extra-index-url`                                          |
| pip 下载 `torch` wheel 很慢                                    | 直连 pypi.org                               | 设置 `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` 后重跑第 3 节 |
| oneshot 量化完后报 `Allocator for npu is not a DeviceAllocator` | torch_npu 还不支持 `torch.accelerator` 显存 API | 用第 5 节里那两行空实现                                                        |


