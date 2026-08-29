# 快速开始：在昇腾 NPU 上跑通 ColossalAI 的一次极小训练

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**：装上与 CANN 匹配的 PyTorch NPU 栈和 ColossalAI，在单卡 NPU 上用 Booster + TorchDDP 对 Qwen2.5-0.5B 做一步训练。

[ColossalAI](https://github.com/hpcaitech/ColossalAI) 通过 `colossalai.accelerator` 选择设备。机器上能 `import torch_npu` 且 `torch.npu.is_available()` 为真时，会选中 `npu`，通信后端是 HCCL。上游 `requirements.txt` 把 `torch` 钉在 `>=2.2.0,<=2.5.1`。本文用的镜像是 CANN 9.1.0，对应的是 `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，所以**不要**让 pip 按那条上限去装 ColossalAI，否则会把刚装好的 NPU 栈换掉。

本文从 PyPI 安装 `colossalai==0.5.0`，并加上 `--no-deps`。`--no-deps` 只跳过依赖解析，不跳过 ColossalAI 自己的代码。后面会把 Booster 真正要用的包单独装上。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| Python | 3.12 |
| PyTorch | `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，见下文安装 |
| ColossalAI | 从 PyPI 安装 `colossalai==0.5.0 --no-deps`，见下文 |
| 模型 | [Qwen/Qwen2.5-0.5B](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B)，首次运行会从 ModelScope 下载 |

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

---

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
```

`PYTHONNOUSERSITE=1` 让 Python 忽略用户目录里的包。本机如果曾经 `pip install --user` 过 CANN 相关包，不设这个变量时，pip 解析器可能被带偏。

---

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

**预期**：命令退出码为 0，并打印设备列表。表格中的功耗、HBM 占用每次不同，**不必**与任何样例逐字一致。

若 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载（如 `/dev/davinci0`）。

### 2.2 确认工具可用

```shell #test-setup
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
```

**预期**：`ASCEND_HOME_PATH` 非空；`npu-smi` 能找到。

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

昇腾上的 `torch_npu` 要从华为 PyPI 额外索引安装，并钉死与 CANN 匹配的版本。`numpy` 和 `pyyaml` 也要一起装。`torch_npu` 的 wheel **没有声明**这两项依赖，但 `import torch` 会自动加载 `torch_npu`，缺了会在你显式 `import torch_npu` 之前就失败。

```shell #test id="install-torch"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  torch==2.9.0 torch_npu==2.9.0.post2 numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-torch"
...
torch 2.9.0...
torch_npu 2.9.0.post2
npu_available True
```

`npu_available` 必须是 `True`。`False` 时不要继续，先查 CANN、驱动和可见设备。

---

## 4. 安装 ColossalAI

不要写 `pip install colossalai`。那条命令会按上游元数据去满足 `torch<=2.5.1`，把第 3 节的 NPU 栈换成 CUDA/CPU 的旧 torch。先装好 NPU 栈，再装带 `--no-deps` 的 `colossalai==0.5.0`。

`--no-deps` 之后，Booster 这条路径还缺几个**导入期**依赖，不是可选项：

- `peft`：`colossalai.booster` 在导入模型接口时会加载它
- `galore_torch`：`colossalai.nn.optimizer` 在导入时会加载它
- `bitsandbytes`：`galore_torch` 在导入时会加载它。这里装的是 PyPI 上的通用包，只为了让导入成功。本文不用它的 8-bit 训练路径
- `einops`：`colossalai.shardformer` 在导入注意力层时会加载它

再装 `transformers` 和 `modelscope`，下一节要用它们拉 Qwen2.5-0.5B。

```shell #test id="install-colossalai"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
python -m pip install colossalai==0.5.0 --no-deps
python -m pip install transformers peft galore_torch bitsandbytes einops modelscope
python -c "import torch, torch_npu, colossalai; from colossalai.accelerator import get_accelerator; print('torch', torch.__version__); print('colossalai', colossalai.__version__); acc = get_accelerator(); print('accel_name', acc.name); print('accel_device', acc.get_current_device()); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-colossalai"
...
torch 2.9.0...
colossalai 0.5.0
accel_name npu
accel_device npu:0
npu_available True
```

`accel_name` 为 `npu`、`accel_device` 为 `npu:0` 只说明设备探测成功。真正的工作负载在下一节。pip 可能会提示 ColossalAI 声明的 `torch<=2.5.1` 与当前 `torch 2.9.0` 冲突，这是预期现象，不是安装失败。

若这里打印 `accel_name cpu` 或 `cuda`，先回到第 3 节确认 `torch_npu` 和可见设备，不要继续训练。`torch` 若不再以 `2.9.0` 开头，说明后面的包把 NPU 栈换掉了，卸掉后按第 3–4 节重装。

---

## 5. 在 NPU 上做一步训练

下面这段用 ColossalAI 的 `launch` + `Booster(plugin=TorchDDPPlugin())` 包住 Qwen2.5-0.5B，做一次前向、反向和 `optimizer.step()`。上游没有现成的昇腾示例，所以这是写给你复制的最小脚本，不是官方 tutorial 原文。

`torch_dtype=torch.bfloat16` 用来降低显存。单卡 910B4 跑 0.5B 用 float32 也放得下，第一次验证不必改。`launch` 的 `world_size=1` 是单进程；多卡时再改 `rank` / `world_size`。端口 `29599` 只要本机没被占用即可。

权重从 ModelScope 拉 [Qwen/Qwen2.5-0.5B](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B)。第一次会下载约 1 GB，之后走本地缓存。

**怎样算成功**

1. 进程退出码为 0；
2. 打印 `accel_name npu` 和 `accel_device npu:0`；
3. 打印 `boosted_param_device npu:0`。这是 Booster 把模型参数放到 NPU 上的证据。只看到上一节的探测结果、参数却在 CPU 上，仍算失败。

```shell #test id="train"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
python - <<'PY'
import torch
from torch.optim import AdamW
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import colossalai
from colossalai.accelerator import get_accelerator
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin

model_path = snapshot_download("Qwen/Qwen2.5-0.5B")
colossalai.launch(rank=0, world_size=1, host="127.0.0.1", port=29599)
acc = get_accelerator()
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16,
)
optimizer = AdamW(model.parameters(), lr=1e-5)
booster = Booster(plugin=TorchDDPPlugin())
model, optimizer, _, _, _ = booster.boost(model, optimizer)
print("accel_name", acc.name)
print("accel_device", acc.get_current_device())
print("boosted_param_device", next(model.parameters()).device)
enc = tokenizer("ColossalAI on Ascend NPU", return_tensors="pt")
enc = {k: v.to(acc.get_current_device()) for k, v in enc.items()}
loss = model(**enc, labels=enc["input_ids"]).loss
booster.backward(loss, optimizer)
optimizer.step()
optimizer.zero_grad()
print("train_ok", True)
PY
```

输出结果如下：

```shell #test-result id="train"
...
accel_name npu
accel_device npu:0
boosted_param_device npu:0
train_ok True
```

导入 ColossalAI 时可能会看到 `tensornvme`、`apex` 的警告。那是上游给 NVIDIA 栈留的提示，昇腾上可以忽略。`boosted_param_device` 才是参数真正所在的设备。

---

## 6. 本文没有覆盖的能力

这些路径不在第一次跑通范围内，正文里也没有对应的可复制命令块：

- Gemini、HybridParallel、MoE 等其它 Booster plugin
- 多卡 / 多机 `launch`
- `pip install colossalai`（不带 `--no-deps`，会降级 torch）
- 上游元数据里的 `torch<=2.5.1` 全量依赖树（ray、fastapi、diffusers 等）
- 8-bit 优化器和 bitsandbytes 的量化训练

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `import torch` 报缺 `numpy` 或 `yaml` | `torch_npu` 未声明这两项依赖 | 与 torch 栈一起安装 `numpy` `pyyaml` |
| `torch.npu.is_available()` 为 `False` | 未 `source set_env.sh`，或设备未挂进容器 | 重做第 1–2 节 |
| `accel_name` 为 `cpu` | `torch_npu` 没装上，或 `torch.npu.is_available()` 为假 | 重做第 3 节 |
| `accel_name` 为 `cuda` | 当前进程里 `torch.cuda.is_available()` 为真，框架会优先选 CUDA | 卸掉 CUDA 版 torch |
| pip 把 `torch` 降到 `2.5.1` 或更低 | 安装 ColossalAI 时没加 `--no-deps` | 卸掉后按第 3–4 节重装 |
| `ModuleNotFoundError: galore_torch`、`peft` 或 `einops` | 只装了 `--no-deps` 的 ColossalAI，没装第 4 节的运行时包 | 按第 4 节把导入期依赖装上 |
| `boosted_param_device` 为 `cpu` 但退出码 0 | 模型没放到 NPU | 检查可见设备和 `torch.npu.is_available()` |
| `Address already in use` | 端口 `29599` 被占用 | 换一个端口，或结束占用该端口的进程 |
| ModelScope 下载超时或落到 HTML | 出口到 modelscope.cn 失败 | 检查网络后重跑；不要改成 Hugging Face 直连 |
