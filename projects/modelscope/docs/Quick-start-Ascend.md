# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑通 [ModelScope](https://modelscope.cn) 的最小链路：从源码安装 modelscope，通过 ModelScope Hub 下载 Qwen2.5-0.5B-Instruct，并在 NPU 上做一次文本生成推理。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品（Ascend 910B4 / 910B 等），并按需完成物理机或容器内的设备挂载。

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
| transformers | `<5.0` |
| modelscope | 最新 release 的源码 |
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct) |

## 前置安装

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

安装 `torch` / `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.9.0
uv pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi torch_npu==2.9.0.post2
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

## 安装 modelscope

### 使用 pip 进行安装（二进制，用户自选）

```shell
uv pip install modelscope
python -c "import modelscope; print('modelscope', modelscope.__version__)"
```

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证。`.[framework]` extra 会连带装上 `transformers` / `datasets` 等做 LLM 推理需要的依赖（modelscope 基础安装 `requirements/hub.txt` 不含 torch / transformers）：

```shell #test id="modelscope-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/modelscope/modelscope.git
cd modelscope
uv pip install -e '.[framework]'
uv pip install 'transformers<5.0'
python -c "import modelscope; print('modelscope', modelscope.__version__)"
```
\<ref> 为安装的最新的 release 分支

输出结果类似如下：

```shell #test-result id="modelscope-install-source" fuzzy='xxx'
modelscope xxx
```

- xxx 表示最新的版本号

## 使用样例

在单卡昇腾 NPU 上用 modelscope 下载 Qwen2.5-0.5B-Instruct 并做一次文本生成。

> **NPU 设备放哪**：modelscope 的 `pipeline(..., device=...)` 目前只接受 `cpu` / `cuda` / `gpu`（`modelscope/utils/device.py` 的 `verify_device` 会拒绝 `npu`），所以本文档走 `AutoModelForCausalLM` 加载后显式 `.to('npu:0')` —— 模型权重下载由 modelscope 的 `from_pretrained` 补丁完成（`snapshot_download`），设备放置由 torch_npu 接管。

下载模型到 ModelScope Hub 缓存（大陆网络更稳；CI 上 `~/.cache/modelscope` 已 bind-mount 持久化，重复运行命中缓存）：

```shell #test-setup
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct
```

在 NPU 上做文本生成推理：

```shell #test id="ms-npu-infer"
python - <<'PY'
import torch
import torch_npu
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True).to('npu:0')
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print('model device:', next(model.parameters()).device)

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '用一句话介绍 ModelScope。'},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors='pt').to('npu:0')
generated_ids = model.generate(**model_inputs, max_new_tokens=64)
generated_ids = [
    out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print('generated:', response)
PY
```

输出结果类似如下：

```shell #test-result id="ms-npu-infer" fuzzy='xxx'
model device: npu:0
generated: xxx
```

> `model device: npu:0` 说明模型权重确实落在昇腾设备上，不是 CPU 兜底。generated 是模型生成内容，随采样变化，这里只校验「生成成功且非空」。

## 快速上手（README_zh 镜像）：分词 pipeline

镜像自上游 `README_zh.md` 的「快速上手」，把 python REPL 片段改写成可执行的 shell 契约。modelscope 的 `pipeline(..., device=...)` 只接受 `cpu` / `cuda` / `gpu`（`verify_device` 会拒绝 `npu`），本示例走默认 device（NPU 上自动回落 CPU），验证 `pipeline()` API 链路可用；昇腾设备上的真实推理见上一节。

下载分词模型（复用 `~/.cache/modelscope` 缓存）：

```shell #test-setup
modelscope download --model damo/nlp_structbert_word-segmentation_chinese-base
```

按 README 原样跑分词：

```shell #test id="ms-pipeline-word-seg"
python - <<'PY'
from modelscope.pipelines import pipeline

word_segmentation = pipeline(
    'word-segmentation',
    model='damo/nlp_structbert_word-segmentation_chinese-base')
result = word_segmentation(' 今天天气不错，适合出去游玩 ')
print('output:', result['output'])
PY
```

输出结果类似如下（分词 token 确定，但首尾空格 / 标点 / 间隔可能随模型版本漂移，用 `...` 通配只锁 token 顺序）：

```shell #test-result id="ms-pipeline-word-seg"
output: ...今天...天气...不错...适合...出去...游玩...
```
