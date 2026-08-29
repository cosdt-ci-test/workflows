# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上快速验证 xllm 离线推理。

## 前置条件

### 硬件

Atlas 900 A2 训练系列产品或者 Ascend 910B 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11 |
| CANN | 9.1.0 |
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| xllm | 从源码编译安装（CANN 基础镜像） |
| 模型 | [Qwen2-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2-7B-Instruct) |

> 说明：CI 使用通用 CANN 基础镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`，在运行环境内**从源码编译安装 xllm**（含 C++ 扩展、vcpkg 依赖与 TileLang 内核），首次冷编译约 1–2 小时；`xllm` Python 包通过 `pip install` 该 wheel 提供，`examples/` 目录则来自 clone 的源码树。

### 前置安装

确认能看到 NPU 设备：

```shell #test id="check-npu"
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
| 0     910B4               | OK            | 89.9        39                0    / 0             |
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
Python 3.11.xxx
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.9.0
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

## 验证 xllm 安装

xllm 由 CI 从源码编译安装，验证版本：

```shell #test id="check-xllm"
python -c "import xllm; print('xllm version:', xllm.__version__)"
```

输出结果如下：

```shell #test-result id="check-xllm" fuzzy='xxx'
xllm version: xxx
```

## 离线推理示例

使用单卡 NPU 运行生成示例（最大生成 10 token，快速验证链路）：

```shell #test id="generate"
ASCEND_RT_VISIBLE_DEVICES=0 python -m examples.generate --model /root/.cache/modelscope/Qwen2-7B-Instruct --max_tokens 10
```

输出结果如下：

```shell #test-result id="generate" fuzzy='xxx' fuzzy='...'
Prompt: 'Hello, my name is', Generated text: 'xxx'
Prompt: 'The president of the United States is', Generated text: 'xxx'
Prompt: 'The capital of France is', Generated text: 'xxx'
Prompt: 'The future of AI is', Generated text: 'xxx'
...
llm finished
```

> 注意：模型路径 `/root/.cache/modelscope/Qwen2-7B-Instruct` 是 CI 环境通过 ModelScope 预先下载的目录（挂载自 CI 缓存 `/data/ci-cache/modelscope/xllm`）。本地运行时请用 `modelscope` 自行下载该模型到对应目录。

## Beam Search 生成示例

```shell #test id="generate-beam"
ASCEND_RT_VISIBLE_DEVICES=0 python -m examples.generate_beam_search --model /root/.cache/modelscope/Qwen2-7B-Instruct --max_tokens 10
```

输出结果如下：

```shell #test-result id="generate-beam" fuzzy='xxx' fuzzy='...'
Prompt: 'Hello, my name is', Generated text: 'xxx'
...
```

## Embedding 生成示例

```shell #test id="generate-embedding"
ASCEND_RT_VISIBLE_DEVICES=0 python -m examples.generate_embedding --model /root/.cache/modelscope/Qwen2-7B-Instruct
```

输出结果如下：

```shell #test-result id="generate-embedding" fuzzy='xxx' fuzzy='...'
Embedding shape: xxx
...
```

## VLM 示例（如果模型支持）

```shell #test id="generate-vlm"
ASCEND_RT_VISIBLE_DEVICES=0 python -m examples.generate_vlm --model /root/.cache/modelscope/Qwen2-7B-Instruct --max_tokens 10
```

输出结果如下：

```shell #test-result id="generate-vlm" fuzzy='xxx' fuzzy='...'
Prompt: 'xxx', Generated text: 'xxx'
...
```

## Sample 示例

```shell #test id="sample"
ASCEND_RT_VISIBLE_DEVICES=0 python -m examples.sample --model /root/.cache/modelscope/Qwen2-7B-Instruct --max_tokens 10
```

输出结果如下：

```shell #test-result id="sample" fuzzy='xxx' fuzzy='...'
Prompt: 'xxx', Generated text: 'xxx'
...
```