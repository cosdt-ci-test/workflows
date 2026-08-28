# Quick Start (Ascend NPU)

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 2）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.10.0 |
| torch_npu | 2.10.0.post4 |
| triton | 最新release |
| modelscope | 最新release |
| torchtitan | 最新 release |
| 训练配置 | `torchtitan/models/llama3/train_configs/debug_model.toml`（dim=256 / 6 层 / 16 head Llama 3 缩水版） |


### 检查前置是否满足

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

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch" fuzzy='xxx'
torch= xxx
torch_npu= xxx
is_available: True
count: 2
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

### 安装 modelscope

```shell #test-setup
uv pip install modelscope
```

打印安装版本：
```shell #test id="modelscope-install"
python -c "import modelscope; print('modelscope', modelscope.__version__)"
```

输出结果如下：
```shell #test-result id="modelscope-install" fuzzy='xxx'
modelscope xxx
```

### 安装triton

```shell #test-setup
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple/ triton
```

打印安装版本：
```shell #test id="triton-install"
python -c "from importlib.metadata import version; print('triton', version('triton'))"
```

输出结果如下：
```shell #test-result id="triton-install" fuzzy='xxx'
triton xxx
```

## 安装 torchtitan

### 通过 uv 安装

torchtitan 上 PyPI 有 `py3-none-any` 的纯 Python wheel（450 KB），不依赖 native build——`uv pip install torchtitan` 直接装最新 stable release，不用管版本号：

```shell #test id="torchtitan-install-binary"
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple/ torchtitan
python -c "import torchtitan; print('torchtitan', torchtitan.__version__)"
```

输出结果类似如下：

```shell #test-result id="torchtitan-install-binary" fuzzy='xxx'
torchtitan xxx
```
- xxx 表示最新的版本号

<!--
```shell #test-setup
uv pip uninstall torchtitan -y
```
-->

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="torchtitan-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/pytorch/torchtitan.git
cd torchtitan
uv pip install -e .
python -c "import torchtitan; print('torchtitan', torchtitan.__version__)"
```
\<ref> 为工作流注入的最新 release tag

输出结果类似如下：

```shell #test-result id="torchtitan-install-source" fuzzy='xxx'
torchtitan xxx
```
- xxx 表示最新的版本号

## 跑训练

### 下载 tokenizer

```shell #test-setup id="modelscope-download-tokenizer" store="ms_tokenizer_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('LLM-Research/Llama-3.2-1B', allow_patterns=['*.json', '*.model', 'tokenizer*']))" | tail -n 1
```

> 输出的路径用于后续「单卡训练」和「多卡训练」章节。

验证 tokenizer 关键文件都落盘：

```shell #test id="modelscope-verify-tokenizer" load="ms_tokenizer_path>>ms_tokenizer_path"
ls -la <ms_tokenizer_path> | head -10
```

```shell #test-result id="modelscope-verify-tokenizer" fuzzy='xxx' fuzzy='...'
xxx xxx xxx xxx xxx config.json
...
xxx xxx xxx xxx xxx generation_config.json
...
xxx xxx xxx xxx xxx special_tokens_map.json
...
xxx xxx xxx xxx xxx tokenizer.json
...
xxx xxx xxx xxx xxx tokenizer_config.json
...
xxx xxx xxx xxx xxx tokenizer.model
...
```

`xxx` 匹配 `ls -la` 的元信息列，文件名是 Llama 3 tokenizer 必备文件，确认 snapshot_download 命中正确。

### 单卡训练

用 fake process group 在 1 张 NPU 上跑 debugmodel，验证配置解析、初始化、加载 tokenizer、build dataloader、`trainer.train()` 整条链路能跑通。`--training.steps 0` 让训练循环直接退出，跳过实际 forward / backward：

```shell #test id="torchtitan-train-debug" load="upstream_ref>>ref" load="ms_tokenizer_path>>ms_tokenizer_path"
cd torchtitan && git checkout <ref>
NGPU=1 ASCEND_RT_VISIBLE_DEVICES=0 LOCAL_RANK=0 \
python -m torchtitan.train \
    --job.config-file ./torchtitan/models/llama3/train_configs/debug_model.toml \
    --model.hf-assets-path <ms_tokenizer_path> \
    --comm.mode fake_backend \
    --training.steps 0 \
    --training.local-batch-size 1 \
    --training.seq-len 256 \
    --metrics.log-freq 1 \
    --metrics.disable-color-printing \
    --job.dump-folder /tmp/torchtitan-quickstart
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-debug" fuzzy='xxx' fuzzy='...'
[titan] xxx - root - INFO - Starting job: Llama 3 debug training
...
[titan] xxx - root - INFO - Training starts at step xxx
[titan] xxx - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] xxx - root - INFO - Training completed
[titan] xxx - root - INFO - Process group destroyed
```

### 多卡训练

用 torchrun 起 2 个 rank 跑 debugmodel 真分布式训练，`--training.steps 0` 跳过实际训练：

```shell #test id="torchtitan-train-2card" load="upstream_ref>>ref" load="ms_tokenizer_path>>ms_tokenizer_path"
cd torchtitan && git checkout <ref>
ASCEND_RT_VISIBLE_DEVICES=0,1 \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 \
    --tee 3 \
    --module torchtitan.train \
    --job.config-file ./torchtitan/models/llama3/train_configs/debug_model.toml \
    --model.hf-assets-path <ms_tokenizer_path> \
    --comm.mode default \
    --training.steps 0 \
    --training.local-batch-size 1 \
    --training.seq-len 256 \
    --metrics.log-freq 1 \
    --metrics.disable-color-printing \
    --job.dump-folder /tmp/torchtitan-quickstart-2card
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-2card" fuzzy='xxx' fuzzy='...'
[default0]:[titan] xxx - root - INFO - Starting job: Llama 3 debug training
...
[default0]:[titan] xxx - root - INFO - Training starts at step xxx
[default0]:[titan] xxx - root - INFO - Training completed
[default0]:[titan] xxx - root - INFO - Process group destroyed
```
