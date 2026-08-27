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
| triton | 最新（`uv pip install triton`，满足 torchtitan `moe/kernels.py` 的 `import triton`，Llama 3 训练不走 triton kernel） |
| modelscope | 最新（`uv pip install modelscope`，不锁版本） |
| torchtitan | 最新 release（v0.2.x 风格 tyro CLI + toml 配置；最新 tag 由 workflow 注入，见下方 `UPSTREAM_REF`） |
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

torchtitan v0.2.1+ 在 `torchtitan/models/moe/kernels.py` 第 8 行硬编码 `import triton`，pyproject.toml 没声明 triton 依赖（隐式依赖）。**只装 plain `triton` 即可，不需要 `triton-ascend`**——Llama 3 模型（debug_model.toml）走 torch 原生 SDPA + FlexAttention，`moe/kernels.py` 的 `@triton.jit` 装饰器是惰性的，只有真正调用 `_fill_indices_kernel[grid](...)` 时才编译；本 quick start 跑 Llama 3 debug 不触发 MoE kernel，不需要 ascend 后端。triton-ascend 3.2.2 的 wheel 把 `triton/_C/libtriton` 装成单文件而标准 triton 是目录，结构性冲突会让 `triton._C.libtriton.ascend` import 失败。

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

> NPU 优化插件层 `cann/torchtitan-npu`（pypi: `0.2.2.post1`）提供 NPU 融合算子、图下沉、AutoFuse、HiFloat8、FSDP 大 EP 切分等生产训练能力——它是 upstream `pytorch/torchtitan` + `ModelConverter` 扩展机制的叠加层，**本 quick-start 不安装它**：Llama 3 debug（dim=256 / 6 层）走原生 SDPA + FlexAttention，不需要这些优化；要训 DeepSeek-V4 / 真实 Llama 3 70B 等大模型再叠加装它。

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装依赖 + 可编辑安装 + 验证：

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

torchtitan 出厂支持 Llama 3 系列训练。本文档的 `debug_model.toml` 配的是 toy 维度（dim=256 / 6 层 / 16 heads），但用真实 Llama 3 tokenizer——让 forward/backward 真的跑 token 嵌入/输出层（vocab_size=128256），而不是 bundled toy tokenizer。下载走 ModelScope SDK，落 `${MODELSCOPE_CACHE:-~/.cache/modelscope}/hub/`（跟 ms-swift 同源），命中已缓存就直接返回。

下载 Llama 3.2 1B 的 tokenizer 到 ModelScope 缓存——下一个小一点的 Llama，比 Llama 3.1 8B 轻很多；用 `allow_patterns` 只下 tokenizer 相关文件（`config.json` / `tokenizer.json` / `tokenizer.model` / `tokenizer_config.json` / `special_tokens_map.json` / `generation_config.json`，~5 MB），不拉 2.5 GB 权重：

```shell #test-setup id="modelscope-download-tokenizer" store="ms_tokenizer_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('LLM-Research/Llama-3.2-1B', allow_patterns=['*.json', '*.model', 'tokenizer*']))" | tail -n 1
```

> 输出的路径用于后续「单卡训练」和「多卡训练」章节。

验证 tokenizer 真的落盘：

```shell #test id="modelscope-verify-tokenizer" load="ms_tokenizer_path>>ms_tokenizer_path"
ls -la <ms_tokenizer_path> | head -10
```

```shell #test-result id="modelscope-verify-tokenizer" fuzzy='xxx' fuzzy='...'
xxx config.json
...
```

### 单卡训练

`--comm.mode fake_backend` 让 torchtitan 跳过 NCCL/HCCL 集合通信初始化、用 fake process group 跑 1 个 rank 的纯 NPU 计算——验证 toml 配置解析 + 模型搬到 NPU + forward / backward / optimizer 这条**最小**链路，多卡 / 真分布式属于另一个配置面：

```shell #test id="torchtitan-train-debug" load="upstream_ref>>ref" load="ms_tokenizer_path>>ms_tokenizer_path"
cd torchtitan && git checkout <ref>
ASCEND_RT_VISIBLE_DEVICES=0 LOCAL_RANK=0 \
python -c "import torch_npu, runpy; runpy.run_module('torchtitan.train', run_name='__main__')" \
    --job.config_file ./torchtitan/models/llama3/train_configs/debug_model.toml \
    --model.hf-assets-path <ms_tokenizer_path> \
    --comm.mode fake_backend \
    --training.steps 1 \
    --training.local_batch_size 1 \
    --training.seq_len 256 \
    --metrics.log_freq 1 \
    --metrics.disable_color_printing true \
    --job.dump_folder /tmp/torchtitan-quickstart
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-debug" fuzzy='xxx' fuzzy='...'
[titan] xxx - root - INFO - Starting job: Llama 3 debug training
...
[titan] xxx - root - INFO - Training starts at step xxx
[titan] xxx - root - INFO - step:  xxx  loss:  xxx...
[titan] xxx - root - INFO - Training completed
[titan] xxx - root - INFO - Process group destroyed
```

`torchtitan.train` 训练收尾最后一步是 `torch.distributed.checkpoint.save(...)` 把权重 / optimizer state / tokenizer config 写到 `--job.dump_folder`：

```shell #test id="torchtitan-dump-debug"
find /tmp/torchtitan-quickstart -mindepth 1 -maxdepth 3 -printf '%p\n' | head -20
```

```shell #test-result id="torchtitan-dump-debug" fuzzy='xxx' fuzzy='...'
/tmp/torchtitan-quickstart/xxx
...
```

### 多卡训练

`--comm.mode hierarchical` 走真实 HCCS 集合通信（单机内多卡用 HCCS 做 ring-allreduce），配合 `torchrun --nproc_per_node=2` 起 2 个 rank 的 DDP——验证 collective 初始化 + DDP 梯度同步 + ProcessGroup 销毁这条**真实分布式**链路：

```shell #test id="torchtitan-train-2card" load="upstream_ref>>ref" load="ms_tokenizer_path>>ms_tokenizer_path"
cd torchtitan && git checkout <ref>
ASCEND_RT_VISIBLE_DEVICES=0,1 \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE="http://localhost:29510" \
torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 \
    --role rank \
    --tee 3 \
    python -c "import torch_npu, runpy; runpy.run_module('torchtitan.train', run_name='__main__')" \
    --job.config_file ./torchtitan/models/llama3/train_configs/debug_model.toml \
    --model.hf-assets-path <ms_tokenizer_path> \
    --comm.mode hierarchical \
    --training.steps 1 \
    --training.local_batch_size 1 \
    --training.seq_len 256 \
    --metrics.log_freq 1 \
    --metrics.disable_color_printing true \
    --job.dump_folder /tmp/torchtitan-quickstart-2card
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-2card" fuzzy='xxx' fuzzy='...'
[titan] xxx - root - INFO - Starting job: Llama 3 debug training
...
[titan] xxx - root - INFO - Training starts at step xxx
[titan] xxx - root - INFO - step:  xxx  loss:  xxx...
[titan] xxx - root - INFO - Training completed
[titan] xxx - root - INFO - Process group destroyed
```

`torchtitan.train` 在 DDP 训练收尾后通过 `torch.distributed.checkpoint.save(...)` 把每个 rank 的权重 shard + optimizer state 写到 `--job.dump_folder`：

```shell #test id="torchtitan-dump-2card"
find /tmp/torchtitan-quickstart-2card -mindepth 1 -maxdepth 3 -printf '%p\n' | head -20
```

```shell #test-result id="torchtitan-dump-2card" fuzzy='xxx' fuzzy='...'
/tmp/torchtitan-quickstart-2card/xxx
...
```
