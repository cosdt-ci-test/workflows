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

### 通过 pip 安装（二进制 wheel）

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

`--comm.mode fake_backend` 让 torchtitan 跳过 NCCL/HCCL 集合通信初始化、用 fake process group 跑 1 个 rank 的纯 NPU 计算——验证 toml 配置解析 + init_distributed + 模型搬到 NPU + dataloader ready + trainer.train() 收尾这条**最小**路径（多卡 / 真分布式属于另一个配置面）。配 `--training.steps 0` 让 `should_continue_training()` 直接返回 False、不进 `while` 循环，从而跳过 `train_step` 的实际 forward + backward：

> v0.2.2 在 `--comm.mode fake_backend` 分支强制要求 `NGPU=<world_size>` env var（见 `torchtitan/distributed/utils.py:307`），否则 `init_distributed` 直接 `raise ValueError`。fake mode 把 `NGPU` 当 fake world size，不读 `WORLD_SIZE`（那是真分布式路径由 torchrun 注入）。

> **为什么不真跑 train_step（forward + backward）**：upstream `debug_model.toml` 的 `flavor = "debugmodel"` 把 `vocab_size` 写死成 **2048**，但本 quick-start 用的真 Llama 3 tokenizer `vocab_size = 128256`——vocab mismatch。`c4_test` 数据集含 `token_id > 2048` 的词，forward 时 `nn.Embedding` gather 越界访问 NPU GM 内存；torch_npu 的 embedding kernel 不做 index bound check，越界 OOB 读不立即抛 `IndexError`，等到 `loss.detach().item()` sync 等 NPU stream 时 NPU 报 `vector core error 0x800000` / `MTE accesses an invalid GM address`，几乎所有 vector core 集体挂。tyro 把 `vocab_size` 当作 `llama3_args` dict 的 key（不在 `Model` config 的 CLI 命名空间），无法用 `--model.vocab-size` CLI override；upstream 也没 vocab=128256 的小 flavor——真跑 train_step 需要 fork torchtitan 添加新 flavor（例如 `dim=256, n_layers=6, n_heads=16, vocab_size=128256`）。

> **为什么不验证 `--job.dump-folder` 落盘**：单卡 fake_backend 路径下 `checkpointer.save()` 会经 `torch.distributed.checkpoint.save → reduce_scatter → gather_object → torch_npu._gather_object → _tensor_to_object`，而 `torch_npu._gather_object` 在 fake_backend 上的 pickle roundtrip 异常（`_pickle.UnpicklingError: invalid load key, '\x00'`，CI 33065970592 traceback）。真分布式 `gather_object` 才能走 HCCL 路径——但本 quick-start 不真跑 train_step（前段解释），save 也不触发。本 quick-start 单卡范围只在 stdout 日志层验证 init + dataloader ready + `trainer.train()` 收尾链路；save 落盘验证属于生产训练配置面，需要先解决上游 vocab 限制再叠真分布式，超出 quick-start 范围。

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

`--comm.mode hierarchical` 走真实 HCCS 集合通信（单机内多卡用 HCCS 做 ring-allreduce），配合 `torchrun --nproc_per_node=2` 起 2 个 rank 的 DDP——验证 collective 初始化 + DDP 梯度同步 + ProcessGroup 销毁这条**真实分布式**链路。多卡场景下 `--checkpoint.create-seed-checkpoint` 不可用（upstream assert `WORLD_SIZE == 1`，见 `torchtitan/train.py:760`），且本 quick-start 不真跑 train_step（理由同上：debugmodel vocab=2048 vs tokenizer vocab=128256 不匹配），所以 train_step 的 while 循环 0 次迭代、save 不触发——本 quick-start 多卡范围只在 stdout 日志层验证 collective 链路：

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
    --module torchtitan.train \
    --job.config-file ./torchtitan/models/llama3/train_configs/debug_model.toml \
    --model.hf-assets-path <ms_tokenizer_path> \
    --comm.mode hierarchical \
    --training.steps 0 \
    --training.local-batch-size 1 \
    --training.seq-len 256 \
    --metrics.log-freq 1 \
    --metrics.disable-color-printing \
    --job.dump-folder /tmp/torchtitan-quickstart-2card
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-2card" fuzzy='xxx' fuzzy='...'
[titan] xxx - root - INFO - Starting job: Llama 3 debug training
...
[titan] xxx - root - INFO - Training starts at step xxx
[titan] xxx - root - INFO - Training completed
[titan] xxx - root - INFO - Process group destroyed
```

> 多卡场景下**不验证 `--job.dump-folder` 落盘**：steps=0 不触发 `checkpointer.save`（save 在 while 循环内部，`torchtitan/train.py:680`），且 2-rank sharded checkpoint 的保存需要先解决上游 vocab 限制（fork torchtitan 加 vocab=128256 toy flavor）才跑 forward→save——属于生产训练配置面，超出本 quick-start 范围。
