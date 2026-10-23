# Quick Start (Ascend NPU)

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 2 — CANN / 驱动预留约 2.5 GB，每张卡 PyTorch 实测可用 ~30 GB）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.2.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.2.0 |
| torch | 2.14.0 |
| torch_npu | 2.14.0 |
| triton-ascend | 3.6.0 |
| modelscope | 1.40.2 |
| torchtitan | 最新 release |
| 训练配置 | 单卡 / 多卡均使用 `torchtitan/models/llama3/config_registry.py::llama3_debugmodel_ce_loss`（上游提供的 CrossEntropyLoss 变体，基于 debugmodel：dim=256 / 6 层 / 16 head / vocab 2048，~6 M 参数；多卡为 FSDP shard=2 + bf16） |


### 检查前置是否满足

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出类似：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.1                   Version: 25.5.1                                               |
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

> 如果 `HBM-Usage` 异常偏高（如 >30 GB 且进程表为空），通常是宿主机侧的其他占用（容器内看不到持有进程），本文档 debugmodel 规模（~6 M 参数）在剩余 ~2 GB HBM 下实测仍可完成。

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

装 `torch` + `torch_npu` ：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.14.0
uv pip install torch_npu==2.14.0
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

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。注意 torch_npu 2.14 配套 CANN 9.2.0，旧镜像里的 CANN 9.1.0 会出现 `libhccl.so` 找不到或 aclnn 符号缺失等加载错误。

### 安装 modelscope

```shell #test-setup
uv pip install modelscope==1.40.2
```

打印安装版本：
```shell #test id="modelscope-install"
python -c "import modelscope; print('modelscope', modelscope.__version__)"
```

输出结果如下：
```shell #test-result id="modelscope-install"
modelscope 1.40.2
```
### 安装 triton-ascend

NPU 上 `torch.compile`/inductor 走到 `torch_npu._inductor` 时需要 Ascend 的 Triton fork **triton-ascend**（为 `triton` 模块提供 Ascend 后端）；社区版 `triton` 只有 CUDA 后端，装了会在首次编译 kernel 时报 `RuntimeError: 0 active drivers ([]). There should only be one.`。triton-ascend 的版本号对齐它 fork 的 triton 基线：torch 2.14 配套 triton 3.6，使用华为云 test 源的 3.6.0 正式版 wheel。`--no-deps` 仍然必要——不让 wheel 声明的社区版 `triton` 依赖进环境，`triton/` 目录只归属 triton-ascend 一个包（混装后 `import triton` 会报符号缺失）。`--no-deps` 跳过的依赖里，被 Ascend 后端运行期 import 的只有 **pybind11**（后端 driver 在首次编译 kernel 时用它把 `npu_utils.cpp` 现场编成扩展，需要环境里有 g++/clang++ 和已 source 的 CANN 环境），单独补装；其余（numpy/pytest/pandas 等）是上游打包夹带的测试依赖，运行期不 import：

```shell #test-setup
uv pip install --no-deps --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi/test triton-ascend==3.6.0
uv pip install pybind11
```

打印安装版本：
```shell #test id="triton-install"
python -c "from importlib.metadata import version; print('triton-ascend', version('triton-ascend'))"
```

输出结果如下：
```shell #test-result id="triton-install"
triton-ascend 3.6.0
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

```{admonition} Note
:class: note
`xxx` 为模糊占位符，表示安装的最新版本号。
```

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

```{admonition} Note
:class: note
`<ref>` 不是字面文本，是上文 `store="upstream_ref"` 块捕获的注入参数——工作流把 `UPSTREAM_REF` 环境变量（pytorch/torchtitan 最新 release tag）注入后捕获，`load="upstream_ref>>ref"` 再把它替换进命令里的 `<ref>` 占位符。手工执行时直接替换成目标 tag（如 `v0.3.0`）。
```

输出结果类似如下：

```shell #test-result id="torchtitan-install-source" fuzzy='xxx'
torchtitan xxx
```

```{admonition} Note
:class: note
`xxx` 为模糊占位符，表示安装的最新版本号。
```

## 跑训练

### 下载 tokenizer

```shell #test-setup id="modelscope-download-tokenizer" store="ms_tokenizer_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('LLM-Research/Llama-3.2-1B', allow_patterns=['*.json', '*.model', 'tokenizer*']))" | tail -n 1
```

> 该下载演示 modelscope `snapshot_download` 的标准用法（训练自身使用 torchtitan 自带的 tests/assets/tokenizer，不依赖此路径）。

验证 tokenizer 关键文件都落盘：

```shell #test id="modelscope-verify-tokenizer" load="ms_tokenizer_path>>ms_tokenizer_path"
ls <ms_tokenizer_path>
```

```{admonition} Note
:class: note
`<ms_tokenizer_path>` 不是字面文本，是上一节 `store="ms_tokenizer_path"` 块捕获的注入参数——该块运行 `snapshot_download` 并把 stdout 最后一行（模型落盘路径）捕获存为变量，`load="ms_tokenizer_path>>ms_tokenizer_path"` 再把它替换进本命令的占位符。手工执行时直接替换成上一步打印的路径。
```

```shell #test-result id="modelscope-verify-tokenizer" fuzzy='...' 
config.json
configuration.json
generation_config.json
original
special_tokens_map.json
tokenizer.json
tokenizer_config.json
```

文件名是 Llama 3 tokenizer 必备文件，确认 snapshot_download 命中正确。

### 兼容性说明与 NPU override

torchtitan v0.3.0 + torch 2.14 + torch_npu 2.14.0 + CANN 9.2.0 + triton-ascend 3.6.0 组合上，旧版文档的六条限制中三条已随版本升级消失，两条残留硬墙通过 torchtitan **官方 override 机制**（`--override.imports`，见 `torchtitan/config/override.py`——上游明确设计为 "Swaps any Configurable for an alternative implementation — without modifying config_registry functions or any other in-repo code"）在不改动上游源码的前提下替换组件配置，loss 墙用上游 registry 的现成变体配置规避。本文档全程**零 sed、零 monkey-patch**，torchtitan 源码树保持 git clean。


**8B 规模仍未验证。** 8B 走 `--training.enable-cpu-offload` 路径时权重在 CPU 上经 DTensor dispatch 逐参数 `init_weights`，旧栈实测单个 `trunc_normal_` 超过 1.5 小时不完成（接近活锁），新栈未复测。多卡章节因此仍用 `debugmodel` 双卡（FSDP shard=2 + HCCL 双卡 + DTensor 参数分布 + bf16 全部覆盖），8B 规模验证留待后续。


### 安装 NPU override 模块

override 模块是一个普通的 Python 文件，放在 torchtitan 仓库**之外**（不污染源码树），通过 `PYTHONPATH` + `--override.imports module.function` 激活。`@override(target=...)` 按 Config 类型匹配、按节点 FQN 替换，对 debugmodel 会精确命中全部 6 层的 `model_spec.model.layers.*.attention.rope` 和 `.inner_attention` 节点：

```shell #test-setup
mkdir -p "$HOME/tt-npu-overrides"
cat > "$HOME/tt-npu-overrides/npu_compat.py" << 'PYEOF'
"""NPU compat overrides for torchtitan v0.3.0 on Ascend (CANN 9.2.0 stack).

Uses torchtitan's official @override mechanism (--override.imports).
No upstream source file is modified.

- ComplexRoPE -> CosSinRoPE: aclnnIndex has no DT_COMPLEX64 impl (CANN 9.2.0).
- FlexAttention -> ScaledDotProductAttention: NPU flex path gaps
  (mask_out template MLIR fail; document-masking aten.index subgraph).
  SDPA dispatches to aclnn flash attention.
"""
from torchtitan.config.override import derive, override
from torchtitan.models.common.attention import (
    FlexAttention,
    ScaledDotProductAttention,
)
from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE


@override(
    target=ComplexRoPE.Config,
    description="complex64 RoPE cache -> real cos/sin (aclnnIndex lacks DT_COMPLEX64)",
)
def cos_sin_rope(cfg: ComplexRoPE.Config) -> CosSinRoPE.Config:
    return derive(cfg, CosSinRoPE.Config, scaling="none")


@override(
    target=FlexAttention.Config,
    description="flex attention -> SDPA (aclnn flash attention)",
)
def sdpa(cfg: FlexAttention.Config) -> ScaledDotProductAttention.Config:
    return ScaledDotProductAttention.Config()
PYEOF
```

验证两个 override 工厂都完成注册：

```shell #test id="override-register"
PYTHONPATH="$HOME/tt-npu-overrides" python -c "import npu_compat; from torchtitan.config.override import _REGISTRY; print(sorted(_REGISTRY))"
```

输出结果如下：

```shell #test-result id="override-register"
['npu_compat.cos_sin_rope', 'npu_compat.sdpa']
```

### 单卡训练

用 `torchrun --nproc_per_node=1` 在 1 张 NPU 上跑 `debugmodel` 真跑 2 步，验证配置解析、初始化、加载 tokenizer、build dataloader、forward + backward 整条链路能跑通。`llama3_debugmodel_ce_loss` 是上游 registry 自带的 CrossEntropyLoss 变体（~6 M 参数量，单卡 30 GB 完全够装）；`--override.imports` 激活上一步的两个 NPU 兼容替换（rope + attention，日志会打印 12 个节点的替换明细和 `Applied 12 override(s)`）。走真实 HCCL backend（`--comm.mode default`）让 c10d 把 `npu` 路由到 `hccl`，1-rank 下所有集合通信都是 self-barrier，不会真的有跨卡流量。多卡 FSDP shard 训练见下一节「多卡训练」（8B 规模在本栈暂未验证，见「兼容性说明与 NPU override」）：

```shell #test id="torchtitan-train-debug" load="upstream_ref>>ref"
cd torchtitan && git checkout -f <ref>
PYTHONPATH="$HOME/tt-npu-overrides" \
ASCEND_RT_VISIBLE_DEVICES=0 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
torchrun --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    --module llama3 \
    --config llama3_debugmodel_ce_loss \
    --override.imports npu_compat.cos_sin_rope npu_compat.sdpa \
    --comm.mode default \
    --training.steps 2 \
    --training.local-batch-size 1 \
    --training.seq-len 256 \
    --metrics.log-freq 1 \
    --metrics.disable-color-printing \
    --dump-folder /tmp/torchtitan-quickstart
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-debug" fuzzy='xxx' fuzzy='...'
[titan] xxx - root - INFO - torchtitan version: xxx
[titan] xxx - root - INFO - [Override] npu_compat.cos_sin_rope: model_spec.model.layers.0.attention.rope ComplexRoPE.Config -> CosSinRoPE.Config
...
[titan] xxx - root - INFO - Applied 12 override(s)
[titan] xxx - root - INFO - Building llama3 debugmodel
...
[titan] xxx - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] xxx - root - INFO - Training completed
[titan] xxx - root - INFO - Process group destroyed
```

### 多卡训练

用 torchrun 起 2 个 rank 跑 `debugmodel` 真分布式训练，`--training.steps 2` 真跑 2 步。`data_parallel_shard_degree = -1` 在双卡下解析成 2，FSDP 把 params / grads / Adam state 按 shard 分摊到两张卡，验证 HCCL 双卡集合通信 + FSDP shard>1 + DTensor 参数分布整条链路；叠 `--training.dtype bfloat16` 验证混合精度。多卡使用**默认 spmd backend**（torch 2.14 的 FSDP 注解翻译就位，双卡实测通过，无需 `--parallelism.spmd-backend` 覆盖）；`HCCL_NPU_SOCKET_PORT_RANGE` 用于错开同容器两个 rank 的 HCCL 监听端口（否则 `EI0020` 端口冲突，见「兼容性说明与 NPU override」）：

```shell #test id="torchtitan-train-2card" load="upstream_ref>>ref"
cd torchtitan && git checkout -f <ref>
PYTHONPATH="$HOME/tt-npu-overrides" \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
HCCL_NPU_SOCKET_PORT_RANGE=60000-60200 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 \
    --tee 3 \
    -m torchtitan.train \
    --module llama3 \
    --config llama3_debugmodel_ce_loss \
    --override.imports npu_compat.cos_sin_rope npu_compat.sdpa \
    --comm.mode default \
    --training.dtype bfloat16 \
    --training.steps 2 \
    --training.local-batch-size 1 \
    --training.seq-len 256 \
    --metrics.log-freq 1 \
    --metrics.disable-color-printing \
    --dump-folder /tmp/torchtitan-quickstart-2card
```

输出结果类似如下：

```shell #test-result id="torchtitan-train-2card"
[default0]:[titan] ... - root - INFO - torchtitan version: ...
[default0]:[titan] ... - root - INFO - Building device mesh with parallelism: ...
[default0]:[titan] ... - root - INFO - Applied 12 override(s)
[default0]:[titan] ... - root - INFO - Building llama3 debugmodel
...
[default0]:[titan] ... - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
...
[default0]:[titan] ... - root - INFO - Sleeping 2 seconds for other ranks to complete
[default0]:[titan] ... - root - INFO - Training completed
[default0]:[titan] ... - root - INFO - Process group destroyed
```

> 8B 规模随 cpu-offload 初始化问题在新栈复测通过后恢复。
