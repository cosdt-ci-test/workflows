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

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，32 GB × 2 — CANN / 驱动预留约 2.5 GB，每张卡可用约 30 GB）
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
| 训练配置 | 单卡 / 多卡均使用上游默认配置 `torchtitan/models/llama3/config_registry.py::llama3_debugmodel`（默认 `ChunkedLossWrapper` loss；基于 debugmodel：dim=256 / 6 层 / 16 head / vocab 2048，~6 M 参数；多卡为 FSDP shard=2 + bf16） |


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

> 如果 `HBM-Usage` 异常偏高（如 >30 GB 且进程表为空），通常是宿主机侧的其他占用（容器内看不到持有进程），本文档 debugmodel 规模（~6 M 参数）在剩余 ~2 GB HBM 下仍可完成。

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

torchtitan v0.3.0 的语言模型默认使用 flex attention 与 complex64 RoPE 缓存，二者在 NPU 上均不可用：flex 路径在 torch_npu 2.14.0 上编不出来，aclnnIndex 则没有 DT_COMPLEX64 实现。本文档用 torchtitan **官方 override 机制**（`--override.imports`，见 `torchtitan/config/override.py`）替换这两个组件，**全程不修改上游源码**，torchtitan 源码树保持 git clean。

loss 用上游自己的默认配置（`llama3_debugmodel` → `ChunkedLossWrapper`），不做任何绕路。

**flex 路径不可用的准确根因**（2026-10-26 在 910B4 上逐层实测，三层各自独立成立）：

1. `torch_npu` 2.14.0 的 `lowering.make_reduction` 缺 `strict_sum` 形参，而 torch 2.14 的 inductor `sum_` lowering 总是传它 —— NPU 上**任何**被编译的 reduction 都会失败（不限于 flex；torchtitan 的 `compile` 特性整体不可用）。
2. `triton-ascend` 3.6.0 在 `triton/backends/ascend/utils.py:709` 硬编码 `-std=c++17`，而 torch 2.14 的 ATen headers 要求 C++20 —— 运行期扩展 `npu_utils` 编不出来。
3. `torch_npu` 自己的 flex lowering（`torch_npu/_inductor/kernel/flex_attention.py:746`）抛 `SubgraphLoweringException: Buffers cannot be created while lowering a pointwise subgraph` —— 这是真实的能力缺口；①②属于依赖版本错配。

**override 换进去的组件与上游数值等价**（这一点由下一节的等价性检查在每次运行时断言）：

- **RoPE**：上游 `ComplexRoPE` 的 complex64 缓存是用位置 id 直接索引的（`aclnnIndex` 无 `DT_COMPLEX64` 实现）。override 改为把缓存看成 float32 的实部/虚部两半、在实数域做同样的 gather，再交回上游自己的旋转代码 —— 缓存数学（含 llama 频率缩放）和相邻维配对约定都是上游的，实测与上游 `ComplexRoPE` 输出**逐位相同**。
  （注意：上游自带的 `CosSinRoPE` **不能**当替代品 —— 它用 rotate-half 配对约定 `(d0,d8),(d1,d9)…`，且对 `scaling="llama"` 直接抛 `NotImplementedError`，与上游 `ComplexRoPE` 的相对误差约 113%。）
- **逐文档 block-causal 掩码**：上游只给 Flex / Varlen backend 构造 per-document 掩码（`trainer.py:738-748`），而 upstream 的 SDPA 被显式禁止用作语言模型 backend（`get_attention_config("sdpa")` 直接抛错，理由就是它吃不了 per-document 位置）。override 换成一个带**显式 dense block-causal 掩码**的 SDPA（aclnn SDPA 支持 `attn_mask`），保留上游的掩码语义：一个 token 只关注同一文档中在它之前的 token。

**8B 规模不在本文档的验证范围。** 8B（~8.03 B 参数）在双卡 32 GB 上需要 `--training.enable-cpu-offload` 才装得下（FSDP shard=2 后每卡约 48 GB > 32 GB HBM），而该组合会触发 PyTorch 上游已知缺陷：DTensor 的 RNG tracker 对「device 类型 mesh + CPU local tensor」不推进 CPU RNG，导致 `nn.init.trunc_normal_` 的拒绝采样循环不收敛（`pytorch/pytorch#180088`、`pytorch/pytorch#196072`，修复方向 `pytorch/pytorch#199209`）。因此本文档的多卡章节使用 `debugmodel` 双卡（FSDP shard=2 + HCCL 双卡 + DTensor 参数分布 + bf16 全部覆盖）。


### 安装 NPU override 模块

override 模块是本仓（cosdt-ci-test/workflows）里的两个普通 Python 文件，放在 torchtitan 仓库**之外**（不污染源码树），通过 `PYTHONPATH` + `--override.imports module.function` 激活：

| 文件 | 作用 |
| --- | --- |
| `projects/torchtitan/example/npu_compat.py` | override 模块本身：注册一个 `npu_attention` 工厂，替换 `GQAttention.Config` 节点 |
| `projects/torchtitan/example/check_npu_overrides.py` | 等价性自检脚本（CPU）：断言替换进去的组件与上游组件数值等价 |

`@override(target=...)` 按 Config 类型匹配、按节点 FQN 替换。因为上游框架**禁止嵌套 override**（一个 override 声明了另一个 override 节点的祖先就会报错），RoPE 与 inner attention 两处替换折进同一个 `GQAttention.Config` override 里 —— 对 debugmodel 会精确命中全部 6 层的 `model_spec.model.layers.*.attention` 节点（每个节点内部再换 rope / inner_attention）：

```shell #test-setup
mkdir -p "$HOME/tt-npu-overrides"
BASE="https://api.github.com/repos/cosdt-ci-test/workflows/contents/projects/torchtitan/example"
REF="${DOC_REF:-main}"
AUTH=()
[ -n "${GH_TOKEN:-}" ] && AUTH+=(-H "Authorization: Bearer $GH_TOKEN")
for f in npu_compat.py check_npu_overrides.py; do
  curl -fsSL --retry 3 --retry-connrefused --retry-delay 2 \
    -H "Accept: application/vnd.github.raw" "${AUTH[@]}" \
    "$BASE/$f?ref=$REF" -o "$HOME/tt-npu-overrides/$f"
done
```

> `${DOC_REF:-main}` 是被本文档自身所在的 ref（CI 里由测试注入 `DOC_REF`，手工执行时默认 `main`），保证拉到的 override 模块与本文档来自同一个版本。
> 走 contents API 而不是 raw.githubusercontent.com：NPU runner 到 raw 端点不通，到 api 端点通（base class 里也是这么取文档的）；仓库是 public，匿名即可，`GH_TOKEN` 存在时带上以提高限流额度。

验证 override 工厂完成注册：

```shell #test id="override-register"
PYTHONPATH="$HOME/tt-npu-overrides" python -c "import npu_compat; from torchtitan.config.override import _REGISTRY; print(sorted(_REGISTRY))"
```

输出结果如下：

```shell #test-result id="override-register"
['npu_compat.npu_attention']
```

再断言替换进去的组件与上游组件**数值等价**（CPU 即可跑，不需要 NPU）：RoPE 与上游 `ComplexRoPE` 的输出必须逐位一致，attention 与朴素实现的逐文档 causal 掩码注意力必须一致。

```shell #test id="override-equivalence"
python "$HOME/tt-npu-overrides/check_npu_overrides.py"
```

输出结果如下：

```shell #test-result id="override-equivalence"
npu override equivalence: rope OK, mask OK, rope selection OK
```

### 单卡训练

用 `torchrun --nproc_per_node=1` 在 1 张 NPU 上跑 `debugmodel` 真跑 2 步，验证配置解析、初始化、加载 tokenizer、build dataloader、forward + backward 整条链路能跑通。`llama3_debugmodel` 是上游 registry 的默认配置（loss = 默认 `ChunkedLossWrapper`，~6 M 参数量，单卡 30 GB 完全够装）；`--override.imports` 激活上一步的 NPU 兼容替换（日志会打印 6 个节点的替换明细和 `Applied 6 override(s)`）。走真实 HCCL backend（`--comm.mode default`）让 c10d 把 `npu` 路由到 `hccl`，1-rank 下所有集合通信都是 self-barrier，不会真的有跨卡流量。多卡 FSDP shard 训练见下一节「多卡训练」（8B 规模不在本文档验证范围，见「兼容性说明与 NPU override」）：

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
    --config llama3_debugmodel \
    --override.imports npu_compat.npu_attention \
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
[titan] xxx - root - INFO - [Override] npu_compat.npu_attention: model_spec.model.layers.0.attention GQAttention.Config -> NpuGQAttention.Config
...
[titan] xxx - root - INFO - Applied 6 override(s)
[titan] xxx - root - INFO - Building llama3 debugmodel
...
[titan] xxx - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] xxx - root - INFO - Training completed
[titan] xxx - root - INFO - Process group destroyed
```

### 多卡训练

用 torchrun 起 2 个 rank 跑 `debugmodel` 真分布式训练，`--training.steps 2` 真跑 2 步。`data_parallel_shard_degree = -1` 在双卡下解析成 2，FSDP 把 params / grads / Adam state 按 shard 分摊到两张卡，验证 HCCL 双卡集合通信 + FSDP shard>1 + DTensor 参数分布整条链路；叠 `--training.dtype bfloat16` 验证混合精度。多卡使用**默认 spmd backend**（torch ≥2.13 的 FSDP 注解翻译已就位，无需 `--parallelism.spmd-backend` 覆盖）；`HCCL_NPU_SOCKET_PORT_RANGE` 用于错开同容器两个 rank 的 HCCL 监听端口（否则报 `EI0020` 端口冲突）：

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
    --config llama3_debugmodel \
    --override.imports npu_compat.npu_attention \
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
[default0]:[titan] ... - root - INFO - Applied 6 override(s)
[default0]:[titan] ... - root - INFO - Building llama3 debugmodel
...
[default0]:[titan] ... - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
...
[default0]:[titan] ... - root - INFO - Sleeping 2 seconds for other ranks to complete
[default0]:[titan] ... - root - INFO - Training completed
[default0]:[titan] ... - root - INFO - Process group destroyed
```
