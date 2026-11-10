# torchtitan

[TorchTitan](https://github.com/pytorch/torchtitan) 是 PyTorch 原生的 LLM 预训练平台，通过 `torch_npu` 把 FSDP / TP / PP / CP 等并行策略的计算调度到昇腾 NPU。

本文在昇腾 NPU 上跑通 TorchTitan 的安装与训练链路：

- **安装**：用 uv 装 PyPI 上的二进制 wheel（最新 stable release），再从**同一个 release tag** 源码安装（可编辑模式），训练用源码树。
- **NPU 兼容 override**：用 torchtitan 官方 override 机制替换 attention 栈（RoPE + attention），训练前先断言替换组件与上游组件数值等价。
- **torch 版本兼容**：torchtitan v0.4.0 用了两个 torch 2.15 才有的符号，昇腾的 torch 只到 2.14.0 —— 用一层启动时自动装载的兼容垫片补齐（同样不改上游源码）。
- **单卡训练**：本文自带的 recipe（llama3 `debugmodel`，~6 M 参数）真跑 2 步，验证配置解析、dataloader 构造、forward + backward 整条链路。
- **多卡训练**：双卡 FSDP shard + bfloat16 训练，验证 HCCL 集合通信与 DTensor 参数分布。

多卡训练需要至少 2 张 NPU。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境（3.11 及以上，TorchTitan 要求 `requires-python >= 3.11`）
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

```{admonition} Note
:class: note
CANN 最低版本为 8.0.rc1，安装 CANN 时请同时安装 Kernel 算子包。
```

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
| torch | 2.14.0+cpu |
| torch_npu | 2.14.0 |
| triton-ascend | 3.6.0 |
| torchtitan | 最新 stable release（二进制 wheel 与源码安装用同一个 tag，当前 v0.4.0） |
| torch 兼容垫片 | `npu_torch_compat.py` + `sitecustomize.py`：torchtitan v0.4.0 需要 torch ≥ 2.15 的两个 `torch.distributed.pipelining` 符号，而昇腾的 torch 上限是 2.14.0 |
| 训练配置 | 本文自带的 recipe `npu_recipe.py::llama3_npu_debugmodel`（单卡 fp32）/ `llama3_npu_debugmodel_bf16`（多卡 bf16）：模型是 llama3 `debugmodel`（dim=256 / 6 层 / 16 head / vocab 2048，~6 M 参数），数据是上游自带的 `tests/assets/c4_test`，tokenizer 是上游自带的 `tests/assets/tokenizer`（vocab 2020），不下载任何权重 |

```{admonition} Note
:class: note
**上游的配置是 recipe 式的**：torchtitan 不再提供 `torchtitan/models/*/config_registry.py` 里的配置注册表，
配置改成「一个返回 `Trainer.Config` 的普通 Python 函数」，命令行用 `--module <模块> --config <函数名>` 选择；
`--training.steps` / `--comm.mode` / `--dump-folder` 这类通用开关也一并消失，只保留
`--module` / `--config` / `--override` / `--comm-backend` / `--output-dir` / `--resume-step` / `--print-config`。
所以本文自带一个 recipe（`npu_recipe.py`），把模型、数据、步数写死，不依赖上游的配置文件 ——
换上游版本时只要 recipe 里的 API 还对得上就行。
```

## 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出类似：

```text
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.1                   Version: 25.5.1                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+---------------------------+---------------+-----------------------------+------------------------+
| No running processes found in NPU 5                                                            |
+---------------------------+---------------+-----------------------------+------------------------+
```

```{admonition} Note
:class: note
如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。
```

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

装 `torch` + `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.14.0
uv pip install torch_npu==2.14.0
```

检查 torch / torch_npu 是否装好且 NPU 设备可用(下面的命令用 Python 执行)：

```python #test id="check-torch"
import torch, torch_npu
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
```

输出结果如下：

```shell #test-result id="check-torch"
torch= 2.14.0+cpu
torch_npu= 2.14.0
is_available: True
count: 2
```

```{admonition} Note
:class: note
如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。
```

### 安装 triton-ascend

`triton-ascend` 是 Ascend 的 Triton fork，只发布在华为云的 Ascend PyPI 源
`https://repo.huaweicloud.com/ascend/repos/pypi`（该源同时提供 3.2.x / 3.6.x 的
cp39–cp313 aarch64 wheel）。`--no-deps` 是必需的：triton-ascend 的 wheel 声明了社区版
`triton` 依赖，装上后两个包会同时写 `triton/` 目录，混装会在 `import triton` 时报
`cannot import name 'Language'`。

```shell #test-setup
set -euo pipefail
uv pip install --no-deps --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi triton-ascend==3.6.0
uv pip install pybind11
```

打印安装版本(下面的命令用 Python 执行)：

```python #test id="triton-install"
from importlib.metadata import version
print('triton-ascend', version('triton-ascend'))
```

输出结果如下：

```shell #test-result id="triton-install"
triton-ascend 3.6.0
```

## 安装 torchtitan

### 通过 uv 安装

torchtitan 上 PyPI 有 `py3-none-any` 的纯 Python wheel，不依赖 native build——`uv pip install torchtitan` 直接装最新 stable release，不用管版本号：

```shell #test-setup
printf 'attn-gym\n' > /tmp/torchtitan-excludes.txt
uv pip install --excludes /tmp/torchtitan-excludes.txt --extra-index-url https://mirrors.aliyun.com/pypi/simple/ torchtitan
```

```{admonition} Note
:class: note
`--excludes` 排除的是 `attn-gym`：上游把 `attn-gym[linear,cudnn]==0.0.17` 写成了硬依赖，
而它的 `cudnn` / `linear` extra 会拉 `cuda-python>=13` 与 `nvidia-cutlass-dsl[cu13]` —— 两者都是 CUDA 专有组件，
在昇腾 NPU 上既用不到，也会白占几个 GB 磁盘。`attn-gym` 本体是纯 Python wheel，它提供的只是 CUDA attention
kernel；torchtitan 对此有官方降级支持（`torchtitan/models/common/attn_gym.py` 用占位 kernel 兜底，
只有真的要用 GDN / KDA 这类线性注意力模型时才会报错），所以在 NPU 上直接排除它最省事。
```

验证装上的版本(下面的命令用 Python 执行)：

```python #test id="torchtitan-install-binary"
import torchtitan
print('torchtitan', torchtitan.__version__)
```

输出结果类似如下：

```shell #test-result id="torchtitan-install-binary" fuzzy='xxx'
torchtitan xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号。
```

### 从源码安装

二进制 wheel 只用来验证「最新版本能不能在昇腾 NPU 上装上」；训练要用源码树（recipe 与 override 都放在
源码树外面，通过 `PYTHONPATH` 挂进去），所以再从**同一个 release tag** 装一次可编辑模式。

先解析出要用的 release tag：CI 里由工作流注入 `UPSTREAM_REF`（上游最新 release 的 tag），
本地手工执行时回落到 GitHub API 查最新 release：

```shell #test-setup store="upstream_ref"
set -o pipefail
if [ -n "${UPSTREAM_REF:-}" ]; then
  echo "${UPSTREAM_REF}"
else
  curl -fsSL --retry 3 --retry-connrefused --retry-delay 2 \
    https://api.github.com/repos/pytorch/torchtitan/releases/latest \
    | python -c "import json,sys; print(json.load(sys.stdin)['tag_name'])"
fi
```

克隆上游仓库、`uv pip install -e .` 安装：

```shell #test-setup load="upstream_ref>>ref"
set -euo pipefail
git clone --depth 1 --branch <ref> https://github.com/pytorch/torchtitan.git
cd torchtitan
uv pip install --excludes /tmp/torchtitan-excludes.txt -e .
```

```{admonition} Note
:class: note
`<ref>` 是上一步解析出的 release tag（当前 `v0.4.0`）。用 release tag 而不是 `main`：
recipe 用到的 API（`Trainer.Config` 的字段、`GQAttention.Config` 的结构、override 机制）在
release 上是稳定的，`main` 上游随时可能改。

`uv pip install -e .` 会把上一步装的二进制 wheel 换成可编辑安装（同一个 distribution 名）。
`--excludes` 沿用上一节排除 `attn-gym` 的理由：源码安装同样会解析它的硬依赖。
```

验证装上的版本(下面的命令用 Python 执行，需在克隆出来的源码目录里跑)：

```shell #test id="torchtitan-install-source"
cd torchtitan && python -c "import torchtitan; print('torchtitan', torchtitan.__version__)"
```

输出结果类似如下：

```shell #test-result id="torchtitan-install-source" fuzzy='xxx'
torchtitan xxx
```

```{admonition} Note
:class: note
xxx 表示最新的版本号。
```

## 跑训练

### 安装 NPU override 模块与 recipe

override 模块和 recipe 都是本仓（cosdt-ci-test/workflows）里的普通 Python 文件，放在 torchtitan 仓库**之外**，
通过 `PYTHONPATH` 挂进去：

| 文件 | 作用 |
| --- | --- |
| `projects/torchtitan/example/npu_compat.py` | override 模块本身：注册一个 `npu_attention` 工厂，替换 `GQAttention.Config` 节点 |
| `projects/torchtitan/example/npu_compat_deepseek.py` | deepseek_v3 专用的 override（它自己的 attention 类不是 `GQAttention`） |
| `projects/torchtitan/example/npu_recipe.py` | quick-start 的 recipe：返回 `Trainer.Config`（模型 / 数据 / 步数 / loss） |
| `projects/torchtitan/example/check_npu_overrides.py` | 等价性自检脚本（CPU）：断言替换进去的组件与上游组件数值等价 |
| `projects/torchtitan/example/npu_torch_compat.py` | torch 版本兼容垫片：补上 torchtitan 需要、但 torch 2.14 没有的两个 `torch.distributed.pipelining` 符号 |
| `projects/torchtitan/example/sitecustomize.py` | 启动钩子：解释器启动时自动装上兼容垫片（`PYTHONPATH` 指向本目录即生效） |

override 模块由 torchtitan 官方的 override 机制激活（`--override npu_compat.npu_attention`），
**不修改上游源码**——torchtitan 的 checkout 保持 git clean。

```{admonition} Note
:class: note
`${DOC_REF:-main}` 是被本文档自身所在的 ref（CI 里由测试注入 `DOC_REF`，手工执行时默认 `main`），
保证拉到的 override 模块与本文档来自同一个版本；走 contents API 而不是 raw.githubusercontent.com：
NPU runner 到 raw 端点不通，到 api 端点通（base class 里也是这么取文档的）；仓库是 public，匿名即可，
`GH_TOKEN` 存在时带上以提高限流额度。
```

```shell #test-setup
mkdir -p "$HOME/tt-npu-overrides"
BASE="https://api.github.com/repos/cosdt-ci-test/workflows/contents/projects/torchtitan/example"
REF="${DOC_REF:-main}"
AUTH=()
[ -n "${GH_TOKEN:-}" ] && AUTH+=(-H "Authorization: Bearer $GH_TOKEN")
for f in npu_compat.py npu_compat_deepseek.py npu_recipe.py check_npu_overrides.py \
         npu_torch_compat.py sitecustomize.py; do
  curl -fsSL --retry 3 --retry-connrefused --retry-delay 2 \
    -H "Accept: application/vnd.github.raw" "${AUTH[@]}" \
    "$BASE/$f?ref=$REF" -o "$HOME/tt-npu-overrides/$f"
done
```

### torch 2.15 兼容垫片

torchtitan v0.4.0 会**顺带**导入它可选的 Dist-MoE 支持，而那个包的 runtime 模块第一行就是：

```python
from torch.distributed.pipelining import (
    analyze_pipeline_activation_liveness,
    PipelineStageInfo,
)
```

这两个符号是 **torch 2.15.0** 才加的。而昇腾这边 `torch_npu` 最新只到 **2.14.0**（并且它把
`torch` 钉死在 `2.14.0.*`），所以 torch 2.14.0 就是 NPU 栈能跑到的天花板。在 torch 2.14 上，
**任何** torchtitan model flavor 的 import 链都会在建模之前就断掉：

```
torchtitan.train → torchtitan.trainer → torchtitan.components.data
  → torchtitan.models.deepseek_v3 → torchtitan.config.transform
  → torchtitan.models.common.dist_moe → runtime
  → ImportError: cannot import name 'analyze_pipeline_activation_liveness'
```

还有第二处：torchtitan v0.4.0 是对着 **torch 2.15** 开发的（它自己的 CI 就钉
`torch==2.15.0`），而 FSDP2 的 mixed-precision policy 在 2.15 多了一个按参数覆盖 dtype 的回调：

```python
# torchtitan/distributed/fsdp.py
mp_policy = MixedPrecisionPolicy(
    param_dtype=param_dtype,
    reduce_dtype=reduce_dtype,
    cast_forward_inputs=False,
    param_dtype_override_fn=param_dtype_override_fn,   # ← torch 2.15 才有
)
```

torch 2.14 的 `MixedPrecisionPolicy` 是只有 `param_dtype` / `reduce_dtype` / `output_dtype` /
`cast_forward_inputs` 四个字段的 frozen dataclass，于是建模阶段直接炸：

```
torchtitan.trainer → training_engine._initialize_model → model.parallelize
  → decoder._apply_fsdp → distributed.fsdp.apply_fsdp_to_decoder
  → TypeError: MixedPrecisionPolicy.__init__() got an unexpected keyword argument
    'param_dtype_override_fn'
```

`npu_torch_compat.py` 把这两处一起补掉：

- 把 `PipelineStageInfo`（照抄 torch 2.15 的定义）和 `analyze_pipeline_activation_liveness`
  （占位实现，调用即报错）塞进 `torch.distributed.pipelining`。Dist-MoE 这个功能本身在这里
  仍然不可用——它还需要 torch ≥ 2.15 和可选的 `dist_moe` 包——但 quick-start 不跑 Dist-MoE，
  import 能过即可。
- 给 `MixedPrecisionPolicy.__init__` 包一层，接受并丢弃 `param_dtype_override_fn`。上游只有在
  这个回调为 `None` 时才走到生产路径（参数默认值就是 `None`，唯一的真实调用方是上游自己的
  单测），所以丢弃 `None` 与 torch 2.15 的行为完全一致；万一传进来一个非 `None` 的回调则直接
  报错——torch 2.14 的 FSDP2 根本没有按参数覆盖 dtype 的能力，静默忽略会改掉数值。

垫片必须在**任何 torchtitan import 之前**装上，所以由 `sitecustomize.py` 承担：CPython 在解释器
启动时会 import `sitecustomize`，只要本目录在 `PYTHONPATH` 上就会生效。下面这条命令把
`PYTHONPATH` 指向本目录，并确认两处补丁都已就位：

```shell #test id="torch-compat"
cd torchtitan && PYTHONPATH="$HOME/tt-npu-overrides" python -c "import torch, torch.distributed.pipelining as p; from torch.distributed.fsdp import MixedPrecisionPolicy; print('analyze_pipeline_activation_liveness:', hasattr(p, 'analyze_pipeline_activation_liveness')); print('PipelineStageInfo:', hasattr(p, 'PipelineStageInfo')); print('param_dtype_override_fn:', MixedPrecisionPolicy(param_dtype=torch.bfloat16, param_dtype_override_fn=None).param_dtype)"
```

输出结果如下：

```shell #test-result id="torch-compat"
analyze_pipeline_activation_liveness: True
PipelineStageInfo: True
param_dtype_override_fn: torch.bfloat16
```

```{admonition} Note
:class: note
后面的每条 torchtitan 命令都带 `PYTHONPATH="$HOME/tt-npu-overrides"`：它同时负责
override 模块 / recipe 的可见性，和这个兼容垫片的自动装载。
```

### 验证 override 与 recipe

验证 override 工厂完成注册：

```shell #test id="override-register"
cd torchtitan && PYTHONPATH="$HOME/tt-npu-overrides" python -c "import npu_compat; from torchtitan.config.override import _REGISTRY; print(sorted(_REGISTRY))"
```

输出结果如下：

```shell #test-result id="override-register"
['npu_compat.npu_attention']
```

再断言替换进去的组件与上游组件**数值等价**（CPU 即可跑，不需要 NPU）：RoPE 与上游 `ComplexRoPE` 的输出必须逐位一致，attention 与朴素实现的逐文档 causal 掩码注意力必须一致，同时确认 override 只替换 `ComplexRoPE` 系模型（qwen3 等用 `CosSinRoPE` 的模型不受影响）。

```shell #test id="override-equivalence"
PYTHONPATH="$HOME/tt-npu-overrides" python "$HOME/tt-npu-overrides/check_npu_overrides.py"
```

输出结果如下：

```shell #test-result id="override-equivalence"
npu override equivalence: rope OK, mask OK, rope selection OK
```

验证 recipe 能被 `--module` 选中，并返回一份可训练的 `Trainer.Config`：

```shell #test id="recipe-import"
cd torchtitan && PYTHONPATH="$HOME/tt-npu-overrides" python -c "import npu_recipe; c = npu_recipe.llama3_npu_debugmodel(); print('recipe ok, steps =', c.training.steps)"
```

输出结果如下：

```shell #test-result id="recipe-import"
recipe ok, steps = 2
```

### 单卡训练

在 1 张 NPU 上用 `torchrun` 真跑 2 步 `debugmodel`（~6 M 参数，单卡足够），验证完整的训练链路。
`--override` 激活上一步的 NPU 兼容替换，日志会打印 `Applied 6 override(s)`（6 层各替换一次）。
多卡训练见下一节「多卡训练」（8B 规模不在本文档验证范围）：

```shell #test id="torchtitan-train-debug"
cd torchtitan && PYTHONPATH="$HOME/tt-npu-overrides" \
ASCEND_RT_VISIBLE_DEVICES=0 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
torchrun --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    -m torchtitan.train \
    --module npu_recipe \
    --config llama3_npu_debugmodel \
    --override npu_compat.npu_attention \
    --output-dir /tmp/torchtitan-quickstart
```

输出结果类似如下：

torchtitan v0.4.0 起每个模块用自己的 logger（`logging.getLogger(__name__)`），日志格式是
`[titan] <时间> - <logger 名> - <级别> - <消息>`。入口是 `python -m torchtitan.train`，所以入口自己打的
两行日志（版本号、`Process group destroyed`）logger 名是 `__main__`；模块里的日志则是模块路径
（`torchtitan.trainer`、`torchtitan.config.override`、`torchtitan.observability.metrics`）：

```shell #test-result id="torchtitan-train-debug" fuzzy='xxx' fuzzy='...'
[titan] xxx - __main__ - INFO - torchtitan version: xxx
[titan] xxx - torchtitan.config.override - INFO - [Override] npu_compat.npu_attention: model.layers.0.attention GQAttention.Config -> NpuGQAttention.Config
[titan] xxx - torchtitan.config.override - INFO - Applied 6 override(s)
[titan] xxx - torchtitan.observability.metrics - INFO - step: xxx
[titan] xxx - torchtitan.trainer - INFO - Training completed
[titan] xxx - __main__ - INFO - Process group destroyed
...
```

### 多卡训练

在 2 张 NPU 上用 `torchrun` 起真分布式训练，同样真跑 2 步：FSDP 把模型参数和优化器状态分片到两张卡，
并叠 bf16 参数（`llama3_npu_debugmodel_bf16`）验证混合精度。`HCCL_NPU_SOCKET_PORT_RANGE` 用于错开两个 rank 的
HCCL 端口（否则会报 `EI0020` 端口冲突）：

```shell #test id="torchtitan-train-2card"
cd torchtitan && PYTHONPATH="$HOME/tt-npu-overrides" \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
HCCL_NPU_SOCKET_PORT_RANGE=60000-60200 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
torchrun --nproc_per_node=2 \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 \
    --tee 3 \
    -m torchtitan.train \
    --module npu_recipe \
    --config llama3_npu_debugmodel_bf16 \
    --override npu_compat.npu_attention \
    --output-dir /tmp/torchtitan-quickstart-2card
```

输出结果类似如下：

`--tee 3` 把每个 worker 的 stdout/stderr 复制到控制台，行首加 `[<角色名><local_rank>]:` 前缀（这里就是
`[default0]:`）；`--local-ranks-filter 0` 只让 rank 0 的日志上控制台——rank 1 照常参与训练和集合通信，
只是它的日志不打印，所以下面的输出不会被两个 rank 交错：

```shell #test-result id="torchtitan-train-2card" fuzzy='xxx' fuzzy='...'
[default0]:[titan] xxx - __main__ - INFO - torchtitan version: xxx
[default0]:[titan] xxx - torchtitan.config.override - INFO - Applied 6 override(s)
[default0]:[titan] xxx - torchtitan.observability.metrics - INFO - step: xxx
[default0]:[titan] xxx - torchtitan.trainer - INFO - Training completed
[default0]:[titan] xxx - __main__ - INFO - Process group destroyed
...
```

## 外部链接

- GitHub：[pytorch/torchtitan](https://github.com/pytorch/torchtitan)
- 论文：[PyTorch Native LLM Training](https://arxiv.org/abs/2410.06511)
