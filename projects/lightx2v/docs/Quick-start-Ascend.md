# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑通 [LightX2V](https://github.com/ModelTC/LightX2V) 的最小链路：安装源码 + 拉权重 + 4 步蒸馏图生视频（I2V）烟囱测试。权重下载走 [ModelScope](https://modelscope.cn/)（国内网络更稳），HF 仓库同名映射。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品（Ascend 910B4 / 910B 等），并按需完成物理机或容器内的设备挂载。

### 基础软件

跑本文档**之前**，机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Ascend 910B4 × 1（32 GB HBM,集群共享 NPU,本文档默认只占 1 张卡）
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
| modelscope | 1.37.0 |
| lightx2v | GitHub main 分支（上游零 release 零 tag，滚动 main） |

## 前置安装

### 拉起 CANN 环境

CANN toolkit（`/usr/local/Ascend/cann-9.1.0/`）在镜像里**自带**，但 `ASCEND_HOME` / `LD_LIBRARY_PATH` / `PATH` 这些环境变量**只在交互式 shell 自动 export**，SSH 进的非交互式 shell 没自动加载。直接 `import torch_npu` 会报 `libhccl.so: cannot open shared object file`。

每个 `#test` 块前都得 `source` 一次（CI 每个 `bash -c` 块都是独立进程，set 进 env 的 `export` 不继承）：

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

> 后续所有 `#test-setup` 块默认已经把 CANN env 加载好（重复 `source` 是幂等的）。

确认能看到 NPU 设备：

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
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
| 5     910B4               | OK            | 89.9        39                0    / 0             2922 / 32768 |
| 0                         | 0000:41:00.0  | 0           0    / 0                               |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。

检查 Python 版本：

```shell #test id="check-py"
python --version
```

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

对齐上游 pin 装 `torch`（CPU build,NPU 加速走 `torch_npu`） + `torch_npu`：

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.9.0
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple torch_npu==2.9.0.post2
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

```shell #test-result id="check-torch"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败,先确认 CANN env 已 source（`echo $LD_LIBRARY_PATH | grep cann`）,再回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

装 `modelscope`（本文档下载权重走 ModelScope,国内网络更稳）：

```shell #test-setup
uv pip install modelscope==1.37.0
```

```shell #test id="modelscope-version"
python -c "import modelscope; print('modelscope=', modelscope.__version__)"
```

```shell #test-result id="modelscope-version" disable_fuzzy
modelscope= 1.37.0
```

### （aarch64 机器）打 cv2 / decord / torchaudio 三个 stub + 装真 triton

仅 aarch64 镜像（如华为云 C76 容器）需要这一步。x86_64 镜像有现成 wheel,直接 `uv pip install opencv-python decord torchaudio triton` 就行,跳到下一节。

LightX2V 在 import 时会触发这些模块：

| 模块 | 触发链 | aarch64 症状 |
| --- | --- | --- |
| `cv2` | `lightx2v.models.video_encoders` 顶层 import | `ImportError: libxcb.so.1`（base image 缺 GUI X11 lib,headless 变体也救不回来） |
| `decord` | 视频解码 | PyPI 没 aarch64 wheel |
| `torchaudio` | 跟着 `torch` 一起装到 2.11.0 | `OSError: Could not load ..._torchaudio.abi3.so`（编的是 torch CUDA,跟 torch_npu 不兼容） |
| `triton` | `lightx2v.common.ops.attn.kernels.sla_kernel` 顶层 `import triton`;更重的链路是 `torch_npu.utils._graph_tree -> torch._inductor.compile_fx -> ... -> runtime/triton_heuristics` | 老镜像缺 triton;但 **不能拿空 stub 糊弄**（见下） |

策略：cv2 / decord / torchaudio 源码 `--no-deps` 安装时跳过,在 site-packages 前面塞同名 stub 包（`PYTHONPATH` 优先,`sys.modules` 命中空 stub,真 .so 不会被加载）。LightX2V 不真正做 GUI / 视频解码 / 音频 IO,空 stub 够用。

**triton 不 stub,装真包**。run#1~#3 证明 stub 路线是打地鼠：torch `_inductor/runtime/triton_heuristics` 会 `from .triton_compat import Config` 然后 `class InductorConfig(Config)` —— 拿 `triton.Config` 当**基类**继承;stub 的模块对象哪怕补了属性访问、dunder、`__call__`、`__mro_entries__`,也堵不住真实 triton 的类/函数语义（torch 在模块 import 期就实例化 `Config`、读 `GPUTarget`、查 `knobs`）。triton 3.5.x 起 PyPI 直接发 aarch64 wheel（cp312 manylinux,run#1 时代"没 wheel"的结论已过期,且它**零运行时依赖**,不会把 CUDA 栈拖回来）,正好也是 torch 2.9 官方配套线（torch 2.9 配 triton 3.5）。`sla_kernel` 的 CUDA kernel 依然不会在 NPU 上执行（torchada 接管）,装真包只为让 import 链语义完整。

```shell #test-setup
# rm -rf 防御:老镜像/跑过旧版文档的机器上可能残留 /tmp/stubs/triton,
# 不删的话 PYTHONPATH 会拿它遮蔽下面装的(和已有的)真 triton
rm -rf /tmp/stubs/triton
mkdir -p /tmp/stubs/cv2 /tmp/stubs/decord /tmp/stubs/torchaudio
```

```shell #test-setup
cat > /tmp/stubs/cv2/__init__.py <<'PY'
# Stub cv2 for aarch64 wheels missing libxcb.so.1
import sys, types

INTER_LINEAR = 1
INTER_NEAREST = 0
INTER_CUBIC = 2
INTER_AREA = 3
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 4
COLOR_BGR2GRAY = 6
COLOR_RGB2GRAY = 7
IMREAD_COLOR = 1
CAP_PROP_FRAME_COUNT = 7
CAP_PROP_FPS = 5

def _stub(*a, **kw): raise NotImplementedError("cv2 stub on aarch64")
class _Stub:
    def __getattr__(self, name): return _Stub()
    def __call__(self, *a, **k): return _Stub()
VideoCapture = type("VideoCapture", (), {"__init__": lambda s,*a,**k: None, "read": _stub, "isOpened": lambda s: False, "release": lambda s: None})
VideoWriter = type("VideoWriter", (), {"__init__": lambda s,*a,**k: None, "write": _stub, "release": lambda s: None})
imread = imwrite = resize = cvtColor = _stub
sys.modules[__name__].__getattr__ = lambda name: _Stub() if name not in globals() else globals()[name]
PY
```

```shell #test-setup
cat > /tmp/stubs/decord/__init__.py <<'PY'
class VideoReader:
    def __init__(self, *a, **kw): raise NotImplementedError("decord stub on aarch64")
PY
```

```shell #test-setup
cat > /tmp/stubs/torchaudio/__init__.py <<'PY'
class _Stub:
    def __getattr__(self, name): return _Stub()
    def __call__(self, *a, **k): return _Stub()
import sys as _s
_s.modules[__name__].__getattr__ = lambda n: _Stub()
PY
```

`triton` **装真包,不 stub**:torch 2.9 配套 triton 3.5.x,3.5 起 PyPI 有 cp312 aarch64 manylinux wheel,且零运行时依赖(不会把 nvidia/cuda 栈拖回来,与 CUDA 排除清单无冲突)。空 stub 路线走不通:stub 让 `import triton` "成功",把 torch 推进真实代码路径——triton_heuristics 拿 triton.Config 当基类继承、读 GPUTarget/knobs,stub 语义对不上逐层炸(AttributeError → TypeError → module() takes at most 2 arguments);而镜像原本"没装 triton"反而没事,torch 的 has_triton_package() 捕获 ImportError 走 HAS_TRITON=False 的无 triton 兜底路径。装真包两边都对。

```shell #test-setup
uv pip install 'triton==3.5.*'
```

```shell #test-setup
# 装 LightX2V 时跳过三个 stub 依赖（具体名字以 setup.py / pyproject.toml 为准,
# 找不到时一个个 --no-deps 单独装也能绕过）。PYTHONPATH 在每条 python 命令前 export。
echo "stubs ready at /tmp/stubs/{cv2,decord,torchaudio}"
ls /tmp/stubs/cv2/__init__.py /tmp/stubs/decord/__init__.py /tmp/stubs/torchaudio/__init__.py
```

```shell #test id="stubs-verify"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/tmp/stubs:$PYTHONPATH
python -c "
import cv2, decord, torchaudio
print('cv2.INTER_LINEAR=', cv2.INTER_LINEAR)
print('cv2.COLOR_BGR2RGB=', cv2.COLOR_BGR2RGB)
print('decord.VideoReader=', decord.VideoReader)
print('torchaudio.load=', torchaudio.load)
import triton
print('triton.__version__=', triton.__version__)
import triton.language as tl
print('triton.language.math=', tl.math)
class ProbeConfig(triton.Config):
    pass
print('Config subclass ok=', ProbeConfig.__name__)
import torch, torch_npu
print('torch/torch_npu chain ok=', torch.__version__)
"
```

```shell #test-result id="stubs-verify" fuzzy='xxx'
cv2.INTER_LINEAR= xxx
cv2.COLOR_BGR2RGB= xxx
decord.VideoReader= xxx
torchaudio.load= xxx
triton.__version__= xxx
triton.language.math= xxx
Config subclass ok= xxx
torch/torch_npu chain ok= xxx
```

## 安装 LightX2V

源码安装(克隆 GitHub 仓库 + `pip install --no-deps -v .`)。源码可改、可调试、便于排错。

### 项目隔离布局(重要)

权重不要塞到 LightX2V 源码 clone 里(`./models/`),源码可以随时 `rm -rf` 重 clone 而权重不丢。LightX2V/./models 作为符号链接 → `../models/`,保持 LightX2V 自己的相对路径约定不破。

```
/home/coder/work/lightx2v-test/          <- 项目根,持久卷(/home/coder/ rebuild 不丢)
├── LightX2V/                            <- 源码 clone(可重置)
│   ├── configs/ ...
│   ├── examples/ ...
│   ├── assets/ ...
│   └── models -> ../models              <- 软链到 ../models
├── models/                              <- 模型权重(项目隔离,持久)
│   ├── Wan2.2-I2V-A14B/
│   ├── Wan2.1-I2V-14B-480P/
│   └── Wan2.2-Distill-Models/
└── save_results/                        <- 推理输出
```

<!-- 工作流注入的 UPSTREAM_REF（上游零 tag,固定 main）通过这个隐藏的 #test-setup 捕获并注入到下方 clone 命令中；markdown 渲染器会丢掉注释，读者看不到，runner 仍会执行 -->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

项目根定在持久卷上（默认 `/home/coder/work/lightx2v-test`，CI 注入 `PROJECT_ROOT` 走别的路径也行）；源码 clone 到 `src` 子目录（避免和 `models/` `save_results/` 平级混淆）；`<UPSTREAM_REF>` 换成目标 分支/tag/commit——上游零 release 零 tag，默认 main，看护流水线会把本次要测的 ref 注入 `UPSTREAM_REF` 环境变量；`ln -sfn ../models models` 软链让 LightX2V 内部 `./models/` 相对路径依然命中；`--no-deps` 装源码时缺 wheel 的几个依赖（cv2 / decord / torchaudio 等）会跳过，真要的依赖看下面 `uv pip install` 单独装，aarch64 走 stub 即可。

```shell #test-setup id="lightx2v-install-source" load="upstream_ref>>UPSTREAM_REF"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

git clone --depth 1 --branch <UPSTREAM_REF> https://github.com/ModelTC/LightX2V.git src
cd src
ln -sfn ../models models

uv pip install --no-deps -v .
```

依赖安装说明：直接依赖（除 stub 的三个外）从上游 pyproject.toml 同步过来；故意不写 `uv pip install .` 重做依赖解析——之前已 `--no-deps` 把 lightx2v 装上，再用 constraint 列表一次装齐其余 deps；CUDA 排除清单（`_CUDA_CONSTRAINTS`）通过 `PIP_CONSTRAINT`/`UV_CONSTRAINT` 已在进程环境，自动屏蔽 nvidia-* / cuda-* 等。例外的三个 stub：opencv-python → stub cv2（`/tmp/stubs/cv2`）、decord → stub decord（`/tmp/stubs/decord`）、torchaudio → stub torchaudio（`/tmp/stubs/torchaudio`，torchada 仍是 NPU 版）；triton 走真包（上一节已装 `triton==3.5.*`），不在例外里；torch + torch_npu 在 check-torch 已装 2.9.0，这里不重复。pyzmq 单独补装：上游 lightx2v.disagg.conn 顶层无条件 `import zmq`（pipeline → qwen_image_runner → disagg_mixin → conn），但上游 pyproject.toml 漏声明，`pip install .` 后 import lightx2v 一样挂；aarch64 有 cp312 manylinux wheel，补装，import 名是 zmq。

```shell #test-setup id="lightx2v-install-deps"
uv pip install \
    numpy scipy diffusers transformers tokenizers tqdm accelerate safetensors \
    imageio imageio-ffmpeg einops loguru omegaconf peft \
    swanlab qtorch 'comfy-kitchen>=0.2.15' ftfy gradio \
    aiohttp pydantic prometheus-client gguf fastapi uvicorn PyJWT requests \
    aio-pika 'asyncpg>=0.27.0' 'aioboto3>=12.0.0' \
    'alibabacloud_dypnsapi20170525==1.2.2' 'redis==6.4.0' tos \
    av 'torchada>=0.1.10' pyzmq
```

```shell #test id="lightx2v-install-verify"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
export PYTHONPATH=/tmp/stubs:$PYTHONPATH  # aarch64 才需要,x86_64 跳过
export PLATFORM=ascend_npu              # lightx2v import 时 read env 选后端,默认 cuda
python -c "
import lightx2v
spec = __import__('importlib.util').util.find_spec('lightx2v')
print('lightx2v spec:', spec.origin if spec else 'MISSING')
print('lightx2v version:', getattr(lightx2v, '__version__', 'unknown'))
"
```

```shell #test-result id="lightx2v-install-verify" fuzzy='xxx' fuzzy='...'
...
lightx2v spec: xxx
lightx2v version: xxx
```

> 开头的 `...` 吞掉 `import lightx2v` 期间可选加速后端（sage_attn）未安装时的提示行（如 `spas_sage_attn is not installed.`），它与验证无关、且会随安装环境出现/消失，不进断言。
>
> LightX2V 内部通过 `os.getenv('PLATFORM', 'cuda')` 选后端(`lightx2v/shot_runner/shot_base.py:43`)。**`PLATFORM=ascend_npu`**(不是 `npu`,不是 `ascend`)。后续每条 `python -m lightx2v.infer` / `python examples/...py` 命令都要 export。

## 拉权重（ModelScope）

LightX2V 跑 I2V 蒸馏需要三组权重（**务必带 include 过滤**，全量仓库有几百 GB 无关权重）：

1. **base 模型**（`Wan-AI/Wan2.2-I2V-A14B`）：T5 text encoder（`models_t5_umt5-xxl-enc-bf16.pth`，11.4 GB）+ VAE（`Wan2.1_VAE.pth`，0.5 GB）+ tokenizer（`google/`）。**不要下 `high_noise_model/` + `low_noise_model/` 两个目录**（各 60 GB 的原始 BF16 权重,14B MoE 蒸馏场景被下面的量化 ckpt 完全替代）,只留两份 `config.json`。
2. **CLIP vision encoder**（`Wan-AI/Wan2.1-I2V-14B-480P`）：`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`（4.8 GB）+ tokenizer（`xlm-roberta-large/`）。Wan2.2-I2V-A14B 仓库**不带 CLIP**,I2V 的 image encoder 走的是 Wan2.1 同款 XLM-RoBERTa-large ViT-Huge,从 Wan2.1-I2V 仓库单独拉这一个文件即可,不用下整个 14B 模型。
3. **蒸馏量化 ckpt**（`lightx2v/Wan2.2-Distill-Models`）：MoE 架构分 high noise + low noise 两份,4 步推理只用 DIT 的 int8 量化 split-block 目录。仓库里同时存了单文件形态（15 GB/份）和 split 目录形态（42 个 `block_*.safetensors`,合计 15 GB/份）以及 fp8/BF16 变体——**只拉 I2V int8 的两个 split 目录**。

```shell #test-setup id="lightx2v-pull-wan22-base"
# base:T5 + VAE + tokenizer + 高低噪声 config.json(~12 GB,跳过 120 GB 原始 DIT shards)
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
mkdir -p "$PROJECT_ROOT/models"
modelscope download \
    --model Wan-AI/Wan2.2-I2V-A14B \
    --local_dir "$PROJECT_ROOT/models/Wan2.2-I2V-A14B" \
    --include "models_t5_umt5-xxl-enc-bf16.pth" "Wan2.1_VAE.pth" \
               "google/*" "high_noise_model/config.json" "low_noise_model/config.json"
```

```shell #test-setup id="lightx2v-pull-clip"
# CLIP vision encoder:Wan2.2 仓库不带,从 Wan2.1-I2V-14B-480P 单拉(~4.8 GB)
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
modelscope download \
    --model Wan-AI/Wan2.1-I2V-14B-480P \
    --local_dir "$PROJECT_ROOT/models/Wan2.1-I2V-14B-480P" \
    --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" "xlm-roberta-large/*"
```

验证口径：目录名断言（`find -maxdepth 1 -type d`，排除 ModelScope 记账目录 `._____temp` 等隐藏条目）+ 下载完整性断言（>100M 大文件字节数精确：T5 11.4 GB + VAE 507 MB），`LC_ALL=C sort` 保证跨机器/跨 locale 排序一致（ls 的排序随 collation 变），权限位/属主/时间戳不进断言。

```shell #test id="lightx2v-pull-wan22-base-verify"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
find "$PROJECT_ROOT/models/Wan2.2-I2V-A14B/" -maxdepth 1 -type d ! -name '.*' ! -name 'Wan2.2-I2V-A14B' -printf '%f\n' | LC_ALL=C sort
echo "---"
find "$PROJECT_ROOT/models/Wan2.2-I2V-A14B/" -maxdepth 1 -type f -size +100M -printf '%s %f\n' | LC_ALL=C sort
echo "---"
head -5 "$PROJECT_ROOT/models/Wan2.2-I2V-A14B/google/umt5-xxl/tokenizer_config.json"
```

```shell #test-result id="lightx2v-pull-wan22-base-verify" fuzzy='xxx'
google
high_noise_model
low_noise_model
---
11361920418 models_t5_umt5-xxl-enc-bf16.pth
507609880 Wan2.1_VAE.pth
---
{
  "added_tokens_decoder": {
    "0": {
      "content": xxx,
      "lstrip": xxx,
```

### 拉 I2V 蒸馏 ckpt（split blocks）

只拉 I2V int8 两个 split 目录（~30 GB）。仓库还有单文件/fp8/BF16 变体共 335 GB，include 过滤必须带，否则全量下载直接把磁盘打爆；lightx2v 的 load_safetensors 原生支持目录形态（utils.py: load_safetensors_from_dir 遍历目录下所有 block_*.safetensors），split 目录直接当 ckpt 路径用，不用先 merge。

```shell #test-setup id="lightx2v-pull-wan22-i2v-distill"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
modelscope download \
    --model lightx2v/Wan2.2-Distill-Models \
    --local_dir "$PROJECT_ROOT/models/Wan2.2-Distill-Models" \
    --include "wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030_split/*" \
               "wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step_split/*"
```

```shell #test id="lightx2v-pull-wan22-i2v-distill-verify"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
# 展示 split-block 目录列表(每个目录里有 block_0.safetensors ~ block_*.safetensors)
ls "$PROJECT_ROOT/models/Wan2.2-Distill-Models/" | grep split
echo "---"
ls "$PROJECT_ROOT/models/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030_split/" | LC_ALL=C sort | head -5
echo "---"
ls "$PROJECT_ROOT/models/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030_split/" | wc -l
```

```shell #test-result id="lightx2v-pull-wan22-i2v-distill-verify" fuzzy='xxx'
wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030_split
wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step_split
---
block_0.safetensors
block_1.safetensors
block_10.safetensors
block_11.safetensors
block_12.safetensors
---
42
```

> ⚠️ 上面 split 目录名里的 `_1030` 后缀是 distill 权重的版本日期（10月30日）,high noise 带、low noise 不带——**名字以 ModelScope 仓库实际为准**,跑之前先 `ls` 确认。
>
> HF 上的同名 repo [lightx2v/Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models) 内容相同,还含 T2V 蒸馏(`wan2.2_t2v_A14b_*_4step.safetensors`,ModelScope 没同步),T2V 路线需要时再走 HF。

## 列出 Wan2.2 4 步蒸馏 config

LightX2V 把推理参数(步数、flow shift、offload 策略、LoRA 路径、量化方案等)写到 config JSON 里,`python -m lightx2v.infer --config_json ...` 一次性喂进去。在 `$PROJECT_ROOT/src` 下做精确存在性断言——`ls` 多参数按参数序输出,不随上游新增 config 漂移。

```shell #test id="lightx2v-list-wan22-configs"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
ls configs/wan22/wan_moe_i2v.json configs/wan22/wan_moe_t2v.json
echo "---"
ls configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json
```

输出结果如下：

```shell #test-result id="lightx2v-list-wan22-configs" fuzzy='xxx'
configs/wan22/wan_moe_i2v.json
configs/wan22/wan_moe_t2v.json
---
configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json
```

下面 I2V demo 用 **`configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json`**(NPU int8 量化版本,`self_attn_1_type=rainfusion_attn` + `cross_attn_*=npu_flash_attn`,直走 NPU 算子)。**但默认 config 是为「910B + 2 卡 + 720P」设计的**(720P + 81 帧 + T5/VAE 全常驻 + ulysses 并行 + rife 视频插帧),**单卡 32 GB 必须改**:720→480 + 开 t5/vae cpu_offload + 删 parallel + 删 video_frame_interpolation。改动在下面 I2V smoke 块里完成。

> 公共要点:config JSON 里 `high/low_noise_quantized_ckpt` 默认是 `models/Wan2.2-Distill-Models/...` 相对路径(LightX2V 仓库内 `./models/` 约定)。本文 doc 把它们改成 `$PROJECT_ROOT/models/...` 绝对路径的 **split 目录**,配合源码 clone 里 `./models -> ../models` 的软链两种寻址都能命中。

## I2V 烟囱测试：4 步蒸馏图生视频

参考官方脚本 [`scripts/wan22/distill/run_wan22_moe_i2v_distill_lora_4step.sh`](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/distill/run_wan22_moe_i2v_distill_lora_4step.sh) 的标准入口是 `python -m lightx2v.infer --model_cls wan2.2_moe_distill --task i2v --model_path <base> --config_json <cfg> --prompt ... --image_path ... --save_result_path ...`。Wan2.2-I2V-A14B 是 MoE 架构,high noise + low noise 两份 ckpt 都由 config JSON 内部指定(走 lazy block load)。

### 单卡 32 GB 适配：改 config

适配逻辑：`high/low_noise_quantized_ckpt` 换成 `$PROJECT_ROOT/models/` 下 split 目录绝对路径（config 默认值是单文件形态 `models/Wan2.2-Distill-Models/wan2.2_i2v_A14b_*_int8_*.safetensors`，ModelScope 只同步了 split 目录形态，不换路径直接 FileNotFoundError；load_safetensors 原生支持目录）；`clip_original_ckpt` 指向从 Wan2.1-I2V-14B-480P 单拉的 CLIP 文件（Wan2.2 base 仓库不带这文件，不指定的话 runner 走 find_torch_model_path 在 model_path 下找，两处都找不到直接 FileNotFoundError）；720P → 480P（activation/KV 省 ~40%）；开 T5/VAE cpu_offload（不释放这俩 32GB 必 OOM）+ block 粒度 DIT offload 进一步省；单卡 ulysses 没意义删掉 parallel；rife 插帧要单独下模型，不需要就删。

```shell #test-setup id="lightx2v-i2v-cfg-adapt"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
python -c "
import json, pathlib, os
cfg_path = pathlib.Path('configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json')
cfg = json.loads(cfg_path.read_text())
proj_models = os.environ['PROJECT_ROOT'] + '/models'
cfg['high_noise_quantized_ckpt'] = proj_models + '/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030_split'
cfg['low_noise_quantized_ckpt']  = proj_models + '/Wan2.2-Distill-Models/wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step_split'
cfg['clip_original_ckpt'] = proj_models + '/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
cfg['target_height'] = 480
cfg['target_width']  = 832
cfg['t5_cpu_offload']  = True
cfg['vae_cpu_offload'] = True
cfg['cpu_offload']             = True
cfg['offload_granularity']     = 'block'
cfg.pop('parallel', None)
cfg.pop('video_frame_interpolation', None)
cfg_path.write_text(json.dumps(cfg, indent=4))
print('cfg adapted (单卡 32GB 适配):')
for k in ['target_height', 'target_width', 'cpu_offload', 't5_cpu_offload', 'vae_cpu_offload', 'high_noise_quantized_ckpt', 'clip_original_ckpt']:
    print(f'  {k}: {cfg[k]}')
"
```

### 跑 I2V

```shell #test-setup id="lightx2v-i2v-smoke-run"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
mkdir -p "$PROJECT_ROOT/save_results"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=/tmp/stubs:${PYTHONPATH:-}     # aarch64 才有,x86_64 删掉
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

python -m lightx2v.infer \
    --model_cls wan2.2_moe_distill \
    --task i2v \
    --model_path "$PROJECT_ROOT/models/Wan2.2-I2V-A14B" \
    --config_json configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds." \
    --negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --image_path "$PROJECT_ROOT/src/assets/inputs/imgs/img_0.jpg" \
    --save_result_path "$PROJECT_ROOT/save_results/output_wan22_moe_i2v_distill_int8_npu_480p.mp4"
```

```shell #test id="lightx2v-i2v-output"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
OUT="$PROJECT_ROOT/save_results/output_wan22_moe_i2v_distill_int8_npu_480p.mp4"
# 只断言 字节数 + basename(不带时间戳/属主/绝对路径,跨机器跨时间可复现)
ls -l "$OUT" | awk '{print $5, $NF}'
file "$OUT"
# 优先用 ffprobe 读视频流(宽/帧数/时长),ffprobe 不存在时退到 ftyp magic 验证
ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration \
    -of default=noprint_wrappers=1 "$OUT" 2>/dev/null || \
    python -c "b=open('$OUT','rb').read(1024*1024); print('size_on_disk:', len(b), 'bytes; ftyp:', b[4:8].decode('ascii','ignore'))"
```

```shell #test-result id="lightx2v-i2v-output" fuzzy='xxx'
xxx output_wan22_moe_i2v_distill_int8_npu_480p.mp4
xxx: ISO Media, MP4 Base Media v2 [ISO 14496-12:2015]
codec_name=xxx
width=xxx
height=xxx
r_frame_rate=xxx
nb_frames=xxx
duration=xxx
```

> **关键 env**：
> - `PLATFORM=ascend_npu`(LightX2V 内部 `os.getenv('PLATFORM','cuda')`,值必须是 `ascend_npu`,不是 `ascend` 也不是 `npu`)
> - `ASCEND_RT_VISIBLE_DEVICES=0`(对应 CUDA_VISIBLE_DEVICES 的 NPU 版本,只用 1 张卡,留另一张给别的实验)
> - `source /usr/local/Ascend/ascend-toolkit/set_env.sh`(`libhccl.so` 在这个目录里,非交互式 SSH 不会自动 load)
> - aarch64 还要 `export PYTHONPATH=/tmp/stubs:$PYTHONPATH`(x86_64 跳过)
>
> **单卡 vs 多卡**：默认 config 是 2 卡 ulysses 并行,本文 doc 单卡跑通后要再扩到多卡,把 `parallel.seq_p_size` 加回去并设成 `torch.npu.device_count()`。

## 更多模型路线（本文档未覆盖）

跑通 I2V 之后,其它任务/模型的调用模式完全一致(`python -m lightx2v.infer --model_cls ... --task ... --model_path ... --config_json ...`),差异只在权重和 config:

- **T2V(Wan2.2 蒸馏)**:`--task t2v` + `configs/wan22/wan_moe_t2v_distill.json`。⚠️ ModelScope 的 `lightx2v/Wan2.2-Distill-Models` **没同步 T2V 蒸馏权重**(只有 I2V),要跑得走 HF([lightx2v/Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models),`wan2.2_t2v_A14b_*_4step.safetensors`)或者跑不蒸馏的 `wan_moe_t2v` 主线(50 步,慢)。
- **T2V(Wan2.1 轻量)**:`--model_cls wan2.1` + `configs/wan22/` 同款 config。官方昇腾入口是 [`scripts/platforms/ascend_npu/run_wan21_t2v.sh`](https://github.com/ModelTC/LightX2V/blob/main/scripts/platforms/ascend_npu/run_wan21_t2v.sh)(配 `configs/platforms/ascend_npu/wan_t2v.json`,`npu_flash_attn` + cpu_offload),模型用 [Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)(~17.6 GB,单卡无压力)。
- **T2I(Qwen-Image-Edit)**:`--model_cls qwen-image-edit-2511` + `--task i2i`(命名是 **i2i 不是 t2i**)。单卡 32 GB 要走 FP8 蒸馏版(`configs/qwen_image/qwen_image_i2i_2511_distill_fp8.json` + [lightx2v/Qwen-Image-Edit-2511-Lightning](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning),FP8 merged ckpt 单文件就有 20 GB,include 过滤挑 `qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors`,**别全量拉**,repo 共 87 GB)。text encoder / VAE 视 Lightning repo 是否携带,缺了再补。

## 下一步

- **量化路径**：`int8-npu` / `scaled-fp8-e4m3` 见 [量化教程](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/quantization.html)
- **特征缓存**：TeaCache / MagCache 消除冗余计算,见 [缓存教程](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/cache.html)
- **多卡并行**：CFG 并行 / Ulysses 序列并行 / 张量并行见 [并行教程](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/parallel.html)
- **多模态音频**：MiniMax-H3 Turbo 跑带同步音频的视频(参考 [官方 README](https://github.com/ModelTC/LightX2V/blob/main/README_zh.md))
- **ComfyUI 节点**：[ComfyUI 部署文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_comfyui.html)
- **服务化部署**：[生产级 API 服务部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_service.html)
- **低资源部署**：本文档基于单卡 32 GB HBM(910B4)调好 offload,想再压低(16/24 GB)见 [低资源部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/for_low_resource.html)(参考的是 CUDA 8 GB 路线,NPU 可借鉴三级 offload 思路)
- **多卡扩展**：本文 doc 单卡跑通后,把 `parallel.seq_p_size = torch.npu.device_count()` 加回 config(或者用 `*_ulysses_*.json` 类自带并行的 config),见 [并行教程](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/parallel.html)
- **训练框架**：[GenRL](https://github.com/ModelTC/GenRL) 用 GRPO 对 diffusion/flow 模型做强化学习训练