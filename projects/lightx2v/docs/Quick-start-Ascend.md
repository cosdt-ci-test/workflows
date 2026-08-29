# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑通 [LightX2V](https://github.com/ModelTC/LightX2V) 的最小链路：4 步蒸馏文生视频（T2V）+ 图生视频（I2V）+ 图编辑（T2I）。权重下载走 [ModelScope](https://modelscope.cn/)（国内网络更稳），HF 仓库同名映射。

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
| lightx2v | GitHub main 分支 |

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

### （aarch64 机器）打 cv2 / decord / torchaudio 三个 stub

仅 aarch64 镜像（如华为云 C76 容器）需要这一步。x86_64 镜像有现成 wheel,直接 `uv pip install opencv-python decord torchaudio` 就行,跳到下一节。

LightX2V 在 import 时会触发这三个模块：

| 模块 | 触发链 | aarch64 症状 |
| --- | --- | --- |
| `cv2` | `lightx2v.models.video_encoders` 顶层 import | `ImportError: libxcb.so.1`（base image 缺 GUI X11 lib,headless 变体也救不回来） |
| `decord` | 视频解码 | PyPI 没 aarch64 wheel |
| `torchaudio` | 跟着 `torch` 一起装到 2.11.0 | `OSError: Could not load ..._torchaudio.abi3.so`（编的是 torch CUDA,跟 torch_npu 不兼容） |

策略：源码 `--no-deps` 安装时跳过这三个依赖,然后在 site-packages 里塞同名的 stub 包（`PYTHONPATH` 优先,`sys.modules` 命中空 stub,真 .so 不会被加载）。LightX2V 不真正做 GUI / 视频解码 / 音频 IO,空 stub 够用。

```shell #test-setup
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

```shell #test-setup
# 装 LightX2V 时跳过三个 stub 依赖（具体名字以 setup.py / pyproject.toml 为准,
# 找不到时一个个 --no-deps 单独装也能绕过）。PYTHONPATH 在每条 python 命令前 export。
echo "stubs ready at /tmp/stubs/{cv2,decord,torchaudio}"
ls /tmp/stubs/cv2/__init__.py /tmp/stubs/decord/__init__.py /tmp/stubs/torchaudio/__init__.py
```

```shell #test id="stubs-verify"
export PYTHONPATH=/tmp/stubs:$PYTHONPATH
python -c "
import cv2, decord, torchaudio
print('cv2.INTER_LINEAR=', cv2.INTER_LINEAR)
print('cv2.COLOR_BGR2RGB=', cv2.COLOR_BGR2RGB)
print('decord.VideoReader=', decord.VideoReader)
print('torchaudio.load=', torchaudio.load)
"
```

```shell #test-result id="stubs-verify" fuzzy='xxx'
cv2.INTER_LINEAR= xxx
cv2.COLOR_BGR2RGB= xxx
decord.VideoReader= xxx
torchaudio.load= xxx
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
│   ├── Wan2.2-Distill-Models/
│   ├── Qwen-Image-Edit-2511/
│   └── Qwen-Image-Edit-2511-Lightning/
└── save_results/                        <- 推理输出
```

```shell #test-setup id="lightx2v-install-source"
# 把项目根定在持久卷上,默认 /home/coder/work/lightx2v-test;
# CI 注入 PROJECT_ROOT 走别的路径也行
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 源码 clone 到 src 子目录(不是项目根,避免和 models/ save_results/ 平级混淆)
git clone --depth 1 https://github.com/ModelTC/LightX2V.git src
cd src
# 软链 ./models → ../models,LightX2V 内部 ./models/ 相对路径依然命中
ln -sfn ../models models

# --no-deps 装源码：缺 wheel 的几个依赖（cv2 / decord / torchaudio 等）会跳过,
# 真要的依赖看下面 `uv pip install` 单独装,aarch64 走 stub 即可
uv pip install --no-deps -v .
```

```shell #test-setup id="lightx2v-install-deps"
# LightX2V 直接依赖（除上面 stub 的三个外），从 pyproject.toml 同步过来。
# 故意不写 `uv pip install .` 重做依赖解析：之前已经 --no-deps 把 lightx2v 装上,
# 再用 constraint 列表一次装齐其余 deps;CUDA 排除清单(_CUDA_CONSTRAINTS)
# 通过 PIP_CONSTRAINT/UV_CONSTRAINT 已在 process env,自动屏蔽 nvidia-* / cuda-* 等。
# 例外的三个 stub:
#   - opencv-python → stub cv2(/tmp/stubs/cv2)
#   - decord        → stub decord(/tmp/stubs/decord)
#   - torchaudio    → stub torchaudio(/tmp/stubs/torchaudio),torchada 仍是 NPU 版
# torch + torch_npu 在 step 4 check-torch 已装 2.9.0,这里不重复
uv pip install \
    numpy scipy diffusers transformers tokenizers tqdm accelerate safetensors \
    imageio imageio-ffmpeg einops loguru omegaconf peft \
    swanlab qtorch 'comfy-kitchen>=0.2.15' ftfy gradio \
    aiohttp pydantic prometheus-client gguf fastapi uvicorn PyJWT requests \
    aio-pika 'asyncpg>=0.27.0' 'aioboto3>=12.0.0' \
    'alibabacloud_dypnsapi20170525==1.2.2' 'redis==6.4.0' tos \
    av 'torchada>=0.1.10'
```

```shell #test id="lightx2v-install-verify"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
export PYTHONPATH=/tmp/stubs:$PYTHONPATH  # aarch64 才需要,x86_64 跳过
python -c "
import lightx2v
spec = __import__('importlib.util').util.find_spec('lightx2v')
print('lightx2v spec:', spec.origin if spec else 'MISSING')
print('lightx2v version:', getattr(lightx2v, '__version__', 'unknown'))
"
```

```shell #test-result id="lightx2v-install-verify" fuzzy='xxx'
lightx2v spec: xxx
lightx2v version: xxx
```

> LightX2V 内部通过 `os.getenv('PLATFORM', 'cuda')` 选后端(`lightx2v/shot_runner/shot_base.py:43`)。**`PLATFORM=ascend_npu`**(不是 `npu`,不是 `ascend`)。后续每条 `python -m lightx2v.infer` / `python examples/...py` 命令都要 export。

## 拉权重（ModelScope）

LightX2V 跑蒸馏需要两组权重：

1. **base 模型**：T5 text encoder + CLIP vision encoder + VAE + Tokenizer,从 base 模型目录拿。I2V demo 用 `Wan2.2-I2V-A14B`(约 28 GB)。
2. **distill LoRA / ckpt**：MoE 架构分 high noise + low noise 两份,4 步推理只用 DIT。从 `lightx2v/Wan2.2-Distill-Models` 拿。

> ⚠️ ModelScope 上的 `lightx2v/Wan2.2-Distill-Models` 当前**只有 I2V 蒸馏权重**(T2V 蒸馏还没同步过来),4 个目录全是 `wan2.2_i2v_A14b_*_split/`。HF 上的同名 repo 含 T2V 蒸馏(`wan2.2_t2v_A14b_*_4step.safetensors`),如果只做 T2V,临时走 `huggingface-cli download` 也行。

### 拉 base 模型（I2V）

```shell #test-setup id="lightx2v-pull-wan22-base"
# Wan2.2-I2V-A14B 28 GB 左右。约 8 MB/s 的话 1 小时下完,后台上更稳。
# 落到 $PROJECT_ROOT/models/(持久卷 + 项目隔离),而不是 ./models/
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
mkdir -p "$PROJECT_ROOT/models"
modelscope download \
    --model Wan-AI/Wan2.2-I2V-A14B \
    --local_dir "$PROJECT_ROOT/models/Wan2.2-I2V-A14B"
```

```shell #test id="lightx2v-pull-wan22-base-verify"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
# 展示 base 模型目录结构(diffusion_pytorch_model shards + config.json):
ls -la "$PROJECT_ROOT/models/Wan2.2-I2V-A14B/" | head -20
echo "---"
# head config.json 验 model_type + _name_or_path
head -15 "$PROJECT_ROOT/models/Wan2.2-I2V-A14B/config.json"
```

```shell #test-result id="lightx2v-pull-wan22-base-verify" fuzzy='xxx'
total xxx
drwxr-xr-x 2 xxx xxx       64 Aug 29 16:05 .
drwxr-xr-x 4 xxx xxx      128 Aug 29 16:05 ..
-rw-r--r-- 1 xxx xxx      897 Aug 29 16:05 config.json
-rw-r--r-- 1 xxx xxx        1 Aug 29 16:05 .msc
-rw-r--r-- 1 xxx xxx 2516582400 Aug 29 16:06 diffusion_pytorch_model-00001-of-00007.safetensors
-rw-r--r-- 1 xxx xxx 2516582400 Aug 29 16:06 diffusion_pytorch_model-00002-of-00007.safetensors
... (省略 5-7 shards)
---
{
  "_class_name": "WanImageToVideoPipeline",
  "_diffusers_version": "0.31.0",
  "_name_or_path": "Wan-AI/Wan2.2-I2V-A14B",
  ...
}
```

### 拉 I2V 蒸馏 LoRA（split blocks）

```shell #test-setup id="lightx2v-pull-wan22-i2v-distill"
# ModelScope repo 当前只同步了 I2V 的 split-block 蒸馏 ckpt(4 个目录,
# high/low noise × int8/fp8 共 ~6.5 GB)。lightx2v 的 lazy load 直接读
# 目录里的 block_*.safetensors,不用先 merge 成单个文件。
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
modelscope download \
    --model lightx2v/Wan2.2-Distill-Models \
    --local_dir "$PROJECT_ROOT/models/Wan2.2-Distill-Models"
```

```shell #test id="lightx2v-pull-wan22-i2v-distill-verify"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
# 展示 split-block 目录列表(每个目录里有 block_0.safetensors ~ block_*.safetensors)
ls -la "$PROJECT_ROOT/models/Wan2.2-Distill-Models/" | grep -v "^total"
echo "---"
ls "$PROJECT_ROOT/models/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_1030_split/" | head -15
```

```shell #test-result id="lightx2v-pull-wan22-i2v-distill-verify" fuzzy='xxx'
drwxr-xr-x 2 xxx xxx  4096 Aug 29 08:52 ._____temp
-rw------- 1 xxx xxx  2574 Aug 29 08:54 .msc
drwxr-xr-x 2 xxx xxx  4096 Aug 29 08:53 wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030_split
drwxr-xr-x 2 xxx xxx  4096 Aug 29 08:54 wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step_1030_split
drwxr-xr-x 2 xxx xxx  4096 Aug 29 08:54 wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step_split
drwxr-xr-x 2 xxx xxx  4096 Aug 29 08:54 wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step_split
---
block_0.safetensors
block_1.safetensors
block_10.safetensors
block_11.safetensors
... (split block files)
```

> HF 上的同名 repo 还含 T2V 蒸馏(`wan2.2_t2v_A14b_*_4step.safetensors`,单文件 LoRA 不是 split) + ComfyUI 格式变体,完整清单见 [HF Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)。

### 拉 Qwen-Image-Edit-2511 Lightning（I2I demo 用）

⚠️ **20B BF16 Qwen-Image-Edit-2511 base 跑不动单卡 32 GB**(权重 40 GB,常驻就 OOM)。改走 **FP8 蒸馏版**(`qwen_image_i2i_2511_distill_fp8.json`),只下 Lightning repo 里的 FP8 merged ckpt(**不下 base**):

```shell #test-setup id="lightx2v-pull-qwen-edit"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
# Lightning repo 里 FP8 merged ckpt(~10 GB) + 纯 LoRA 适配器(~0.5 GB)+ 其它辅助文件
modelscope download \
    --model lightx2v/Qwen-Image-Edit-2511-Lightning \
    --local_dir "$PROJECT_ROOT/models/Qwen-Image-Edit-2511-Lightning"
```

```shell #test id="lightx2v-pull-qwen-edit-verify"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
ls -la "$PROJECT_ROOT/models/Qwen-Image-Edit-2511-Lightning/" | grep -v "^total"
```

```shell #test-result id="lightx2v-pull-qwen-edit-verify" fuzzy='xxx'
-rw-r--r-- 1 xxx xxx 540770304 Aug 29 16:19 Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors
-rw-r--r-- 1 xxx xxx 10000000000 Aug 29 16:21 qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors
... (其它辅助文件: text encoder / VAE 等,视 repo 内容)
```

> FP8 ckpt 只量化 DiT(20B → 10 GB)。text encoder(Qwen2.5-VL-7B 等)和 VAE 视 Lightning repo 是否带 → 如果带了就直接用,没带要单独补下。**跑 smoke test 时如果报缺模块,再到 `lightx2v/models/input_encoders/` / `video_encoders/` 看具体依赖**。

## 列出 Wan2.2 4 步蒸馏 config

LightX2V 把推理参数(步数、flow shift、offload 策略、LoRA 路径、量化方案等)写到 config JSON 里,`python -m lightx2v.infer --config_json ...` 一次性喂进去。

```shell #test id="lightx2v-list-wan22-configs"
# 主线 Wan2.2 config(T2V/I2V 各一个,MoE 高低噪合并)
ls configs/wan22/ | grep -E "moe_(t2v|i2v)"
echo "---"
# 蒸馏 + LoRA config(I2V LoRA 版本,NPU int8 量化版本等)
ls configs/distill/wan22/
```

输出结果类似：

```shell #test-result id="lightx2v-list-wan22-configs" fuzzy='xxx'
wan_moe_i2v.json
wan_moe_t2v.json
... (主线 moe 配置)
---
wan_moe_i2v_distill_4090.json
wan_moe_i2v_distill_5090.json
wan_moe_i2v_distill_fp8_4step_cfg_ulysses.json
wan_moe_i2v_distill_int8_4step_ulysses_npu.json   <- NPU int8 量化专用,单卡需改
wan_moe_i2v_distill_lora_4step_cfg_ulysses.json
wan_moe_i2v_distill_model.json
wan_moe_i2v_distill_model_4step_cfg_ulysses.json
wan_moe_i2v_distill_model_4step_cfg_ulysses_rainfusion.json
wan_moe_i2v_distill_quant.json
wan_moe_i2v_distill_with_lora.json                <- 蒸馏 LoRA 通用版
wan_moe_t2v_distill_lora.json
wan_moe_t2v_distill_lora_4step_cfg_ulysses.json
```

下面 I2V demo 用 **`configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json`**(NPU int8 量化版本,`self_attn_1_type=rainfusion_attn` + `cross_attn_*=npu_flash_attn`,直走 NPU 算子)。**但默认 config 是为「910B + 2 卡 + 720P」设计的**(720P+ 81 帧 + T5/VAE 全常驻 + ulysses 并行 + rife 视频插帧),**单卡 32 GB 必须改**:720→480 + 开 t5/vae cpu_offload + 删 parallel + 删 video_frame_interpolation。改动在下面 I2V smoke 块里完成。

I2I demo 用 **`configs/qwen_image/qwen_image_i2i_2511_distill_fp8.json`**(FP8 量化 + Lightning 8 步),**不下载 Qwen-Image-Edit-2511 base 20 GB**(单卡 BF16 跑不下)。

> 公共要点:所有 config JSON 里 `xxx_ckpt` 默认是 HF repo 路径(`lightx2v/Qwen-Image-Edit-2511-Lightning/...` 之类),LightX2V 通过 hf_hub / 本地 cache 解析。本文 doc 把它们改成 `$PROJECT_ROOT/models/...` 绝对路径,避免 hf cache 污染。

## I2V 烟囱测试：4 步蒸馏图生视频

参考官方脚本 [`scripts/wan22/distill/run_wan22_moe_i2v_distill_lora_4step.sh`](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/distill/run_wan22_moe_i2v_distill_lora_4step.sh) 的标准入口是 `python -m lightx2v.infer --model_cls wan2.2_moe_distill --task i2v --model_path <base> --config_json <cfg> --prompt ... --image_path ... --save_result_path ...`。Wan2.2-I2V-A14B 是 MoE 架构,high noise + low noise 两份 ckpt 都由 config JSON 内部指定(走 lazy block load)。

### 单卡 32 GB 适配：改 config

```shell #test-setup id="lightx2v-i2v-cfg-adapt"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
python -c "
import json, pathlib, os
cfg_path = pathlib.Path('configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json')
cfg = json.loads(cfg_path.read_text())
proj_models = os.environ['PROJECT_ROOT'] + '/models'
# split-block 目录:绝对路径,避免 cwd 依赖
cfg['high_noise_quantized_ckpt'] = proj_models + '/Wan2.2-Distill-Models/wan2.2_i2v_A14b_high_noise_int8_lightx2v_4step_1030_split'
cfg['low_noise_quantized_ckpt']  = proj_models + '/Wan2.2-Distill-Models/wan2.2_i2v_A14b_low_noise_int8_lightx2v_4step_split'
# 720P → 480P(activation/KV 省 ~40%)
cfg['target_height'] = 480
cfg['target_width']  = 832
# 开 T5/VAE cpu_offload(关键:不释放这俩,32GB 必 OOM)
cfg['t5_cpu_offload']  = True
cfg['vae_cpu_offload'] = True
# block 粒度 DIT offload(进一步省)
cfg['cpu_offload']             = True
cfg['offload_granularity']     = 'block'
# 单卡 ulysses 没意义,删掉
cfg.pop('parallel', None)
# rife 插帧要单独下模型,不需要就删
cfg.pop('video_frame_interpolation', None)
cfg_path.write_text(json.dumps(cfg, indent=4))
print('cfg adapted (单卡 32GB 适配):')
for k in ['target_height', 'target_width', 'cpu_offload', 't5_cpu_offload', 'vae_cpu_offload', 'high_noise_quantized_ckpt']:
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
ls -la "$OUT"
file "$OUT"
# 优先用 ffprobe 读视频流(宽/帧数/时长),ffprobe 不存在时退到 ftyp magic 验证
ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration \
    -of default=noprint_wrappers=1 "$OUT" 2>/dev/null || \
    python -c "b=open('$OUT','rb').read(1024*1024); print('size_on_disk:', len(b), 'bytes; ftyp:', b[4:8].decode('ascii','ignore'))"
```

```shell #test-result id="lightx2v-i2v-output" fuzzy='xxx'
-rw-r--r-- 1 xxx xxx xxx Aug 29 16:14 /home/coder/work/lightx2v-test/save_results/output_wan22_moe_i2v_distill_int8_npu_480p.mp4
/home/coder/work/lightx2v-test/save_results/output_wan22_moe_i2v_distill_int8_npu_480p.mp4: ISO Media, MP4 Base Media v2 [ISO 14496-12:2015]
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

## T2V 烟囱测试：4 步蒸馏文生视频

`configs/wan22/wan_moe_t2v_distill.json` 已经写好(`infer_steps: 4`, `high_noise_original_ckpt` + `low_noise_original_ckpt` 两份 DIT ckpt,MoE 合并)。**单卡 32 GB 同样要做 I2V 那套适配**:

```shell #test-setup id="lightx2v-t2v-cfg-adapt"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
python -c "
import json, pathlib, os
cfg_path = pathlib.Path('configs/wan22/wan_moe_t2v_distill.json')
cfg = json.loads(cfg_path.read_text())
print('---wan_moe_t2v_distill.json 原始值---')
print('infer_steps:', cfg['infer_steps'])
print('high_noise_original_ckpt:', cfg.get('high_noise_original_ckpt'))
print('low_noise_original_ckpt:',  cfg.get('low_noise_original_ckpt'))
print('sample_guide_scale:', cfg.get('sample_guide_scale'))
print('denoising_step_list:', cfg.get('denoising_step_list'))
"
```

```shell #test-result id="lightx2v-t2v-cfg-adapt" fuzzy='xxx'
infer_steps: xxx
high_noise_original_ckpt: xxx
low_noise_original_ckpt: xxx
sample_guide_scale: xxx
denoising_step_list: xxx
```

⚠️ **ModelScope 的 `lightx2v/Wan2.2-Distill-Models` 当前只同步了 I2V ckpt**(目录全是 `wan2.2_i2v_A14b_*_split/`)。T2V 蒸馏要的两份 `Wan2.2-T2V-A14B/distill_models/{high,low}_noise_model/distill_model.safetensors` 在 ModelScope 上没有,需要绕走:

```shell #test-setup id="lightx2v-pull-wan22-t2v-distill-fallback"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
# 方案 A:HF 上拉(需要 HF token + 网络通畅)
huggingface-cli download lightx2v/Wan2.2-Distill-Models \
    --local-dir "$PROJECT_ROOT/models/Wan2.2-Distill-Models-hf" \
    --include "wan2.2_t2v_A14b_high_noise_lightx2v_4step*.safetensors" \
                "wan2.2_t2v_A14b_low_noise_lightx2v_4step*.safetensors"

# 方案 B:跑主线 wan_moe_t2v(不蒸馏,50 步,慢但不需要额外蒸馏权重)
# python -m lightx2v.infer --model_cls wan2.2_moe --task t2v --model_path ... --config_json configs/wan22/wan_moe_t2v.json
```

走通之后跟 I2V 同款调用模式(`--task t2v`, 去掉 `--image_path`, 改 `--config_json`)。完整 T2V 烟囱测试本文档略,留给后续按需补。

## T2I 烟囱测试：Qwen-Image-Edit Lightning FP8 图编辑

参考 [examples/qwen_image/qwen_2511_with_distill_lora.py](https://github.com/ModelTC/LightX2V/blob/main/examples/qwen_image/qwen_2511_with_distill_lora.py),Qwen-Image-Edit 用 `model_cls=qwen-image-edit-2511` + `task=i2i`(LightX2V 命名是 i2i,**不是 t2i**)。Lightning 是 **8 步蒸馏**(不是 4 步,example docstring 的 4steps 名字是别的版本)。

### 跑 I2I(FP8 蒸馏版,不下 20 GB base)

```shell #test-setup id="lightx2v-i2i-cfg-adapt"
# FP8 config 默认 dit_quantized_ckpt 指向 HF repo,改成 $PROJECT_ROOT/models/ 绝对路径
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
python -c "
import json, pathlib, os
cfg_path = pathlib.Path('configs/qwen_image/qwen_image_i2i_2511_distill_fp8.json')
cfg = json.loads(cfg_path.read_text())
proj_models = os.environ['PROJECT_ROOT'] + '/models'
cfg['dit_quantized_ckpt'] = proj_models + '/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors'
cfg_path.write_text(json.dumps(cfg, indent=4))
print('cfg dit_quantized_ckpt:', cfg['dit_quantized_ckpt'])
"
```

```shell #test-setup id="lightx2v-i2i-smoke-run"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
mkdir -p "$PROJECT_ROOT/save_results"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=/tmp/stubs:${PYTHONPATH:-}     # aarch64 才有,x86_64 删掉
export TOKENIZERS_PARALLELISM=false

python -m lightx2v.infer \
    --model_cls qwen-image-edit-2511 \
    --task i2i \
    --model_path "$PROJECT_ROOT/models/Qwen-Image-Edit-2511-Lightning" \
    --config_json configs/qwen_image/qwen_image_i2i_2511_distill_fp8.json \
    --prompt "Replace the polka-dot shirt with a light blue shirt." \
    --image_path "$PROJECT_ROOT/src/assets/inputs/imgs/img_0.jpg" \
    --save_result_path "$PROJECT_ROOT/save_results/output_qwen_image_edit_2511_distill_fp8.png"
```

> **关于 `--model_path`**: FP8 ckpt 是 Lightning repo 里的 merged 文件,config 里 `dit_quantized_ckpt` 已经指向它了;`--model_path` 这里传 Lightning repo 根目录是 LightX2V 找 text encoder / VAE / tokenizer 的位置(视 Lightning repo 是否带这些组件而定)。如果 Lightning repo 不带 text encoder,要补下 Qwen2.5-VL-7B 这类,然后 `--model_path` 改成那个目录。

```shell #test id="lightx2v-i2i-output"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
OUT="$PROJECT_ROOT/save_results/output_qwen_image_edit_2511_distill_fp8.png"
ls -la "$OUT"
file "$OUT"
python -c "
from PIL import Image
im = Image.open('$OUT')
print(f'size: {im.size}, mode: {im.mode}, format: {im.format}')
"
```

```shell #test-result id="lightx2v-i2i-output" fuzzy='xxx'
-rw-r--r-- 1 xxx xxx xxx Aug 29 16:25 /home/coder/work/lightx2v-test/save_results/output_qwen_image_edit_2511_distill_fp8.png
/home/coder/work/lightx2v-test/save_results/output_qwen_image_edit_2511_distill_fp8.png: PNG image data, 1328 x 1328, 8-bit/color RGB, non-interlaced
size: (xxx, xxx), mode: xxx, format: xxx
```

> **I2I 跟 I2V/T2V 的 env 区别**:Qwen demo **不需要 `DTYPE=BF16` / PYTORCH_CUDA_ALLOC_CONF**(没碰 DIT bf16 路径);`enable_cfg: false` 已经是 8 步蒸馏默认,不需要另开。
>
> **跑失败的常见原因**:Lightning repo 不带 text encoder → import / load 时报缺 Qwen2.5-VL 这类模块 → 回到 [LightX2V 模型结构文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/getting_started/model_structure.html)看完整目录布局,缺啥补啥。

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