# LightX2V 快速入门指南（Ascend NPU）

欢迎使用 LightX2V！本指南帮助你在单卡昇腾 NPU 上快速搭建环境并生成视频。

## 🚀 系统要求

- **硬件**：Atlas 900 A2 / A3 训练系列产品（Ascend 910B4 / 910B 等），单卡 32 GB HBM 可跑本文全部内容
- **操作系统**：Linux（Ubuntu 22.04）
- **存储**：至少 30 GB 可用空间

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| CANN | ≥ 8.5.1 | 环境搭建安装（CI 以 9.1.0 验证） |
| Python | ≥ 3.10 | 自备环境（CI 镜像为 3.12） |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 | 环境搭建安装 |
| lightx2v | GitHub main 分支（滚动 main） | 下方安装 |

> 也可用带 CANN 的昇腾镜像（如 [ascendhub cann 镜像](https://www.hiascend.com/developer/ascendhub)）跳过 CANN 安装，其余步骤相同。

## 🐧 Linux 系统环境搭建

### 🐍 环境搭建

**安装 CANN**（≥ 8.5.1，与驱动配套，[快速安装脚本](https://ascend.github.io/docs/sources/ascend/quick_install.html)会自动识别卡型），安装完成后：

```shell
source ~/Ascend/ascend-toolkit/set_env.sh
```

**准备 Python 环境**：Python ≥ 3.10。

**安装 torch + torch_npu**：

```shell #test-setup id="lightx2v-install-torch"
pip install uv
uv pip install "torch==2.9.0" "torchvision==0.24.*" "torch_npu==2.9.0.post2"
```

### 安装 LightX2V

以下步骤通用：

**克隆项目**（先进入想放置项目的目录，后续步骤都在该目录下执行）：

```shell #test-setup id="lightx2v-install-source"
git clone --depth 1 https://github.com/ModelTC/LightX2V.git
```

**安装依赖及代码**（aarch64 上部分依赖无预编译包，无法自动解析依赖，先装源码再补齐）：

```shell #test-setup id="lightx2v-install-deps"
uv pip install --no-deps -v ./LightX2V

uv pip install \
    numpy scipy diffusers transformers tokenizers tqdm accelerate safetensors \
    imageio imageio-ffmpeg einops loguru omegaconf peft \
    swanlab qtorch 'comfy-kitchen>=0.2.15' ftfy gradio \
    aiohttp pydantic prometheus-client gguf fastapi uvicorn PyJWT requests \
    aio-pika 'asyncpg>=0.27.0' 'aioboto3>=12.0.0' \
    'alibabacloud_dypnsapi20170525==1.2.2' 'redis==6.4.0' tos \
    av 'torchada>=0.1.10' pyzmq soundfile \
    'modelscope==1.37.0' 'triton==3.5.*'
```

**验证安装**（没有 `uv` 先 `pip install uv`）：

```shell #test id="lightx2v-install-verify"
python -c "
import os
os.environ.setdefault('PLATFORM', 'ascend_npu')   # lightx2v 按此选后端,默认 cuda
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


## 🎯 推理使用

### 📥 模型准备

本文使用 [Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)（~17.6 GB）：

```shell #test-setup id="lightx2v-pull-wan21-t2v"
modelscope download \
    --model Wan-AI/Wan2.1-T2V-1.3B \
    --local_dir models/Wan2.1-T2V-1.3B \
    --include "diffusion_pytorch_model.safetensors" "models_t5_umt5-xxl-enc-bf16.pth" \
               "Wan2.1_VAE.pth" "google*" "config.json"
```

### 🚀 开始推理

通过 [Python API](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/getting_started/quickstart.html) 生成视频：

```shell #test-setup id="lightx2v-i2v-smoke-run"
python - <<'PY'
import os
os.environ.setdefault('PLATFORM', 'ascend_npu')      # lightx2v 按此选后端
from lightx2v import LightX2VPipeline

root = os.getcwd()
model_path = root + '/models/Wan2.1-T2V-1.3B'
save_result_path = root + '/save_results/output_lightx2v_wan_t2v.mp4'
os.makedirs(os.path.dirname(save_result_path), exist_ok=True)

pipe = LightX2VPipeline(
    model_path=model_path,
    model_cls='wan2.1',
    task='t2v',
)
# 仓库自带的 NPU 专用配置（npu_flash_attn / 50 步 / cpu_offload 已预调好）
pipe.config_json = root + '/LightX2V/configs/platforms/ascend_npu/wan_t2v.json'
pipe.create_generator()

seed = 42
prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

pipe.generate(
    seed=seed,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=save_result_path,
)
PY
```

### 输出校验

```shell #test id="lightx2v-i2v-output"
python - <<'PY'
import os, re, subprocess
root = os.getcwd()
out = os.path.join(root, 'save_results', 'output_lightx2v_wan_t2v.mp4')
size = os.path.getsize(out)
assert size > 1_000_000, f'output too small: {size} bytes'
with open(out, 'rb') as fh:
    magic = fh.read(12)
assert magic[4:8] == b'ftyp', f'not an mp4: {magic!r}'
print('size:', size)
print('ftyp_brand:', magic[8:12].decode('ascii', 'ignore'))
try:
    import imageio_ffmpeg
    err = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), '-i', out, '-f', 'null', '-'],
        capture_output=True, text=True, timeout=300,
    ).stderr
    m = re.search(r'Video: (\w+)', err)
    d = re.search(r'Duration: (\d+):(\d+):([\d.]+)', err)
    fr = re.findall(r'frame=\s*(\d+)', err)
    print('codec:', m.group(1) if m else 'unknown')
    print('duration_s:',
          round(int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)), 2) if d else 'unknown')
    print('frames:', fr[-1] if fr else 'unknown')
except Exception as exc:
    print('codec: probe-skip', type(exc).__name__)
    print('duration_s: probe-skip')
    print('frames: probe-skip')
PY
```

```shell #test-result id="lightx2v-i2v-output" fuzzy='xxx'
size: xxx
ftyp_brand: xxx
codec: xxx
duration_s: xxx
frames: xxx
```

> **注意**：如新开终端执行推理，先 `source ~/Ascend/ascend-toolkit/set_env.sh`。
