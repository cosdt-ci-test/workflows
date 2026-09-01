# LightX2V 快速入门指南（Ascend NPU）

欢迎使用 LightX2V！本指南帮助你在单卡昇腾 NPU 上快速搭建环境并生成视频。整体流程与[官方快速入门](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/getting_started/quickstart.html)一致，仅依赖安装与模型下载按 NPU 环境适配。权重下载走 [ModelScope](https://modelscope.cn/)（国内网络更稳），HF 仓库同名映射。

## 🚀 系统要求

- **硬件**：Atlas 900 A2 / A3 训练系列产品（Ascend 910B4 / 910B 等），单卡 32 GB HBM 可跑本文全部内容
- **操作系统**：Linux（Ubuntu 22.04）
- **存储**：至少 60 GB 可用空间

**本文配套镜像**（[ascendhub 官方 mindiesd 镜像](https://www.hiascend.com/developer/ascendhub/detail/7c3b1b7c5151469a98ac08b868dab45f)，基于 vllm-omni 底座）：

```
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindiesd:v3.0.0-A2-ubuntu22.04-py3.11-aarch64
```

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| CANN | 8.5.1 | 随镜像 |
| Python | 3.11 | 随镜像 |
| torch / torch_npu | 2.9.0（以镜像为准） | 随 vllm-omni 底座 |
| mindiesd | 3.0.0 | 随镜像 |
| lightx2v | GitHub main 分支（滚动 main） | 步骤 2 安装 |

> 非容器（裸机）也可以：自带可用的 CANN + torch + torch_npu（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)）即可跳过环境搭建，后续步骤完全相同。

## 🐧 Linux 系统环境搭建

### 🐳 Docker 环境（推荐）

1. 拉取镜像：

```shell
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindiesd:v3.0.0-A2-ubuntu22.04-py3.11-aarch64
```

2. 运行容器（NPU 设备与宿主机驱动库必须挂载，多卡按需加 `--device /dev/davinciN`）：

```shell
docker run -it --rm --name=lightx2v --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/mindiesd:v3.0.0-A2-ubuntu22.04-py3.11-aarch64 \
    bash
```

容器内 CANN 环境变量已 export；若非交互式 shell 里 `import torch_npu` 报 `libhccl.so` 缺失，手动 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`（幂等，本文示例命令里都带着）。

### 安装 LightX2V（Docker / 裸机通用）

以下步骤 Docker 容器与裸机**通用**（mindiesd 镜像不含 LightX2V 源码，容器内同样执行）：

1. 克隆项目（`<UPSTREAM_REF>` 换成目标分支/tag/commit，上游零 release 零 tag 默认 `main`；权重放 `$PROJECT_ROOT/models/`，`src/models -> ../models` 软链保持 LightX2V 的 `./models/` 相对路径约定）：

<!-- 工作流注入的 UPSTREAM_REF（上游零 tag,固定 main）通过这个隐藏的 #test-setup 捕获并注入到下方 clone 命令中；markdown 渲染器会丢掉注释，读者看不到，runner 仍会执行 -->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test-setup id="lightx2v-install-source" load="upstream_ref>>UPSTREAM_REF"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

git clone --depth 1 --branch <UPSTREAM_REF> https://github.com/ModelTC/LightX2V.git src
cd src
ln -sfn ../models models
```

2. 安装依赖及代码（NPU 适配：aarch64 缺 wheel 的 `cv2` / `decord` / `torchaudio` 打空 stub 占位——LightX2V import 链会触发它们但 smoke 不真正调用；`--no-deps` 装源码后按上游 pyproject 补齐其余依赖，`nvidia-*`/`cuda-*` 由排除清单自动屏蔽）：

```shell #test-setup
export PYTHONPATH=/tmp/stubs:${PYTHONPATH:-}
rm -rf /tmp/stubs && mkdir -p /tmp/stubs
python - <<'PY'
import os

STUB = (
    "class _Stub:\n"
    "    def __getattr__(self, name): return _Stub()\n"
    "    def __call__(self, *a, **k): return _Stub()\n"
    "import sys as _s\n"
    "_s.modules[__name__].__getattr__ = lambda name: _Stub()\n"
)
for m in ('cv2', 'decord', 'torchaudio'):
    d = os.path.join('/tmp/stubs', m)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, '__init__.py'), 'w') as fh:
        fh.write(STUB)
print('stubs ready:', sorted(os.listdir('/tmp/stubs')))
PY
```

```shell #test-setup id="lightx2v-install-deps"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
uv pip install --no-deps -v .

uv pip install \
    numpy scipy diffusers transformers tokenizers tqdm accelerate safetensors \
    imageio imageio-ffmpeg einops loguru omegaconf peft \
    swanlab qtorch 'comfy-kitchen>=0.2.15' ftfy gradio \
    aiohttp pydantic prometheus-client gguf fastapi uvicorn PyJWT requests \
    aio-pika 'asyncpg>=0.27.0' 'aioboto3>=12.0.0' \
    'alibabacloud_dypnsapi20170525==1.2.2' 'redis==6.4.0' tos \
    av 'torchada>=0.1.10' pyzmq soundfile \
    'modelscope==1.37.0' triton
```

3. 安装注意力/量化算子：**NPU 无需安装**。torchada 内置 `npu_flash_attn` / `npu_rope` / `npu_layer_norm` 等 NPU 算子，config 里用 `self_attn_1_type` / `cross_attn_*_type` 等键选择；int8-npu 量化算子同样已含于 torchada。镜像已含 mindiesd——如需 rainfusion 稀疏注意力加速，把 `self_attn_1_type` 改回 `rainfusion_attn` 即可。

4. 验证安装（没有 `uv` 先 `pip install uv`）：

```shell #test id="lightx2v-install-verify"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
export PYTHONPATH=/tmp/stubs:$PYTHONPATH
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

> 开头的 `...` 吞掉 `import lightx2v` 期间可选加速后端未安装时的提示行（如 `spas_sage_attn is not installed.`），不进断言。

## 🎯 推理使用

### 📥 模型准备

官方昇腾路线用 [Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)（~17.6 GB：T5 + VAE + 1.3B DiT + tokenizer，单卡无压力）：

```shell #test-setup id="lightx2v-pull-wan21-t2v"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
mkdir -p "$PROJECT_ROOT/models"
modelscope download \
    --model Wan-AI/Wan2.1-T2V-1.3B \
    --local_dir "$PROJECT_ROOT/models/Wan2.1-T2V-1.3B" \
    --include "diffusion_pytorch_model.safetensors" "models_t5_umt5-xxl-enc-bf16.pth" \
               "Wan2.1_VAE.pth" "google/*" "config.json"
```

### 📁 配置文件与脚本

官方为昇腾准备了现成 config：`configs/platforms/ascend_npu/wan_t2v.json`（`npu_flash_attn` 三路注意力、480×832、50 步、cpu_offload 全配好），直接用、零适配；[config](https://github.com/ModelTC/LightX2V/tree/main/configs) 与 [scripts](https://github.com/ModelTC/LightX2V/tree/main/scripts) 都在上游仓库。

### 🚀 开始推理

官方 NPU 路线同款（[`scripts/platforms/ascend_npu/run_wan21_t2v.sh`](https://github.com/ModelTC/LightX2V/blob/main/scripts/platforms/ascend_npu/run_wan21_t2v.sh) 的 Python API 版；与官方 quickstart 示例同构，差异仅 `config_json` 用官方昇腾 config——官方示例的 `sage_attn2` 是 CUDA 算子、NPU 不可用）：

```shell #test-setup id="lightx2v-i2v-smoke-run"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
mkdir -p "$PROJECT_ROOT/save_results"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PLATFORM=ascend_npu
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=/tmp/stubs:${PYTHONPATH:-}

python - <<'PY'
import os
from lightx2v import LightX2VPipeline

root = os.environ['PROJECT_ROOT']
model_path = root + '/models/Wan2.1-T2V-1.3B'
save_result_path = root + '/save_results/output_lightx2v_wan_t2v.mp4'

pipe = LightX2VPipeline(
    model_path=model_path,
    model_cls='wan2.1',
    task='t2v',
)
pipe.create_generator(
    config_json='configs/platforms/ascend_npu/wan_t2v.json'
)

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
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
# 确定性结构校验:形状由下面的 print 自控(跨机器/跨时间可复现),值全掩码;
# 硬断言 size 下界 + ftyp magic;深度信息(编码/时长/帧数)经 imageio-ffmpeg
# 自带的静态 ffmpeg 解析,解析不了就打 probe-skip,形状不破
python - <<'PY'
import os, re, subprocess
out = os.path.join(os.environ['PROJECT_ROOT'], 'save_results',
                   'output_lightx2v_wan_t2v.mp4')
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

> **关键 env**（都在上方 smoke 块里）：`PLATFORM=ascend_npu`（值必须是 `ascend_npu`，不是 `ascend`/`npu`）、`ASCEND_RT_VISIBLE_DEVICES=0`、`source set_env.sh`（非交互式 shell 需要）、`PYTHONPATH=/tmp/stubs`。

## 📞 获取帮助

1. 在 [GitHub Issues](https://github.com/ModelTC/LightX2V/issues) 搜索或提交问题
2. 更多教程：[量化](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/quantization.html) / [特征缓存](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/cache.html) / [多卡并行](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/parallel.html)（单卡跑通后把 `parallel.seq_p_size` 加回 config 并设为 `torch.npu.device_count()`）/ [服务化部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_service.html)
3. 其它模型路线（Wan2.2 MoE I2V 4 步蒸馏 int8、Qwen-Image-Edit T2I 等）见[官方 README](https://github.com/ModelTC/LightX2V/blob/main/README_zh.md) 与 [configs/](https://github.com/ModelTC/LightX2V/tree/main/configs)
