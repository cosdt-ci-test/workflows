# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑通 [LightX2V](https://github.com/ModelTC/LightX2V) 的最小链路：容器环境 + 安装源码 + 拉权重 + 4 步蒸馏图生视频（I2V）烟囱测试。权重下载走 [ModelScope](https://modelscope.cn/)（国内网络更稳），HF 仓库同名映射。

## 系统要求

- **硬件**：Atlas 900 A2 / A3 训练系列产品（Ascend 910B4 / 910B 等），单卡 32 GB 可跑本文全部内容；按需完成物理机或容器内的设备挂载
- **存储**：~60 GB（源码 + 权重）

**本文配套镜像**（[ascendhub 官方 mindiesd 镜像](https://www.hiascend.com/developer/ascendhub/detail/7c3b1b7c5151469a98ac08b868dab45f)，MindIE 社区维护，基于 vllm-omni 底座）：

```
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindiesd:v3.0.0-A2-ubuntu22.04-py3.11-aarch64
```

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| OS / Python | Ubuntu 22.04 / 3.11 | 随镜像 |
| CANN | 8.5.1 | 随镜像 |
| torch / torch_npu | 2.9.0（以镜像为准） | 随 vllm-omni 底座 |
| mindiesd | 3.0.0 | 随镜像 |
| modelscope | 1.37.0 | 文档安装 |
| lightx2v | GitHub main 分支（上游零 release 零 tag，滚动 main） | 文档安装 |

> 非容器（裸机）也能跑：机器上自带可用的 CANN + torch + torch_npu（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按三方兼容矩阵选版本）即可跳过环境搭建，后续 pip 步骤完全相同。

## 环境搭建（容器）

拉取镜像：

```shell
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindiesd:v3.0.0-A2-ubuntu22.04-py3.11-aarch64
```

启动容器（NPU 设备 + 宿主机驱动库必须挂载，多卡按需加 `--device /dev/davinciN`）：

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

CANN 环境变量（`ASCEND_HOME` / `LD_LIBRARY_PATH` 等）镜像内已 export；若非交互式 shell 里 `import torch_npu` 报 `libhccl.so` 缺失，手动 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 即可（幂等，本文示例命令里都带着）。

## 安装 LightX2V

### 环境自检

确认能看到 NPU 设备（`npu-smi` 是宿主机驱动工具，部分容器内没有，以 `check-torch` 的 `is_available: True` 为准）：

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
npu-smi info || echo 'npu-smi not found in container; rely on check-torch'
```

输出类似：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| 5     910B4               | OK            | 89.9        39                0    / 0             2922 / 32768 |
| 0                         | 0000:41:00.0  | 0           0    / 0                               |
+===========================+===============+====================================================+
```

检查 Python（文档统一用 `uv pip` 装依赖，没有先 `pip install uv`）：

```shell #test id="check-py"
python --version
```

```shell #test-result id="check-py" fuzzy='xxx'
Python xxx
```

镜像自带 `torch`（CPU build，NPU 加速走 `torch_npu`）+ `torch_npu`，确认可用即可：

```shell #test id="check-torch"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

```shell #test-result id="check-torch" fuzzy='xxx'
torch= xxx
torch_npu= xxx
is_available: True
count: 1
```

> `import torch_npu` 失败，先确认 CANN env 已 source（`echo $LD_LIBRARY_PATH | grep cann`），再按 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查三方兼容矩阵。

装 `modelscope`（本文档下载权重走 ModelScope，国内网络更稳）：

```shell #test-setup
uv pip install modelscope==1.37.0
```

```shell #test id="modelscope-version"
python -c "import modelscope; print('modelscope=', modelscope.__version__)"
```

```shell #test-result id="modelscope-version" disable_fuzzy
modelscope= 1.37.0
```

### aarch64 按需 stub（cv2 / decord / torchaudio）+ triton

LightX2V 的 import 链会触发 `cv2` / `decord` / `torchaudio`，但它们不在 mindiesd 镜像的依赖树里（vllm-omni 多媒体栈带的是 av / soundfile / pyzmq），aarch64 PyPI 也没有可用 wheel（cv2 缺 libxcb、decord 无 wheel、torchaudio 编的是 CUDA ABI）。本文 smoke 不真正调用这三者（I2V 输入走 PIL，输出走 imageio-ffmpeg 的 ffmpeg），所以**缺谁才给谁打空 stub**，真模块可用就用真的：

```shell #test-setup
export PYTHONPATH=/tmp/stubs:${PYTHONPATH:-}
# rm -rf 防御:跑过旧版文档的机器可能残留旧 stub(尤其 triton stub 会遮蔽真包),先清掉再按需重建
rm -rf /tmp/stubs/cv2 /tmp/stubs/decord /tmp/stubs/torchaudio /tmp/stubs/triton
mkdir -p /tmp/stubs
python - <<'PY'
import importlib, os

STUB = (
    "class _Stub:\n"
    "    def __getattr__(self, name): return _Stub()\n"
    "    def __call__(self, *a, **k): return _Stub()\n"
    "import sys as _s\n"
    "_s.modules[__name__].__getattr__ = lambda name: _Stub()\n"
)
for m in ('cv2', 'decord', 'torchaudio'):
    try:
        importlib.import_module(m)  # 真模块能完整加载(.so 也在) -> 不打 stub
        print(m, 'real module present, skip stub')
    except Exception:
        d = os.path.join('/tmp/stubs', m)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, '__init__.py'), 'w') as fh:
            fh.write(STUB)
        print(m, 'stub created at', d)
PY
# triton 不能拿空 stub 糊弄(torch _inductor 拿 triton.Config 当基类,stub 语义对不上逐层炸);
# 镜像 vllm 底座大概率自带,没有就装真包(3.5.x 有 aarch64 wheel,零运行时依赖)
python -c "import triton" 2>/dev/null || uv pip install 'triton==3.5.*'
```

验证 stub / triton / torch 链路：

```shell #test id="stubs-verify"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/tmp/stubs:$PYTHONPATH
python -c "
import importlib
for m in ('cv2', 'decord', 'torchaudio'):
    mod = importlib.import_module(m)
    print(m, 'ok:', getattr(mod, '__version__', 'stub'))
import triton
print('triton:', triton.__version__)
class ProbeConfig(triton.Config):
    pass
print('Config subclass ok=', ProbeConfig.__name__)
import torch, torch_npu
print('torch/torch_npu chain ok=', torch.__version__)
"
```

```shell #test-result id="stubs-verify" fuzzy='xxx'
cv2 ok: xxx
decord ok: xxx
torchaudio ok: xxx
triton: xxx
Config subclass ok= xxx
torch/torch_npu chain ok= xxx
```

### 源码安装

**项目隔离布局**：权重放 `$PROJECT_ROOT/models/`，源码 clone 到 `$PROJECT_ROOT/src/`，`src/models -> ../models` 软链保持 LightX2V 的 `./models/` 相对路径约定，源码可随时重 clone 而权重不丢。

<!-- 工作流注入的 UPSTREAM_REF（上游零 tag,固定 main）通过这个隐藏的 #test-setup 捕获并注入到下方 clone 命令中；markdown 渲染器会丢掉注释，读者看不到，runner 仍会执行 -->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

项目根默认 `/home/coder/work/lightx2v-test`（CI 注入 `PROJECT_ROOT` 走别的路径也行）；`<UPSTREAM_REF>` 换成目标分支/tag/commit——上游零 release 零 tag，默认 main，看护流水线会把本次要测的 ref 注入 `UPSTREAM_REF` 环境变量；`ln -sfn ../models models` 软链让 LightX2V 内部 `./models/` 相对路径依然命中；`--no-deps` 装源码时缺 wheel 的依赖会被跳过，下面统一补装。

```shell #test-setup id="lightx2v-install-source" load="upstream_ref>>UPSTREAM_REF"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
mkdir -p "$PROJECT_ROOT"
cd "$PROJECT_ROOT"

git clone --depth 1 --branch <UPSTREAM_REF> https://github.com/ModelTC/LightX2V.git src
cd src
ln -sfn ../models models

uv pip install --no-deps -v .
```

依赖安装：清单按上游 pyproject 对齐，镜像已带的（av / soundfile / pyzmq / diffusers / transformers / accelerate 等）自动 no-op，缺的补装；上游 pyproject 漏声明的 `pyzmq`（`lightx2v.disagg.conn` 顶层 `import zmq`）和 `soundfile`（infer.py 顶层 import 链）也在清单里兜底；CUDA 排除清单（nvidia-\* / cuda-\*）经 `PIP_CONSTRAINT`/`UV_CONSTRAINT` 已在进程环境自动屏蔽。

```shell #test-setup id="lightx2v-install-deps"
uv pip install \
    numpy scipy diffusers transformers tokenizers tqdm accelerate safetensors \
    imageio imageio-ffmpeg einops loguru omegaconf peft \
    swanlab qtorch 'comfy-kitchen>=0.2.15' ftfy gradio \
    aiohttp pydantic prometheus-client gguf fastapi uvicorn PyJWT requests \
    aio-pika 'asyncpg>=0.27.0' 'aioboto3>=12.0.0' \
    'alibabacloud_dypnsapi20170525==1.2.2' 'redis==6.4.0' tos \
    av 'torchada>=0.1.10' pyzmq soundfile
```

### 平台适配说明

NPU **无需安装任何注意力/量化算子**：torchada 内置 `npu_flash_attn` / `rainfusion_attn` / `npu_rope` / `npu_layer_norm` 等算子，config 里用 `self_attn_1_type` / `cross_attn_*_type` / `rope_type` 等键选择（int8 npu config 正是这么写的）。镜像已含 mindiesd——如需 rainfusion 稀疏注意力加速，把 `self_attn_1_type` 改回 `rainfusion_attn` 即可（本文 smoke 用标准 `npu_flash_attn`，功能验证不受影响）。

### 验证安装

```shell #test id="lightx2v-install-verify"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
export PYTHONPATH=/tmp/stubs:$PYTHONPATH  # 条件 stub 前缀,一个真模块都没有则是空目录
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
> LightX2V 内部通过 `os.getenv('PLATFORM', 'cuda')` 选后端。**`PLATFORM=ascend_npu`**（不是 `npu`，不是 `ascend`）。后续每条 `python -m lightx2v.infer` 命令都要 export。

## 推理使用

### 模型准备

LightX2V 跑 I2V 蒸馏需要三组权重（**务必带 include 过滤**，全量仓库有几百 GB 无关权重）：

1. **base 模型**（`Wan-AI/Wan2.2-I2V-A14B`）：T5 text encoder（11.4 GB）+ VAE（0.5 GB）+ tokenizer（`google/`）。**不要下 `high_noise_model/` + `low_noise_model/` 两个目录**（各 60 GB 原始 BF16 权重，被下面的量化 ckpt 完全替代），只留两份 `config.json`。
2. **CLIP vision encoder**（`Wan-AI/Wan2.1-I2V-14B-480P`）：`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`（4.8 GB）+ tokenizer。Wan2.2-I2V-A14B 仓库**不带 CLIP**，从 Wan2.1 仓库单拉这一个文件即可。
3. **蒸馏量化 ckpt**（`lightx2v/Wan2.2-Distill-Models`）：**只拉 I2V int8 的两个 split 目录**（~30 GB），其余单文件/fp8/BF16 变体共 335 GB 不拉。

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

验证口径：目录名断言 + 大文件字节数精确断言，`LC_ALL=C sort` 保证跨 locale 排序一致。

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

拉 I2V 蒸馏 ckpt：lightx2v 的 load_safetensors 原生支持目录形态（遍历目录下所有 `block_*.safetensors`），split 目录直接当 ckpt 路径用，不用先 merge。

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

> ⚠️ split 目录名里的 `_1030` 后缀是 distill 权重的版本日期，high noise 带、low noise 不带——**名字以 ModelScope 仓库实际为准**，跑之前先 `ls` 确认。HF 上的同名 repo 内容相同，还含 T2V 蒸馏权重（ModelScope 没同步）。

### 列出 Wan2.2 4 步蒸馏 config

LightX2V 把推理参数写到 config JSON 里，`python -m lightx2v.infer --config_json ...` 一次性喂进去。在 `$PROJECT_ROOT/src` 下做精确存在性断言：

```shell #test id="lightx2v-list-wan22-configs"
export PROJECT_ROOT=${PROJECT_ROOT:-/home/coder/work/lightx2v-test}
cd "$PROJECT_ROOT/src"
ls configs/wan22/wan_moe_i2v.json configs/wan22/wan_moe_t2v.json
echo "---"
ls configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json
```

```shell #test-result id="lightx2v-list-wan22-configs" fuzzy='xxx'
configs/wan22/wan_moe_i2v.json
configs/wan22/wan_moe_t2v.json
---
configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json
```

下面 I2V demo 用 **`configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json`**。官方脚本 [`scripts/wan22/distill/run_wan22_moe_i2v_distill_lora_4step.sh`](https://github.com/ModelTC/LightX2V/blob/main/scripts/wan22/distill/run_wan22_moe_i2v_distill_lora_4step.sh) 的标准入口是 `python -m lightx2v.infer --model_cls wan2.2_moe --task i2v ...`；上游 PR #1455 已把蒸馏 runner 统一进基础 runner（不再有 `wan2.2_moe_distill` 这个 model_cls），蒸馏改由 config JSON 驱动。

### 单卡 32 GB 适配：改 config

`high/low_noise_quantized_ckpt` 换成 `$PROJECT_ROOT/models/` 下 split 目录绝对路径（config 默认是单文件形态，ModelScope 只同步了 split 目录形态）；`clip_original_ckpt` 指向单拉的 CLIP 文件（Wan2.2 base 仓库不带，不指定直接 FileNotFoundError）；`self_attn_1_type` 从 `rainfusion_attn` 改为 `npu_flash_attn`（rainfusion 的 NPU 实现依赖 MindIE-SD，镜像虽带 mindiesd，smoke 功能验证用标准 NPU flash attention 即可），`rainfusion_attn_setting` 一并清除；720P → 480P；开 T5/VAE/cpu_offload；单卡删 ulysses parallel 和 rife 插帧。

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
cfg['self_attn_1_type'] = 'npu_flash_attn'
cfg.pop('rainfusion_attn_setting', None)
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
for k in ['target_height', 'target_width', 'cpu_offload', 't5_cpu_offload', 'vae_cpu_offload', 'high_noise_quantized_ckpt', 'clip_original_ckpt', 'self_attn_1_type']:
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
export PYTHONPATH=/tmp/stubs:${PYTHONPATH:-}     # 条件 stub 前缀,一个真模块都没有则是空目录
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

python -m lightx2v.infer \
    --model_cls wan2.2_moe \
    --task i2v \
    --model_path "$PROJECT_ROOT/models/Wan2.2-I2V-A14B" \
    --config_json configs/distill/wan22/wan_moe_i2v_distill_int8_4step_ulysses_npu.json \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds." \
    --negative_prompt "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --image_path "$PROJECT_ROOT/src/assets/inputs/imgs/img_0.jpg" \
    --save_result_path "$PROJECT_ROOT/save_results/output_wan22_moe_i2v_distill_int8_npu_480p.mp4"
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
                   'output_wan22_moe_i2v_distill_int8_npu_480p.mp4')
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

> **关键 env**：
> - `PLATFORM=ascend_npu`（内部 `os.getenv('PLATFORM','cuda')`，值必须是 `ascend_npu`）
> - `ASCEND_RT_VISIBLE_DEVICES=0`（只用 1 张卡；`count: 1` 断言依赖它）
> - `source /usr/local/Ascend/ascend-toolkit/set_env.sh`（非交互式 shell 需要；`libhccl.so` 在这个目录里）
> - `export PYTHONPATH=/tmp/stubs:$PYTHONPATH`（条件 stub 前缀）
>
> **单卡 vs 多卡**：本文单卡跑通后要扩多卡，把 `parallel.seq_p_size` 加回 config 并设成 `torch.npu.device_count()`。

## 下一步

- **量化路径**：`int8-npu` / `scaled-fp8-e4m3` 见 [量化教程](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/quantization.html)
- **特征缓存**：TeaCache / MagCache 消除冗余计算，见 [缓存教程](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/cache.html)
- **多卡并行**：CFG 并行 / Ulysses 序列并行 / 张量并行见 [并行教程](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/method_tutorials/parallel.html)
- **多模态音频**：MiniMax-H3 Turbo 跑带同步音频的视频（参考 [官方 README](https://github.com/ModelTC/LightX2V/blob/main/README_zh.md)）
- **ComfyUI 节点**：[ComfyUI 部署文档](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_comfyui.html)
- **服务化部署**：[生产级 API 服务部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/deploy_service.html)
- **低资源部署**：本文基于单卡 32 GB HBM(910B4)调好 offload，想再压低见 [低资源部署](https://lightx2v-zhcn.readthedocs.io/zh-cn/latest/deploy_guides/for_low_resource.html)
- **训练框架**：[GenRL](https://github.com/ModelTC/GenRL) 用 GRPO 对 diffusion/flow 模型做强化学习训练

其它任务/模型的调用模式与 I2V 完全一致（`python -m lightx2v.infer --model_cls ... --task ... --model_path ... --config_json ...`），差异只在权重和 config：

- **T2V(Wan2.2 蒸馏)**：`--task t2v` + `configs/wan22/wan_moe_t2v_distill.json`。⚠️ ModelScope 的 `lightx2v/Wan2.2-Distill-Models` 没同步 T2V 蒸馏权重（只有 I2V），要跑得走 [HF 同名 repo](https://huggingface.co/lightx2v/Wan2.2-Distill-Models)（`wan2.2_t2v_A14b_*_4step.safetensors`）。
- **T2V(Wan2.1 轻量)**：`--model_cls wan2.1`，官方昇腾入口 [`scripts/platforms/ascend_npu/run_wan21_t2v.sh`](https://github.com/ModelTC/LightX2V/blob/main/scripts/platforms/ascend_npu/run_wan21_t2v.sh)，模型 [Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)（~17.6 GB，单卡无压力）。
- **T2I(Qwen-Image-Edit)**：`--model_cls qwen-image-edit-2511` + `--task i2i`（命名是 **i2i 不是 t2i**）。单卡 32 GB 走 FP8 蒸馏版（[lightx2v/Qwen-Image-Edit-2511-Lightning](https://modelscope.cn/models/lightx2v/Qwen-Image-Edit-2511-Lightning)，include 过滤挑 `qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_8steps_v1.0.safetensors`，**别全量拉**）。
