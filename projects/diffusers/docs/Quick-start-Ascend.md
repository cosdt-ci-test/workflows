# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑通 Diffusers 文生图全链路：SD 1.5 生成 / 换调度器 / 组件拆解，以及 Qwen-Image 20B 的显存优化（cpu offload vs 全量加载实测对比）。
使用 `DiffusionPipeline.from_pretrained` 加载文生图管线并生成图像，用 `UniPCMultistepScheduler` 等调度器控制生成速度与质量。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1）
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
| transformers | `<5.0` |
| accelerate | `>=1.0,<2.0` |
| modelscope | 1.37.0 |
| diffusers | 最新 release 的源码/二进制 |
| 模型 | [AI-ModelScope/stable-diffusion-v1-5](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5)（对应 HuggingFace 的 `stable-diffusion-v1-5/stable-diffusion-v1-5`）；显存优化小节用 [Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image)（20B DiT + 7B text encoder，~57 GB，bf16 全量峰值贴近 64 GB HBM 上限） |

### 前置安装
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

```shell #test-result id="check-torch"
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

安装 `transformers` / `accelerate` / `modelscope`：

```shell #test-setup
pip install 'transformers<5.0' 'accelerate>=1.0,<2.0' 'modelscope==1.37.0'
```

打印安装版本：
```shell #test id="install-deps"
python -c "import transformers, accelerate, modelscope; print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('modelscope', modelscope.__version__)"
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
transformers xxx
accelerate xxx
modelscope 1.37.0
```

## 安装 Diffusers

### 使用 uv 进行安装

```shell #test id="diffusers-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple diffusers
python -c "import diffusers; print('diffusers', diffusers.__version__)"
```

输出结果类似如下：

```shell #test-result id="diffusers-install-binary" fuzzy='xxx'
diffusers xxx
```
- xxx 表示最新的版本号
<!--
```shell #test-setup
uv pip uninstall diffusers -y
```
-->

### 从源码安装
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证

```shell #test id="diffusers-install-source" load="upstream_ref>>ref"
git clone https://github.com/huggingface/diffusers.git
cd diffusers && git checkout <ref>
uv pip install -e .
python -c "import diffusers; print('diffusers', diffusers.__version__)"
```
\<ref> 为安装的最新的 release tag

输出结果类似如下：

```shell #test-result id="diffusers-install-source" fuzzy='xxx'
diffusers xxx
```
- xxx 表示最新的版本号

## 使用样例

~5 分钟在单卡昇腾 NPU 上跑通 SD 1.5「文生图 → 更换调度器 → 组件拆解」整条链路（不含模型下载时间；想调生成效果可修改 `prompt` / `num_inference_steps` / `guidance_scale`，参数含义与官方 Quicktour 完全一致）。Qwen-Image 显存优化小节见下文。

### 下载基础模型

默认使用 **ModelScope** 进行模型下载。

```shell #test-setup store="model_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('AI-ModelScope/stable-diffusion-v1-5'))" | tail -n 1
```

输出类似：

```
/root/.cache/modelscope/hub/models/AI-ModelScope/stable-diffusion-v1-5
```

### 文生图（DiffusionPipeline）

`DiffusionPipeline.from_pretrained` 按模型仓库的 `model_index.json` 自动组装 text_encoder / unet / vae / tokenizer / scheduler，加载后传入 prompt 即可出图：

```shell #test id="text-to-image" load="model_path>>model_path"
python << 'PY' 2>&1 | tail -1
import os

import torch
import torch_npu
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "<model_path>", dtype=torch.bfloat16, safety_checker=None,
).to("npu:0")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=30).images[0]

os.makedirs("output", exist_ok=True)
image.save("output/astronaut_rides_horse.png")
print("saved:", "output/astronaut_rides_horse.png", image.size)
PY
```

输出结果如下：

```shell #test-result id="text-to-image"
saved: output/astronaut_rides_horse.png (512, 512)
```

### 更换调度器（UniPCMultistepScheduler）

调度器决定逐步去噪的算法，直接影响生成速度与质量。用 `UniPCMultistepScheduler.from_config(pipe.scheduler.config)` 复用原调度器配置，即可把 pipeline 热替换为更少步数出图的调度器：

```shell #test id="swap-scheduler" load="model_path>>model_path"
python << 'PY' 2>&1 | tail -1
import os

import torch
import torch_npu
from diffusers import DiffusionPipeline, UniPCMultistepScheduler

pipe = DiffusionPipeline.from_pretrained(
    "<model_path>", dtype=torch.bfloat16, safety_checker=None,
).to("npu:0")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=20).images[0]

os.makedirs("output", exist_ok=True)
image.save("output/astronaut_unipc.png")
print("saved:", "output/astronaut_unipc.png", image.size, "via", type(pipe.scheduler).__name__)
PY
```

输出结果如下：

```shell #test-result id="swap-scheduler"
saved: output/astronaut_unipc.png (512, 512) via UniPCMultistepScheduler
```

### 拆解 pipeline 组件

`DiffusionPipeline` 只是组件的打包器，每个组件都能从各自子目录单独加载（对应 Quicktour 的 Models / Schedulers 一节）：

```shell #test id="pipeline-components" load="model_path>>model_path"
python << 'PY' 2>&1 | tail -3
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler

unet = UNet2DConditionModel.from_pretrained("<model_path>", subfolder="unet")
vae = AutoencoderKL.from_pretrained("<model_path>", subfolder="vae")
scheduler = DDPMScheduler.from_pretrained("<model_path>", subfolder="scheduler")
print("unet sample_size:", unet.config.sample_size)
print("vae latent_channels:", vae.config.latent_channels)
print("scheduler:", type(scheduler).__name__, scheduler.config.num_train_timesteps)
PY
```

输出结果如下：

```shell #test-result id="pipeline-components"
unet sample_size: 64
vae latent_channels: 4
scheduler: DDPMScheduler 1000
```

## 大模型显存优化（Qwen-Image 20B）

上面 SD1.5 的所有组件加起来 ~2 GB，单卡 64 GB 显存放得下，不需要任何优化。但对应 Quicktour 的 Optimizations 一节：现代扩散模型（如 Qwen-Image 的 20B DiT + 7B text encoder，全量 bf16 权重 ~57 GB）在单卡上放不下时，`enable_model_cpu_offload()` 让组件**逐个上下场**——text encoder 编码完搬回 CPU 内存，再请 DiT 上 NPU 去噪，最后 VAE 解码——显存峰值从「所有组件之和」降到「最大单个组件」。

本节用 Qwen-Image 实测两种加载方式的 NPU 显存峰值对比（`torch.npu.max_memory_allocated`，对应 Quicktour 量化小节打印 `torch.cuda.max_memory_allocated` 的做法）：

### 下载 Qwen-Image

Qwen-Image 约 58 GB，首次下载耗时长；CI 使用宿主机持久缓存（容器挂载 `/root/.cache/modelscope`），后续运行直接命中本地文件。持久缓存中可能残留之前中断下载产生的残缺权重文件，测试框架会在下载前做 safetensors 完整性校验，损坏的模型目录会被整体清除并重新下载。

```shell #test-setup store="qwen_image_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen-Image'))" | tail -n 1
```

输出类似：

```
/root/.cache/modelscope/hub/models/Qwen/Qwen-Image
```

### cpu offload 模式生成

`enable_model_cpu_offload()` 之后不要手动 `.to("npu:0")`——offload hook 自己管理组件的进出，手动搬运会和 hook 冲突（对应 Quicktour「Memory usage」一节去掉 `device_map` 的写法）：

```shell #test id="qwen-image-offload" load="qwen_image_path>>qwen_path"
python << 'PY' 2>&1 | tail -2
import torch
import torch_npu
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "<qwen_path>", dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

image = pipe(
    "a black falcon twist in the air, close-up photo",
    num_inference_steps=8,
).images[0]
image.save("output/qwen_image_offload.png")
torch.npu.synchronize()
peak = torch.npu.max_memory_allocated() / 1024**3
print("saved: output/qwen_image_offload.png", image.size)
print(f"offload peak NPU memory: {peak:.2f} GB")
PY
```

输出结果类似（offload 峰值应明显低于全量权重 57 GB，接近最大的单个组件 transformer ~41 GB）：

```shell #test-result id="qwen-image-offload" fuzzy='xxx'
saved: output/qwen_image_offload.png (1328, 1328)
offload peak NPU memory: xxx GB
```

### 全量加载对比

同一模型不带 offload 直接 `.to("npu:0")` 全量驻留显存（64 GB HBM 恰好装得下 57 GB 权重，但已接近上限）：

```shell #test id="qwen-image-full" load="qwen_image_path>>qwen_path"
python << 'PY' 2>&1 | tail -2
import torch
import torch_npu
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "<qwen_path>", dtype=torch.bfloat16,
).to("npu:0")

image = pipe(
    "a black falcon twist in the air, close-up photo",
    num_inference_steps=8,
).images[0]
image.save("output/qwen_image_full.png")
torch.npu.synchronize()
peak = torch.npu.max_memory_allocated() / 1024**3
print("saved: output/qwen_image_full.png", image.size)
print(f"full-load peak NPU memory: {peak:.2f} GB")
PY
```

输出结果类似（全量峰值 ≈ 全部权重 57 GB 驻留 + 激活）：

```shell #test-result id="qwen-image-full" fuzzy='xxx'
saved: output/qwen_image_full.png (1328, 1328)
full-load peak NPU memory: xxx GB
```

小贴士：

- `safety_checker=None` 跳过 NSFW 安全检查器（可少加载一个 CLIP vision 模型）；需要安全过滤时去掉该参数即可。
- 昇腾上建议 `dtype=torch.bfloat16`：bf16 数值范围与 fp32 一致，可避开 Stable Diffusion VAE 在 fp16 下可能的 NaN（黑图）问题。
- 调度器可随时热替换，完整列表见 [Diffusers Schedulers API](https://huggingface.co/docs/diffusers/api/schedulers/overview)；`UniPCMultistepScheduler` 20 步即可接近默认 `PNDMScheduler` 50 步的质量。
- Quicktour 的 LoRA（`pipe.load_lora_weights(...)`）与量化（bitsandbytes / torchao）小节依赖 CUDA 后端，昇腾上暂不适用；LoRA 权重加载本身与设备无关，可自行尝试。
- Qwen-Image 的推理步数是 FlowMatch 调度器语义（默认 50 步）；本文档压到 8 步只为控制 CI 时长，追求质量请用默认值。text encoder（Qwen2.5-VL）会截断 prompt 到 `max_sequence_length`（默认 512）。
- 显存不够全量加载时（比如 32 GB 卡跑 Qwen-Image），把上文全量块换成 offload 块即可；offload 用 CPU 内存换显存，代价是组件来回搬运的额外耗时。
- 想看 pipeline 内部的逐步去噪循环（Quicktour 的 DiffusionPipeline explained 一节），把 `pipe` 拆成 `text_encoder` / `unet` / `vae` / `scheduler` 手动调用即可，本文档的组件拆解就是它的第一步。
