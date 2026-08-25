# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑通 Diffusers 文生图全链路：SD 1.5 生成 / 换调度器，Qwen-Image 20B 的 LoRA 风格加载，以及显存优化（cpu offload vs 全量加载实测对比）。
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
| transformers | `>=5.0,<6.0`（diffusers 0.40+ 依赖 huggingface-hub>=1.23，而 transformers 4.x 要求 hub<1.0，二者冲突；transformers 5.x 起适配 hub 1.x） |
| accelerate | `>=1.0,<2.0` |
| peft | `>=0.6` |
| modelscope | 1.37.0 |
| diffusers | 最新 release 的源码/二进制 |
| 模型 | [AI-ModelScope/stable-diffusion-v1-5](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5)；显存优化小节用 [Qwen/Qwen-Image](https://www.modelscope.cn/models/Qwen/Qwen-Image) + LoRA [flymy-ai/qwen-image-realism-lora](https://www.modelscope.cn/models/flymy-ai/qwen-image-realism-lora) |

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

```shell #test-result id="check-torch" fuzzy='xxx'
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: xxx
```
- xxx 表示容器内可见的 NPU 数量（本文档所有示例只用 1 张卡，即 device `npu:0`）

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

安装 `transformers` / `accelerate` / `peft` / `modelscope`：

```shell #test-setup
pip install 'transformers>=5.0,<6.0' 'accelerate>=1.0,<2.0' 'peft>=0.6' 'modelscope==1.37.0'
```

打印安装版本：
```shell #test id="install-deps"
python -c "import transformers, accelerate, peft, modelscope; print('transformers', transformers.__version__); print('accelerate', accelerate.__version__); print('peft', peft.__version__); print('modelscope', modelscope.__version__)"
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
transformers xxx
accelerate xxx
peft xxx
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

~5 分钟在单卡昇腾 NPU 上跑通 SD 1.5「文生图 → 更换调度器」整条链路（不含模型下载时间；想调生成效果可修改 `prompt` / `num_inference_steps` / `guidance_scale`，参数含义与官方 Quicktour 完全一致）。Qwen-Image 显存优化小节见下文。

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
mkdir -p output
python << 'PY' 2>&1
import torch
import torch_npu
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "<model_path>", dtype=torch.bfloat16, safety_checker=None,
).to("npu:0")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=30).images[0]
image.save("output/astronaut_rides_horse.png")
print("generated:", image.size)
PY
ls -l output/astronaut_rides_horse.png | awk '{print $5, $9}'
```

输出结果如下（`...` 吞掉加载进度等前导日志，`xxx` 为实际文件字节数）：

```shell #test-result id="text-to-image" fuzzy='xxx' fuzzy='...'
...
generated: (512, 512)
xxx output/astronaut_rides_horse.png
```

### 更换调度器（UniPCMultistepScheduler）

调度器决定逐步去噪的算法，直接影响生成速度与质量。用 `UniPCMultistepScheduler.from_config(pipe.scheduler.config)` 复用原调度器配置，即可把 pipeline 热替换为更少步数出图的调度器：

```shell #test id="swap-scheduler" load="model_path>>model_path"
mkdir -p output
python << 'PY' 2>&1
import torch
import torch_npu
from diffusers import DiffusionPipeline, UniPCMultistepScheduler

pipe = DiffusionPipeline.from_pretrained(
    "<model_path>", dtype=torch.bfloat16, safety_checker=None,
).to("npu:0")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=20).images[0]
image.save("output/astronaut_unipc.png")
print("generated:", image.size, "via", type(pipe.scheduler).__name__)
PY
ls -l output/astronaut_unipc.png | awk '{print $5, $9}'
```

输出结果如下（`...` 吞掉前导日志，`xxx` 为实际文件字节数）：

```shell #test-result id="swap-scheduler" fuzzy='xxx' fuzzy='...'
...
generated: (512, 512) via UniPCMultistepScheduler
xxx output/astronaut_unipc.png
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

### 加载 LoRA（Quicktour LoRA 一节）

LoRA 适配器只往基础模型插入少量可训练参数（本例 ~94 MB 对 20B 的 DiT），推理时把 LoRA 权重加载/合并进 transformer 即可切换生成风格。对应 Quicktour 的 LoRA 一节：使用 `load_lora_weights` 加载适配器，prompt 里带上触发词 `Realism` 激活风格。

先下载 LoRA 权重（同样走 ModelScope，落入持久缓存）：

```shell #test-setup store="lora_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('flymy-ai/qwen-image-realism-lora'))" | tail -n 1
```

输出类似：

```
/root/.cache/modelscope/hub/models/flymy-ai/qwen-image-realism-lora
```

在 Qwen-Image pipeline 上加载 LoRA 并用触发词生成（`load_lora_weights` 走 PEFT 的权重合并路径，与设备无关）：

```shell #test id="qwen-image-lora" load="qwen_image_path>>qwen_path" load="lora_path>>lora_path"
mkdir -p output
python << 'PY' 2>&1
import torch
import torch_npu
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "<qwen_path>", dtype=torch.bfloat16,
).to("npu:0")
pipe.load_lora_weights("<lora_path>", weight_name="flymy_realism.safetensors")

image = pipe(
    "Super Realism close-up photo of a black falcon twist in the air",
    num_inference_steps=8,
).images[0]
image.save("output/qwen_image_lora.png")
pipe.unload_lora_weights()
print("generated:", image.size, "lora loaded & unloaded")
PY
ls -l output/qwen_image_lora.png | awk '{print $5, $9}'
```

输出结果类似（`...` 吞掉前导日志，`xxx` 为实际文件字节数）：

```shell #test-result id="qwen-image-lora" fuzzy='xxx' fuzzy='...'
...
generated: (1328, 1328) lora loaded & unloaded
xxx output/qwen_image_lora.png
```

### cpu offload 模式生成

`enable_model_cpu_offload()` 之后不要手动 `.to("npu:0")`——offload hook 自己管理组件的进出，手动搬运会和 hook 冲突（对应 Quicktour「Memory usage」一节去掉 `device_map` 的写法）：

```shell #test id="qwen-image-offload" load="qwen_image_path>>qwen_path"
mkdir -p output
python << 'PY' 2>&1
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
print("generated:", image.size)
print(f"offload peak NPU memory: {peak:.2f} GB")
PY
ls -l output/qwen_image_offload.png | awk '{print $5, $9}'
```

输出结果类似（offload 峰值应明显低于全量权重 57 GB，接近最大的单个组件 transformer ~41 GB；`...` 吞掉前导日志，`xxx` 为实际文件字节数 / 显存峰值）：

```shell #test-result id="qwen-image-offload" fuzzy='xxx' fuzzy='...'
...
generated: (1328, 1328)
offload peak NPU memory: xxx GB
xxx output/qwen_image_offload.png
```

### 全量加载对比

同一模型不带 offload 直接 `.to("npu:0")` 全量驻留显存（64 GB HBM 恰好装得下 57 GB 权重，但已接近上限）：

```shell #test id="qwen-image-full" load="qwen_image_path>>qwen_path"
mkdir -p output
python << 'PY' 2>&1
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
print("generated:", image.size)
print(f"full-load peak NPU memory: {peak:.2f} GB")
PY
ls -l output/qwen_image_full.png | awk '{print $5, $9}'
```

输出结果类似（全量峰值 ≈ 全部权重 57 GB 驻留 + 激活；`...` 吞掉前导日志，`xxx` 为实际文件字节数 / 显存峰值）：

```shell #test-result id="qwen-image-full" fuzzy='xxx' fuzzy='...'
...
generated: (1328, 1328)
full-load peak NPU memory: xxx GB
xxx output/qwen_image_full.png
```
