# 快速开始：在昇腾 NPU 上用 DiffSynth-Studio 生成一张图

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**：装上与 CANN 匹配的 PyTorch NPU 栈和 DiffSynth-Studio，在单卡 NPU 上用 Stable Diffusion 1.5 生成一张图。

[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 是 ModelScope 的扩散模型框架。昇腾路径通过 `diffsynth.core.device` 把设备选成 `npu`。上游安装文档里的 `[npu_aarch64]` extra 会把 `torch` 钉在 `2.7.1`。本文用的镜像是 CANN 9.1.0，对应的是 `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，所以**不要**装那个 extra，先装 NPU 栈，再装不带 extra 的 `diffsynth`。

GitHub 上最新 Release 是 `v1.1.9`。那个 tag 里还没有 `diffsynth/core/device/`。本文从 PyPI 安装当前发布的 `diffsynth`（撰写时是 `2.1.3`），这份包已经带昇腾设备代码。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| Python | 3.12 |
| PyTorch | `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，见下文安装 |
| DiffSynth-Studio | 从 PyPI 安装 `diffsynth`，见下文 |
| 模型 | [AI-ModelScope/stable-diffusion-v1-5](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5)，首次运行会从 ModelScope 下载 |

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4）。**配套镜像**：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

---

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。常见容器里 `npu-smi` 在 `/usr/local/sbin`，需要把该目录加入 `PATH`。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
```

`PYTHONNOUSERSITE=1` 让 Python 忽略用户目录里的包。本机如果曾经 `pip install --user` 过 CANN 相关包，不设这个变量时，pip 解析器可能被带偏。

---

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

**预期**：命令退出码为 0，并打印设备列表。表格中的功耗、HBM 占用每次不同，**不必**与任何样例逐字一致。

若 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载（如 `/dev/davinci0`）。

### 2.2 确认工具可用

```shell #test-setup
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
```

**预期**：`ASCEND_HOME_PATH` 非空；`npu-smi` 能找到。

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy="xxx"
Python 3.12.xxx
```

---

## 3. 安装 PyTorch NPU 栈

昇腾上的 `torch_npu` 要从华为 PyPI 额外索引安装，并钉死与 CANN 匹配的版本。`numpy` 和 `pyyaml` 也要一起装。`torch_npu` 的 wheel **没有声明**这两项依赖，但 `import torch` 会自动加载 `torch_npu`，缺了会在你显式 `import torch_npu` 之前就失败。`torchvision` 是 DiffSynth 的依赖，和 torch 一起装，避免 pip 稍后拉到不匹配的 CUDA 轮子。

```shell #test id="install-torch"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
  torch==2.9.0 torch_npu==2.9.0.post2 torchvision numpy pyyaml
python -c "import numpy, yaml, torch, torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_available', torch.npu.is_available())"
```

输出结果如下：

```shell #test-result id="install-torch"
...
torch 2.9.0...
torch_npu 2.9.0.post2
npu_available True
```

`npu_available` 必须是 `True`。`False` 时不要继续，先查 CANN、驱动和可见设备。

---

## 4. 安装 DiffSynth-Studio

不要写 `pip install -e ".[npu_aarch64]"`。那个 extra 会把刚装好的 `torch 2.9.0` 降回 `2.7.1`，和 CANN 9.1 对不上。先装好上一节的 NPU 栈，再装**不带 extra** 的 `diffsynth`。

```shell #test id="install-diffsynth"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
python -m pip install diffsynth
python -c "from diffsynth.core.device.npu_compatible_device import get_device_name, get_device_type; print('device_type', get_device_type()); print('device_name', get_device_name())"
```

输出结果如下：

```shell #test-result id="install-diffsynth"
...
device_type npu
device_name npu:0
```

`device_name` 为 `npu:0` 只说明设备探测成功。真正的工作负载在下一节。

若这里打印 `device_type cpu` 或 `cuda`，先回到第 3 节确认 `torch_npu` 和可见设备，不要继续生成。

---

## 5. 在 NPU 上生成一张图

下面这段改写自上游 `examples/stable_diffusion/model_inference/stable-diffusion-v1-5.py`。CUDA 示例把 `rand_device` 写成 `"cuda"`。昇腾上要改成 `"npu"`，否则噪声张量会建在一张不存在的 CUDA 设备上。`from_pretrained` 也要显式传 `device="npu"`。上游默认会调用 `get_device_type()`，在没挂 CUDA 的机器上通常已经是 `npu`，写出来是为了你复制时不会漏。

`num_inference_steps=5` 是第一次跑通的步数。50 步出图更清楚，但第一次验证安装不必等那么久。正式出图时把步数改回 `50`。

权重从 ModelScope 拉 [AI-ModelScope/stable-diffusion-v1-5](https://www.modelscope.cn/models/AI-ModelScope/stable-diffusion-v1-5)。第一次会下载约 4 GB，之后走本地缓存。

**怎样算成功**

1. 进程退出码为 0；
2. 写出 `image.jpg`，尺寸是 `(512, 512)`；
3. 日志里出现 `unet.device npu:0`。这是权重已经在 NPU 上的证据。只看到上一节的 `device_name npu:0`、进程却把模型放在 CPU 上，仍算失败。

```shell #test id="generate"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
python - <<'PY'
import torch
from diffsynth.core import ModelConfig
from diffsynth.core.device.npu_compatible_device import get_device_name
from diffsynth.pipelines.stable_diffusion import StableDiffusionPipeline

print("device_name", get_device_name())
pipe = StableDiffusionPipeline.from_pretrained(
    torch_dtype=torch.float32,
    device="npu",
    model_configs=[
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="unet/diffusion_pytorch_model.safetensors"),
        ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="AI-ModelScope/stable-diffusion-v1-5", origin_file_pattern="tokenizer/"),
)
print("pipe.device", pipe.device)
print("unet.device", next(pipe.unet.parameters()).device)
image = pipe(
    prompt="a photo of an astronaut riding a horse on mars, high quality, detailed",
    negative_prompt="blurry, low quality, deformed",
    cfg_scale=7.5,
    height=512,
    width=512,
    seed=42,
    rand_device="npu",
    num_inference_steps=5,
)
image.save("image.jpg")
print("image_size", image.size)
PY
```

输出结果如下：

```shell #test-result id="generate"
device_name npu:0
...
pipe.device npu
unet.device npu:0
image_size (512, 512)
```

`pipe.device` 打印的是你传入的字符串 `npu`。`unet.device` 才是参数真正所在的设备。

---

## 6. 本文没有覆盖的能力

这些路径不在第一次跑通范围内，正文里也没有对应的可复制命令块：

- 视频生成（Wan、LTX 等）和音频模型
- 训练、USP 序列并行、`[quant]` 量化 extra
- Z-Image / Flux 等更大的图像模型
- 上游文档里的 `pip install -e ".[npu_aarch64]"`（会降级 torch）
- GitHub Release `v1.1.9` 的源码树（没有昇腾设备模块）

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `import torch` 报缺 `numpy` 或 `yaml` | `torch_npu` 未声明这两项依赖 | 与 torch 栈一起安装 `numpy` `pyyaml` |
| `torch.npu.is_available()` 为 `False` | 未 `source set_env.sh`，或设备未挂进容器 | 重做第 1–2 节 |
| `device_type` 为 `cpu` | `torch_npu` 没装上，或 `torch.npu.is_available()` 为假 | 重做第 3 节 |
| `device_type` 为 `cuda` | 当前进程里 `torch.cuda.is_available()` 为真，框架会优先选 CUDA | 卸掉 CUDA 版 torch，或显式传 `device="npu"` |
| pip 把 `torch` 降回 `2.7.1` | 安装了 `[npu_aarch64]` extra | 卸掉 extra，按第 3–4 节重装 |
| `unet.device` 为 `cpu` 但退出码 0 | 模型没放到 NPU | 检查 `device="npu"` 和可见设备 |
| `rand_device="npu"` 报 Generator 不支持 | 当前 `torch_npu` 不能在 NPU 上建 RNG | 改回 `rand_device="cpu"`。噪声仍会 `.to(npu)`，但锚点看 `unet.device` |
| ModelScope 下载超时或落到 HTML | 出口到 modelscope.cn 失败 | 检查网络后重跑；不要改成 Hugging Face 直连 |
