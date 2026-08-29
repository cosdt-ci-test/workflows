# 快速开始：在昇腾 NPU 上用 DiffSynth-Studio 生成一张图

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 是 ModelScope 的扩散模型框架。在昇腾上，它通过 `diffsynth.core.device` 把计算设备选成 `npu`。

上游[安装说明](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Pipeline_Usage/Setup.md#ascend-npu)里常见 [`pip install -e ".[npu_aarch64]"`](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Pipeline_Usage/Setup.md#ascend-npu) 这种写法。方括号里的 [`[npu_aarch64]`](https://github.com/modelscope/DiffSynth-Studio/blob/main/pyproject.toml#L64-L68) 是 pip 的**可选依赖组**（也叫 extra）：安装时会一并拉取一组为昇腾 aarch64 预选的包，并把 `torch` 固定为 `2.7.1`。本文使用的镜像是 CANN **9.1.0**（见下文[前置条件](#前置条件)），需要 [`torch==2.9.0`](https://gitcode.com/Ascend/pytorch) 与 [`torch_npu==2.9.0.post2`](https://gitcode.com/Ascend/pytorch)，版本对不上。**请不要用带 `[npu_aarch64]` 的安装方式**；按下面[第 3 节](#3-安装-pytorch-npu-栈)、[第 4 节](#4-安装-diffsynth-studio)，先单独安装 NPU 栈，再执行 [`pip install diffsynth`](https://pypi.org/project/diffsynth/)。

GitHub 上最新 Release 是 `v1.1.9`。那个 tag 里还没有 `diffsynth/core/device/`。本文从 PyPI 安装当前发布的 `diffsynth`（撰写时是 `2.1.3`），这份包已经带昇腾设备代码。

<!--
```shell #test-setup store="package_version"
printf '%s\n' "$UPSTREAM_REF"
```
-->

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


```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
```

`PYTHONNOUSERSITE=1` 让 Python 忽略用户目录里的包。若本机曾用 `pip install --user` 安装过 CANN 相关包，不设此变量时 Python 仍会把 `~/.local` 纳入 import 与 pip 的解析路径，可能选中与当前 CANN 栈不匹配的版本。

---

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

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

不要运行 `pip install -e ".[npu_aarch64]"`。那个 extra 会把刚装好的 `torch 2.9.0` 降回 `2.7.1`，和 CANN 9.1 对不上。先装好上一节的 NPU 栈，再装**不带 extra** 的 `diffsynth`。

```shell #test id="install-diffsynth"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
export PYTHONNOUSERSITE=1
python -m pip install diffsynth
python -c "from importlib.metadata import version; from diffsynth.core.device.npu_compatible_device import get_device_name, get_device_type; print('diffsynth', version('diffsynth')); print('device_type', get_device_type()); print('device_name', get_device_name())"
```

输出结果如下：

```shell #test-result id="install-diffsynth" load="package_version>>version"
...
diffsynth <version>
device_type npu
device_name npu:0
```

`device_name` 为 `npu:0` 只说明设备探测成功。

若这里打印 `device_type cpu` 或 `cuda`，先回到第 3 节确认 `torch_npu` 和可见设备。

---

## 5. 在 NPU 上生成一张图

`num_inference_steps=5` 是第一次跑通的步数。50 步出图更清楚，但第一次验证安装不必等那么久。正式出图时把步数改回 `50`。


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
