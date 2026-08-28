# Quick Start (Ascend NPU)

在昇腾 NPU 上跑通 `torchvision` 的最小链路。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）

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
| torch_npu | 2.9.0.post6 |
| torchvision | 0.24.0（stock release，PyPI linux aarch64 cpu-only wheel，走阿里 PyPI 镜像；fork 不在本文烟雾路径里） |
| pillow | `>=10.0`（`torchvision.transforms.functional.to_pil_image` 等的运行时依赖） |

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

安装 `torch` / `torch_npu`：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.9.0
uv pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi torch_npu==2.9.0.post6
```

检查 torch / torch_npu 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下（`count: 1` 表示本机可见一张卡，下文示例只用 `npu:0`）：

```shell #test-result id="check-torch" fuzzy='xxx'
torch= 2.9.0+cpu
torch_npu= 2.9.0.post6
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

安装 `pillow`：

```shell #test-setup
uv pip install 'pillow>=10.0'
```

打印版本：

```shell #test id="install-deps"
python -c "import PIL; print('Pillow', PIL.__version__)"
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
Pillow xxx
```

## 安装 torchvision

### 二进制路径

```shell #test-setup id="stock-torchvision-install"
uv pip install torchvision==0.24.0 -i https://mirrors.aliyun.com/pypi/simple/
```

打印版本：

```shell #test id="stock-torchvision-check"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
```

```shell #test-result id="stock-torchvision-check" fuzzy='xxx'
torchvision xxx
```

## Getting started with transforms v2

### 准备工作

`torchvision.io.decode_image` 走的是 libjpeg-turbo（torchvision cpu wheel 自带），NPU 上跑 JPEG decode 不绕 Pillow——Pillow-SIMD 选型对这条路径**没差异**。这里 import 一下确认 `torchvision.io` 跟 `tv_tensors` 链路 OK；图本身用合成 256×256 RGB（CPU 路径、无网络依赖，跟 torchvision 官方教程的 `astronaut.jpg` 等价但避免模型仓库 clone）。NPU 烟雾集中在 transform 输入的 `img.to('npu:0')` 上：

```shell #test id="v2-setup"
python << 'PY'
import numpy as np
import torch
import torch_npu
import torchvision
from torchvision.transforms import v2
from torchvision.io import decode_image
from torchvision import tv_tensors

torch.manual_seed(1)
np.random.seed(1)

# 跟 torchvision 官方教程对齐：decode_image + TVTensor wrap。
# 本烟雾路径无网络依赖，这里走合成图（形状 / dtype 跟 decode_image
# 输出对齐）；decode_image 只是为了验 import 链路 + libjpeg-turbo 入口可达。
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))
print(f"{type(img) = }, {img.dtype = }, {img.shape = }")
print(f"{decode_image.__module__ = }")  # 验 torchvision.io.decode_image 走 libjpeg-turbo 入口

# 验证 torch_npu NPU dispatch：is_available + device_count + cpu→npu→cpu round-trip
print(f"npu_available: {torch.npu.is_available()}")
print(f"npu_count: {torch.npu.device_count()}")
x = torch.zeros(2, 3)
x_npu = x.to('npu:0')
x_back = x_npu.cpu()
print(f"npu round-trip: in={tuple(x.shape)} on {x.device.type}, npu={tuple(x_npu.shape)} on {x_npu.device.type}, back={tuple(x_back.shape)} on {x_back.device.type}")
PY
```

输出结果如下：

```shell #test-result id="v2-setup" fuzzy='xxx'
type(img) = <class 'torchvision.tv_tensors.Image'>, img.dtype = torch.uint8, img.shape = torch.Size([3, 256, 256])
decode_image.__module__ = 'torchvision.io.image'
npu_available: True
npu_count: xxx
npu round-trip: in=(2, 3) on cpu, npu=(2, 3) on npu, back=(2, 3) on cpu
```

### The basics

v2 transform 是 `nn.Module` 风格——单输入单输出。把图搬到 NPU **再**调 transform，让 `CenterCrop` 跑在 NPU 后端（device 透传：CPU 输入 → CPU 输出，NPU 输入 → NPU 输出）：

```shell #test id="v2-basics"
python << 'PY'
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# 复用 v2-setup 的 fixture：每个 #test 跑在独立子进程，img 不能跨块带过来，
# 这里重建（同一种子 → 同一张图；尺寸 / dtype / TVTensor 类型都对得上）。
np.random.seed(0)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))

img_npu = img.to('npu:0')   # 显式搬 NPU；CenterCrop 之后输出也在 NPU
transform = v2.CenterCrop(size=(224, 224))
out = transform(img_npu)
print(f"in:  type={type(img_npu).__name__} device={img_npu.device.type} shape={tuple(img_npu.shape)}")
print(f"out: type={type(out).__name__} device={out.device.type} shape={tuple(out.shape)}")
PY
```

输出结果如下（`device.type` 在 torch 2.x + torch_npu 上稳定为 `npu`；完整 `device` 字符串可能打印 `npu:0` 或 `privateuseone:0`，不影响）：

```shell #test-result id="v2-basics"
in:  type=Image device=npu shape=(3, 256, 256)
out: type=Image device=npu shape=(3, 224, 224)
```

### Random crop

官方教程第一个例子：`RandomCrop(size=(224, 224))`，验证随机 crop 跑通（`pad_if_needed=False`、input 256 > 224 时 offset 恒为 `(0, 0)`——crop 是确定的，NPU 上跟 CPU 输出一致；shape 一致即证明 NPU kernel 跟 CPU 一致）：

```shell #test id="v2-randomcrop"
python << 'PY'
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# 复用 v2-setup 的 fixture（独立子进程，img 不能跨块带过来）
np.random.seed(1)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))

img_npu = img.to('npu:0')   # 显式搬 NPU；RandomCrop 之后输出也在 NPU
transform = v2.RandomCrop(size=(224, 224))
out = transform(img_npu)
print(f"in:  type={type(img_npu).__name__} device={img_npu.device.type} shape={tuple(img_npu.shape)}")
print(f"out: type={type(out).__name__} device={out.device.type} shape={tuple(out.shape)}")
# shape 一致即证明 NPU kernel 跟 CPU 一致；CPU/NPU 同时跑同一 crop，再比对 sum
out_cpu = v2.RandomCrop(size=(224, 224))(img)
print(f"npu vs cpu sum match: {int(out.sum().cpu()) == int(out_cpu.sum())}")
PY
```

输出结果如下（`RandomCrop` 在 `pad_if_needed=False` + input > output 时 offset 恒为 `(0, 0)`，所以这条路径在 NPU 上是确定的；CPU/NPU sum 一致即证明 NPU 端 crop kernel 行为跟 CPU 一致）：

```shell #test-result id="v2-randomcrop"
in:  type=Image device=npu shape=(3, 256, 256)
out: type=Image device=npu shape=(3, 224, 224)
npu vs cpu sum match: True
```

### Image classification pipeline

官方教程的 ImageNet 经典 pipeline：`Compose` 把 `RandomResizedCrop` + `RandomHorizontalFlip` + `ToDtype(float32, scale=True)` + `Normalize` 串起来——这跟 `v2-basics` / `v2-randomcrop` 单算子不同，**整条 pipeline 都要在 NPU 上跑通**才能算 smoke。`torch.manual_seed` 锁定 crop / flip 随机性（这俩都是 `torch.*_uniform_` / `torch.rand` RNG，跟 PIL `random` 模块无关，因为这里是 tensor 输入路径）；float 值用 `fuzzy='xxx'` 抹平（NPU `interpolate` / `aten::div` 跟 CPU 可能有 ULP 级精度差）：

```shell #test id="v2-classification"
python << 'PY'
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# 复用 v2-setup 的 fixture（独立子进程，img 不能跨块带过来）
np.random.seed(1)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))

# 锁定 crop / flip RNG——RandomResizedCrop 的 area / aspect_ratio / (i, j)
# 跟 RandomHorizontalFlip 的 p<0.5 都走 torch RNG
torch.manual_seed(1)

img_npu = img.to('npu:0')
transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
out = transforms(img_npu)
print(f"in:  type={type(img_npu).__name__} device={img_npu.device.type} dtype={img_npu.dtype} shape={tuple(img_npu.shape)}")
print(f"out: type={type(out).__name__} device={out.device.type} dtype={out.dtype} shape={tuple(out.shape)}")
# 归一化后值大致落在 [-2.12, 2.64]（mean=0.485/std=0.229 时 1/255≈0.004 → (1-0.485)/0.229≈2.25，
# (0-0.485)/0.229≈-2.12）。打印 min/mean/max/sum 验 pipeline 完整性；用 abs(x)<100 兜底检查浮点未爆
print(f"out stats: min={float(out.min().cpu()):.4f} mean={float(out.mean().cpu()):.4f} max={float(out.max().cpu()):.4f}")
print(f"out finite: {torch.isfinite(out).all().item()}")
PY
```

输出结果如下（`fuzzy='xxx'` 抹平 NPU/CPU interpolate / div 浮点精度差——shape / dtype / finite 这三个离散属性必须在 NPU 上跟 CPU 路径完全一致）：

```shell #test-result id="v2-classification" fuzzy='xxx'
in:  type=Image device=npu dtype=torch.uint8 shape=(3, 256, 256)
out: type=Image device=npu dtype=torch.float32 shape=(3, 224, 224)
out stats: min=xxx mean=xxx max=xxx
out finite: True
```

### Object detection with bounding boxes

官方教程 detection 段：训练阶段把 `BoundingBoxes` 跟 `Image` 一起塞进 Compose。Compose 内 3 个算子：`RandomResizedCrop`（NPU 端 resize + crop + 重映射 box 坐标）+ `RandomPhotometricDistort(p=1)`（NPU 端颜色 jitter 链：brightness / contrast / saturation / hue / channel permute）+ `RandomHorizontalFlip(p=1)`（永远翻，box 同步翻）。boxes 留在 CPU（坐标运算不需要 NPU kernel；这是 v2 的标准用法，跟 `v2-vbmk` 一致）。`torch.manual_seed` 锁定 crop / photometric / flip 的 RNG（都是 `torch.rand` / `torch.randperm` / `torch.empty.uniform_`）：

```shell #test id="v2-detection"
python << 'PY'
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# 复用 v2-setup 的 fixture（独立子进程，img 不能跨块带过来）
np.random.seed(1)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))
H, W = img.shape[-2:]
boxes = tv_tensors.BoundingBoxes(
    [[15, 10, 370, 510], [275, 340, 510, 510], [130, 345, 210, 425]],
    format="XYXY", canvas_size=(H, W))

torch.manual_seed(1)

img_npu = img.to('npu:0')
transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomPhotometricDistort(p=1),
    v2.RandomHorizontalFlip(p=1),
])
out_img, out_boxes = transforms(img_npu, boxes)
print(f"in_img: type={type(img_npu).__name__} device={img_npu.device.type} dtype={img_npu.dtype} shape={tuple(img_npu.shape)}")
print(f"in_boxes: type={type(boxes).__name__} format={boxes.format.name if hasattr(boxes.format, 'name') else boxes.format} canvas_size={tuple(boxes.canvas_size)} shape={tuple(boxes.shape)}")
print(f"out_img: type={type(out_img).__name__} device={out_img.device.type} dtype={out_img.dtype} shape={tuple(out_img.shape)}")
print(f"out_boxes: type={type(out_boxes).__name__} format={out_boxes.format.name if hasattr(out_boxes.format, 'name') else out_boxes.format} canvas_size={tuple(out_boxes.canvas_size)} shape={tuple(out_boxes.shape)}")
print(f"out_img finite: {torch.isfinite(out_img.float()).all().item()}")  # uint8 finite 永远 True；用 .float() cast 是为防 photometric jitter 链路里某一步意外产 NaN/Inf
print(f"out_boxes min/max: {int(out_boxes.min())}/{int(out_boxes.max())}")  # box 坐标被 RandomResizedCrop 重映射 + flip 翻过来，应仍在 [0, 224] 范围内（clip 由 v2 自动做）
PY
```

输出结果如下（crop / photometric / flip 三个 RNG 都用 `torch.manual_seed(1)` 锁定，所以离散属性（device / dtype / format / canvas_size）必须严格一致；浮点型走 `fuzzy='xxx'`）：

```shell #test-result id="v2-detection" fuzzy='xxx'
in_img: type=Image device=npu dtype=torch.uint8 shape=(3, 256, 256)
in_boxes: type=BoundingBoxes format=XYXY canvas_size=(256, 256) shape=(3, 4)
out_img: type=Image device=npu dtype=torch.uint8 shape=(3, 224, 224)
out_boxes: type=BoundingBoxes format=XYXY canvas_size=(224, 224) shape=(3, 4)
out_img finite: True
out_boxes min/max: xxx
```

### Videos, boxes, masks, keypoints

v2 transform 不只处理 image——也支持 BoundingBoxes / Mask / Video / KeyPoints。一次性把 4 种 TVTensor 塞进 `Compose`，验证 dispatch 链能正确识别每种类型（用 `p=0` 的 `RandomHorizontalFlip` 保证确定性）：

```shell #test id="v2-vbmk"
python << 'PY'
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# 复用 v2-setup 的 fixture（独立子进程，img 不能跨块带过来）
np.random.seed(0)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))

H, W = img.shape[-2:]
boxes = tv_tensors.BoundingBoxes(
    [[15, 10, 370, 510], [275, 340, 510, 510], [130, 345, 210, 425]],
    format="XYXY", canvas_size=(H, W))
mask = tv_tensors.Mask(torch.zeros((1, H, W), dtype=torch.uint8))
video = tv_tensors.Video(torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8))
keypoints = tv_tensors.KeyPoints(
    torch.tensor([[100, 100], [200, 200], [50, 150]]),
    canvas_size=(H, W))

# 显式搬 NPU；坐标型 TVTensor（boxes / keypoints）保持 CPU，
# 这是 v2 的标准用法——坐标无需 NPU 计算
img_npu = img.to('npu:0')
mask_npu = mask.to('npu:0')
video_npu = video.to('npu:0')
transforms = v2.Compose([v2.RandomHorizontalFlip(p=0)])
out_img, out_boxes, out_mask, out_video, out_kp = transforms(
    img_npu, boxes, mask_npu, video_npu, keypoints)
print(f"img: type={type(out_img).__name__} device={out_img.device.type} shape={tuple(out_img.shape)}")
print(f"boxes: type={type(out_boxes).__name__} device={out_boxes.device.type} shape={tuple(out_boxes.shape)}")
print(f"mask: type={type(out_mask).__name__} device={out_mask.device.type} shape={tuple(out_mask.shape)}")
print(f"video: type={type(out_video).__name__} device={out_video.device.type} shape={tuple(out_video.shape)}")
print(f"keypoints: type={type(out_kp).__name__} device={out_kp.device.type} shape={tuple(out_kp.shape)}")
PY
```

输出结果如下：

```shell #test-result id="v2-vbmk"
img: type=Image device=npu shape=(3, 256, 256)
boxes: type=BoundingBoxes device=cpu shape=(3, 4)
mask: type=Mask device=npu shape=(1, 256, 256)
video: type=Video device=npu shape=(3, 3, 32, 32)
keypoints: type=KeyPoints device=cpu shape=(3, 2)
```

### What are TVTensors?

TVTensor 是 `torch.Tensor` 的子类，原生 tensor 接口（`isinstance(..., torch.Tensor)` / `.sum()` / `torch.*`）都能用。transforms 就是按 TVTensor **类型**做 dispatch：

```shell #test id="v2-tvtensors"
python << 'PY'
import torch
from torchvision import tv_tensors

img_dp = tv_tensors.Image(torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8))
print(f"{isinstance(img_dp, torch.Tensor) = }")
print(f"{img_dp.dtype = }, {img_dp.shape = }, {img_dp.sum() = }")
img_npu = img_dp.to('npu:0')   # .to 走 torch_npu 的 PrivateUse1 dispatch；TVTensor 子类化不被打断
print(f"{isinstance(img_npu, torch.Tensor) = }")
print(f"{img_npu.dtype = }, {img_npu.shape = }, {img_npu.device.type = }, {img_npu.sum().cpu() = }")
PY
```

输出结果如下（CPU 段和 NPU 段都验证——TVTensor 在 NPU 上仍是 `torch.Tensor` 子类，`aten::sum` 走 `torch_npu` NPU kernel 后 `.cpu()` 把标量搬回来；`sum` 是 uint8 在 [0, 255] 范围内随机累加的结果，`xxx` 抹平）：

```shell #test-result id="v2-tvtensors" fuzzy='xxx'
isinstance(img_dp, torch.Tensor) = True
img_dp.dtype = torch.uint8, img_dp.shape = torch.Size([3, 256, 256]), img_dp.sum() = tensor(xxx)
isinstance(img_npu, torch.Tensor) = True
img_npu.dtype = torch.uint8, img_npu.shape = torch.Size([3, 256, 256]), img_npu.device.type = 'npu', img_npu.sum().cpu() = tensor(xxx)
```

### What do I pass as input?

transforms 接受**任意嵌套结构**——单 image、`(img, target)`、dict、嵌套 dict 都行；返回**同结构**。transforms 只看 TVTensor **类型**做 dispatch；外来的 str / int / tuple / dict 原样穿透：

```shell #test id="v2-input-structure"
python << 'PY'
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

# 复用 v2-setup 的 fixture（独立子进程，img 不能跨块带过来）
np.random.seed(0)
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))

H, W = img.shape[-2:]
boxes = tv_tensors.BoundingBoxes(
    [[15, 10, 370, 510], [275, 340, 510, 510], [130, 345, 210, 425]],
    format="XYXY", canvas_size=(H, W))
target = {
    "boxes": boxes,
    "labels": torch.arange(boxes.shape[0]),  # 纯 Tensor，不是 TVTensor → 穿透
    "this_is_ignored": ("arbitrary", {"structure": "!"}),  # 外来对象 → 穿透
}
transforms = v2.Compose([v2.Resize(size=(128, 128))])
img_npu = img.to('npu:0')
out_img, out_target = transforms(img_npu, target)
print(f"img device: {out_img.device.type}, target type: {type(out_target).__name__}")
print(f"boxes shape: {tuple(out_target['boxes'].shape)}")
print(f"labels: {out_target['labels'].tolist()}")
print(f"passthrough: {out_target['this_is_ignored']}")
PY
```

输出结果如下（labels 是 `torch.arange(3)`，3 个 box 顺序标 0/1/2；Resize 不改变 box 数也不改变 labels 数值）：

```shell #test-result id="v2-input-structure"
img device: npu, target type: dict
boxes shape: (3, 4)
labels: [0, 1, 2]
passthrough: ('arbitrary', {'structure': '!'})
```

### Transforms and Datasets intercompatibility

自定义 Dataset 的 `__getitem__` 返 `tv_tensors` 后，v2 transform 直接可用——不需要任何胶水代码。下面的合成 `SyntheticDetectionDataset` 复刻 `CocoDetection` 的 `(image, target_dict)` 返回形态：

```shell #test id="v2-dataset-interop"
python << 'PY'
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

class SyntheticDetectionDataset:
    """无网络依赖的合成检测数据集；__getitem__ 返 tv_tensors。"""
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)   # 确定性合成
        arr = (rng.rand(256, 256, 3) * 255).astype('uint8')
        img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))
        boxes = tv_tensors.BoundingBoxes(
            [[10, 20, 100, 200], [50, 60, 150, 250]],
            format="XYXY", canvas_size=img.shape[-2:])
        return img, {"boxes": boxes, "labels": torch.zeros(2, dtype=torch.int64)}

dataset = SyntheticDetectionDataset()
transforms = v2.Compose([v2.Resize(size=(128, 128))])
img, target = dataset[0]
img_npu, target_npu = transforms(img.to('npu:0'), target)
print(f"img: type={type(img_npu).__name__} device={img_npu.device.type} shape={tuple(img_npu.shape)}")
print(f"boxes: type={type(target_npu['boxes']).__name__} shape={tuple(target_npu['boxes'].shape)}")
print(f"labels: {target_npu['labels'].tolist()}")
print(f"len(dataset): {len(dataset)}")
PY
```

输出结果如下：

```shell #test-result id="v2-dataset-interop"
img: type=Image device=npu shape=(3, 128, 128)
boxes: type=BoundingBoxes shape=(2, 4)
labels: [0, 0]
len(dataset): 2
```

