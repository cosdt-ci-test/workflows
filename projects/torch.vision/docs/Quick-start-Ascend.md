# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑通 `torchvision` 的最小链路：从 PyPI 装 stock `torchvision==0.24.0`（linux aarch64 cpu-only wheel，走阿里 PyPI 镜像），验 `torchvision.transforms.v2` 的导入与版本，再把一张合成图走完「PIL → 张量 → NPU 端到端」的烟雾流程。NPU 端的 kernel 命中走 `torch_npu` wheel 自带的 `aten::*` PrivateUse1 dispatch + CPU fallback，无需 [Ascend/vision](https://github.com/Ascend/vision) fork 的 C++ bridge（`deform_conv` / `roi_pool` patch 那些 op 跟 transforms.v2 不沾边）。配套版本按 [Ascend PyTorch Compatibility 矩阵](https://gitcode.com/Ascend/pytorch/blob/main/COMPATIBILITY.en.md)：`torch==2.9.0 + torch_npu==2.9.0.post6 + CANN==9.1.0`。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- `torch` + `torch_npu` + `torchvision` **由本文档用 `uv pip install` 显式装**，不依赖 image 预装。配套版本按 [Ascend PyTorch Compatibility 矩阵](https://gitcode.com/Ascend/pytorch/blob/main/COMPATIBILITY.en.md)：`torch==2.9.0 + torch_npu==2.9.0.post6 + CANN==9.1.0` 是 CANN 9.1.0 的推荐组合（per table 行 `2.9.0.post6 / v2.9.0-26.1.0 / 2.9.0 / CANN 9.1.0`）

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
torch= 2.9.0xxx
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

### 二进制路径（stock）

stock torchvision 从阿里 PyPI 镜像装（0.22+ 的 torchvision 在 PyPI 上的 linux aarch64 wheel 本身就是 cpu-only）：

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

> 二进制路径只覆盖 Python-level v2 API + `aten::*` NPU dispatch（`torch_npu` wheel 自带），**不验** fork 那边 C++ 扩展的 NPU 算子注册——后者由 [Ascend/vision](https://github.com/Ascend/vision) fork 的 `torchvision_npu.ops` 子包（`deform_conv` / `roi_pool` patch）承担，但本文档的烟雾路径（`transforms.v2.CenterCrop` / `RandomHorizontalFlip` / `Resize` / TVTensor dispatch）不命中这些 op，所以源码 build 没必要：stock cpu wheel 走 `torch_npu` PrivateUse1 dispatch 已经够。
>
> 如果后续要把 torchvision 推到 `model.eval().to('npu:0')` 做端到端推理，再单独开一个 fork 路径的 doc（`uv pip install --no-build-isolation -e .` 那个 21 分钟 cold-cache C++ 编译）；那时候也要切到带 DVPP dev headers 的镜像或者像本次调试那样把 `npu_decode_video_kernel.{cpp,hpp}` 挪出 build 路径（CANN base 镜像 `cann:9.1.0-910b-ubuntu22.04-py3.12` 不带 `<acl/dvpp/hi_dvpp.h>`，fork 默认会 `fatal error`）。

## 验证 transforms.v2

下面 6 节按 torchvision 官方 [Transforms v2 Getting Started](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html) 的章节顺序走，把示例里的 PIL 资产换成合成图（CPU 路径、无网络依赖），每一节都把 transform 输入**显式搬到 NPU**（`img.to('npu:0')`），让 `torch_npu` 注册的 PrivateUse1 dispatch + `aten::*` 的 NPU kernel 真被命中。这 6 节跑在 stock torchvision cpu wheel 上——`transforms.v2` 的 Python-level 调用 + `aten::npu::*` 算子都在 `torch_npu` wheel 自带的 dispatch 表里，不需要 fork patch。每一节一个 `#test` 块，跑过即走完整个 v2 API 表面。

### Setup

导入 `tv_tensors` + `v2`，合成一张 256×256 RGB 图包成 `tv_tensors.Image`，同时验证 torchvision 顶层包 + 子模块 import OK（解包 + 链接 C++ 扩展）：

```shell #test id="v2-setup"
python << 'PY' 2>&1
import numpy as np
import torch
import torch_npu
import torchvision
import torchvision.transforms as T
import torchvision.transforms.v2 as T2
import torchvision.io as IO
import torchvision.models as M
from torchvision import tv_tensors

torch.manual_seed(0)
np.random.seed(0)

# 1) 合成 3x256x256 RGB 图（CPU 路径，无网络依赖）
arr = (np.random.rand(256, 256, 3) * 255).astype('uint8')
img = tv_tensors.Image(torch.from_numpy(arr).permute(2, 0, 1))
print(f"torchvision: {torchvision.__version__}")
print(f"transforms: {T.__name__}")
print(f"transforms.v2: {T2.__name__}")
print(f"io: {IO.__name__}")
print(f"models: {M.__name__}")
print(f"img: type={type(img).__name__} dtype={img.dtype} shape={tuple(img.shape)}")
# 2) 验证 torch_npu NPU dispatch：is_available + device_count + cpu→npu→cpu round-trip
print(f"npu_available: {torch.npu.is_available()}")
print(f"npu_count: {torch.npu.device_count()}")
x = torch.zeros(2, 3)
x_npu = x.to('npu:0')
x_back = x_npu.cpu()
print(f"npu round-trip: in={tuple(x.shape)} on {x.device.type}, npu={tuple(x_npu.shape)} on {x_npu.device.type}, back={tuple(x_back.shape)} on {x_back.device.type}")
PY
```

输出结果如下（`tv_tensors.Image` 把 `(H, W, C)` numpy 转 `(C, H, W)` uint8 张量；`torchvision` 行用 `xxx` 抹平 patch 版本后缀；第 2) 段验证 `torch_npu` 注册 + `aten::to` NPU kernel 双向通）：

```shell #test-result id="v2-setup" fuzzy='xxx'
torchvision: xxx
transforms: torchvision.transforms
transforms.v2: torchvision.transforms.v2
io: torchvision.io
models: torchvision.models
img: type=Image dtype=torch.uint8 shape=(3, 256, 256)
npu_available: True
npu_count: xxx
npu round-trip: in=(2, 3) on cpu, npu=(2, 3) on npu, back=(2, 3) on cpu
```

### The basics

v2 transform 是 `nn.Module` 风格——单输入单输出。把图搬到 NPU **再**调 transform，让 `CenterCrop` 跑在 NPU 后端（device 透传：CPU 输入 → CPU 输出，NPU 输入 → NPU 输出）：

```shell #test id="v2-basics"
python << 'PY' 2>&1
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

### Videos, boxes, masks, keypoints

v2 transform 不只处理 image——也支持 BoundingBoxes / Mask / Video / KeyPoints。一次性把 4 种 TVTensor 塞进 `Compose`，验证 dispatch 链能正确识别每种类型（用 `p=0` 的 `RandomHorizontalFlip` 保证确定性）：

```shell #test id="v2-vbmk"
python << 'PY' 2>&1
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
python << 'PY' 2>&1
import torch
from torchvision import tv_tensors

img_dp = tv_tensors.Image(torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8))
print(f"isinstance Tensor: {isinstance(img_dp, torch.Tensor)}")
print(f"cpu: dtype={img_dp.dtype} shape={tuple(img_dp.shape)} sum={int(img_dp.sum())}")
img_npu = img_dp.to('npu:0')   # .to 走 torch_npu 的 PrivateUse1 dispatch；TVTensor 子类化不被打断
print(f"isinstance Tensor (npu): {isinstance(img_npu, torch.Tensor)}")
print(f"npu: dtype={img_npu.dtype} shape={tuple(img_npu.shape)} device={img_npu.device.type} sum={int(img_npu.sum().cpu())}")
PY
```

输出结果如下（CPU 段和 NPU 段都验证——TVTensor 在 NPU 上仍是 `torch.Tensor` 子类，`aten::sum` 走 `torch_npu` NPU kernel 后 `.cpu()` 把标量搬回来；`sum` 是 uint8 在 [0, 255] 范围内随机累加的结果，`xxx` 抹平）：

```shell #test-result id="v2-tvtensors" fuzzy='xxx'
isinstance Tensor: True
cpu: dtype=torch.uint8 shape=(3, 256, 256) sum=xxx
isinstance Tensor (npu): True
npu: dtype=torch.uint8 shape=(3, 256, 256) device=npu sum=xxx
```

### What do I pass as input?

transforms 接受**任意嵌套结构**——单 image、`(img, target)`、dict、嵌套 dict 都行；返回**同结构**。transforms 只看 TVTensor **类型**做 dispatch；外来的 str / int / tuple / dict 原样穿透：

```shell #test id="v2-input-structure"
python << 'PY' 2>&1
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
python << 'PY' 2>&1
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

