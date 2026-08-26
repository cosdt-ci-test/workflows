# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑通 [timm](https://github.com/huggingface/pytorch-image-models) 的核心链路：安装 timm（二进制 + 源码两条路径），逐节验证 `timm.list_models()` / `timm.create_model()` / forward / `forward_features()` / `timm.data.resolve_data_config()` / `timm.data.create_transform()` 的最小图像模型入口——全部在 `npu:0` 上验证 `torch_npu` 路由正确、无回落。

> 本文档的可执行「核心链路」示例统一使用 `pretrained=False`（随机初始化），不依赖 HuggingFace Hub / GitHub Releases 下载预训练权重。原因是自托管 NPU runner 处于集群防火墙内，无法访问这两处；而 `pretrained=True` 会触发权重下载。文档因此只验证「安装 + 模型构造 + 前向 + 特征提取 + transform 构建」，不验证实际的 ImageNet 权重加载。若你在联网机器上想加载预训练权重，把 `pretrained=False` 改成 `pretrained=True` 即可（timm 会自动下载到 `~/.cache/torch/hub/checkpoints`），本文其余链路不变。


## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

timm 通过 `torch_npu` 间接支持昇腾 NPU：timm 自身的模型前向 / 特征提取 / 数据变换全部建立在 `torch.nn` 与 `torchvision` 之上，`torch_npu` 把这些算子正确路由到 NPU，timm 自身不需要额外的 NPU 适配层。

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
| torchvision | 0.24.0 |
| timm | 最新 release 的源码/二进制 |

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

## 安装 timm

timm 同时支持 PyPI 二进制安装与 GitHub 源码安装，两条路径都把核心模块（`timm.models` / `timm.data` / `timm.optim` / `timm.scheduler` 等）一起打包。timm 依赖 `torch` / `torchvision` / `pyyaml` / `huggingface_hub` / `safetensors`，安装 timm 时这些依赖会一并解析。

### 使用 uv 进行安装

```shell #test id="timm-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple timm
python -c "import timm; print('timm', timm.__version__)"
```

输出结果类似如下：

```shell #test-result id="timm-install-binary" fuzzy='xxx'
timm xxx
```
- xxx 表示最新的版本号

<!--
```shell #test-setup
uv pip uninstall timm -y
```
-->

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

用 `git clone --depth 1 --branch <ref>` 直接浅克隆工作流注入的最新 release tag，安装并且验证：

```shell #test id="timm-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/huggingface/pytorch-image-models.git
cd pytorch-image-models
uv pip install -e .
python -c "import timm; print('timm', timm.__version__)"
```

\<ref> 为安装的最新的 release tag。

输出结果类似如下：

```shell #test-result id="timm-install-source" fuzzy='xxx'
timm xxx
```
- xxx 表示最新的版本号

## 核心链路验证

以下示例统一用 `pretrained=False`（随机初始化）在 `npu:0` 上构造模型，验证最小图像模型链路，不触发任何权重下载。

### 1. 模型列表 — `timm.list_models()`

`timm.list_models('resnet*')` 列出匹配前缀的所有模型名，这里只校验 `resnet18` 在列表中：

```shell #test id="timm-list-models"
python -c "import timm; print('resnet18' in timm.list_models('resnet*'))"
```

输出结果如下：

```shell #test-result id="timm-list-models"
True
```

### 2. 构造模型 + 前向 — `create_model()` + forward

`timm.create_model('resnet18', pretrained=False)` 随机初始化 ResNet-18，`.to('npu:0')` 搬到 NPU 上做一次前向；输入 `(2, 3, 224, 224)`，输出 `(2, 1000)`（ImageNet 1000 类 logits）：

```shell #test id="timm-create-forward"
python -c "
import torch, torch_npu, timm
m = timm.create_model('resnet18', pretrained=False).to('npu:0')
m.eval()
x = torch.randn(2, 3, 224, 224, device='npu:0')
with torch.no_grad():
    y = m(x)
print('class', type(m).__name__)
print('y_shape', tuple(y.shape))
"
```

输出结果如下：

```shell #test-result id="timm-create-forward"
class ResNet
y_shape (2, 1000)
```

### 3. 特征提取 — `forward_features()`

`forward_features()` 返回全局池化前的特征图，ResNet-18 对 `(1, 3, 224, 224)` 输入输出 `(1, 512, 7, 7)`：

```shell #test id="timm-forward-features"
python -c "
import torch, torch_npu, timm
m = timm.create_model('resnet18', pretrained=False).to('npu:0')
m.eval()
x = torch.randn(1, 3, 224, 224, device='npu:0')
with torch.no_grad():
    f = m.forward_features(x)
print('f_shape', tuple(f.shape))
"
```

输出结果如下：

```shell #test-result id="timm-forward-features"
f_shape (1, 512, 7, 7)
```

### 4. 数据配置 — `timm.data.resolve_data_config()`

`resolve_data_config()` 从模型注册的配置里解析出推理/训练所需的数据配置（输入尺寸 / 插值方式 / 均值 / 标准差 / crop 比例 / crop 模式）：

```shell #test id="timm-data-config"
python -c "
import timm
m = timm.create_model('resnet18', pretrained=False)
cfg = timm.data.resolve_data_config(model=m)
print('keys', sorted(cfg.keys()))
print('input_size', tuple(cfg['input_size']))
"
```

输出结果如下：

```shell #test-result id="timm-data-config"
keys ['crop_mode', 'crop_pct', 'input_size', 'interpolation', 'mean', 'std']
input_size (3, 224, 224)
```

### 5. 构建 transform — `timm.data.create_transform()`

`create_transform()` 用上一步的 `cfg` 直接构建推理 transform（`torchvision.transforms.Compose`），这是上游 `timm.data.create_transform(**data_config)` 的标准用法：

```shell #test id="timm-create-transform"
python -c "
import timm
m = timm.create_model('resnet18', pretrained=False)
cfg = timm.data.resolve_data_config(model=m)
t = timm.data.create_transform(**cfg, is_training=False)
print('transform_type', type(t).__name__)
print('callable', callable(t))
"
```

输出结果如下：

```shell #test-result id="timm-create-transform"
transform_type Compose
callable True
```