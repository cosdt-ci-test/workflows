# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上快速跑通 [timm](https://github.com/huggingface/pytorch-image-models) 的训练 / 验证 / 推理全流程：完成环境准备与安装后，在入口脚本导入 `torch_npu`，即可把算子全部路由到 `npu:0`。

> 文中可执行示例统一使用 `pretrained=False`（随机初始化），无需下载预训练权重与数据集，开箱即跑；联网环境改为 `pretrained=True` 即可自动加载权重。

## 环境准备

- **硬件**：Atlas 900 A2 / A3 或 Ascend 950 系列，已挂载设备。
- **软件**：已装好 CANN，以及与 CANN 匹配的 `torch` + `torch_npu`（`torch.npu.is_available() == True`）。参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)与 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)。
- **本文档示例版本**：Python 3.12 · CANN 9.1.0 · torch 2.9.0+cpu · torch\_npu 2.9.0.post2 · torchvision 0.24.0 · timm 最新 release。

检查 Python 与 torch / torch\_npu 环境：

```shell
python --version
```

```shell
Python 3.12.xxx
```

```shell
python -c "
import torch, torch_npu
print('torch=', torch.__version__)
print('torch_npu=', torch_npu.__version__)
print('is_available:', torch.npu.is_available())
print('count:', torch.npu.device_count())
"
```

```shell
torch= 2.9.0+cpu
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

> `import torch_npu` 失败时，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵排查版本。

## 安装 timm

timm 依赖 `torch` / `torchvision` / `pyyaml` / `huggingface_hub` / `safetensors`，安装时一并解析。

```shell
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple timm
python -c "
import timm
print('timm', timm.__version__)
"
```

```shell
timm xxx
```

## 快速开始

在入口脚本（如 timm 仓库的 `train.py` / `validate.py` / `inference.py`）中先 `import torch`，再 `import torch_npu`，即可用 `--device npu` 在 NPU 上训练、验证与推理。下面按官方流程给出完整命令与无需下载数据 / 权重的最小可执行示例。

### 1. 导入 torch\_npu

```shell
python -c "
import torch
import torch_npu
x = torch.ones(2, 3, device='npu:0')
print('x_device', x.device)
print('x_sum', x.sum().item())
"
```

```shell
x_device npu:0
x_sum 6.0
```

### 2. 单卡/分布式训练

以 `ImageNet-1000` 图像分类训练为例（`num_npus` 换成实际 NPU 卡数，`--model` 与数据集路径按需替换）：

```shell
num_npus=1
./distributed_train.sh $num_npus path/to/dataset/ImageNet-1000 \
    --device npu \
    --model seresnet34 \
    --sched cosine \
    --epochs 150 \
    --warmup-epochs 5 \
    --lr 0.4 \
    --reprob 0.5 \
    --remode pixel \
    --batch-size 256 \
    --amp -j 4
```

无需数据的最小训练示例（ResNet-18 完成一次「前向 → 损失 → 反向 → 优化器更新」）：

```shell
python -c "
import torch, torch_npu, timm
model = timm.create_model('resnet18', pretrained=False).to('npu:0')
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
x = torch.randn(4, 3, 224, 224, device='npu:0')
y = torch.randint(0, 1000, (4,), device='npu:0')
optimizer.zero_grad()
out = model(x)
loss = torch.nn.functional.cross_entropy(out, y)
loss.backward()
optimizer.step()
print('loss_dtype', loss.dtype)
print('loss_device', loss.device)
print('out_shape', tuple(out.shape))
"
```

```shell
loss_dtype torch.float32
loss_device npu:0
out_shape (4, 1000)
```

### 3. 模型验证

用 `validate.py` 在验证集上评估，`--pretrained` 加载预训练权重：

```shell
python validate.py path/to/data --device npu --model path/to/model --batch-size 64 --pretrained
```

无需数据的最小验证示例（eval 前向，`argmax` 对应 `Acc@1`、`topk(5)` 对应 `Acc@5`）：

```shell
python -c "
import torch, torch_npu, timm
model = timm.create_model('resnet18', pretrained=False).to('npu:0')
model.eval()
x = torch.randn(8, 3, 224, 224, device='npu:0')
with torch.no_grad():
    out = model(x)
pred = out.argmax(dim=1)
top5 = out.topk(5, dim=1).indices
print('out_shape', tuple(out.shape))
print('pred_shape', tuple(pred.shape))
print('top5_shape', tuple(top5.shape))
"
```

```shell
out_shape (8, 1000)
pred_shape (8,)
top5_shape (8, 5)
```

### 4. 模型推理

用 `inference.py` 对图片做推理分类，`--topk 5` 输出 top-5 类别：

```shell
python inference.py ../open_clip/data/ImageNet-1000/val/ \
    --device npu \
    --batch-size 64 \
    --model ./model_ckpts/tiny_vit_21m_512 \
    --label-type detail \
    --topk 5
```

无需数据的最小推理示例（单张输入，`softmax` 后取 top-5）：

```shell
python -c "
import torch, torch_npu, timm
model = timm.create_model('resnet18', pretrained=False).to('npu:0')
model.eval()
x = torch.randn(1, 3, 224, 224, device='npu:0')
with torch.no_grad():
    out = model(x)
probs = torch.softmax(out, dim=1)
topk = probs.topk(5, dim=1)
print('out_shape', tuple(out.shape))
print('topk_values_shape', tuple(topk.values.shape))
print('topk_indices_shape', tuple(topk.indices.shape))
"
```

```shell
out_shape (1, 1000)
topk_values_shape (1, 5)
topk_indices_shape (1, 5)
```

