# Quick Start: DeepSpeed on Ascend NPU

在昇腾 NPU 上安装 DeepSpeed，用 CIFAR10 图像分类跑通第一次训练，并理解 DeepSpeed 的工作流程。

## 前置条件

- **硬件**：Atlas 800T / 900 A2 训练服务器，搭载 Ascend 910B NPU，单卡。
- **软件**：已装好 CANN，以及与 CANN 匹配的 `torch` + `torch_npu`（`torch.npu.is_available() == True`）。参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)与 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)。
- **示例版本**：Python 3.10 · CANN 8.x · torch 2.9.x · torch_npu 2.9.x · torchvision 0.24.x · deepspeed 0.19.x。

## 安装 DeepSpeed

**安装 DeepSpeed。** 通过 pip 安装。

```shell #test id="install-deepspeed"
pip install deepspeed
python -c "import deepspeed; print('DeepSpeed', deepspeed.__version__)"
```

```shell #test-result id="install-deepspeed" fuzzy='...' fuzzy='xxx'
...
DeepSpeed xxx
```

**验证 DeepSpeed 已识别昇腾 NPU 加速器。** 输出 accelerator: npu 即接入成功。

```shell #test id="verify-accelerator"
python -c "from deepspeed.accelerator import get_accelerator; print('accelerator:', get_accelerator()._name)"
```

```shell #test-result id="verify-accelerator"
accelerator: npu
```

## 编写训练脚本

下面这段 CIFAR10 训练脚本分 4 个模块，展示了 DeepSpeed 的完整工作流程。脚本直接执行，CIFAR10 数据集会在首次运行时自动下载。

```shell #test id="run-train"
python - <<'PY'
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import deepspeed

# ===== 模块 1：准备数据 =====
# CIFAR10 数据集首次运行时自动下载到当前目录，无需手动准备。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

# ===== 模块 2：定义模型 =====
# 一个用于图像分类的小型 CNN。DeepSpeed 兼容任意 PyTorch 模型，无需修改模型代码。
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ===== 模块 3：配置 DeepSpeed 并初始化引擎 =====
# deepspeed.initialize() 是 DeepSpeed 的入口，ZeRO 显存优化和 BF16 混合精度在此注入，返回封装后的模型引擎。
ds_config = {
    'train_batch_size': 16,
    'zero_optimization': {'stage': 1},
    'bf16': {'enabled': True},
}
model = Net()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, model_parameters=model.parameters(), config=ds_config)

# ===== 模块 4：训练循环 =====
# model_engine.backward() 和 model_engine.step() 替换了原生 API，DeepSpeed 在此接管梯度计算与参数更新。
criterion = nn.CrossEntropyLoss()
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(model_engine.device), data[1].to(model_engine.device)
        outputs = model_engine(inputs)
        loss = criterion(outputs, labels)
        model_engine.backward(loss)
        model_engine.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0
print('Finished Training')
PY
```

**验证训练结果。** 末尾输出包含 Finished Training 即训练成功。

```shell #test-result id="run-train" fuzzy='...'
...
Finished Training
```

## 更多用法

更多用法见 DeepSpeedExamples：https://github.com/deepspeedai/DeepSpeedExamples