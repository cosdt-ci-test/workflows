# 快速开始：在昇腾 NPU 上用 DeepSpeed 做分布式训练

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**：从源码安装 DeepSpeed，验证 NPU 加速器，在单卡 NPU 上完成一次最小化训练。

[DeepSpeed](https://github.com/deepspeedai/DeepSpeed) 是微软开源的深度学习训练优化库，通过 **NPU 加速器**（`accelerator/npu_accelerator.py`）自动适配昇腾硬件，加速器名称为 `npu`。

<!--
```shell #test-setup store="upstream_ref"
echo $UPSTREAM_REF
```
-->
<!--
```shell #test-setup
apt-get update && apt-get install -y libopenmpi-dev
pip install mpi4py
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
| PyTorch | `torch` + `torch_npu` 已安装且 `torch.npu.is_available() == True` |
| pip | 可访问昇腾 PyPI 镜像（`https://repo.huaweicloud.com/ascend/repos/pypi`） |
| deepspeed | 本文从 GitHub 源码安装，见下文 |

---

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。`npu-smi` 在常见容器布局下需手动加入 `PATH`。

```shell #test id="load-cann"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

```shell #test-result id="load-cann"
...
```

---

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell #test id="check-npu"
npu-smi info
```

```shell #test-result id="check-npu"
...
```

**预期**：命令退出码为 0，并打印设备列表。表格中的功耗、HBM 占用每次不同，**不必**与任何样例逐字一致。

### 2.2 确认 PyTorch 与 torch_npu

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

```shell #test-result id="check-torch" fuzzy='xxx'
torch: 2.9.0+cpu
torch_npu: 2.9.0.post2
is_available: True
count: xxx
```

---

## 3. 获取源码并安装

克隆上游仓库，检出你要用的 ref，然后 `pip install -e .` 安装。将 `<UPSTREAM_REF>` 换成目标**分支、tag 或 commit**（上游默认分支为 `master`）。

```shell #test id="install-deepspeed" load="upstream_ref>>UPSTREAM_REF"
git clone https://github.com/deepspeedai/DeepSpeed.git
cd DeepSpeed
git checkout <UPSTREAM_REF>
pip install -e .
python -c "import deepspeed; print('DeepSpeed:', deepspeed.__version__)"
```

```shell #test-result id="install-deepspeed" fuzzy='...' fuzzy='xxx'
...
DeepSpeed: xxx
```

---

## 4. 验证 NPU 加速器

`ds_report` 应列出 `torch_npu` 和 `ascend_cann` 版本。`get_accelerator()._name` 应为 `npu`。

```shell #test id="verify-accelerator"
ds_report 2>&1 | grep -i 'torch_npu\|ascend_cann'
python -c "from deepspeed.accelerator import get_accelerator; print('accelerator:', get_accelerator()._name)"
```

```shell #test-result id="verify-accelerator"
...
accelerator: npu
```

---

## 5. 运行最小化训练

跑一个最小化训练脚本：3 层 Linear 网络（32 → 64 → 32），ZeRO-1 + BF16，5 步训练。模型和数据均在 NPU 上，不依赖外部数据集。

```shell #test id="train-minimal"
python - <<'PY'
import torch
import torch.nn as nn
import deepspeed

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
    def forward(self, x):
        return self.net(x)

model = Net()
ds_config = {
    'train_batch_size': 4,
    'train_micro_batch_size_per_gpu': 4,
    'zero_optimization': {'stage': 1},
    'optimizer': {'type': 'Adam', 'params': {'lr': 0.001}},
    'bf16': {'enabled': True},
}
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, model_parameters=model.parameters(), config=ds_config)

for step in range(1, 6):
    x = torch.randn(4, 32, device=model_engine.device)
    loss = model_engine(x).sum()
    model_engine.backward(loss)
    model_engine.step()
    print(f'step {step}/5 loss={loss.item():.6f}')

print('Quick-start test PASSED')
PY
```

```shell #test-result id="train-minimal"
...
Quick-start test PASSED
```

**怎样算成功**：进程退出码为 0，且输出末尾出现 `Quick-start test PASSED`。

---

## 6. 下一步

| 目标 | 参考 |
| --- | --- |
| 深度学习训练入门 | 上游 [Getting Started](https://www.deepspeed.ai/getting-started/) |
| 昇腾加速器完整文档 | 上游 [Accelerator Setup Guide — Huawei Ascend NPU](https://www.deepspeed.ai/tutorials/accelerator-setup-guide/) |
| 更多训练示例 | [DeepSpeedExamples](https://github.com/deepspeedai/DeepSpeedExamples) 仓库（CIFAR、HelloDeepSpeed 等） |

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `import torch_npu` 失败 | torch_npu 未安装或版本不匹配 | 检查 torch ↔ torch_npu ↔ CANN 三方兼容矩阵 |
| `ds_report` 无 `npu` 加速器 | 安装时 torch_npu 不可 import | 确保 `source set_env.sh` 后 `python -c "import torch_npu"` 成功 |
| 训练退出 0 但无 `npu` 加速器 | 设备未挂载或驱动异常 | 检查 `/dev/davinci0` 是否存在，`npu-smi info` 是否正常 |
| 训练卡住不输出 | 分布式通信异常 | 检查 `ASCEND_RT_VISIBLE_DEVICES` 是否设置正确