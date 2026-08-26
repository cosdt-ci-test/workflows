# 快速开始：在昇腾 NPU 上使用 DeepSpeed

> 本文帮助你在 **昇腾 NPU** 上从零开始完成第一次 DeepSpeed 训练：加载环境 → 安装 DeepSpeed → 确认 NPU 可用 → 在 NPU 上跑通一个最小训练示例 → 使用 `deepspeed` 命令启动分布式训练。阅读前请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。更多通用用法（训练 API、配置项说明、ZeRO 进阶）请参考官方 [Getting Started](https://www.deepspeed.ai/getting-started/)。

[DeepSpeed](https://github.com/deepspeedai/DeepSpeed) 是一款开源的深度学习训练加速库，支持大规模分布式训练与显存优化。在昇腾 NPU 上，DeepSpeed 会自动选择 `npu` 作为加速器后端，无需额外配置即可使用。

---

## 开始之前

完成本文全部步骤后，你将能够在昇腾 NPU 上成功运行 DeepSpeed 训练，并知道如何用 `deepspeed` 命令启动单卡和多卡训练。

### 适用硬件

Atlas **800T** / **900 A2** 训练系列（搭载 Ascend **910B** NPU）。本文前半部分以单卡为例，多卡用法见 §6。

### 需要你提前准备好的软件

| 组件 | 说明 | 如何确认已装好 |
| --- | --- | --- |
| NPU 驱动与固件 | 让操作系统能识别 NPU 硬件 | 命令行输入 `npu-smi info` 能看到设备列表 |
| CANN toolkit | 昇腾计算架构软件栈 | 安装后存在 `/usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Python 3 | 运行 DeepSpeed 所需 | `python --version` 能看到 3.x 版本号 |
| PyTorch + torch_npu | 深度学习框架及昇腾适配 | `python -c "import torch, torch_npu; print(torch.npu.is_available())"` 输出 `True` |
| pip 源 | 可访问昇腾 PyPI 镜像 | 可通过 `-i https://repo.huaweicloud.com/ascend/repos/pypi` 加速下载 |

### 参考版本

以下是本文撰写时验证通过的版本组合，仅作参考。如果你的版本不完全一致，通常只要 torch 与 torch_npu 主版本号匹配即可。

| 组件 | 参考版本 |
| --- | --- |
| torch | 2.9.x |
| torch_npu | 2.9.x |
| CANN toolkit | 8.x |

---

## 1. 加载昇腾环境

新打开的终端不会自动加载 CANN 环境变量，需要先执行一次加载脚本；`npu-smi` 在常见容器布局下也需要手动加入 PATH。

```shell #test id="load-cann"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

```shell #test-result id="load-cann"
...
```

> 提示：每次新开终端都需要重新执行上面两条命令，建议写入 `~/.bashrc`。

---

## 2. 确认环境就绪

### 2.1 查看 NPU 设备

```shell #test id="check-npu"
npu-smi info
```

```shell #test-result id="check-npu"
...
```

如果能看到设备列表表格，说明驱动和 NPU 工作正常。表格中的功耗、显存占用每次运行都会变化，不用与任何示例逐字一致。

### 2.2 确认 PyTorch 能识别 NPU

```shell #test id="check-torch"
python -c "import torch, torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

```shell #test-result id="check-torch" fuzzy='xxx'
torch: 2.9.0+cpu
torch_npu: 2.9.0.post2
is_available: True
count: xxx
```

`is_available: True` 表示 PyTorch 已正确识别 NPU；`count` 显示当前可见的 NPU 数量。

---

## 3. 安装 DeepSpeed

DeepSpeed 已原生支持昇腾 NPU，直接通过 pip 安装即可：

```shell #test id="install-deepspeed"
pip install deepspeed
python -c "import deepspeed; print('DeepSpeed 安装成功，版本：', deepspeed.__version__)"
```

```shell #test-result id="install-deepspeed" fuzzy='...' fuzzy='xxx'
...
DeepSpeed 安装成功，版本： xxx
```

如果默认 pip 源下载较慢，可使用昇腾 PyPI 镜像：

```shell
pip install deepspeed -i https://repo.huaweicloud.com/ascend/repos/pypi
```

DeepSpeed 需要通过 MPI 通信库发现分布式环境，安装完 DeepSpeed 后请一并安装：

```shell #test-setup
apt-get update && apt-get install -y libopenmpi-dev
pip install mpi4py
```

---

## 4. 确认 DeepSpeed 识别到 NPU

安装完成后，用 `ds_report` 命令查看 DeepSpeed 当前识别到的加速硬件。在昇腾上，输出中应包含 `torch_npu` 和 `ascend_cann` 的版本信息；DeepSpeed 会自动选择 `npu` 作为加速器后端。

```shell #test id="verify-accelerator"
ds_report 2>&1 | grep -i 'torch_npu\|ascend_cann'
python -c "from deepspeed.accelerator import get_accelerator; print('accelerator:', get_accelerator()._name)"
```

```shell #test-result id="verify-accelerator"
...
accelerator: npu
```

看到最后一行 `accelerator: npu` 即表示 DeepSpeed 已正确接入昇腾。

---

## 5. 跑通第一个训练

下面这段代码会在 NPU 上训练一个 3 层的小型全连接网络（启用 ZeRO-1 显存优化和 BF16 混合精度），共训练 5 步。模型和随机数据都直接放在 NPU 上，不需要你准备任何数据集，把代码原样粘贴到终端执行即可。

> 在昇腾 NPU 上，模型会自动运行在 NPU 上，不需要像 CUDA 代码那样手动调用 `.to('cuda')`，示例中通过 `model_engine.device` 获取当前设备。

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

model_engine.save_checkpoint('./ds_ckpt', 'step5', client_state={'step': 5})
print('Quick-start test PASSED')
PY
```

```shell #test-result id="train-minimal"
...
Quick-start test PASSED
```

看到 `Quick-start test PASSED` 字样就表示训练跑通了。同时当前目录下会生成一个 `ds_ckpt/` 文件夹，里面是训练过程中自动保存的断点，可用于下次继续训练。

### 5.1 保存与继续训练

DeepSpeed 支持把训练进度保存为断点，下次可以从断点继续训练而不必从头开始。下面演示如何加载刚才保存的断点并继续训练 1 步：

```shell #test id="ckpt-load"
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

_, client_state = model_engine.load_checkpoint('./ds_ckpt', 'step5')
print(f'已加载断点，当前训练步数：{client_state["step"]}')

x = torch.randn(4, 32, device=model_engine.device)
loss = model_engine(x).sum()
model_engine.backward(loss)
model_engine.step()
print('Checkpoint load PASSED')
PY
```

```shell #test-result id="ckpt-load" fuzzy='...'
已加载断点，当前训练步数：5
Checkpoint load PASSED
```

### 5.2 使用配置文件

除了在 Python 代码里直接写配置，DeepSpeed 也支持使用独立的 JSON 配置文件，这是更常见的做法。下面把上面用到的最小配置写成 `ds_config.json` 并验证文件格式正确：

```shell #test-setup
cat > ds_config.json <<'EOF'
{
  "train_batch_size": 4,
  "train_micro_batch_size_per_gpu": 4,
  "zero_optimization": {"stage": 1},
  "optimizer": {"type": "Adam", "params": {"lr": 0.001}},
  "bf16": {"enabled": true}
}
EOF
```

```shell #test id="ds-config"
python -c "import json; cfg=json.load(open('ds_config.json')); print('train_batch_size:', cfg['train_batch_size']); print('bf16 enabled:', cfg['bf16']['enabled']); print('zero stage:', cfg['zero_optimization']['stage'])"
```

```shell #test-result id="ds-config"
train_batch_size: 4
bf16 enabled: True
zero stage: 1
```

> 提示：昇腾 910B 原生支持 BF16 混合精度，建议在配置中开启 `bf16.enabled: true` 以获得更好的训练稳定性和性能。配置文件的完整字段说明请参考官方 [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/)。

### 5.3 手动初始化分布式环境

`deepspeed.initialize()` 默认会自动初始化分布式环境，无需手动处理。只有当你的代码需要在初始化前使用 `torch.distributed`（例如获取进程编号）时，才需要把原来的 `torch.distributed.init_process_group` 替换为 `deepspeed.init_distributed()`，DeepSpeed 会自动识别昇腾的 `hccl` 后端：

```shell #test id="init-distributed"
python - <<'PY'
import torch
import deepspeed

deepspeed.init_distributed()
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
backend = torch.distributed.get_backend()
print(f'rank={rank} world_size={world_size} backend={backend}')
print('init_distributed PASSED')
PY
```

```shell #test-result id="init-distributed"
rank=0 world_size=1 backend=hccl
init_distributed PASSED
```

> 说明：`deepspeed.init_distributed()` 会自动选择昇腾的 `hccl` 通信后端，无需手动传入 `dist_backend`。单卡运行时进程总数 `world_size` 为 1。多机训练时该调用会自动通过 MPI 发现进程编号并传播到所有节点，详见官方文档。

---

## 6. 在昇腾上启动分布式训练

前面的示例直接用 `python` 命令运行，适合调试。生产环境里通常使用 DeepSpeed 提供的 `deepspeed` 命令启动训练，它会自动在多张 NPU 上启动多个进程并分配进程编号，你无需手动管理。

### 6.0 准备多卡训练依赖

MPI 依赖已在 §3 安装，接下来把 §5 的训练脚本保存成文件 `train_minimal.py`，供 `deepspeed` 命令调用：

```shell #test-setup
cat > train_minimal.py <<'EOF'
import os
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

ds_config = {
    'train_batch_size': 4,
    'train_micro_batch_size_per_gpu': 4,
    'zero_optimization': {'stage': 1},
    'optimizer': {'type': 'Adam', 'params': {'lr': 0.001}},
    'bf16': {'enabled': True},
}

model = Net()
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model, model_parameters=model.parameters(), config=ds_config)

print(f'RANK={os.environ.get("RANK","?")} LOCAL_RANK={os.environ.get("LOCAL_RANK","?")} '
      f'WORLD_SIZE={os.environ.get("WORLD_SIZE","?")} device={model_engine.device}')

for step in range(1, 6):
    x = torch.randn(4, 32, device=model_engine.device)
    loss = model_engine(x).sum()
    model_engine.backward(loss)
    model_engine.step()
    if model_engine.global_rank == 0:
        print(f'step {step}/5 loss={loss.item():.6f}')

print('Quick-start test PASSED')
EOF
```

```shell #test-setup store="npu_count"
python -c "import torch; print(torch.npu.device_count())"
```

### 6.1 单卡启动

使用 1 张 NPU 启动训练：

```shell #test id="launch-single"
deepspeed --num_gpus 1 train_minimal.py 2>&1 | tail -n 20
```

```shell #test-result id="launch-single" fuzzy='...'
...
Quick-start test PASSED
...
```

### 6.2 多卡启动

通过 `--num_gpus` 指定使用几张 NPU。下面的命令会先检测可用 NPU 数量：有 2 张及以上就用 2 张启动，否则自动退化为 1 张。

```shell #test id="launch-multi" load="npu_count>>NPU_COUNT"
NG=$(python -c "print(2 if int('<NPU_COUNT>') >= 2 else 1)"); echo "检测到 <NPU_COUNT> 张 NPU，将使用 $NG 张启动"; deepspeed --num_gpus $NG train_minimal.py 2>&1 | tail -n 30
```

```shell #test-result id="launch-multi" fuzzy='...'
检测到 ... 张 NPU，将使用 ... 张启动
...
Quick-start test PASSED
...
```

> 提示：多张 NPU 训练时，每个进程都会打印日志，输出会交错在一起，这是正常现象。

### 6.3 指定使用哪几张 NPU

如果你只想使用某几张 NPU（例如只用 0 号和 1 号卡），可以通过 **`ASCEND_RT_VISIBLE_DEVICES`** 环境变量控制。这与 CUDA 场景下的 `CUDA_VISIBLE_DEVICES` 作用相同，在昇腾上请使用前者。下面两条命令等价，都只让 0 号和 1 号 NPU 对当前训练可见：

```shell
ASCEND_RT_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2 train_minimal.py
deepspeed --include localhost:0,1 train_minimal.py
```

---

## 7. 多机训练

如果需要在多台服务器之间做分布式训练，DeepSpeed 提供了几种方式：

- **hostfile 模式**：通过 `--hostfile myhostfile` 指定节点列表（与 OpenMPI / Horovod 兼容），每行形如 `worker-1 slots=8`，可结合 `--num_nodes`、`--include`、`--exclude` 做资源过滤。
- **无 SSH 模式**：在 Kubernetes 等容器环境里，可使用 `--no_ssh --node_rank=<n> --master_addr=<addr> --master_port=<port>` 在每个节点上分别启动，行为类似 `torchrun`。
- **环境变量传播**：多机训练时通常需要把 `ASCEND_RT_VISIBLE_DEVICES`、CANN 的库路径等环境变量同步到所有节点，可通过 `.deepspeed_env` 文件（或 `DS_ENV_FILE` 指定路径）配置。
- **MPI 启动**：如果使用 `mpirun` 启动训练，请先安装 `mpi4py` 包（`pip install mpi4py`），DeepSpeed 会通过 mpi4py 自动获取进程编号和总进程数。

详细用法请参考官方 [Launching DeepSpeed Training](https://www.deepspeed.ai/getting-started/#launching-deepspeed-training)。
