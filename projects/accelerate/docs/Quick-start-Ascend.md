# 快速上手（昇腾 NPU）

在双卡昇腾 NPU 上跑通 [Accelerate](https://github.com/huggingface/accelerate) 的三大核心能力：

- **训练代码适配**（`acc-launch` / `acc-launch-bf16` / `acc-train-single` / `acc-prepare`）：多卡 DDP 训练 + 单卡训练 + 单卡推理三档场景都验证 `Accelerator.prepare` / `Accelerator.backward` 在 NPU 上跑通；
- **分布式评估**（`acc-infer-multi` / `acc-gather-multi`）：DDP 双进程下 forward-only 推理 + `gather_for_metrics` 真的跨卡 `all_gather`（hccl 后端），而不是退回单进程 identity；
- **大模型推理**（`acc-empty-weights`）：`init_empty_weights` 把大模型骨架以 meta 占位符方式创建，0 显存分配。

DDP 训练与跨卡集体通讯都依赖至少 2 张 NPU；单卡 runner 上 `accelerate launch --num_processes 2` 会直接报「visible devices 不够」。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

Accelerate 通过 `torch_npu` 间接支持昇腾 NPU（无需在 Accelerate 端引入额外后端），只要 `torch` / `torch_npu` 已就绪，Accelerate 的 `Accelerator` 类就能正确探测到设备并把张量放到 NPU 上。

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 2）
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
| transformers | `<5.0` |
| accelerate | 本仓库当前 main 分支源码（`uv pip install -e .`，对应 release tag 见下） |

### 检查前置是否满足

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
torch= xxx
torch_npu= xxx
is_available: True
count: xxx
```

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出如下类似结果：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
| 1     910B4               | OK            | 89.9        39                0    / 0             |
| 1                         | 0000:42:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 0                                                            |
| No running processes found in NPU 1                                                            |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动；如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

## 安装 accelerate

### 使用 uv 进行安装

通过 PyPI 镜像直接装最新 release 的二进制 wheel：

```shell #test id="acc-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple accelerate
python -c "import accelerate; print('accelerate', accelerate.__version__)"
```

输出结果类似如下：

```shell #test-result id="acc-install-binary" fuzzy='xxx'
accelerate xxx
```
- xxx 表示最新的版本号
<!--
```shell #test-setup
uv pip uninstall accelerate -y
```
-->

### 从源码安装

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 install 命令中-->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到最新 release tag，安装并且验证：

```shell #test id="acc-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/huggingface/accelerate.git
cd accelerate
uv pip install -e .
python -c "import accelerate; print('accelerate', accelerate.__version__)"
```
\<ref> 为安装的最新的 release tag

输出结果类似如下：

```shell #test-result id="acc-install-source" fuzzy='xxx'
accelerate xxx
```
- xxx 表示最新的版本号

> 如果机器在国内、无稳定外网，请参考 [huawei pypi 镜像](https://repo.huaweicloud.com/ascend/repos/pypi) 配置 `pip install -i ...`。

### CLI 自检：`accelerate env`

`accelerate env` 打印出当前环境对 PyTorch / 分布式后端 / 设备信息的探测结果。在昇腾 NPU 上跑应当能看到 `torch_npu` 已被识别：

```shell #test id="acc-env"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate env
```

输出结果类似如下：

```shell #test-result id="acc-env" fuzzy='xxx' fuzzy='...'
...
Copy-and-paste the text below in your GitHub issue

- `Accelerate` version: xxx
- Platform: xxx
- `accelerate` bash location: xxx
- Python version: xxx
- Numpy version: xxx
- PyTorch version: xxx
- PyTorch accelerator: NPU
- System RAM: xxx
- CANN version: xxx
...
```

其中 `PyTorch accelerator: NPU` 是 accelerate 探测到 `torch_npu` 后给出的标识；`CANN version` 一行只有在 NPU 环境才会出现。如果 `PyTorch accelerator` 不是 `NPU`，多半是 `torch_npu` 没被 import 到——回到「基础软件」一节检查。

## 训练代码适配

下面把上游 [Quicktour](https://github.com/huggingface/accelerate/blob/main/docs/source/quicktour.md) 里「训练代码适配」一节的训练循环压到最小，目标是验证 `Accelerator.prepare` / `Accelerator.backward` 在 NPU 上跑通。模型只用一个小线性层，但走的是 Accelerate 的全套适配路径。

### 第一步：写最小训练脚本

保存为 `train_npu.py`：

```shell #test-setup store="script_path"
cat > train_npu.py <<'PY'
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator


def build():
    # Toy regression: y = 2 * x + 1, learned by a single linear layer.
    x = torch.linspace(-1.0, 1.0, 64).unsqueeze(1)
    y = 2 * x + 1 + 0.05 * torch.randn_like(x)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    model = nn.Linear(1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    return loader, model, optim


def main():
    accelerator = Accelerator()
    loader, model, optim = build()
    model, optim, loader = accelerator.prepare(model, optim, loader)
    loss_fn = nn.MSELoss()
    for step, (xb, yb) in enumerate(loader):
        preds = model(xb)
        loss = loss_fn(preds, yb)
        accelerator.backward(loss)
        optim.step()
        optim.zero_grad()
        # 3 steps are enough to drive loss below 0.5 on this toy task.
        if step == 2:
            break
    # accelerator.print only emits on the main process, so the #test-result
    # below sees exactly one line regardless of --num_processes.
    accelerator.print(
        f"device={accelerator.device.type} "
        f"final_loss={loss.item():.4f}"
    )


if __name__ == "__main__":
    main()
PY
echo "${PWD}/train_npu.py"
```

### 第二步：用 `accelerate launch` 启动

```shell #test id="acc-launch" load="script_path>>path"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision no <path>
```

其中 `<path>` 是第一步生成的脚本绝对路径（即 `${PWD}/train_npu.py`）

输出结果类似：

```shell #test-result id="acc-launch" fuzzy='xxx'
device=npu final_loss=xxx
```

### 单卡训练

`accelerate launch` 不上场，直接 `Accelerator()` + 完整训练循环（forward + `accelerator.backward` + `optim.step`），验证单卡 NPU 上完整训练链路：

```shell #test id="acc-train-single"
python -c "
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

x = torch.linspace(-1.0, 1.0, 64).unsqueeze(1)
y = 2 * x + 1 + 0.05 * torch.randn_like(x)
ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=8, shuffle=True)
model = nn.Linear(1, 1)
optim = torch.optim.SGD(model.parameters(), lr=0.1)

accelerator = Accelerator()
model, optim, loader = accelerator.prepare(model, optim, loader)
loss_fn = nn.MSELoss()
for step, (xb, yb) in enumerate(loader):
    preds = model(xb)
    loss = loss_fn(preds, yb)
    accelerator.backward(loss)
    optim.step()
    optim.zero_grad()
    if step == 2:
        break
print(f'device={accelerator.device.type} final_loss={loss.item():.4f}')
"
```

输出结果如下（与 `acc-launch` 同一行格式；loss 受随机种子影响，用 `fuzzy='xxx'` 兜底）：

```shell #test-result id="acc-train-single" fuzzy='xxx'
device=npu final_loss=xxx
```

> 这是 `acc-launch` 的单卡对应——同一份训练逻辑，不走 `accelerate launch`、不依赖多卡，CI 单卡 runner 也能跑通完整 backward + optim.step 链路。和 `acc-prepare`（只跑 forward 不算 backward）形成互补。

### 独立 `Accelerator.prepare`（非分布式）

`accelerate launch` 不上场，直接 `Accelerator()` + `accelerator.prepare(model)` 跑一次 forward，验证 Accelerate 在 NPU 上：

1. **设备探测**：`Accelerator()` 自动识别 `torch_npu`、拿到 `device='npu:0'`；
2. **NPU 放置**：`accelerator.prepare(model)` 在非 distributed 上下文只做 `model.to(self.device)`，权重真在 NPU 上分配；
3. **真 kernel 跑**：`prepared(x)` 在 `npu:0` 上跑一次 1×1 matmul + bias add，**不是只 alloc**——NPU kernel 真跑了一次。

```shell #test id="acc-prepare"
python -c "
import torch
from torch import nn
from accelerate import Accelerator

accelerator = Accelerator()
model = nn.Linear(1, 1)
prepared = accelerator.prepare(model)
# prepare moves the model to the device, NOT your inputs —
# data tensors must be placed explicitly.
x = torch.tensor([[0.5]], device=accelerator.device)
y = prepared(x)
print(f'device={y.device.type}')
print(f'shape={list(y.shape)}')"
```

输出结果如下：

```shell #test-result id="acc-prepare"
device=npu
shape=[1, 1]
```

`accelerator.prepare(model)` 这一行是 Accelerate 的核心 abstraction——同一份脚本不写一行 `.cuda()` / `.npu()`，能自动适配 CPU / CUDA / NPU / XPU / MPS。分布式路径下的 `gather_for_metrics` 在下一节「分布式评估」验。

### 启动 bf16 混合精度路径

把第一步的 `train_npu.py` 再跑一遍，但把 `--mixed_precision` 从 `no` 切到 `bf16`，验证 Accelerate 的 autocast 包装在 NPU 上不出错。脚本不动一行——Accelerate 自动包 `torch.autocast`（`<path>` 仍是第一步的 `train_npu.py`）：

```shell #test id="acc-launch-bf16" load="script_path>>path"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 <path>
```

输出结果如下（与 `acc-launch` 同一行格式；`final_loss` 因为 bf16 精度差异可能与 `acc-launch` 不完全一致，断言只断言「不崩 + 落点是 npu」）：

```shell #test-result id="acc-launch-bf16" fuzzy='xxx'
device=npu final_loss=xxx
```

## 分布式评估

把上游 [Quicktour](https://github.com/huggingface/accelerate/blob/main/docs/source/quicktour.md)「分布式评估」一节拆成两块：「分布式推理」先在 DDP 双进程里跑一次 forward-only 推理（不训练、不算 metric），下一节 `acc-gather-multi` 再加 `gather_for_metrics` 跨卡汇总。

### 分布式推理（仅 forward）

`accelerate launch --num_processes 2` 拉起双进程，每个 rank 独立 forward 一个本地 batch——没有 loss、没有 backward、没有 `optim.step`。这是 inference 路径与训练路径（`acc-launch`）的关键分界：Accelerate 的 DDP 容器只走 `forward`，不触发梯度同步。

注意 `accelerate launch` 只接受脚本文件路径、不支持 `python -c` 内联代码，所以先落盘再启动：

```shell #test-setup store="infer_script_path"
cat > infer_npu.py <<'PY'
import torch
from torch import nn
from accelerate import Accelerator

accelerator = Accelerator()
model = nn.Linear(1, 1)
prepared = accelerator.prepare(model)
# Forward only — no loss, no backward, no optim.step
x = torch.tensor([[0.5], [1.0], [-1.0]], device=accelerator.device)
preds = prepared(x)
# accelerator.print only emits on the main process, so the #test-result
# below sees exactly one line regardless of --num_processes.
accelerator.print(
    f"world={accelerator.num_processes} "
    f"device={preds.device.type} "
    f"shape={list(preds.shape)}"
)
PY
echo "${PWD}/infer_npu.py"
```

```shell #test id="acc-infer-multi" load="infer_script_path>>path"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 <path>
```

`<path>` 同样是上方 setup 块捕获的 `${PWD}/infer_npu.py` 绝对路径，由 runner 自动代入；手动跑时替换为实际路径。

输出结果如下（两个 rank 各自 forward 本地 3 个样本，shape 仍是 `[3, 1]`，但 accelerator.print 只让主进程输出，所以 #test-result 只看到一行）：

```shell #test-result id="acc-infer-multi"
world=2 device=npu shape=[3, 1]
```

> inference 路径与训练路径（`acc-launch` / `acc-train-single`）的区别是：没有 `loss` / `backward` / `optim.step`。DDP 容器在 inference 时只做 `forward`，梯度同步那一步直接跳过。下一节 `acc-gather-multi` 在这个基础上加 `gather_for_metrics`，把各 rank 的结果汇总到主进程——这才是「分布式评估」的完整语义。

### 跨卡 `gather_for_metrics`（DDP 双进程）

在 `acc-infer-multi` 的基础上加 `gather_for_metrics`，验证它在 NPU 上真的跨卡做 `all_gather`（hccl 后端）而不是退回 identity：

```shell #test-setup store="gather_script_path"
cat > gather_npu.py <<'PY'
import torch
from accelerate import Accelerator

accelerator = Accelerator()
x = torch.tensor([1, 2, 3], device=accelerator.device)
(gathered,) = accelerator.gather_for_metrics((x,))
# accelerator.print only emits on the main process — without this guard
# the #test-result below would see two copies of each line.
accelerator.print(f'world={accelerator.num_processes}')
accelerator.print(f'device={gathered.device.type}')
accelerator.print(f'gathered={gathered.tolist()}')
PY
echo "${PWD}/gather_npu.py"
```

```shell #test id="acc-gather-multi" load="gather_script_path>>path"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 <path>
```

`<path>` 同样是上方 setup 块捕获的 `${PWD}/gather_npu.py` 绝对路径，由 runner 自动代入；手动跑时替换为实际路径。

输出结果如下（两个 rank 都喂 `[1, 2, 3]`，all_gather 沿 dim=0 串成 `[1, 2, 3, 1, 2, 3]`，长度 6 = `world * 3`）：

```shell #test-result id="acc-gather-multi" fuzzy='...'
world=2
device=npu
gathered=[1, 2, 3, 1, 2, 3]
```

> 这条才是真正「跨卡 collective 跑通」的可执行断言——单进程路径下 `gather_for_metrics` 文档化保证退化为 identity，要看 `all_gather` 必须 ≥2 张 NPU。多进程路径（`pad_across_processes` / 跨卡 `gather`）详见上游 [分布式评估](https://huggingface.co/docs/accelerate/basic_tutorials/evaluation) 教程。

## 大模型推理

把上游 [Quicktour](https://github.com/huggingface/accelerate/blob/main/docs/source/quicktour.md)「大模型推理」一节的「空权重初始化」抽出来用最小模型跑一遍。

### 空权重初始化

```shell #test id="acc-empty-weights"
python -c "
from transformers import LlamaConfig, LlamaForCausalLM
from accelerate import init_empty_weights

# Inline config: 1 层 / 1 head / hidden=64 —— 没有 hub 网络依赖。
config = LlamaConfig(
    vocab_size=32,
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=1,
    max_position_embeddings=64,
)
with init_empty_weights():
    model = LlamaForCausalLM(config)
n_params = sum(p.numel() for p in model.state_dict().values())
first_dev = next(model.parameters()).device
print(f'empty_model_params={n_params}')
print(f'device={first_dev}')
"
```

输出结果如下（参数数固定、device 固定为 meta —— 用 `fuzzy='xxx'` 让版本无关的字段差异不会挂）：

```shell #test-result id="acc-empty-weights" fuzzy='xxx'
empty_model_params=xxx
device=meta
```

`device=meta` 是 `init_empty_weights` 的合同行为：参数不在真实设备上，而是 PyTorch 的 meta 占位符，所以这一步**不占 NPU 显存**，CI 上跑稳。

> 「Load and dispatch weights」这一半需要真权重（Mixtral-8x7B 之类 ~90 GB），CI 不试——文档正文里链接到上游 [Big model inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference) 教程。
