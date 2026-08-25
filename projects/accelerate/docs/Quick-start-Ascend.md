# Quick Start (Ascend NPU)

在双卡昇腾 NPU 上跑通 [Accelerate](https://github.com/huggingface/accelerate) 的两个核心能力：`accelerate launch` 启动入口，以及把一个最小训练脚本改造成 `Accelerator` 适配版。DDP 训练 (`acc-launch` / `acc-launch-bf16`) 与跨卡集体通讯 (`acc-gather-multi`) 都依赖至少 2 张 NPU；单卡 runner 上 `accelerate launch --num_processes 2` 会直接报「visible devices 不够」。

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
Python 3.xxx
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

### 从源码安装

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 install 命令中；markdown 渲染器会丢掉注释里全部内容，读者看不到这段代码，但 runner 仍然执行它并 store="upstream_ref" -->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证：

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

输出结果类似如下（`accelerate env` 的 stdout 起始段；开头会先打印一个空行，结尾的 `...` 覆盖 `Accelerate default config`——未生成默认配置时值为 `Not found`——以及后续字段）：

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

其中 `PyTorch accelerator: NPU` 是 accelerate 探测到 `torch_npu` 后给出的标识（`accelerate/utils/environment.py` 里把 `is_npu_available()` 命中时赋值为 `"NPU"`）；`CANN version` 一行只有在 NPU 环境才会出现，CUDA / XPU 等环境会有 `GPU type` / `XPU type` 等行代替。如果 `PyTorch accelerator` 不是 `NPU`，多半是 `torch_npu` 没被 import 到——回到「基础软件」一节检查。

### 空权重初始化（Big Model Inference）

把上游 [Quicktour](https://github.com/huggingface/accelerate/blob/main/docs/source/quicktour.md)「Big Model Inference」一节的 `init_empty_weights` 抽出来用最小模型跑一遍。配置直接 inline 写死，避免再向 HF Hub 拉一个 `from_pretrained` 的 config：

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

`device=meta` 是 `init_empty_weights` 的合同行为：参数不在真实设备上，而是 PyTorch 的 meta 占位符，所以这一步**不占 NPU 显存**，CI 上跑稳。`load_checkpoint_and_dispatch` 这一半需要真权重（Mixtral-8x7B 之类 ~90 GB），CI 不试——文档正文里链接到上游教程。

## 使用样例：最小 `Accelerator` 训练脚本

下面把上游 [Quicktour](https://github.com/huggingface/accelerate/blob/main/docs/source/quicktour.md) 里「Adapt training code」一节的训练循环压到最小，目标是验证 `Accelerator.prepare` / `Accelerator.backward` 在 NPU 上跑通。模型只用一个小线性层，但走的是 Accelerate 的全套适配路径。

### Step 1：写最小训练脚本

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

### Step 2：用 `accelerate launch` 启动

```shell #test id="acc-launch" load="script_path>>path"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision no <path>
```

输出结果类似：

```shell #test-result id="acc-launch" fuzzy='xxx'
device=npu final_loss=xxx
```

> `--num_processes 2 --mixed_precision no` 把 Accelerate 锁到「双卡 DDP、不走 AMP」模式——`Accelerator.prepare(model)` 自动包一层 `DistributedDataParallel`，每张卡拿 64 个样本的子集跑 SGD；`accelerator.device.type` 在两个 rank 上都是 `npu`。多卡请参考上游 [Launch distributed code](https://huggingface.co/docs/accelerate/basic_tutorials/launch) 教程，把 `num_processes` / `num_machines` / `gpu_ids` 等参数填对。

### Standalone `Accelerator.prepare`（非 distributed）

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

`accelerator.prepare(model)` 这一行是 Accelerate 的核心 abstraction——同一份脚本不写一行 `.cuda()` / `.npu()`，能自动适配 CPU / CUDA / NPU / XPU / MPS。DDP 训练在下一节 `acc-launch` 验，跨卡集体通讯在 `acc-gather-multi` 验。

### 跨卡 `gather_for_metrics`（DDP 双进程）

把 `gather_for_metrics` 放进 `accelerate launch` 拉起的双进程里跑，验证它在 NPU 上真的跨卡做 `all_gather`（hccl 后端）而不是退回 identity：

```shell #test id="acc-gather-multi"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 python -c "
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
"
```

输出结果如下（两个 rank 都喂 `[1, 2, 3]`，all_gather 沿 dim=0 串成 `[1, 2, 3, 1, 2, 3]`，长度 6 = `world * 3`）：

```shell #test-result id="acc-gather-multi" fuzzy='...'
world=2
device=npu
gathered=[1, 2, 3, 1, 2, 3]
```

> 这条才是真正「跨卡 collective 跑通」的可执行断言——单进程路径下 `gather_for_metrics` 文档化保证退化为 identity，要看 `all_gather` 必须 ≥2 张 NPU。多进程路径（`pad_across_processes` / 跨卡 `gather`）详见上游 [Distributed evaluation](https://huggingface.co/docs/accelerate/basic_tutorials/evaluation) 教程。

### 启动 bf16 混合精度路径

把玩具脚本再跑一遍，但把 `--mixed_precision` 从 `no` 切到 `bf16`，验证 Accelerate 的 autocast 包装在 NPU 上不出错。脚本不动一行——Accelerate 自动包 `torch.autocast`：

```shell #test id="acc-launch-bf16" load="script_path>>path"
ASCEND_RT_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision bf16 <path>
```

输出结果如下（与 `acc-launch` 同一行格式；`final_loss` 因为 bf16 精度差异可能与 `acc-launch` 不完全一致，断言只断言「不崩 + 落点是 npu」）：

```shell #test-result id="acc-launch-bf16" fuzzy='xxx'
device=npu final_loss=xxx
```

### Step 3：清理（可选）

玩具脚本 `train_npu.py` 是当前目录下的临时文件——CI 容器是 ephemeral 的，每次跑都是全新环境，不需要清理；如果你是在本地照着文档手动跑，结束时 `rm -f train_npu.py` 即可。

## 小贴士

- 如果你习惯 `torchrun`，Accelerate 也支持 `accelerate launch --use_torchrun` 走 PyTorch 原生弹性启动器。
- 在多卡机器上想验证 DDP 路径，把 `--num_processes` 调到 `torch.npu.device_count()`，脚本无需改一行——`Accelerator` 会自动包一层分布式容器。
- 需要 bf16 时把 `--mixed_precision bf16` 加上，并保证 `torch` / `torch_npu` 在昇腾侧支持 bf16 路径（参考 [NPU 最佳实践文档](https://github.com/modelscope/ms-swift/blob/main/docs/source/BestPractices/NPU-support.md)）。
- 如果只想做单机推理（不训练），`Accelerator.prepare` 同样能用：`model = accelerator.prepare(model)` 会把模型搬到 NPU。
