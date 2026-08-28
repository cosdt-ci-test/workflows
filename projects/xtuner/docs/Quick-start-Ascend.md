# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑通 [xtuner](https://github.com/InternLM/xtuner) 的最小链路。


## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)，按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）

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
| xtuner | GitHub 最新 stable release（当前 `v0.2.0`，2025-07-11；`v1.0.1` 是 prerelease，引擎的 fallback 链按 release > prerelease > tag 解析，会优先选 stable） |

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

```shell #test-result id="check-torch" fuzzy='xxx'
torch= 2.9.0xxx
torch_npu= 2.9.0.post2
is_available: True
count: 1
```

> 如果 `import torch_npu` 失败，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查 torch / torch_npu / CANN 三方兼容矩阵。

## 安装 xtuner

xtuner 同时支持 PyPI 二进制安装与源码安装。

### 使用 uv 进行安装（PyPI 二进制）

```shell #test id="xtuner-install-binary"
uv pip install --index-url https://mirrors.aliyun.com/pypi/simple --no-deps xtuner
uv pip install 'mmengine==0.11.0rc2' 'transformers==5.2.0' 'peft>=0.14.0' 'datasets>=3.2.0,<4.0.0' einops loguru openpyxl 'scikit-image' scipy SentencePiece tiktoken transformers_stream_generator cyclopts 'opencv-python-headless<=4.12.0.88' timm pyarrow pydantic tensorboard xxhash imageio 'py-libnuma' GitPython
python -c "import xtuner; from xtuner.version import __version__; print('xtuner', __version__)"
```

输出结果类似如下：

```shell #test-result id="xtuner-install-binary" fuzzy='xxx'
xtuner xxx
```
- xxx 表示最新的版本号

<!--
```shell #test-setup
uv pip uninstall xtuner -y
```
-->

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证：

```shell #test id="xtuner-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/InternLM/xtuner.git
cd xtuner
uv pip install --no-deps -e .
uv pip install 'mmengine==0.11.0rc2' 'transformers==5.2.0' 'peft>=0.14.0' 'datasets>=3.2.0,<4.0.0' einops loguru openpyxl 'scikit-image' scipy SentencePiece tiktoken transformers_stream_generator cyclopts 'opencv-python-headless<=4.12.0.88' timm pyarrow pydantic tensorboard xxhash imageio 'py-libnuma' GitPython
python -c "import xtuner; from xtuner.version import __version__; print('xtuner', __version__)"
```
\<ref> 为安装的最新的 release tag。

输出结果类似如下：

```shell #test-result id="xtuner-install-source" fuzzy='xxx'
xtuner xxx
```

## 导入校验

源码装好后做一次 `importlib.util.find_spec` 烟囱测试，验证顶层包、v1.0.1 已有的子模块路径、还有 V1 训练入口都能被解析：

```shell #test id="xtuner-import-check"
python -c "
import importlib.util as u
for mod in ['xtuner', 'xtuner.dataset', 'xtuner.model', 'xtuner.engine', 'xtuner.v1', 'xtuner.v1.train.cli.sft']:
    spec = u.find_spec(mod)
    print(f'{mod}: {\"ok\" if spec is not None else \"MISSING\"}')
"
```

输出结果如下（用 `disable_fuzzy` 关掉默认非贪婪通配，按字面比对 — 任何一个 `MISSING` 都说明源码树装缺了）：

```shell #test-result id="xtuner-import-check" disable_fuzzy
xtuner: ok
xtuner.dataset: ok
xtuner.model: ok
xtuner.engine: ok
xtuner.v1: ok
xtuner.v1.train.cli.sft: ok
```

CLI 表面 sanity check——`xtuner.entry_point.MODES` 是 v0.2.0 legacy 训练入口硬编码的子命令列表，源码装好后这个常量必须等于（顺序也保持）：

```shell #test id="xtuner-cli-modes"
python -c "
from xtuner.entry_point import MODES
print('modes:', ' '.join(MODES))
"
```

输出结果如下（`MODES` 是上游硬编码的子命令列表，按字面比对 — 顺序也得保持）：

```shell #test-result id="xtuner-cli-modes" disable_fuzzy
modes: list-cfg copy-cfg log-dataset check-custom-dataset train test chat convert preprocess mmbench eval_refcoco
```

## LLM 大模型微调

### 写 NPU 训练 config

```shell #test-setup store="xtuner_llm_cfg_path"
cat > /tmp/xtuner_npu_llm_cfg.py <<'PY'
"""xtuner V1 LLM SFT smoke config for Ascend NPU.

"""
from xtuner.v1.config import AdamWConfig, LRConfig
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Qwen3Dense8BConfig
from xtuner.v1.train import TrainerConfig


model_cfg = Qwen3Dense8BConfig(num_hidden_layers=3, hidden_size=512)

dataset_cfg = DatasetConfig(
    name="openai_sft",
    anno_path="tests/resource/openai_sft.jsonl",
    sample_ratio=1.0,
)
tokenize_fn_cfg = OpenaiTokenizeFunctionConfig(
    chat_template="qwen3",
    max_length=4096,
)
dataloader_cfg = DataloaderConfig(
    dataset_config_list=[
        {"dataset": dataset_cfg, "tokenize_fn": tokenize_fn_cfg},
    ],
    pack_max_length=4096,
    collator="sft_llm_collator",
)

optim_cfg = AdamWConfig(lr=1e-4, foreach=False)
lr_cfg = LRConfig(lr_type="constant", warmup_ratio=0)

trainer = TrainerConfig(
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_cfg,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024),
    global_batch_size=1,
    total_step=3,
    work_dir="/tmp/xtuner_sft_llm_out",
    dist_backend="npu:hccl",
)
PY
echo "/tmp/xtuner_npu_llm_cfg.py"
```

### 跑 `torchrun xtuner/v1/train/cli/sft.py`

```shell #test id="xtuner-llm-sft" load="xtuner_llm_cfg_path>>cfg"
# pipefail so torchrun's exit code (not tail's) is what the test framework
# sees — without it a backend / NCCL init / OOM failure would surface as
# an "output mismatch" instead of "command failed (rc=...)", which costs
# the entire traceback.
set -o pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
cd xtuner
ASCEND_RT_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 xtuner/v1/train/cli/sft.py \
  --config <cfg> 2>&1 | tail -20
```

其中 `<cfg>` 是 「写 NPU 训练 config」写入的 `/tmp/xtuner_npu_llm_cfg.py` 绝对路径

输出结果类似：

```shell #test-result id="xtuner-llm-sft" fuzzy='xxx'
[XTuner][RANK 0]xxxStep 1/3xxx
[XTuner][RANK 0]xxxStep 3/3xxx
```

## MLLM 多模态大模型微调

### 派生一份 NPU 训练 config

```shell #test-setup store="xtuner_mllm_cfg_path"
cat > /tmp/xtuner_npu_mllm_cfg.py <<'PY'
"""xtuner V1 MLLM SFT smoke config for Ascend NPU.

Imports the upstream ``examples/v1/config/sft_intern_s1_tiny_config.py``
(which is already a complete ``TrainerConfig`` but defaults to the
torch-distributed auto backend — useless on NPU) and overrides only the
three NPU-specific knobs: ``dist_backend="npu:hccl"`` to force HCCL
collectives, ``total_step=3`` to cap the run, and ``work_dir`` to
land under /tmp. Every other field — the Intern-S1 tiny model arch,
the dual ``pure_text`` + ``media`` dataset split, the
``intern_s1_vl_sft_collator``, the chunked CE loss — is left as the
upstream maintainers set it.
"""
import sys

from xtuner.v1.train import TrainerConfig

# xtuner/ was `cd`'d into by the run block before this config is
# loaded, so the relative import path below resolves against CWD.
sys.path.insert(0, "examples/v1/config")
import sft_intern_s1_tiny_config as upstream  # noqa: E402

assert isinstance(upstream.trainer, TrainerConfig), (
    f"upstream.trainer must be a TrainerConfig, got {type(upstream.trainer)}"
)
upstream.trainer.dist_backend = "npu:hccl"
upstream.trainer.total_step = 3
upstream.trainer.work_dir = "/tmp/xtuner_sft_mllm_out"

trainer = upstream.trainer
PY
echo "/tmp/xtuner_npu_mllm_cfg.py"
```

### 跑 `torchrun xtuner/v1/train/cli/sft.py`（MLLM）

```shell #test id="xtuner-mllm-sft" load="xtuner_mllm_cfg_path>>cfg"
# pipefail so torchrun's exit code (not tail's) is what the test framework
# sees — without it a backend / NCCL init / OOM failure would surface as
# an "output mismatch" instead of "command failed (rc=...)", which costs
# the entire traceback.
set -o pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_USE_HCCL=1
cd xtuner
ASCEND_RT_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 xtuner/v1/train/cli/sft.py \
  --config <cfg> 2>&1 | tail -30
```

其中 `<cfg>` 是 「写 NPU 训练 config」写入的 `/tmp/xtuner_npu_mllm_cfg.py` 绝对路径

输出结果类似：

```shell #test-result id="xtuner-mllm-sft" fuzzy='xxx'
Using toy tokenizer:xxx
[XTuner][RANK 0]xxxStep 1/3xxx
```