# Quick Start (Ascend NPU)

在单张昇腾 NPU 上运行 [VBench 官方 README 的 human_action 示例](https://github.com/Vchitect/VBench#usage)，
验证安装、视频解码、UMT 动作识别以及 JSON 结果输出。本文使用官方 LaVie 视频中一个
原始 Prompt 的 5 段视频，是流程验证子集，不用于复现完整 VBench 排行榜分数。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品，至少一张可用 Ascend NPU，驱动与设备配置完好。
CI 使用 `linux-aarch64-a2-1`，Runner 自动提供一张 NPU。

### 基础软件

- Ubuntu 22.04 aarch64、Python 3.12、CANN toolkit 与驱动；
- CANN 环境已通过 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载；
- Git、C/C++ 编译器、CMake、FFmpeg 开发库，用于构建 CPU Decord；
- 能访问 Python 包源、GitHub、PyTorch CPU wheel、官方 UMT 权重和 Google Drive；
- 模型及视频缓存目录可写，建议预留至少 10 GB。

依赖安装、patch 和资源准备脚本位于本看护仓库的 `projects/vbench/`。
以下命令从该目录执行；独立使用时先获取该项目目录，再设置
`UPSTREAM_REF` 为目标 tag、分支或 commit。CI 会注入共享模板解析出的最新 release
（当前为 `v0.1.4`）。patch 同时对当前 `master` 做了应用检查。

### 本文档示例使用的版本

| 组件 | 版本 |
| --- | --- |
| Python / CANN | 3.12 / 9.1.0 |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 |
| torchvision | 0.24.0，CPU 构建，负责 CPU 预处理 |
| timm / numpy | 0.9.12 / 1.26.4 |
| Decord | v0.6.0，CPU 源码构建 |
| VBench | 工作流解析的最新 release 源码 + 本项目 Ascend patch |
| 评测器 | 官方 UMT `l16_ptk710_ftk710_ftk400_f16_res224.pth` |
| 样本 | 官方 LaVie `A person is strumming guitar`，原始索引 0～4 |

## 检查并安装运行环境

```shell #test id="check-python"
python --version
```

```text #test-result id="check-python"
Python 3.12...
```

先准备匹配的 PyTorch-NPU 栈；torchvision 从 PyTorch CPU 索引安装。它的图像预处理
在 CPU 上执行，UMT 前向计算在 NPU 上执行。

```shell #test id="install-torch"
set -euo pipefail
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu \
    "torch==2.9.0" "torch_npu==2.9.0.post2" "numpy==1.26.4" pyyaml Pillow setuptools wheel
PIP_EXTRA_INDEX_URL= python -m pip install --no-deps \
    --index-url https://download.pytorch.org/whl/cpu "torchvision==0.24.0"
python - <<'PY'
import torch
import torch_npu
import torchvision

assert torch.npu.is_available()
assert torch.npu.device_count() == 1
print("NPU available:", torch.npu.is_available())
print("torchvision:", torchvision.__version__)
PY
```

```text #test-result id="install-torch"
...NPU available: True
torchvision: 0.24.0...
```

## 获取上游并应用 Ascend patch

<!--
```shell #test-setup store="upstream_ref"
printf '%s\n' "${UPSTREAM_REF:?Set UPSTREAM_REF to the upstream ref}"
```
-->

```shell #test id="checkout-and-patch" load="upstream_ref>>ref"
set -euo pipefail
git init vbench-src
git -C vbench-src remote add origin https://github.com/Vchitect/VBench.git
git -C vbench-src fetch --depth 1 origin <ref>
git -C vbench-src checkout --detach FETCH_HEAD
python scripts/apply_patches.py vbench-src
git -C vbench-src diff --stat
echo "VBench Ascend patches applied"
```

```text #test-result id="checkout-and-patch"
...VBench Ascend patches applied
```

patch 增加 `npu:0` 设备选择、单进程评测、`human_action` 安装依赖集合和设备结果记录；
保留 UMT 网络、16 帧采样、预处理、sigmoid、top-5 与 0.85 阈值。
显式请求 NPU 而设备不可用时直接失败。上游增加了 CUDA 安装检查的版本还会应用对应补丁。
所有 patch 先通过 `git apply --check`，上游结构不匹配时停止并要求审阅补丁。

## 安装 VBench

Decord 0.6.0 没有官方 Linux aarch64 wheel，使用上游源码构建 CPU 解码器。
仅安装本例调用链所需依赖，避免引入其他评测器的 Detectron2 等依赖。

```shell #test id="install-vbench"
set -euo pipefail
bash scripts/install_decord.sh
VBENCH_INSTALL_PROFILE=human_action uv pip install --no-build-isolation \
    -e ./vbench-src "gdown==6.4.2"
python - <<'PY'
import decord
import vbench.human_action

print("Decord:", decord.__version__)
print("VBench human_action import: OK")
PY
vbench evaluate --help >/dev/null
```

```text #test-result id="install-vbench"
...Decord: 0.6.0
VBench human_action import: OK
```

## 准备官方权重和视频

UMT 权重从上游指定的 `OpenGVLab/VBench_Used_Models` 下载，固定 revision 并校验 SHA256。
LaVie 视频使用[官方分发的 Google Drive 压缩包](https://github.com/Vchitect/VBench/tree/master/sampled_videos)，
只解压原始 `human_action` 的指定 5 段视频，保留文件名和 Prompt。
脚本从上游 `VBench_full_info.json` 选出对应原始记录，生成本次评测清单。

```shell #test id="prepare-assets"
set -euo pipefail
python -u scripts/prepare_assets.py vbench-src
```

```text #test-result id="prepare-assets"
...Original LaVie human_action videos: 5
```

CI 持久化 `/root/.cache/vbench`（宿主机 `/data/ci-cache/vbench`）；其中包括 `umt_model/`、
`lavie.zip` 和 `lavie/human_action/` 中的 5 段原始视频。首次需要下载，后续复用缓存；
checkpoint 校验失败时重新下载。Hugging Face 的 LaVie 镜像需要数据集访问授权，
因此这里采用上游 README 公布的 Google Drive 来源。

Google Drive 可能触发下载配额限制。此时需提前把官方 `lavie.zip` 放入缓存根目录，
或者从有访问权限的官方 HF 数据集中取得这 5 段原始视频，放入
`/data/ci-cache/vbench/lavie/human_action/`。文件名必须为
`A person is strumming guitar-0.mp4` 至 `A person is strumming guitar-4.mp4`。
已有这 5 段缓存时不会访问 Google Drive。准备失败会让 CI 失败，不会跳过评测。

## 使用官方 CLI 进行 human_action 评测

在上游命令基础上添加 NPU 设备参数，并指定本次的原始 Prompt 子集清单。

```shell #test id="human-action-cli"
set -euo pipefail
vbench evaluate \
    --videos_path ./vbench-src/sampled_videos/lavie/human_action \
    --dimension human_action \
    --device npu:0 \
    --full_json_dir ./vbench-src/human_action_full_info.json \
    --output_path ./output/vbench/cli
```

```text #test-result id="human-action-cli"
...UMT device: npu:0
...done
```

## 使用官方 Python API 并验证结果

同一示例还验证 README 中的 Python API。两个入口使用同一组权重和视频。
评分可以为零，测试不要求预设准确率；要求原始视频确实经过 NPU 评测、输出有限分数，
并且每个结果都包含完整的 5 段视频。

```python #test id="human-action-api"
import json
import math
from pathlib import Path

import torch
import torch_npu
from vbench import VBench

my_VBench = VBench(
    torch.device("npu:0"),
    "vbench-src/human_action_full_info.json",
    "output/vbench/api",
)
my_VBench.evaluate(
    videos_path="vbench-src/sampled_videos/lavie/human_action",
    name="lavie_human_action",
    dimension_list=["human_action"],
)
torch.npu.synchronize()

for entry in ("cli", "api"):
    files = list(Path(f"output/vbench/{entry}").glob("*_eval_results.json"))
    assert len(files) == 1, (entry, files)
    score, videos = json.loads(files[0].read_text())["human_action"]
    assert math.isfinite(score) and 0 <= score <= 1
    assert len(videos) == 5
    assert len({v["video_path"] for v in videos}) == 5
    assert all(v["device"] == "npu:0" for v in videos)
    assert all(isinstance(v["video_results"], bool) for v in videos)
    print(f"{entry}: {len(videos)} videos evaluated on npu:0")
print("VBench human_action NPU Quick Start PASSED")
```

```text #test-result id="human-action-api"
...cli: 5 videos evaluated on npu:0
api: 5 videos evaluated on npu:0
VBench human_action NPU Quick Start PASSED
```

本文的支持范围是初代 VBench 的单卡 `human_action` 示例；其他维度、NPU 多卡、
VBench++ 与 VBench-2.0 需要各自验证。真实 NPU 是否通过，以看护工作流运行结果为准。
