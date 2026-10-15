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
- 能访问 Python 包源、GitHub、PyTorch CPU wheel 和官方 UMT 权重；
- 视频由仓库 cache-seed bundle 或预置缓存提供；两者都没有时，需要访问官方 Google Drive；
- 模型及视频缓存目录可写，建议预留至少 10 GB。

依赖安装、patch 和资源准备脚本位于本看护仓库的 `projects/vbench/`。
以下命令从该目录执行；使用 cache-seed 时需保留看护仓库的完整目录结构，
包括公共 `scripts/cache_seed.py` 和 `cache-seed/vbench/`。独立使用时设置
`UPSTREAM_REF` 为目标 tag、分支或 commit。CI 会注入共享模板解析出的最新 release
（当前为 `v0.1.4`）。补丁以最新 release 验证，不再单独维护 master 兼容分支。

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

patch 增加 CLI 的 `npu:0` 设备选择、NPU 单进程启动和 `human_action` 安装依赖集合；
保留 UMT 网络、16 帧采样、预处理、sigmoid、top-5 与 0.85 阈值。
官方 Python API 原本就透传设备，无需修改；共享分布式逻辑和 CUDA 启动行为也保持原样。
显式请求 NPU 而设备不可用时直接失败。单份补丁先通过 `git apply --check`；
未来 release 的源码或安装条件发生变化时，失败后审阅并更新补丁，不切换到旧版本。

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
LaVie 视频优先从 `cache-seed/vbench/manifest.yaml` 声明的仓库 bundle 投递，
复用公共 `scripts/cache_seed.py`，显式指定 VBench 缓存根目录而非默认 HF 缓存。
仅接受原始 `human_action` 的指定 5 段视频；投递时按 manifest 校验 SHA256，
已有文件校验通过则复用，损坏则从 bundle 恢复。manifest 不完整、源文件缺失或
校验失败会终止准备步骤，不回退到网络下载掩盖问题。

没有 bundle manifest 时保留原来的手动预置／官方下载方式：已有 5 段缓存则直接复用，
否则下载[官方 Google Drive 压缩包](https://github.com/Vchitect/VBench/tree/master/sampled_videos)，
只解压指定视频，保留文件名和 Prompt。
脚本从上游 `VBench_full_info.json` 选出对应原始记录，生成本次评测清单。

```shell #test id="prepare-assets"
set -euo pipefail
python -u scripts/prepare_assets.py vbench-src
```

```text #test-result id="prepare-assets"
...Original LaVie human_action videos: 5
```

CI 持久化 `/root/.cache/vbench`（宿主机 `/data/ci-cache/vbench`）；其中包括 `umt_model/`、
`lavie/human_action/` 中的 5 段原始视频，以及使用官方下载时产生的 `lavie.zip`。
bundle 随看护仓库 checkout 到达 Runner，不要求为 Runner 配置代理，也不要求
在 Runner 中放置 HF Token。权重仍按原来的方式下载并校验，视频 bundle 不包含权重。
当前看护仓库已包含这 5 段原始视频及生成的 manifest，SHA256 已与官方数据集
revision `8c89bb0218a12323d04821d11ec03c780d0dfb9c` 的 LFS 元数据核对。
bundle 的来源记录及重建流程见项目 README 的 cache-seed 章节。

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
`runtime_checks` 是看护侧验证工具，通过 PyTorch forward hook 观察真实计算，并检查
checkpoint 与模型的权重对应关系，不替换模型、输入或评分过程。评分可以为零，
测试不要求预设准确率；要求 5 次前向的模型、输入和输出都位于 NPU，输出有限，
并且结果保持上游 JSON 格式。

```python #test id="human-action-api"
import json
import math
import os
from pathlib import Path

import torch
import torch_npu
import vbench.human_action as human_action
from vbench import VBench
from scripts.prepare_assets import UMT_FILENAME
from scripts.runtime_checks import watch_human_action

my_VBench = VBench(
    torch.device("npu:0"),
    "vbench-src/human_action_full_info.json",
    "output/vbench/api",
)
cache = Path(os.environ.get("VBENCH_CACHE_DIR", Path.home() / ".cache/vbench"))
with watch_human_action(human_action, cache / "umt_model" / UMT_FILENAME) as observed:
    my_VBench.evaluate(
        videos_path="vbench-src/sampled_videos/lavie/human_action",
        name="lavie_human_action",
        dimension_list=["human_action"],
    )
torch.npu.synchronize()
assert observed == ["npu:0"] * 5, observed

for entry in ("cli", "api"):
    files = list(Path(f"output/vbench/{entry}").glob("*_eval_results.json"))
    assert len(files) == 1, (entry, files)
    score, videos = json.loads(files[0].read_text())["human_action"]
    assert math.isfinite(score) and 0 <= score <= 1
    assert len(videos) == 5
    assert len({v["video_path"] for v in videos}) == 5
    assert all(set(v) == {"video_path", "video_results", "cor_num_per_video"} for v in videos)
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
