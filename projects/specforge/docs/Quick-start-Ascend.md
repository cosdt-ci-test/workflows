# Quick Start (Ascend NPU)

在 4 卡昇腾 NPU 上把 [SpecForge](https://github.com/sgl-project/SpecForge) 端到端跑通：`pip install specforge` 满足上游 `pyproject.toml` 钉死的 `torch==2.11.0` / `sglang==0.5.14`，从 ModelScope 拉 `Qwen/Qwen3.5-4B`，起 `mooncake_master` + SGLang capture server + `specforge train` 三件套，跑 1 步训练作为 smoke。

## 前置条件

### 硬件

- **Atlas 900 A2 / A3 训练系列产品**或 **Ascend 950 系列产品**，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。
- **至少 4 卡**：本文 smoke 把 capture server 放卡 0、trainer 放卡 1，卡 2/3 留空。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 至少 1 张可见的 NPU 设备（`npu-smi info` 能看到 ≥ 4 卡）

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 4）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.11.0+cpu |
| torch_npu | 2.11.0 |
| sglang | 0.5.14（需 apply specforge 仓内的 capture 补丁） |
| specforge | 最新 release 的源码（>= #722 修 NPU 传输绑定） |
| modelscope | 1.37.0 |
| mooncake | main 分支 latest（master server 二进制需单独编译，smoke 阶段可用 pip 从源码装 transfer engine 部分） |
| 模型 | [Qwen/Qwen3.5-4B](https://www.modelscope.cn/Qwen/Qwen3.5-4B)（同时存在于 HF Hub；ModelScope 镜像同 ID） |
| 配方 | `examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml`（来自 specforge 源码仓） |

> 上游 `pyproject.toml` 把 `torch==2.11.0` / `transformers==5.8.1` / `sglang==0.5.14` 写死——本文档在装 specforge **之前**就装齐这三个版本，让 `pip install specforge`（无 `--no-deps`）满足依赖解析即可。

### 前置安装

确认能看到 ≥ 4 张 NPU 设备：`npu-smi info` 输出应至少列出 4 张 `910B4`，状态 OK。如果 `npu-smi` 不存在或 < 4 卡，回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动；本文档跑不动。

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

对齐 specforge 上游 pin 装 `torch` / `torch_npu` / `sglang`（cluster 镜像里 sglang 0.5.14 wheel 的 `Requires-Dist: cuda-python` 被重打包成 `<0` 当"exclude 哨兵"——uv 的 pubgrub 不识别这个模式，会去找 <0.0.0 的 cuda-python 找不到而报 unsatisfiable，所以 sglang 必须 `--no-deps`；specforge 跟上游 [ascend_npu.md](https://github.com/sgl-project/SpecForge/blob/main/docs/basic_usage/Ascend/ascend_npu.md) 一致也 `--no-deps`，避免再把 sglang 拉进来重解析）：

```shell #test-setup
uv pip install -f https://mirrors.aliyun.com/pytorch-wheels/cpu torch==2.11.0
uv pip install --extra-index-url https://mirrors.aliyun.com/pypi/simple torch_npu==2.11.0
# specforge 的依赖（无 CUDA 哨兵，正常装）
uv pip install transformers==5.8.1 datasets tqdm accelerate huggingface-hub numpy openai-harmony pydantic psutil pyyaml safetensors requests tensorboard typing-extensions wandb yunchang fastapi uvicorn aiohttp pyzmq python-multipart
# sglang 0.5.14 上游 requires_dist 里非 CUDA-only 项；里头 quack-kernels 自己带 nvidia-cutlass-dsl<0 哨兵，
# torch / numpy / pydantic / 等基础 dep 已经装好，整批 --no-deps 装。IPython 单独再装（不带 --no-deps）：
# specforge 的 scripts/apply_sglang_spec_capture_patch.sh 里 `python -c "import sglang; print(sglang.__version__)"`
# 会触发 sglang.__init__ → sglang.lang → IPython → traitlets 这条 import 链，
# 而 traitlets 是 IPython 的硬依赖，不在 sglang 自己的 requires_dist 里、也不会被 --no-deps 拉进来。
uv pip install --no-deps orjson anthropic apache-tvm-ffi av blobfile build compressed-tensors decord2 distro easydict einops gguf interegular kernels llguidance mistral_common msgspec ninja outlines packaging partial_json_parser pillow prometheus-client py-spy pybase64 quack-kernels scipy sentencepiece setproctitle sgl-deep-gemm starlette triton
uv pip install IPython
# sglang wheel 本身 --no-deps 装（cluster 镜像把它的 Requires-Dist cuda-python 改成 <0 哨兵，绕开解析）
uv pip install --no-deps --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi sglang==0.5.14
# openai<2.0.0 单独装，不带 --no-deps：>=2.0 切到了 pydantic 团队的 httpx2 fork（run 33268955540: `import httpx2._config`），
# 集群镜像没 httpx2、shim 也补不全 sub-module。openai 是 sglang server_args → openai.protocol → openai._models._utils
# 的硬依赖（sniffio / anyio / jiter / httpx 这些都从 openai 的 Requires-Dist 拉）；上一行 sglang --no-deps 没带它们，
# 所以这里让 openai 正常装来补齐 transitive deps。belt-and-suspenders：下面再显式装一次 sniffio / anyio / jiter / httpx，
# 万一 cluster 镜像里 openai<2.0.0 的 METADATA 被改过、Requires-Dist 不准，这些常被 openai 间接 import 的包也不会缺。
uv pip install 'openai<2.0.0'
uv pip install sniffio anyio jiter 'httpx<1'
# torchvision stub：sglang srt/utils/common.py line 92 `from torchvision.io import decode_jpeg` 在 import sglang 时硬依赖，
# 但 torchvision 顶层 __init__.py 跑 @torch.library.register_fake("torchvision::nms") 时会因 CPU torch 2.11.0 没注册该 op 而抛
# RuntimeError: operator torchvision::nms does not exist。Qwen3.5-4B 文本 smoke 不走 image path，stub 出 torchvision + torchvision.io
# 让 import 通过；decode_jpeg 不会被调用。另外 sglang.srt.configs.__init__ 直接 `from sglang.srt.configs.deepseekvl2 import DeepseekVL2Config`，
# deepseekvl2.py 顶部 `from torchvision.io import ImageReadMode`（PIL.ImageMode 风格 enum）→ run 33263935680
# 在 launch_server 启动早期就报 `ImportError: cannot import name 'ImageReadMode'`，把 ImageReadMode 也补上。
# 再加 torchvision.transforms.InterpolationMode（run 33265261220）：sglang.launch_server → server_args
# → configs/__init__ → deepseekvl2.py 顶部 `from transformers import (...)` → transformers 内部
# image_utils.py:55 `from torchvision.transforms import InterpolationMode` → ModuleNotFoundError →
# transformers 的 AutoProcessor lazy loader 报"Could not import module 'AutoProcessor'"（实际根因是 torchvision.transforms）。
# 还需 torchvision.transforms.functional 子模块（run 33266519990）：configs/__init__ 还导入 deepseek_ocr.py，
# 顶部 `from torchvision.transforms import functional as TF` → ModuleNotFoundError（functional 子模块不存在）。
# 顺便把 v2.functional 也 stub 上（sglang NPU 路径有 `import torchvision.transforms.v2.functional as tvF`，
# 文本 smoke 不走 VL 路径但 configs 链路 import 时可能引入）。transformers 5.8.1 还用 pil_to_tensor（image_utils.py:56），
# functional 里加个 raise NotImplementedError 占位。
python - <<'PY'
import os, site
sp = site.getsitepackages()[0]
pkg = os.path.join(sp, 'torchvision')
io = os.path.join(pkg, 'io')
tx = os.path.join(pkg, 'transforms')
txf = os.path.join(tx, 'functional')
txv2 = os.path.join(tx, 'v2')
txv2f = os.path.join(txv2, 'functional')
os.makedirs(io, exist_ok=True)
os.makedirs(txf, exist_ok=True)
os.makedirs(txv2f, exist_ok=True)
open(os.path.join(pkg, '__init__.py'), 'w').close()
open(os.path.join(io, '__init__.py'), 'w').write(
    'class ImageReadMode:\n'
    '    UNCHANGED = 0\n'
    '    GRAY = 1\n'
    '    RGB = 2\n'
    '\n'
    'def decode_jpeg(*args, **kwargs):\n'
    '    raise NotImplementedError("torchvision stub: not used in this text-only smoke")\n'
    '\n'
    'def decode_image(*args, **kwargs):\n'
    '    raise NotImplementedError("torchvision stub: not used in this text-only smoke")\n'
)
open(os.path.join(tx, '__init__.py'), 'w').write(
    'from torchvision.transforms import functional as _F  # re-export submodule\n'
    '\n'
    'class InterpolationMode:\n'
    '    NEAREST = "nearest"\n'
    '    NEAREST_EXACT = "nearest-exact"\n'
    '    BILINEAR = "bilinear"\n'
    '    BICUBIC = "bicubic"\n'
    '    BOX = "box"\n'
    '    HAMMING = "hamming"\n'
    '    LANCZOS = "lanczos"\n'
    '\n'
    'functional = _F\n'
)
# Sub-module so `from torchvision.transforms import functional` / `import torchvision.transforms.functional as TF` works.
open(os.path.join(txf, '__init__.py'), 'w').write(
    'class InterpolationMode:\n'
    '    NEAREST = "nearest"\n'
    '    NEAREST_EXACT = "nearest-exact"\n'
    '    BILINEAR = "bilinear"\n'
    '    BICUBIC = "bicubic"\n'
    '    BOX = "box"\n'
    '    HAMMING = "hamming"\n'
    '    LANCZOS = "lanczos"\n'
    '\n'
    'def pil_to_tensor(*args, **kwargs):\n'
    '    raise NotImplementedError("torchvision stub: not used in this text-only smoke")\n'
    '\n'
    'def resize(*args, **kwargs):\n'
    '    raise NotImplementedError("torchvision stub: not used in this text-only smoke")\n'
    '\n'
    'def center_crop(*args, **kwargs):\n'
    '    raise NotImplementedError("torchvision stub: not used in this text-only smoke")\n'
)
# torchvision.transforms.v2 也要 stub（v2/__init__.py 让 `from torchvision.transforms.v2 import functional` 能 import）。
open(os.path.join(txv2, '__init__.py'), 'w').write(
    'from torchvision.transforms.v2 import functional\n'
)
open(os.path.join(txv2f, '__init__.py'), 'w').write(
    'def __getattr__(name):\n'
    '    raise NotImplementedError(f"torchvision stub: torchvision.transforms.v2.functional.{name} not used in this text-only smoke")\n'
)
print(f'torchvision stub installed at {pkg}')
PY
```

检查 torch / torch_npu / sglang 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; from importlib.metadata import version; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('sglang', version('sglang')); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch" fuzzy='xxx'
torch= 2.11.0+cpu
torch_npu= 2.11.0
sglang xxx
is_available: True
count: 4
```

> 如果 `import torch_npu` 失败或 `count` 不是 4，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查三方兼容矩阵；`sglang` 必须有 `--attention-backend ascend` 支持（普通 PyPI 轮子不支持，需要 vendor 镜像或 NPU 编译产物）。

装 `modelscope`（走 ModelScope 镜像拉底座模型 + datasets）+ `mooncake` 传输引擎（master server 二进制留给生产环境，本文档 smoke 仅用其 Python 客户端路径）：

```shell #test-setup
uv pip install 'modelscope==1.37.0'
# mooncake-transfer-engine 是 specforge 在线训练里 specforge runtime 的 client 端；
# master server 二进制（mooncake_master / mooncake_client / transfer_engine_bench）必须
# 随 wheel 一起分发。main 分支的 mooncake-wheel/ setup.py 只编译 _fast_copy 扩展，
# 不带那三个预编译二进制 → mooncake_master 启动后 execv 找不到 binary，bind 失败，
# smoke 30s 后 nc -z 35551 全部 timeout。所以从 release tarball 拿预编译的 wheel。
#
# 用 tsinghua 镜像：直连 GitHub release 在集群网络下不稳（run 33254357756 90min timeout），
# aliyun 镜像只有 manylinux_2_39 aarch64（CI image 是 ubuntu22.04 glibc 2.35，跑不了 2.39 wheel），
# tsinghua 镜像有 v0.3.13 manylinux_2_28 aarch64 cp312 wheel（与 GitHub release 同字节）。
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'mooncake-transfer-engine==0.3.13'
```

打印安装版本：
```shell #test id="install-deps"
python -c "import modelscope; print('modelscope', modelscope.__version__)"
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
modelscope xxx
```

## 安装 specforge

### 从源码安装（拿到 `examples/configs/` 配方）

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证：

```shell #test id="specforge-install-source" load="upstream_ref>>ref"
if [[ "<ref>" =~ ^[0-9a-f]{40}$ ]]; then
    # commit SHA 路径：sgl-project/SpecForge 没有 release/tag，monitor 走 /commits/HEAD fallback 拿 main HEAD SHA；
    # git clone --branch 不接 SHA，先浅克隆再 fetch + checkout FETCH_HEAD。
    git clone --depth 1 https://github.com/sgl-project/SpecForge.git SpecForge
    git -C SpecForge fetch --depth 1 origin <ref>
    git -C SpecForge checkout FETCH_HEAD
else
    # tag / 分支名路径
    git clone --depth 1 --branch <ref> https://github.com/sgl-project/SpecForge.git SpecForge
fi
cd SpecForge
uv pip install --no-deps .
python -c "from importlib.metadata import version; print('specforge, version', version('specforge'))"
```

\<ref> 为最新的 release tag / 分支名 / commit SHA（监控自动 fallback）。

输出结果类似如下：

```shell #test-result id="specforge-install-source" fuzzy='xxx'
specforge, version xxx
```

> 从源码装是因为本文 smoke 脚本要拿 `examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml` 配方 + `scripts/apply_sglang_spec_capture_patch.sh` + `patches/sglang/v0.5.14/spec-capture-ascend-mount.patch`。PyPI 二进制 wheel 不会带 examples/ 与 patches/。

## CLI 自检

包导入自检先做一遍——`specforge` 在 NPU torch 栈上的模块加载在 install 之后立刻验证，省得 smoke 跑到 SGLang graph compile 才发现：

```shell #test id="specforge-import"
python -c "import specforge, torch, torch_npu; print('specforge', getattr(specforge, '__version__', 'unknown')); print('torch', torch.__version__); print('torch.npu.is_available', torch.npu.is_available())"
```

输出结果类似如下：

```shell #test-result id="specforge-import" fuzzy='xxx'
specforge xxx
torch 2.11.0+cpu
torch.npu.is_available True
```

`specforge --help` 列出子命令：

```shell #test id="specforge-help"
specforge --help
```

输出结果类似如下：

```shell #test-result id="specforge-help"
usage: specforge [-h] {train,export,benchmark} ...

positional arguments:
  {train,export,benchmark}
    train               train a draft model from a typed config
    export              materialize a runtime checkpoint as a model directory
    benchmark           benchmark a running SGLang server

options:
  -h, --help            show this help message and exit
```

`specforge train --help` 展示 typed run config 入口：

```shell #test id="specforge-train-help"
specforge train --help
```

输出结果类似如下：

```shell #test-result id="specforge-train-help"
usage: specforge train [-h] -c CONFIG
                       [--role {auto,all,producer,consumer,both}]
                       [--node-rank NODE_RANK] [--plan]
                       [overrides ...]

positional arguments:
  overrides             dotted overrides, e.g. training.learning_rate=1e-4

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        YAML or JSON run config
  --role {auto,all,producer,consumer,both}
                        launch selection (default: offline local all or
                        online/disaggregated producer+consumer)
  --node-rank NODE_RANK
                        node-local rank for an explicit multi-node trainer
                        launch
  --plan                print the resolved process plan without starting
                        workers
```

## 端到端 smoke：1 步训练

Smoke 把 `mooncake_master` / SGLang capture server / `specforge train` 串在同一个 `#test` 块里：装好 specforge 后整段执行，跑 1 步训练（~3 分钟，含 SGLang 首次 graph compile），`set -euo pipefail` + trap 自动清理后台进程。所有默认值通过 `SPECFORGE_*` 环境变量可覆盖。点开下面折叠块看完整命令：

<details>
<summary>展开看完整 smoke 命令（默认折叠）</summary>

```shell #test id="specforge-train-smoke"
set -euxo pipefail
PS4='+${LINENO}: '

# ---- Configuration (overridable via env) ----
MODEL_ID="${SPECFORGE_MODEL_ID:-Qwen/Qwen3.5-4B}"
RECIPE="${SPECFORGE_RECIPE:-examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml}"
SPECFORGE_ROOT="${SPECFORGE_ROOT:-SpecForge}"
CAPTURE_DEVICE="${SPECFORGE_CAPTURE_DEVICE:-0}"
TRAINER_DEVICE="${SPECFORGE_TRAINER_DEVICE:-1}"
SGLANG_PORT="${SPECFORGE_SGLANG_PORT:-30000}"
MOONCAKE_RPC_PORT="${SPECFORGE_MOONCAKE_RPC_PORT:-35551}"
MOONCAKE_HTTP_PORT="${SPECFORGE_MOONCAKE_HTTP_PORT:-35880}"
SGLANG_HEALTH_TIMEOUT="${SPECFORGE_SGLANG_HEALTH_TIMEOUT:-600}"  # 10 min for first compile

# ---- Cleanup trap ----
# pkill -f matches against /proc/<pid>/cmdline; the bash script's own
# cmdline IS the entire script body (it's run as `bash -c "<script>"`),
# so unanchored patterns like "sglang.launch_server" / "mooncake_master"
# / "specforge train" all match bash itself and `pkill -9` SIGKILLs the
# smoke runner mid-script (rc=-9 at the first pkill — that's exactly
# what killed run 33245161460 inside the stale-process sweep, before any
# actual smoke work). Anchor the regexes to ^cmdline: real processes
# start with `python -m sglang.launch_server` / `mooncake_master` /
# `specforge train`, while `bash -c "<script>"` starts with `bash` so
# none of the anchored patterns match the parent shell.
cleanup() {
    echo "smoke: cleanup"
    pkill -9 -f '^python -m sglang\.launch_server' 2>/dev/null || true
    pkill -9 -f '^mooncake_master' 2>/dev/null || true
    pkill -9 -f '^specforge train' 2>/dev/null || true
    rm -rf "$SPECFORGE_ROOT/outputs/qwen3.5-4b-dflash-npu-online" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- 0. Stale process sweep ----
cleanup

# ---- 1. SpecForge source present? ----
if [[ ! -d "$SPECFORGE_ROOT" ]]; then
    echo "smoke: FAILED - $SPECFORGE_ROOT/ missing; run \`git clone\` first"
    exit 1
fi

# ---- 2. Pre-download model from ModelScope ----
echo "smoke: downloading model $MODEL_ID from ModelScope"
MODEL_PATH=$(python -c "from modelscope import snapshot_download; print(snapshot_download('$MODEL_ID'))")
echo "smoke: model at $MODEL_PATH"

# ---- 3. Apply SGLang capture patches (online runs only) ----
echo "smoke: applying SGLang capture patches"
pushd "$SPECFORGE_ROOT" >/dev/null
if [[ -f scripts/apply_sglang_spec_capture_patch.sh ]]; then
    bash scripts/apply_sglang_spec_capture_patch.sh || echo "smoke: base patch already applied (ok)"
else
    echo "smoke: WARNING - scripts/apply_sglang_spec_capture_patch.sh missing; assuming already patched"
fi
# spec-capture-ascend-mount.patch（仓库自带 hunk1/hunk2 @@ line number 是写给 spec-capture.patch
# 在 a8c0993 之前的版本（彼时 setup() 没 rdma_devices/master_server_addr 两行），CI 装的 spec-capture.patch
# 已经是 a8c0993 之后版本 → 实际 spec_capture_sink.py 在 patch 锚点处多 6 行，BSD patch hunk2 直接
# `malformed patch at line 41`（line 100 → 实际 106，line 113 → 实际 119；用 --fuzz=10 也救不回来，
# BSD patch 在 @@ 处直接报 malformation 而不是 fallback 到 fuzzy match）。
#
# 改用 Python 字符串替换做相同改造：锚点字符串都用 spec-capture.patch 引入的多行 unique 子串，
# 不依赖行号；上游 spec-capture.patch 改 setup() 字段时不会让我们失锚。
# 用 `importlib.util.find_spec` 而不是 `import sglang`：sglang.__init__ 会拉 sglang.lang → IPython
# → traitlets；traitlets 不在 sglang 的 requires_dist 里、IPython 是 --no-deps 后单独装，这条链
# 上某环断就会 import 失败。find_spec 只查 module spec 不执行 __init__。
SGLANG_DIR=$(python -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('sglang').origin)))")
SINK_FILE="$SGLANG_DIR/sglang/srt/spec_capture_sink.py"
if [[ -f "$SINK_FILE" ]] && ! grep -q 'segment_to_mount' "$SINK_FILE"; then
    echo "smoke: applying ascend companion (inline python; upstream patch's hunk2 is malformed against current spec-capture.patch)"
    python3 - "$SINK_FILE" <<'PY'
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

# 1. 在 `store = MooncakeDistributedStore()` 之后、`rc = store.setup(` 之前，
#    插入 ascend-aware 的 segment / buffer / protocol 变量 + Ascend 检测。
old_anchor = (
    '            store = MooncakeDistributedStore()\n'
    '            rc = store.setup(\n'
)
new_anchor = (
    '            store = MooncakeDistributedStore()\n'
    '            global_segment_size = int(\n'
    '                os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", 1 << 30)\n'
    '            )\n'
    '            local_buffer_size = int(\n'
    '                os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", 1 << 30)\n'
    '            )\n'
    '            protocol = os.environ.get("MOONCAKE_PROTOCOL", "tcp")\n'
    '            # Ascend Mooncake rejects the wildcard location ("location:* is\n'
    '            # not supported"); skip it in setup() and mount with location="cpu".\n'
    '            ascend_host = bool(os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))\n'
    '            segment_to_mount = global_segment_size if ascend_host else 0\n'
    '            if ascend_host:\n'
    '                global_segment_size = 0\n'
    '                local_buffer_size = 0\n'
    '            rc = store.setup(\n'
)
assert old_anchor in src, "store.setup anchor not found (spec-capture.patch shape changed?)"
src = src.replace(old_anchor, new_anchor, 1)

# 2. setup() 里把三个 `int(os.environ.get(...))` / `os.environ.get(...)` 形参替换成上面定义的变量。
old_args = (
    '                global_segment_size=int(\n'
    '                    os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", 1 << 30)\n'
    '                ),\n'
    '                local_buffer_size=int(\n'
    '                    os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", 1 << 30)\n'
    '                ),\n'
    '                protocol=os.environ.get("MOONCAKE_PROTOCOL", "tcp"),\n'
)
new_args = (
    '                global_segment_size=global_segment_size,\n'
    '                local_buffer_size=local_buffer_size,\n'
    '                protocol=protocol,\n'
)
assert old_args in src, "setup() args anchor not found (spec-capture.patch shape changed?)"
src = src.replace(old_args, new_args, 1)

# 3. setup() 抛错之后插 mount_segment 调用（spec_capture_sink.py 里 if rc is not None 这条分支的紧后）。
old_post_setup = (
    '                raise RuntimeError(f"spec-capture mooncake setup failed (status {rc})")\n'
)
new_post_setup = (
    '                raise RuntimeError(f"spec-capture mooncake setup failed (status {rc})")\n'
    '            if segment_to_mount:\n'
    '                mount = getattr(store, "allocate_and_mount_segment", None)\n'
    '                if mount is None:\n'
    '                    raise RuntimeError(\n'
    '                        "Mooncake build on this Ascend host cannot register a "\n'
    '                        "wildcard segment and has no allocate_and_mount_segment; "\n'
    '                        "upgrade mooncake-transfer-engine"\n'
    '                    )\n'
    '                result = mount(segment_to_mount, protocol, "cpu")\n'
    '                mrc = result.get("ret", -1) if isinstance(result, dict) else result\n'
    '                if mrc is not None and int(mrc) != 0:\n'
    '                    raise RuntimeError(\n'
    '                        f"spec-capture mooncake mount segment failed (status {mrc})"\n'
    '                    )\n'
    '                logger.info(\n'
    '                    "spec-capture mooncake segment mounted with location=cpu "\n'
    '                    "(%d bytes)",\n'
    '                    segment_to_mount,\n'
    '                )\n'
)
assert old_post_setup in src, "raise RuntimeError after setup() not found (spec-capture.patch shape changed?)"
src = src.replace(old_post_setup, new_post_setup, 1)

with open(path, 'w') as f:
    f.write(src)
print("smoke: ascend companion edits applied (3 anchors)")
PY
fi
popd >/dev/null

# mooncake-transfer-engine v0.3.13 PyPI wheel 把 libasio/libgflags/libglog/libjsoncpp/liburing/
# libxxhash/libyaml-cpp/libzstd 八个库改名后打进 mooncake_transfer_engine.libs/，靠
# RPATH `$ORIGIN/../mooncake_transfer_engine.libs` 让 mooncake_master 找到；
# 但 libcurl4 / libibverbs1 / libnuma1 没打进 wheel，要走 apt。
# run 33259780290 直接跑 `mooncake_master` 报
# `error while loading shared libraries: libibverbs.so.1: cannot open shared object file`。
apt-get update -qq && apt-get install -y --no-install-recommends \
    libcurl4 libibverbs1 libnuma1

# ---- 4. Start mooncake_master ----
echo "smoke: starting mooncake_master"
nohup mooncake_master \
    --enable_http_metadata_server=true \
    --rpc_port=$MOONCAKE_RPC_PORT \
    --http_metadata_server_port=$MOONCAKE_HTTP_PORT \
    --metrics_port=35903 \
    --enable_metric_reporting=false \
    >/tmp/smoke-mooncake.log 2>&1 &
MOONCAKE_PID=$!
# CANN base image (ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12) 没装 nc(netcat)，
# `nc -z 127.0.0.1 35551` 直接 command-not-found → 30 次循环每次都 false → smoke
# 误判 mooncake 没 bind；run 33262609924 复现：mooncake_master 实际 log 已经
# `Master service started on port 35551` + `rpc_address=0.0.0.0`，但 nc 不存在。
# 改用 Python socket 检查；Python 3.12 在 py3.12 tag 的 CANN 镜像里一定有。
mooncake_ready() {
    python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
try:
    s.connect(('127.0.0.1', $MOONCAKE_RPC_PORT))
except Exception:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
" 2>/dev/null
}
for _ in $(seq 1 30); do
    if mooncake_ready; then
        echo "smoke: mooncake ready (rpc $MOONCAKE_RPC_PORT, pid=$MOONCAKE_PID)"
        break
    fi
    sleep 1
done
if ! mooncake_ready; then
    echo "smoke: FAILED - mooncake_master did not bind $MOONCAKE_RPC_PORT in 30s"
    tail -50 /tmp/smoke-mooncake.log
    exit 1
fi

# ---- 5. Start SGLang capture server on capture device ----
echo "smoke: starting SGLang capture server on device $CAPTURE_DEVICE"
ASCEND_RT_VISIBLE_DEVICES=$CAPTURE_DEVICE \
MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:$MOONCAKE_HTTP_PORT/metadata \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:$MOONCAKE_RPC_PORT \
MOONCAKE_PROTOCOL=tcp \
MOONCAKE_GLOBAL_SEGMENT_SIZE=$((32<<30)) \
nohup python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --trust-remote-code \
    --skip-tokenizer-init \
    --tp-size 1 \
    --mem-fraction-static 0.5 \
    --context-length 1024 \
    --attention-backend ascend \
    --enable-spec-capture --spec-capture-method dflash \
    --spec-capture-aux-layer-ids 1 8 15 22 29 \
    --host 127.0.0.1 --port "$SGLANG_PORT" \
    >/tmp/smoke-sglang.log 2>&1 &
SGLANG_PID=$!
echo "smoke: waiting for SGLang /health (up to ${SGLANG_HEALTH_TIMEOUT}s)"
HEALTH_DEADLINE=$((SGLANG_HEALTH_TIMEOUT / 5))
for _ in $(seq 1 "$HEALTH_DEADLINE"); do
    if curl -fsS "http://127.0.0.1:$SGLANG_PORT/health" >/dev/null 2>&1; then
        echo "smoke: sglang ready"
        break
    fi
    sleep 5
done
if ! curl -fsS "http://127.0.0.1:$SGLANG_PORT/health" >/dev/null 2>&1; then
    echo "smoke: FAILED - SGLang not healthy after ${SGLANG_HEALTH_TIMEOUT}s"
    tail -50 /tmp/smoke-sglang.log
    exit 1
fi

# ---- 6. Run specforge trainer (1 step) on trainer device ----
echo "smoke: running specforge trainer on device $TRAINER_DEVICE (1 step)"
pushd "$SPECFORGE_ROOT" >/dev/null
ASCEND_RT_VISIBLE_DEVICES=$TRAINER_DEVICE \
HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
specforge train -c "$RECIPE" \
    training.max_steps=1 \
    training.batch_size=1 \
    training.accumulation_steps=1 \
    training.max_length=512 \
    training.num_anchors=32 \
    training.save_interval=0 \
    training.log_interval=1 \
    deployment.trainer.nproc_per_node=1 \
    model.target_model_path="$MODEL_PATH" \
    2>&1 | tee /tmp/smoke-train.log
TRAIN_RC=${PIPESTATUS[0]}
popd >/dev/null

# ---- 7. Verify ----
echo "smoke: training exit=$TRAIN_RC"
tail -30 /tmp/smoke-train.log
if [[ $TRAIN_RC -ne 0 ]]; then
    echo "smoke: FAILED - specforge train exit=$TRAIN_RC"
    exit "$TRAIN_RC"
fi
if ! grep -qE "step.*loss|loss.*step|step N:|train_runtime" /tmp/smoke-train.log; then
    echo "smoke: FAILED - no step/loss output in train log"
    exit 1
fi

echo "smoke: OK - 1-step training completed"
```

</details>

输出结果类似如下（中间省略 SGLang graph compile / model load 的逐行日志）：

```shell #test-result id="specforge-train-smoke" fuzzy='xxx' fuzzy='...'
smoke: downloading model Qwen/Qwen3.5-4B from ModelScope
smoke: model at /root/.cache/modelscope/hub/Qwen/Qwen3.5-4B
smoke: applying SGLang capture patches
smoke: starting mooncake_master
smoke: mooncake ready (rpc 35551)
smoke: starting SGLang capture server on device 0
smoke: waiting for SGLang /health (up to 600s)
smoke: sglang ready
smoke: running specforge trainer on device 1 (1 step)
...
smoke: training exit=0
...
smoke: OK - 1-step training completed
```

> 卡 0 跑 capture server，卡 1 跑 trainer，卡 2/3 空闲给 HCCL buffer。Smoke 的 `--context-length 1024 --mem-fraction-static 0.5` 把 SGLang KV池压住（sglang 0.5.x 把 `--max-model-len` 改名成 `--context-length`，server_args.py `context_length` 字段），`training.max_steps=1 training.batch_size=1 training.max_length=512 training.num_anchors=32 deployment.trainer.nproc_per_node=1` 把训练侧压到 1 步最小数据。