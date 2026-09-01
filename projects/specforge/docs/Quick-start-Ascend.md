# Quick Start (Ascend NPU)

在 4 卡昇腾 NPU 上把 [SpecForge](https://github.com/sgl-project/SpecForge) 端到端跑通：镜像预装 torch 2.10.0 + torch_npu 2.10.0 + sglang 0.5.18 + CANN 9.0.0 + Python 3.11.15，再装 modelscope 1.37.0 + mooncake-transfer-engine 0.3.13 + specforge 源码（`pip install --no-deps .`），从 ModelScope 拉 `Qwen/Qwen3.5-4B`，起 `mooncake_master` + SGLang capture server + `specforge train` 三件套，跑 1 步训练作为 smoke。

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

swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:v0.5.18-cann9.0.0-910b

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.15 |
| CANN | 9.0.0 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0 |
| sglang | 0.5.18 |
| specforge | 最新 release 的源码（>= #722 修 NPU 传输绑定） |
| modelscope | 1.37.0 |
| mooncake-transfer-engine | 0.3.13（PyPI tsinghua 镜像 manylinux_2_28 aarch64 cp312 wheel；mooncake_master 二进制跟着 wheel 走） |
| 模型 | [Qwen/Qwen3.5-4B](https://www.modelscope.cn/Qwen/Qwen3.5-4B) |
| 配方 | `examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml`（来自 specforge 源码仓） |

> 镜像里 torch 2.10.0 / torch_npu 2.10.0 / sglang 0.5.18 已经预装好了（`check-torch` 步输出可看）。specforge 上游 `pyproject.toml` 还钉 `sglang==0.5.14`，但 `pip install --no-deps .` 跳过依赖解析、运行时实际走镜像里的 sglang 0.5.18——spec-capture patch 也按 `--target v0.5.18` 走，所以版本是配套的。如果后续 SpecForge pin 变了，sglang 行要跟着镜像对齐。

### 前置安装

确认能看到 ≥ 4 张 NPU 设备：`npu-smi info` 输出应至少列出 4 张 `910B4`，状态 OK。如果 `npu-smi` 不存在或 < 4 卡，回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动；本文档跑不动。

检查 Python 版本：

```shell #test id="check-py"
python --version
```

输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.11.xxx
```

CANN toolkit 装好且 install.info 可读（后续 smoke 跑 SGLang / torch_npu 都靠 CANN env；ascend-toolkit 下 `latest/` 是软链到具体版本的子目录，`/usr/local/Ascend/ascend-toolkit/set_env.sh` 默认指向它）：

```shell #test id="check-cann"
test -f /usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/ascend_toolkit_install.info && \
    grep '^version=' /usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/ascend_toolkit_install.info || \
    echo "ascend_toolkit_install.info MISSING"
```

> 按[昇腾官方手册](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha3/softwareinstall/instg/atlas_03_0013.html)：CANN 版本号写在 `/usr/local/Ascend/ascend-toolkit/latest/<arch>-linux/ascend_toolkit_install.info` 的 `version=` 行。`$(uname -m)` 在 aarch64 runner 上展开成 `aarch64`（这是镜像里实际的架构）；arm64 / x86_64 镜像同样适用。
>
> 不走 `source set_env.sh`：典型 Ascend set_env.sh 在 `case $- in *i*)` 守护下跳过非交互式运行，`bash -c` 子 shell 里 $- 不带 i → ASCEND_HOME 落空（run 33464343161 复现）。

输出结果类似如下（`9.0.0` 是镜像 tag `cann9.0.0-910b` 标称的版本，钉住——以后 image bump CANN 时这里会立即报错提醒改文档）：

```shell #test-result id="check-cann"
version=9.0.0
```

检查 torch / torch_npu / sglang 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu; from importlib.metadata import version; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('sglang', version('sglang')); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch" fuzzy='xxx'
torch= 2.10.0+cpu
torch_npu= 2.10.0
sglang 0.5.18
is_available: True
count: 4
```

镜像预装的 torch/torch_npu/sglang 版本单独探一次（`check-torch` 输出混进了 patch 路径里的 echo，结构上不够清晰）：

```shell #test id="image-probe"
python -c "
import sys, torch, torch_npu
from importlib.metadata import version
print('python', sys.version.split()[0])
print('torch', torch.__version__)
print('torch_npu', torch_npu.__version__)
print('sglang', version('sglang'))
print('npu_available', torch.npu.is_available())
print('npu_count', torch.npu.device_count())
"
```

输出结果如下：

```shell #test-result id="image-probe" fuzzy='xxx'
python 3.11.xxx
torch xxx
torch_npu xxx
sglang xxx
npu_available True
npu_count 4
```

> `sglang` 实际版本看这里——spec-capture patch 的 target 跟着 sglang 行走（`--target v${SGLANG_VER}`）。当前镜像 `v0.5.18-cann9.0.0-910b` 里 sglang 是 **0.5.18**，跟 `apply_sglang_spec_capture_patch.sh` 的默认 target 一致，直接打 `patches/sglang/v0.5.18/` 下的两个 patch。

> 如果 `import torch_npu` 失败或 `count` 不是 4，回到 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch) 检查三方兼容矩阵；`sglang` 必须有 `--attention-backend ascend` 支持（普通 PyPI 轮子不支持，需要 vendor 镜像或 NPU 编译产物）。

装 `modelscope`+ `mooncake-transfer-engine`：

```shell #test-setup
uv pip install 'modelscope==1.37.0'
#
# 用 tsinghua 镜像：直连 GitHub release 在集群网络下不稳（run 33254357756 90min timeout），
# aliyun 镜像只有 manylinux_2_39 aarch64（CI image 是 ubuntu22.04 glibc 2.35，跑不了 2.39 wheel），
# tsinghua 镜像有 v0.3.13 manylinux_2_28 aarch64 cp311 wheel（与 GitHub release 同字节）。
uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'mooncake-transfer-engine==0.3.13'
```

> smoke 实际用的是 `mooncake_master` 二进制（sglang.spec_capture_sink.py 通过 `mooncake.store.MooncakeDistributedStore` 调用，绑定由 sglang 自带的 wheel 处理），`import mooncake_transfer_engine` 仅作 wheel 完整性的 sanity check。下面用 if/else 让它失败时也走 stdout 一行、不会因为异常走 stderr 而看不到。

打印安装版本：
```shell #test id="install-deps"
python -c "import modelscope; print('modelscope', modelscope.__version__)"
if python -c "import mooncake_transfer_engine" 2>/dev/null; then
    echo "mooncake-transfer-engine ok"
else
    echo "mooncake-transfer-engine not importable"
fi
test -x "$(command -v mooncake_master)" && echo "mooncake_master binary present" || echo "mooncake_master binary MISSING"
```

输出结果如下：

```shell #test-result id="install-deps" fuzzy='xxx'
modelscope xxx
xxx
mooncake_master binary present
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

> 从源码装是因为本文 smoke 脚本要拿 `examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml` 配方 + `scripts/apply_sglang_spec_capture_patch.sh` + `patches/sglang/v0.5.18/spec-capture-ascend-mount.patch`。PyPI 二进制 wheel 不会带 examples/ 与 patches/。

## CLI 自检

包导入自检先做一遍——`specforge` 在 NPU torch 栈上的模块加载在 install 之后立刻验证，省得 smoke 跑到 SGLang graph compile 才发现：

```shell #test id="specforge-import"
python -c "import specforge, torch, torch_npu; print('specforge', getattr(specforge, '__version__', 'unknown')); print('torch', torch.__version__); print('torch.npu.is_available', torch.npu.is_available())"
```

输出结果类似如下：

```shell #test-result id="specforge-import" fuzzy='xxx'
specforge xxx
torch 2.10.0+cpu
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

Smoke 把 `mooncake_master` / SGLang capture server / `specforge train` 串起来：装好 specforge 后逐段执行，跑 1 步训练（~3 分钟，含 SGLang 首次 graph compile），每段是独立的 `#test` 块，失败时定位到具体阶段。所有默认值通过 `SPECFORGE_*` 环境变量可覆盖。

| Step | 块 id | 干啥 |
| --- | --- | --- |
| 1 | `smoke-download-model` | 从 ModelScope 拉 `Qwen/Qwen3.5-4B`，把模型路径存入 store `model_path` |
| 2 | `smoke-apply-patches` | 跑 `apply_sglang_spec_capture_patch.sh` + 内联 ascend companion + `apt-get install libcurl4/libibverbs1/libnuma1` |
| 3 | `smoke-start-mooncake` | `nohup mooncake_master`，socket 探活 35551 |
| 4 | `smoke-start-sglang` | `nohup sglang.launch_server`，curl `/health` 探活 30000 |
| 5 | `smoke-train` | `specforge train ... max_steps=1`，断言 stdout 出现 `step.*loss` |

> 每段独立失败：step 2 报 patch 失锚时 CI 在 `smoke-apply-patches` 红、不会继续起 sglang。mooncake_master / sglang.launch_server 是 nohup 后台进程，前一段 setup 返回 0 后下一个 `#test` 块直接复用——`pkill -f '^mooncake_master'` / `pkill -f '^python -m sglang\.launch_server'` 在每段入口先扫一遍，避免上次残留端口占用。

### Step 1：预下载模型

```shell #test-setup id="smoke-download-model" store="model_path"
set -euo pipefail
MODEL_ID="${SPECFORGE_MODEL_ID:-Qwen/Qwen3.5-4B}"
# store="model_path" 抓的是整个 stdout，必须只让路径落 stdout。echo 走 stderr 做诊断，
# modelscope 自身的下载进度 ('Downloading Model from ... to directory: ...') 也走
# stdout —— run 33507844975 复现：modelscope progress + print(path) 两行都被 store
# 抓住，<MODEL_PATH> 替换出 200+ 字节多行字符串，--model-path 收到垃圾。
# contextlib.redirect_stdout(sys.stderr) 在 with 块内把 sys.stdout 指向 stderr，
# modelscope 进度落 stderr；with 块外的 print(path) 走真 stdout，只有路径被 store
# 抓住。stderr 的进度仍会出现在 CI log 里做诊断。
echo "smoke: downloading model $MODEL_ID from ModelScope" >&2
python - "$MODEL_ID" <<'PY'
import sys, contextlib
MODEL_ID = sys.argv[1]
with contextlib.redirect_stdout(sys.stderr):
    from modelscope import snapshot_download
    path = snapshot_download(MODEL_ID)
print(path)
PY
```

输出结果类似如下（stdout 只有路径，被 store 进 `model_path`，后续 step 4 / 5 用 `<MODEL_PATH>` 引用；stderr 的 echo 不会出现在这里）：

```shell #test-result id="smoke-download-model" load="model_path>>MODEL_PATH"
<MODEL_PATH>
```

### Step 2：打补丁 + apt 依赖

镜像里 sglang 实际是 0.5.18（看 step `image-probe`），`apply_sglang_spec_capture_patch.sh` 默认 target 就是 `v0.5.18`，这里显式传 `--target v${SGLANG_VER}` 以便 image 以后 bump sglang 时自动跟上。脚本会从 specforge 源码仓拉 `patches/sglang/v0.5.18/spec-capture.patch` + `patches/sglang/v0.5.18/spec-capture-ascend-mount.patch`（前者改 `sglang/srt/spec_capture_sink.py` 加 `allocate_and_mount_segment` 等字段，后者再加 ascend 段挂载适配）。Hunk2 行号跟上游 a8c0993 之后版本对不上（BSD patch 直接 `malformed patch at line 41`），ascend companion 用 Python 字符串替换做（不依赖行号），见块里 heredoc。

SpecForge 上游 2026-08-29 退掉了 v0.5.14 patch（commit `b453386827`）；如果以后 image 把 sglang 倒回 0.5.14，这条 step 会 rc!=0 立即红——届时把 sglang 重新钉到 0.5.18、或 checkout `b453386827` 之前的 SpecForge commit，二选一。

```shell #test id="smoke-apply-patches"
set -euo pipefail
SPECFORGE_ROOT="${SPECFORGE_ROOT:-SpecForge}"
if [[ ! -d "$SPECFORGE_ROOT" ]]; then
    echo "smoke: FAILED - $SPECFORGE_ROOT/ missing; specforge-install-source first"
    exit 1
fi
pushd "$SPECFORGE_ROOT" >/dev/null
SGLANG_VER=$(python -c "from importlib.metadata import version; print(version('sglang'))")
echo "smoke: applying spec-capture patches for sglang ${SGLANG_VER}"
# Run base patch (脚本自己的 success msg + ascend companion / apt-get install 的几十行输出都重定向到 /tmp/smoke-*.log，
# test stdout 只留 echo 的总结行——run 33467285504 在 stdout 60+ 行下 expected 2 行 MISMATCH)
if [[ -f scripts/apply_sglang_spec_capture_patch.sh ]]; then
    # 走 --target v${SGLANG_VER}；image 现在 sglang=0.5.18 → v0.5.18 patch 直打。
    bash scripts/apply_sglang_spec_capture_patch.sh --target "v${SGLANG_VER}" >/tmp/smoke-patch.log 2>&1 || {
        echo "smoke: FAILED - apply_sglang_spec_capture_patch.sh --target v${SGLANG_VER} returned non-zero" >&2
        tail -30 /tmp/smoke-patch.log >&2
        exit 1
    }
else
    echo "smoke: WARNING - apply_sglang_spec_capture_patch.sh missing; assuming already patched"
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
    if ! python - "$SINK_FILE" <<'PY' >/tmp/smoke-ascend.log 2>&1
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
PY
    then
        echo "smoke: FAILED - ascend companion python failed" >&2
        tail -30 /tmp/smoke-ascend.log >&2
        exit 1
    fi
fi
popd >/dev/null

# mooncake-transfer-engine v0.3.13 PyPI wheel 把 libasio/libgflags/libglog/libjsoncpp/liburing/
# libxxhash/libyaml-cpp/libzstd 八个库改名后打进 mooncake_transfer_engine.libs/，靠
# RPATH `$ORIGIN/../mooncake_transfer_engine.libs` 让 mooncake_master 找到；
# 但 libcurl4 / libibverbs1 / libnuma1 没打进 wheel，要走 apt。
# run 33259780290 直接跑 `mooncake_master` 报
# `error while loading shared libraries: libibverbs.so.1: cannot open shared object file`。
apt-get update -qq >/dev/null 2>&1
apt-get install -qq -y --no-install-recommends \
    libcurl4 libibverbs1 libnuma1 >/dev/null 2>&1

# 防御性 verify：base patch 必须在 server_args.py 引入 enable_spec_capture /
# spec_capture_aux_layer_ids / spec_capture_method 三个字段，launch_server 的
# argparse 才能识别 --enable-spec-capture 这一组 CLI flag。run 33493594121 复现：
# apply 脚本 stdout 'spec-capture patch v0.5.18 applied at
# /sgl-workspace/sglang/python/sglang'——但同路径 grep 找不到字段。怀疑
# `git -C $SGL_PARENT apply` 在 site-packages（非 git 仓）下报成功但
# server_args.py hunk 没真正落盘。先 dump 现场，再 fallback 到 Python 字
# 符串替换直接插字段（沿用 ascend companion 的不依赖 git 的稳健路径）。
SGLANG_DIR=$(python -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('sglang').origin)))")
SERVER_ARGS="$SGLANG_DIR/sglang/srt/server_args.py"
if ! grep -q 'enable_spec_capture: A\[' "$SERVER_ARGS" \
   || ! grep -q 'spec_capture_aux_layer_ids: A\[' "$SERVER_ARGS" \
   || ! grep -q 'spec_capture_method: A\[' "$SERVER_ARGS"; then
    echo "smoke: FAILED - spec-capture patch did not add enable_spec_capture fields to $SERVER_ARGS" >&2
    echo "smoke: diagnostic dump:" >&2
    echo "smoke:   SGLANG_DIR=$SGLANG_DIR" >&2
    echo "smoke:   SGLANG_DIR/.git present? $(test -d "$SGLANG_DIR/.git" && echo YES || echo NO)" >&2
    echo "smoke:   APPLIED_COPY present? $(test -f "$SGLANG_DIR/sglang/.spec_capture_patch.applied" && echo YES || echo NO)" >&2
    echo "smoke:   server_args.py size: $(wc -c < "$SERVER_ARGS")B" >&2
    echo "smoke:   enable_spec_capture grep count: $(grep -c 'enable_spec_capture' "$SERVER_ARGS" 2>/dev/null || echo 0)" >&2
    echo "smoke:   tail /tmp/smoke-patch.log:" >&2
    tail -50 /tmp/smoke-patch.log >&2 || true
    echo "smoke: last-resort: apply server_args.py hunk directly via Python" >&2
    # 不能写 python ... <<'PY' || { ... }：heredoc body 从 <<'PY' 后一直读到 PY，
    # `||` 和 brace group 都被吞进 stdin 当 python 源码（python 报 syntax error 就
    # exit 了，brace group 永远不执行）。改用 python 退出后 recheck grep 的方式
    # 兜底：python 退出后 grep enable_spec_capture 字段是否落盘，不在就报错并
    # 把 python stderr 一并贴出来。
    python - "$SERVER_ARGS" >/tmp/smoke-py-patch.out 2>/tmp/smoke-py-patch.err <<'PY'
import sys, ast
path = sys.argv[1]
with open(path) as f:
    src = f.read()
# Three fields are inserted as a block right BEFORE 'enable_return_routed_experts: A['
# (matches the v0.5.18 patch hunk context: 'enable_spec_capture: A[' sits immediately
# above 'enable_return_routed_experts: A['). Inserting before the anchor is structural —
# 'enable_return_routed_experts: A[' is the same neighbour regardless of upstream
# sglang's exact line numbers, so this survives 0.5.18 -> 0.5.x minor drifts the BSD
# patch is fragile to. Earlier attempts to insert AFTER 'enable_return_routed_experts'
# via `src.index('\n', end_of_block)` failed: end_of_block already IS a '\n' (the
# newline preceding '    ] = ...'), so str.index returns it unchanged — insertion
# landed between the description line and the closing '    ] = False', orphaning the
# closing ']' and breaking syntax. Anchor BEFORE sidesteps the closing-line offset
# arithmetic entirely.
fields_to_add = """    enable_spec_capture: A[
        bool,
        "Enable server-side speculative-training capture (SpecForge DataFlow layout).",
        NS("exec.features"),
    ] = False
    spec_capture_aux_layer_ids: A[
        Optional[List[int]],
        "Target layer ids whose hidden states are captured for spec-capture requests.",
        NS("exec.features"),
    ] = None
    spec_capture_method: A[
        str,
        "Capture method for --enable-spec-capture: 'eagle3', 'dflash', or 'dspark'.",
        NS("exec.features"),
    ] = "eagle3"
"""
anchor = 'enable_return_routed_experts: A['
if anchor not in src:
    sys.exit("anchor 'enable_return_routed_experts: A[' not found in server_args.py")
if 'enable_spec_capture: A[' in src:
    sys.exit(0)  # already there
# Insert BEFORE the 'enable_return_routed_experts: A[' line. rfind('\n', 0, idx) gives
# the '\n' that ends the PREVIOUS line (i.e., the previous field's closing
# '    ] = <val>\n'). insert_pos = that '\n' + 1 = start of the anchor's line.
idx = src.index(anchor)
prev_newline = src.rfind('\n', 0, idx)
insert_pos = prev_newline + 1
new_src = src[:insert_pos] + fields_to_add + src[insert_pos:]
# Validate syntax BEFORE writing back: ast.parse checks only syntax (names like
# Optional/List/NS/A don't need to be in scope for the check), so an early
# SyntaxError caught here is actionable upstream of the smoke step.
try:
    ast.parse(new_src)
except SyntaxError as e:
    sys.exit(f"smoke: FAILED - inserted code creates SyntaxError at line {e.lineno}: {e.msg} ({e.text!r})")
with open(path, 'w') as f:
    f.write(new_src)
print('smoke: server_args.py patched in-place via python (added 3 fields)')
PY
    if ! grep -q 'enable_spec_capture: A\[' "$SERVER_ARGS"; then
        echo "smoke: FAILED - even direct python patch did not land enable_spec_capture" >&2
        echo "smoke:   python stderr:" >&2
        tail -30 /tmp/smoke-py-patch.err >&2 || true
        exit 1
    fi
fi
echo "smoke: patches + apt deps applied"
```

输出结果类似如下（中间省略 patch 应用逐行日志）：

```shell #test-result id="smoke-apply-patches"
smoke: applying spec-capture patches for sglang 0.5.18
smoke: patches + apt deps applied
```

### Step 3：起 mooncake_master

CANN 镜像没装 nc(netcat)，`nc -z 127.0.0.1 35551` 直接 command-not-found → 30 次循环每次都 false → smoke 误判 mooncake 没 bind；run 33262609924 复现：mooncake_master log 已 `Master service started on port 35551`，但 nc 不存在。改用 Python socket 检查。

```shell #test id="smoke-start-mooncake"
set -euo pipefail
pkill -9 -f '^mooncake_master' 2>/dev/null || true
MOONCAKE_RPC_PORT="${SPECFORGE_MOONCAKE_RPC_PORT:-35551}"
MOONCAKE_HTTP_PORT="${SPECFORGE_MOONCAKE_HTTP_PORT:-35880}"
nohup mooncake_master \
    --enable_http_metadata_server=true \
    --rpc_port=$MOONCAKE_RPC_PORT \
    --http_metadata_server_port=$MOONCAKE_HTTP_PORT \
    --metrics_port=35903 \
    --enable_metric_reporting=false \
    >/tmp/smoke-mooncake.log 2>&1 &
MOONCAKE_PID=$!
mooncake_ready() {
    python -c "
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
        exit 0
    fi
    sleep 1
done
echo "smoke: FAILED - mooncake_master did not bind $MOONCAKE_RPC_PORT in 30s" >&2
tail -50 /tmp/smoke-mooncake.log >&2
exit 1
```

输出结果类似如下：

```shell #test-result id="smoke-start-mooncake" fuzzy='xxx'
smoke: mooncake ready (rpc 35551, pid=xxx)
```

### Step 4：起 SGLang capture server

```shell #test id="smoke-start-sglang" load="model_path>>MODEL_PATH"
set -euo pipefail
pkill -9 -f '^python -m sglang\.launch_server' 2>/dev/null || true
CAPTURE_DEVICE="${SPECFORGE_CAPTURE_DEVICE:-0}"
SGLANG_PORT="${SPECFORGE_SGLANG_PORT:-30000}"
SGLANG_HEALTH_TIMEOUT="${SPECFORGE_SGLANG_HEALTH_TIMEOUT:-600}"
MOONCAKE_RPC_PORT="${SPECFORGE_MOONCAKE_RPC_PORT:-35551}"
MOONCAKE_HTTP_PORT="${SPECFORGE_MOONCAKE_HTTP_PORT:-35880}"
ASCEND_RT_VISIBLE_DEVICES=$CAPTURE_DEVICE \
MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:$MOONCAKE_HTTP_PORT/metadata \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:$MOONCAKE_RPC_PORT \
MOONCAKE_PROTOCOL=tcp \
MOONCAKE_GLOBAL_SEGMENT_SIZE=$((32<<30)) \
nohup python -m sglang.launch_server \
    --model-path "<MODEL_PATH>" \
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
        echo "smoke: sglang ready (pid=$SGLANG_PID)"
        exit 0
    fi
    sleep 5
done
echo "smoke: FAILED - SGLang not healthy after ${SGLANG_HEALTH_TIMEOUT}s" >&2
tail -50 /tmp/smoke-sglang.log >&2
exit 1
```

输出结果类似如下（中间省略 SGLang graph compile / model load 逐行日志）：

```shell #test-result id="smoke-start-sglang" fuzzy='xxx'
smoke: waiting for SGLang /health (up to 600s)
smoke: sglang ready (pid=xxx)
```

### Step 5：跑 specforge 训练（1 步）

```shell #test id="smoke-train" load="model_path>>MODEL_PATH"
set -euo pipefail
SPECFORGE_ROOT="${SPECFORGE_ROOT:-SpecForge}"
TRAINER_DEVICE="${SPECFORGE_TRAINER_DEVICE:-1}"
RECIPE="${SPECFORGE_RECIPE:-examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml}"
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
    model.target_model_path="<MODEL_PATH>" \
    2>&1 | tee /tmp/smoke-train.log
TRAIN_RC=${PIPESTATUS[0]}
popd >/dev/null

if [[ $TRAIN_RC -ne 0 ]]; then
    echo "smoke: FAILED - specforge train exit=$TRAIN_RC"
    tail -30 /tmp/smoke-train.log
    exit "$TRAIN_RC"
fi
if ! grep -qE "step.*loss|loss.*step|step N:|train_runtime" /tmp/smoke-train.log; then
    echo "smoke: FAILED - no step/loss output in train log"
    tail -30 /tmp/smoke-train.log
    exit 1
fi
echo "smoke: OK - 1-step training completed (exit=$TRAIN_RC)"
```

输出结果类似如下（中间省略逐行训练日志）：

```shell #test-result id="smoke-train" fuzzy='...'
smoke: OK - 1-step training completed (exit=0)
```

> 卡 0 跑 capture server，卡 1 跑 trainer，卡 2/3 空闲给 HCCL buffer。Smoke 的 `--context-length 1024 --mem-fraction-static 0.5` 把 SGLang KV池压住（sglang 0.5.x 把 `--max-model-len` 改名成 `--context-length`，server_args.py `context_length` 字段），`training.max_steps=1 training.batch_size=1 training.max_length=512 training.num_anchors=32 deployment.trainer.nproc_per_node=1` 把训练侧压到 1 步最小数据。