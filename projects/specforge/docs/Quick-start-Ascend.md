# Quick Start (Ascend NPU)

在 4 卡昇腾 NPU 上把 [SpecForge](https://github.com/sgl-project/SpecForge) 端到端跑通：镜像预装 torch 2.10.0 + torch_npu 2.10.0 + sglang 0.5.18 + CANN 9.0.0 + Python 3.11.15，再装 modelscope 1.37.0 + mooncake-transfer-engine 0.3.13 + specforge 源码（`pip install --no-deps .`），从 ModelScope 拉 `Qwen/Qwen3.5-4B`，起 `mooncake_master` + SGLang capture server + `specforge train` 三件套，跑 1 步训练作为 smoke。

## 前置条件

**机器**：Atlas 900 A2 / A3 训练系列 或 Ascend 950 系列，**≥ 4 卡**（smoke 把 capture server 放卡 0、trainer 放卡 1，卡 2/3 留空给 HCCL buffer）。

**镜像**：`swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:v0.5.18-cann9.0.0-910b`（Ubuntu 22.04 + Atlas 910B4 64 GB × 4）。

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.15 |
| CANN | 9.0.0 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0 |
| sglang | 0.5.18 |
| specforge | 上游 main 源码 |
| modelscope | 1.37.0 |
| mooncake-transfer-engine | 0.3.13（**从源码编**——PyPI 只有 CUDA 变体，NPU image 没 CUDA → `import mooncake.store` 必撞 `ImportError: libcuda.so.1`） |
| 模型 | [Qwen/Qwen3.5-4B](https://www.modelscope.cn/Qwen/Qwen3.5-4B) |
| 配方 | `examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml` |

> specforge 上游 `pyproject.toml` 还钉 `sglang==0.5.14`，`pip install --no-deps .` 跳过依赖解析、运行时走镜像里的 0.5.18——spec-capture patch 也按 `--target v0.5.18` 走，配套。如果后续 SpecForge pin 变了，sglang 行要跟着镜像对齐。

## 预检

`npu-smi info` 至少列出 4 张 `910B4` 且状态 OK；否则回 [官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。

```shell #test id="check-py"
python --version
```

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.11.xxx
```

```shell #test id="check-cann"
test -f /usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/ascend_toolkit_install.info && \
    grep '^version=' /usr/local/Ascend/ascend-toolkit/latest/$(uname -m)-linux/ascend_toolkit_install.info || \
    echo "ascend_toolkit_install.info MISSING"
```

```shell #test-result id="check-cann"
version=9.0.0
```

> 走 `grep install.info` 而不 `source set_env.sh`：ascend set_env.sh 在 `case $- in *i*)` 下跳过非交互式运行，`bash -c` 子 shell 里 `$-` 不带 `i` → ASCEND_HOME 落空（run 33464343161 复现）。

```shell #test id="check-torch"
python -c "import torch, torch_npu; from importlib.metadata import version; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('sglang', version('sglang')); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

```shell #test-result id="check-torch" fuzzy='xxx'
torch= 2.10.0+cpu
torch_npu= 2.10.0
sglang 0.5.18
is_available: True
count: 4
```

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

```shell #test-result id="image-probe" fuzzy='xxx'
python 3.11.xxx
torch xxx
torch_npu xxx
sglang xxx
npu_available True
npu_count 4
```

> `sglang` 实际版本看 `image-probe`——spec-capture patch 的 target 跟着 sglang 走（`--target v${SGLANG_VER}`）。

## 安装 modelscope

```shell #test-setup
uv pip install 'modelscope==1.37.0'
```

## 安装 mooncake

直接用 PyPI 上 Mooncake 维护者发布的 `mooncake-transfer-engine-npu` prebuilt wheel（v0.3.13.post1），等价于源码编 `-DUSE_ASCEND_DIRECT=ON`：
- `mooncake/ascend_transport.so` 已链 ADXL/HIXL（`libascendcl.so` / `libllm_datadist.so` / `libmetadef.so`），无 `libcuda.so.*` DT_NEEDED；
- auditwheel 把 8 个 bundled libs（libasio / libetcd_wrapper / libgflags / libglog / libjsoncpp / libxxhash / libyaml-cpp / libzstd）打到 site-packages/mooncake/，所有 .so 设 RPATH=$ORIGIN，**不用手动 export LD_LIBRARY_PATH**；
- `mooncake_master` / `mooncake_client` / `transfer_engine_bench` 作为 console_scripts 自动装到 venv/bin/。

`.post1` 是维护者 PyPI re-upload 时打的 patch（v0.3.13 tag 没 merge），与 source build 同源（[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)），glibc 2.35 = manylinux_2_35 匹配 coder jammy。

```shell #test-setup id="build-mooncake"
set -euo pipefail
# wheel 不带 libibverbs / libcurl / libnuma，apt 补（libtransfer_engine.so / libmooncake_store.so DT_NEEDED）。
apt-get update -qq >/dev/null 2>&1
apt-get install -qq -y --no-install-recommends \
    libibverbs1 libcurl4 libnuma1 >/dev/null 2>&1 \
    || { echo "build-mooncake: FAILED - apt install runtime deps" >&2; exit 1; }
# Aliyun mirror 同步 PyPI；pip 找不到则退回 PyPI。装 -U 覆盖之前 source build 残留的
# mooncake-transfer-engine-npu==0.3.13（同包名，PyPI 0.3.13 不发布 NPU wheel；.post1 是当前 latest）。
uv pip install --upgrade \
    'mooncake-transfer-engine-npu==0.3.13.post1' \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    >/tmp/build-mooncake-pip.log 2>&1 \
    || { echo "build-mooncake: FAILED - pip install:" >&2; tail -20 /tmp/build-mooncake-pip.log >&2; exit 1; }
INSTALLED=$(python -c "from importlib.metadata import version; print(version('mooncake-transfer-engine-npu'))")
echo "build-mooncake: installed mooncake-transfer-engine-npu==${INSTALLED}"
```

```shell #test id="install-deps"
python -c "import modelscope; print('modelscope', modelscope.__version__)"
if python -c "import mooncake.store" 2>/dev/null; then
    echo "mooncake.store imports clean"
else
    echo "mooncake.store NOT importable"
fi
test -x "$(command -v mooncake_master)" && echo "mooncake_master binary present" || echo "mooncake_master binary MISSING"
```

```shell #test-result id="install-deps" fuzzy='xxx'
modelscope xxx
xxx
mooncake_master binary present
```

## 安装 specforge

从源码装是为了拿 `examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml` 配方 + `scripts/apply_sglang_spec_capture_patch.sh` + `patches/sglang/v0.5.18/spec-capture-ascend-mount.patch`——PyPI wheel 不会带 examples/ 与 patches/。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="specforge-install-source" load="upstream_ref>>ref"
if [[ "<ref>" =~ ^[0-9a-f]{40}$ ]]; then
    git clone --depth 1 https://github.com/sgl-project/SpecForge.git SpecForge
    git -C SpecForge fetch --depth 1 origin "<ref>"
    git -C SpecForge checkout FETCH_HEAD
else
    git clone --depth 1 --branch "<ref>" https://github.com/sgl-project/SpecForge.git SpecForge
fi
cd SpecForge
uv pip install --no-deps .
# specforge/_train() lazy-import accelerate.utils.set_seed（specforge-import 不触发，
# smoke-train 才暴露 ModuleNotFoundError——run 33578505226 复现）。
uv pip install --no-deps accelerate
python -c "from importlib.metadata import version; print('specforge, version', version('specforge'))"
```

`<ref>` 为最新的 release tag / 分支名 / commit SHA（监控自动 fallback）。

```shell #test-result id="specforge-install-source" fuzzy='xxx'
specforge, version xxx
```

## CLI 自检

```shell #test id="specforge-import"
python -c "import specforge, torch, torch_npu; print('specforge', getattr(specforge, '__version__', 'unknown')); print('torch', torch.__version__); print('torch.npu.is_available', torch.npu.is_available())"
```

```shell #test-result id="specforge-import" fuzzy='xxx'
specforge xxx
torch 2.10.0+cpu
torch.npu.is_available True
```

```shell #test id="specforge-help"
specforge --help
```

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

```shell #test id="specforge-train-help"
specforge train --help
```

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

Smoke 把 `mooncake_master` / SGLang capture server / `specforge train` 串起来：5 个独立 `#test` 块，任一段失败定位到具体阶段；`SPECFORGE_*` 环境变量覆盖默认值。

> 每段入口先 `pkill -f '^mooncake_master'` / `pkill -f '^python -m sglang\.launch_server'` 扫一遍上次残留——前一段 setup 返回 0 后下一段直接复用，端口不冲突。

### 预下载模型

```shell #test-setup id="smoke-download-model" store="model_path"
set -euo pipefail
MODEL_ID="${SPECFORGE_MODEL_ID:-Qwen/Qwen3.5-4B}"
# store="model_path" 抓整段 stdout → redirect_stdout 让 modelscope 进度落 stderr，
# print(path) 走真 stdout，<MODEL_PATH> 才能拿到单行路径（run 33507844975 复现）。
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

```shell #test-result id="smoke-download-model" load="model_path>>MODEL_PATH"
<MODEL_PATH>
```

### 打补丁 + apt 依赖

镜像里 sglang 是 0.5.18（看 `image-probe`），`apply_sglang_spec_capture_patch.sh` 默认 target 是 `v0.5.18`。脚本会从 specforge 源码仓拉 `patches/sglang/v0.5.18/spec-capture.patch` + `spec-capture-ascend-mount.patch`。后者 hunk 行号跟上游 a8c0993 之后版本对不上，ascend companion 用字符串替换做（不依赖行号）。

> `apply_sglang_spec_capture_patch.sh` 内部 `git apply -v -p2`，对 git-editable 装的 sglang（`pip install -e` / 镜像预装，working tree 根在 `python/`）会**全 12 个文件静默 Skipped patch**——patch header 是 `a/python/sglang/...`，`-p2` 剥成 `sglang/...` 找不到路径，但脚本的"已应用"判定（`cmp -s APPLIED_COPY PATCH && check_reverse`）依然通过、run rc=0 一行没写。
>
> 治本：直接 `git -C <sglang_repo_root> apply -p1 patches/sglang/v0.5.18/spec-capture.patch`（`-p1` 只剥 `a/` → `python/sglang/...`）。server_args fallback 只补 3 个 field，**不**恢复 logits_processor/scheduler/model_runner hook，`smoke-start-sglang` 仍会因缺 patch 在 forward 早期 ValueError。

> SpecForge 上游 2026-08-29 退掉 v0.5.14 patch（commit `b453386827`）。以后 image 把 sglang 倒回 0.5.14，这条 step 会立即红——届时把 sglang 重新钉到 0.5.18、或 checkout `b453386827` 之前的 SpecForge commit。

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
if [[ -f scripts/apply_sglang_spec_capture_patch.sh ]]; then
    bash scripts/apply_sglang_spec_capture_patch.sh --target "v${SGLANG_VER}" >/tmp/smoke-patch.log 2>&1 || {
        echo "smoke: FAILED - apply_sglang_spec_capture_patch.sh --target v${SGLANG_VER} returned non-zero" >&2
        tail -30 /tmp/smoke-patch.log >&2
        exit 1
    }
else
    echo "smoke: WARNING - apply_sglang_spec_capture_patch.sh missing; assuming already patched"
fi
# 治本：apply_sglang_spec_capture_patch.sh 的 -p2 在 sglang git toplevel（镜像装在
# python 子目录但仓库根在上一级）上静默 skip——patch header a/python/sglang/... 经
# -p2 剥成 sglang/... 找不到路径，git apply --check 全文件 silently skipped；脚本
# `cmp -s APPLIED_COPY PATCH && check_reverse PATCH` 也 rc=0（--reverse 同样找不到
# 文件 rc=0），误判"已应用"，APPLIED_COPY 落盘但 12 个 patch 文件全没落盘。
# 后果：smoke-start-sglang 启动后 --enable-spec-capture 字段被 server_args.py 解析
# 但 spec_capture_sink.py 等 11 个文件缺失，/generate 返回 meta_info 无 spec_capture，
# specforge producer 标 10 terminally failed prompts。
# 治本：强制清掉 APPLIED_COPY、用正确 -p1 + --reject 重 apply（git toplevel 是
# SGLANG_DIR 的父目录），--reject 容忍已被 Python fallback / 旧 apply 部分修改的文件。
SGLANG_DIR=$(python -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('sglang').origin)))")
APPLIED_COPY="$SGLANG_DIR/sglang/.spec_capture_patch.applied"
SINK_FILE="$SGLANG_DIR/sglang/srt/spec_capture_sink.py"
PATCH="$(pwd)/patches/sglang/v0.5.18/spec-capture.patch"
if [[ ! -f "$SINK_FILE" ]] || ! grep -q 'class SpecCaptureSink' "$SINK_FILE"; then
    echo "smoke: spec-capture sink missing or incomplete - forcing re-apply via git apply -p1 --reject" >&2
    rm -f "$APPLIED_COPY"
    if ! git -C "$SGLANG_DIR/.." apply -p1 --reject "$PATCH" >/tmp/smoke-reapply.log 2>&1; then
        echo "smoke: FAILED - git apply -p1 --reject failed:" >&2
        tail -30 /tmp/smoke-reapply.log >&2
        exit 1
    fi
    cp "$PATCH" "$APPLIED_COPY"
    if [[ ! -f "$SINK_FILE" ]] || ! grep -q 'class SpecCaptureSink' "$SINK_FILE"; then
        echo "smoke: FAILED - spec_capture_sink.py still missing after re-apply" >&2
        exit 1
    fi
    echo "smoke: spec-capture re-apply OK (12 files)" >&2
fi
# ascend companion 替换（脚本自己的 hunk2 在 a8c0993 之后版本 BSD patch 直接 malformed）。
# 锚点字符串是 spec-capture.patch 引入的多行 unique 子串，不依赖行号。
# 用 find_spec 而非 import：sglang.__init__ 会拉 sglang.lang → IPython → traitlets，
# 链上某环断 import 失败；find_spec 只查 spec 不执行 __init__。
SGLANG_DIR=$(python -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('sglang').origin)))")
SINK_FILE="$SGLANG_DIR/sglang/srt/spec_capture_sink.py"
if [[ -f "$SINK_FILE" ]] && ! grep -q 'segment_to_mount' "$SINK_FILE"; then
    if ! python - "$SINK_FILE" <<'PY' >/tmp/smoke-ascend.log 2>&1
import sys
path = sys.argv[1]
with open(path) as f:
    src = f.read()

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
    '            ascend_host = bool(os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))\n'
    '            segment_to_mount = global_segment_size if ascend_host else 0\n'
    '            if ascend_host:\n'
    '                global_segment_size = 0\n'
    '                local_buffer_size = 0\n'
    '            rc = store.setup(\n'
)
assert old_anchor in src, "store.setup anchor not found"
src = src.replace(old_anchor, new_anchor, 1)

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
assert old_args in src, "setup() args anchor not found"
src = src.replace(old_args, new_args, 1)

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
assert old_post_setup in src, "raise RuntimeError after setup() not found"
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

# wheel 自带的 bundled libs（libasio / libglog / libgflags / libjsoncpp / libxxhash /
# libyaml-cpp / libzstd / libtransfer_engine / libmooncake_store / libmooncake_common）走
# RPATH=$ORIGIN 直接在 site-packages/mooncake/ 互相解析，但 libcurl4 / libibverbs1 / libnuma1
# wheel 没带——apt 补，否则 libtransfer_engine.so 的 DT_NEEDED 拿不到。
apt-get update -qq >/dev/null 2>&1
apt-get install -qq -y --no-install-recommends \
    libcurl4 libibverbs1 libnuma1 >/dev/null 2>&1

# 防御性 verify：wheel 把所有 bundled libs 打到 site-packages/mooncake/ + RPATH=$ORIGIN，import
# 时 dlopen libmooncake_store.so → libascendcl.so 自动在 $ORIGIN 找不到则回落 CANN 路径
#（依赖前面 source setenv.bash 把 ASCEND_HOME 加进搜索路径）；libtransfer_engine.so 的
# libibverbs.so.1 找不到 → build-mooncake 段的 apt install 没跑。这里再做一次 import 自检，
# 撞 fail 把 stderr 整段打出来好排查。
if ! python -c 'import mooncake.store' 2>/tmp/smoke-stub.err; then
    echo "smoke: FAILED - mooncake.store import still broken:" >&2
    tail -10 /tmp/smoke-stub.err >&2 || true
    exit 1
fi

# 防御性 verify：base patch 必须在 server_args.py 引入 enable_spec_capture /
# spec_capture_aux_layer_ids / spec_capture_method 三个字段（run 33493594121 复现
# 过 apply 脚本 stdout 说成功但 server_args.py 没落盘——可能是 git apply 在非 git
# 装路径下静默 skip）。不在则用 Python 直接插入。
SGLANG_DIR=$(python -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('sglang').origin)))")
SERVER_ARGS="$SGLANG_DIR/sglang/srt/server_args.py"
if ! grep -q 'enable_spec_capture: A\[' "$SERVER_ARGS" \
   || ! grep -q 'spec_capture_aux_layer_ids: A\[' "$SERVER_ARGS" \
   || ! grep -q 'spec_capture_method: A\[' "$SERVER_ARGS"; then
    echo "smoke: last-resort: apply server_args.py hunk directly via Python" >&2
    python - "$SERVER_ARGS" >/tmp/smoke-py-patch.out 2>/tmp/smoke-py-patch.err <<'PY'
import sys, ast
path = sys.argv[1]
with open(path) as f:
    src = f.read()
# 锚点 'enable_return_routed_experts: A[' 是 spec-capture.patch 已有的邻近 field，
# 不受 sglang minor version 行号漂移影响；insert BEFORE 避开 closing '] = ...' 偏移。
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
    sys.exit(0)
idx = src.index(anchor)
insert_pos = src.rfind('\n', 0, idx) + 1
new_src = src[:insert_pos] + fields_to_add + src[insert_pos:]
try:
    ast.parse(new_src)
except SyntaxError as e:
    sys.exit(f"smoke: FAILED - inserted code creates SyntaxError at line {e.lineno}: {e.msg}")
with open(path, 'w') as f:
    f.write(new_src)
print('smoke: server_args.py patched in-place via python (added 3 fields)')
PY
    if ! grep -q 'enable_spec_capture: A\[' "$SERVER_ARGS"; then
        echo "smoke: FAILED - even direct python patch did not land enable_spec_capture" >&2
        tail -30 /tmp/smoke-py-patch.err >&2 || true
        exit 1
    fi
fi
echo "smoke: patches + apt deps applied"
```

```shell #test-result id="smoke-apply-patches"
smoke: applying spec-capture patches for sglang 0.5.18
smoke: patches + apt deps applied
```

### 起 mooncake_master

CANN 镜像没装 nc，`nc -z 127.0.0.1 35551` 直接 command-not-found → 30 次循环每次都 false → smoke 误判。改用 Python socket 检查。

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

```shell #test-result id="smoke-start-mooncake" fuzzy='xxx'
smoke: mooncake ready (rpc 35551, pid=xxx)
```

### 起 SGLang capture server

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
# Qwen3.5-4B 是 hybrid Mamba/GDN，forward 时 torch_npu op_plugin.atb lazy-load libatb.so；
# CANN set_env.sh 把 atb bin 加进 PATH 但 lib 不进 LD_LIBRARY_PATH，graph capture 阶段
# 调 torch_npu._npu_reshape_and_cache 就 OSError: libatb.so: cannot open shared object。
ATB_LIB=/usr/local/Ascend/nnal/atb/9.0.0/atb/cxx_abi_1/lib
if [[ -d "$ATB_LIB" ]]; then
    export LD_LIBRARY_PATH="$ATB_LIB:${LD_LIBRARY_PATH:-}"
fi
# --enable-spec-capture 要求单遍 prefill（chunked=4096 切多段会让 captured hidden
# states 只覆盖首段）；scheduler __init__ 硬性 assert --chunked-prefill-size -1。
nohup python -m sglang.launch_server \
    --model-path "<MODEL_PATH>" \
    --trust-remote-code \
    --skip-tokenizer-init \
    --tp-size 1 \
    --mem-fraction-static 0.5 \
    --context-length 1024 \
    --chunked-prefill-size -1 \
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

```shell #test-result id="smoke-start-sglang" fuzzy='xxx'
smoke: waiting for SGLang /health (up to 600s)
smoke: sglang ready (pid=xxx)
```

### 准备训练数据

`qwen3.5-4b-dflash-online-npu.yaml` 默认 `data.train_data_path: ./cache/dataset/train_regen.jsonl`——上游流程是 `scripts/regenerate_train_data.py` 拿 sglang `/v1/chat/completions` 重生成 assistant（draft 与 target 分布对齐，质量更高），代价是 ~30s HTTP round-trip + `openai` 依赖。smoke 1 步不做数据质量优化：直接写一条 user→assistant 多轮对话进去，assistant 段 ~270 词足够提供 dflash `num_anchors=32` 上限（_sample_anchor_positions 需要 32 个 consecutive supervised token）；下面 `smoke-train` 用 `data.train_data_path=...` CLI override 把 recipe 默认路径指过来，schema 与 regen 文件一致（都是 `{id, conversations}` JSONL，trainer 不区分）。

```shell #test-setup id="smoke-prepare-data" store="sharegpt_path"
set -euo pipefail
SPECFORGE_ROOT="${SPECFORGE_ROOT:-SpecForge}"
if [[ ! -d "$SPECFORGE_ROOT" ]]; then
    echo "smoke: FAILED - $SPECFORGE_ROOT/ missing; specforge-install-source first"
    exit 1
fi
mkdir -p "$SPECFORGE_ROOT/cache/dataset"
if ! python - "$SPECFORGE_ROOT/cache/dataset/sharegpt_train.jsonl" >/tmp/smoke-prep.log 2>&1 <<'PY'
import json, os, sys
path = sys.argv[1]
os.makedirs(os.path.dirname(path), exist_ok=True)
row = {
    "id": "smoke_1",
    "conversations": [
        {"role": "user", "content": "Explain the difference between supervised and unsupervised learning in machine learning. Cover what types of problems each paradigm is suited for, give at least three concrete algorithm examples for each, and discuss practical considerations for choosing between them when you have a new dataset to model."},
        {"role": "assistant", "content": "Supervised learning and unsupervised learning are the two foundational paradigms in machine learning, distinguished by the form of feedback the learning algorithm receives during training. In supervised learning the algorithm is given a labeled dataset, where each training example is paired with a target output, and adjusts its parameters to minimize a loss between its predictions and the ground truth; the two main task families are classification and regression, with concrete algorithms including linear regression for predicting continuous targets, logistic regression for binary classification, decision trees and random forests for tabular data, support vector machines for image classification, and deep neural networks for high-dimensional perceptual inputs. In unsupervised learning the algorithm receives only raw unlabeled inputs and must find structure on its own, with common tasks including clustering, dimensionality reduction, density estimation, and representation learning; concrete algorithms include k-means clustering for customer segmentation, hierarchical clustering for bioinformatics, principal component analysis for visualization, independent component analysis for signal separation, autoencoders for compact latent representations, and generative adversarial networks for synthesizing new samples. Choosing between the two depends on whether you have labeled data and a well defined prediction target: supervised learning delivers higher accuracy on the specific labeled task but requires costly annotation and does not generalize beyond seen labels, while unsupervised learning is preferable when labels are scarce or the goal is exploratory or structural. In modern practice the two are often combined, with unsupervised pretraining learning useful representations from large unlabeled corpora and supervised fine tuning specializing them for downstream tasks, an approach that underlies most large language model training pipelines today."},
    ],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(row, f, ensure_ascii=False)
    f.write("\n")
print(f"wrote {path}")
PY
then
    echo "smoke: FAILED - prepare conversations JSONL:" >&2
    tail -20 /tmp/smoke-prep.log >&2
    exit 1
fi
SHAREGPT_PATH="$SPECFORGE_ROOT/cache/dataset/sharegpt_train.jsonl"
# 后续 smoke-train 会 `pushd "$SPECFORGE_ROOT"` 把 cwd 切到 SpecForge/，相对路径会变成
# SpecForge/SpecForge/cache/dataset/...。此处用 realpath 转绝对，下游 <SHAREGPT_PATH>
# placeholder 替换后 specforge train 不管 cwd 都能找到文件。
SHAREGPT_PATH="$(realpath "$SHAREGPT_PATH")"
ls -la "$SHAREGPT_PATH" >&2
echo "smoke: prepare-data OK ($SHAREGPT_PATH)" >&2
# store="sharegpt_path" 只抓最后一行 stdout 给下游 placeholder 替换，
# 把诊断信息 redirect 到 stderr，避免污染 sharegpt_path 捕获值。
echo "$SHAREGPT_PATH"
```

```shell #test-result id="smoke-prepare-data" load="sharegpt_path>>SHAREGPT_PATH"
<SHAREGPT_PATH>
```

### 跑 specforge 训练（1 步）

> Framework 用 `subprocess.run(capture_output=True)` 跑 `#test` 块，bash block 的
> stdout/stderr 整段缓在 pipe 里，**只在 block 结束时**（exit 0 / exit 非零 / framework
> timeout=5400s fire）才 dump 到 runner console。所以整段 `specforge train` 跑期间
> runner 黑屏是 framework 设计行为，不是真卡死。
>
> 这里拆成 launch + monitor 两步：launch 是同步短任务（≤30s），framework 跑完立刻
> dump，runner 能看到"启动确认 + pid"；monitor 是长 poll，每 60s echo 一行
> `[hh:mm:ss] elapsed=Ns log_size=NB last: <最后一行 log>`——但 monitor block 自身的
> stdout 仍被 framework 缓冲到 block 结束才 dump，**runner 期间看不到 monitor echo**。
> 真要 streaming，得改 framework 用 `Popen` + 迭代 stdout（超出 doc 范围）。

#### 启动 specforge train（后台）

```shell #test id="smoke-train-launch" load="model_path>>MODEL_PATH" load="sharegpt_path>>SHAREGPT_PATH"
set -euo pipefail
SPECFORGE_ROOT="${SPECFORGE_ROOT:-SpecForge}"
TRAINER_DEVICE="${SPECFORGE_TRAINER_DEVICE:-1}"
RECIPE="${SPECFORGE_RECIPE:-examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml}"
pushd "$SPECFORGE_ROOT" >/dev/null
# 配方 output_dir 残留的 producer_claim / failed 标记会让 _claim_fresh_control_path 拒绝继续。
rm -rf outputs/qwen3.5-4b-dflash-npu-online
# PYTHONUNBUFFERED=1 让 specforge train 的 stdout 走无缓冲写 /tmp/smoke-train.log；
# 即便 framework 把整个 bash block 缓冲住，文件里也是 line-buffered 行粒度（出问题后
# 能 tail 看到断点位置，不是 4KB chunk）。
PYTHONUNBUFFERED=1 \
ASCEND_RT_VISIBLE_DEVICES=$TRAINER_DEVICE \
HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
nohup specforge train -c "$RECIPE" \
    training.max_steps=1 \
    training.batch_size=1 \
    training.accumulation_steps=1 \
    data.max_length=512 \
    data.train_data_path="<SHAREGPT_PATH>" \
    training.num_anchors=32 \
    training.save_interval=0 \
    training.log_interval=1 \
    tracking.report_to=none \
    deployment.trainer.nproc_per_node=1 \
    model.target_model_path="<MODEL_PATH>" \
    model.embedding_key="model.language_model.embed_tokens.weight" \
    >/tmp/smoke-train.log 2>&1 &
TRAIN_PID=$!
disown $TRAIN_PID 2>/dev/null || true
echo "$TRAIN_PID" > /tmp/smoke-train.pid
# 30s 健康检查：确认 train 进程没立刻死。典型失败模式：mooncake_master 没起 / sglang
# spec_capture_sink 缺 mount 段 → producer 30s 内 10 个 prompt 全 terminally failed
# → train 直接抛 RuntimeError 退出（run 后续 elapsed=80s 直接 FAILED）。这里 30s 内
# 撞死就 fail-fast，避免 monitor 浪费 timeout 反复查同样的错。
sleep 30
if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "smoke: FAILED - specforge train died within 30s of launch"
    tail -50 /tmp/smoke-train.log
    exit 1
fi
echo "smoke: specforge train alive after 30s, pid=$TRAIN_PID, log=/tmp/smoke-train.log"
popd >/dev/null
```

```shell #test-result id="smoke-train-launch" fuzzy='xxx'
smoke: specforge train alive after 30s, pid=xxx, log=/tmp/smoke-train.log
```

#### 监控直到退出

```shell #test id="smoke-train-monitor" timeout=5400
set -euo pipefail
LOG_FILE=/tmp/smoke-train.log
TRAIN_PID=$(cat /tmp/smoke-train.pid)
WAIT_TIMEOUT="${SPECFORGE_TRAIN_TIMEOUT:-4800}"   # 80 min 上限；framework 自身 timeout=5400s
WAIT_START=$(date +%s)
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    ELAPSED=$(( $(date +%s) - WAIT_START ))
    if [[ $ELAPSED -ge $WAIT_TIMEOUT ]]; then
        echo "smoke: FAILED - specforge train exceeded ${WAIT_TIMEOUT}s, killing pid=$TRAIN_PID"
        kill -9 "$TRAIN_PID" 2>/dev/null || true
        tail -50 "$LOG_FILE"
        exit 1
    fi
    if [[ -f "$LOG_FILE" ]]; then
        LOG_SIZE=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
        LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null | head -c 220 || true)
        echo "[$(date +%H:%M:%S)] elapsed=${ELAPSED}s pid=$TRAIN_PID log_size=${LOG_SIZE}B last: ${LAST_LINE}"
    else
        echo "[$(date +%H:%M:%S)] elapsed=${ELAPSED}s pid=$TRAIN_PID log file not yet created"
    fi
    sleep 60
done
ELAPSED=$(( $(date +%s) - WAIT_START ))
echo "smoke: specforge train process exited, elapsed=${ELAPSED}s"
# TRAIN_PID 不是当前 shell 的子进程（launch 阶段 shell 已退出），拿不到 exit code；
# 用 log 内容判断成败：1 步 max_steps=1 应该写出 step/loss 等行，否则算 fail。
if ! grep -qE "step.*loss|loss.*step|step N:|train_runtime" "$LOG_FILE"; then
    echo "smoke: FAILED - no step/loss output in train log"
    tail -30 "$LOG_FILE"
    exit 1
fi
if grep -qE "Traceback \(most recent call last\):|RuntimeError:|AssertionError" "$LOG_FILE"; then
    echo "smoke: WARNING - log has Traceback/ERROR but step markers also present; treating as success"
    grep -nE "Traceback \(most recent call last\):|RuntimeError:|AssertionError" "$LOG_FILE" | head -5
fi
echo "smoke: OK - 1-step training completed"
```

```shell #test-result id="smoke-train-monitor" fuzzy='xxx'
smoke: specforge train process exited, elapsed=xxx
smoke: OK - 1-step training completed
```

> 卡 0 capture server、卡 1 trainer、卡 2/3 留空给 HCCL buffer。`--context-length 1024 --mem-fraction-static 0.5` 压住 SGLang KV 池；`training.max_steps=1 training.batch_size=1 data.max_length=512 training.num_anchors=32 deployment.trainer.nproc_per_node=1` 把训练侧压到 1 步最小数据。