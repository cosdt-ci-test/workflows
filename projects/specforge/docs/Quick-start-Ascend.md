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

## 源码编译 mooncake

`mooncake-transfer-engine` **从源码编译**。`-DUSE_ASCEND_DIRECT=ON` 把 transport 切到 ADXL/HIXL（CANN 内置），整链不再链 libcuda；`-DWITH_STORE=ON` 同时产出 Python `mooncake.store` 模块（specforge eager-import 要它）+ `mooncake_master` 二进制。cmake flags 与 `projects/mooncake/docs/Quick-start-Ascend.md` + `scripts/setup_example.sh` + `release-npu.yaml` 一致。

```shell #test-setup id="build-mooncake"
set -euo pipefail
BUILD_DIR="${MOONCAKE_BUILD_DIR:-/tmp/build-mooncake}"
MOONCAKE_REF=v0.3.13
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
# 前 11 个与 mooncake/setup_example.sh 一致（仅编 transfer_engine_ascend_direct_perf 够用）；
# WITH_STORE=ON 额外要 6 个 cmake 包（缺一个 cmake configure 直接 abort）+ 2 个 wheel 打包工具。
apt-get update -qq >/dev/null 2>&1
apt-get install -qq -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libgoogle-glog-dev libgflags-dev libibverbs-dev \
    libjsoncpp-dev libnuma-dev libyaml-cpp-dev \
    libssl-dev libcurl4-openssl-dev \
    libzstd-dev libxxhash-dev libzmq3-dev libasio-dev \
    libboost-dev libmsgpack-dev patchelf file \
    >/dev/null 2>&1 \
    || { echo "build-mooncake: FAILED - apt install build deps" >&2; exit 1; }
git clone --depth 1 https://github.com/kvcache-ai/Mooncake.git \
    >/tmp/build-mooncake-clone.log 2>&1 \
    || { echo "build-mooncake: FAILED - mooncake clone:" >&2; tail -20 /tmp/build-mooncake-clone.log >&2; exit 1; }
cd Mooncake
git fetch --depth 1 origin "$MOONCAKE_REF" >/dev/null 2>&1
git checkout FETCH_HEAD >/dev/null 2>&1
# 直连 GitHub 拉 pybind11 失败时降级到 ghfast.top 镜像（与 mooncake setup_example.sh 同款）。
if ! git submodule update --init --depth 1 extern/pybind11 >/dev/null 2>&1; then
    expect=$(git ls-tree HEAD extern/pybind11 | awk '{print $3}')
    if [[ ! "$expect" =~ ^[0-9a-f]{40}$ ]]; then
        echo "build-mooncake: FAILED - cannot read pybind11 SHA from tree" >&2
        exit 1
    fi
    rm -rf extern/pybind11
    mkdir -p extern/pybind11
    git -C extern/pybind11 init -q
    git -C extern/pybind11 remote add origin https://ghfast.top/https://github.com/pybind/pybind11.git
    git -C extern/pybind11 fetch --depth 1 origin "$expect" >/dev/null 2>&1 \
        || { echo "build-mooncake: FAILED - pybind11 fetch via mirror" >&2; exit 1; }
    git -C extern/pybind11 checkout --detach FETCH_HEAD -q
fi
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_ASCEND_DIRECT=ON \
    -DBUILD_UNIT_TESTS=OFF \
    -DWITH_STORE=ON \
    -DWITH_STORE_RUST=OFF \
    -DWITH_EP=OFF \
    -DWITH_P2P_STORE=OFF \
    -DUSE_ETCD=OFF \
    -DUSE_REDIS=OFF \
    >/tmp/build-mooncake-cmake.log 2>&1 \
    || { echo "build-mooncake: FAILED - cmake configure:" >&2; tail -50 /tmp/build-mooncake-cmake.log >&2; exit 1; }
cmake --build build -j"$(nproc)" \
    >/tmp/build-mooncake-build.log 2>&1 \
    || { echo "build-mooncake: FAILED - cmake build:" >&2; tail -50 /tmp/build-mooncake-build.log >&2; exit 1; }
NPU_BUILD=1 OUTPUT_DIR=dist ./scripts/build_wheel.sh \
    >/tmp/build-mooncake-wheel.log 2>&1 \
    || { echo "build-mooncake: FAILED - build_wheel.sh:" >&2; tail -50 /tmp/build-mooncake-wheel.log >&2; exit 1; }
WHL=$(ls mooncake-wheel/dist/*.whl | head -1)
if [[ -z "$WHL" || ! -f "$WHL" ]]; then
    echo "build-mooncake: FAILED - no wheel produced in mooncake-wheel/dist/" >&2
    ls -la mooncake-wheel/dist/ >&2 || true
    exit 1
fi
uv pip install "$WHL" \
    >/tmp/build-mooncake-pip.log 2>&1 \
    || { echo "build-mooncake: FAILED - pip install wheel:" >&2; tail -20 /tmp/build-mooncake-pip.log >&2; exit 1; }
echo "build-mooncake: installed $WHL"
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

# mooncake_transfer_engine.libs/ 自带 8 个库（asio/gflags/glog/jsoncpp/uring/xxhash/
# yaml-cpp/zstd），但 libcurl4 / libibverbs1 / libnuma1 不在——直接 apt 补。
apt-get update -qq >/dev/null 2>&1
apt-get install -qq -y --no-install-recommends \
    libcurl4 libibverbs1 libnuma1 >/dev/null 2>&1

# store.so DT_NEEDED 链 libcuda.so.1 + libcudart.so.12（pybind buffer mgmt 调 cudaMemcpy
# 等——USE_ASCEND_DIRECT=ON 只切 transport，没去掉 pybind 引用），NPU image 无 CUDA →
# dynamic-link 阶段 `import mooncake.store` 必报缺符号。往 wheel 的 .libs/ 塞两个 SONAME
# 正确的 stub（store.so 永不真调 CUDA，stub 只是给动态链接器找 SONAME 用）。
#
# 用 sysconfig 找 site-packages 而非 `import mooncake.store`——后者会先撞符号缺失连路径都拿不到。
LIBS_DIR=$(python -c "import sysconfig, os; print(os.path.join(sysconfig.get_paths()['purelib'], 'mooncake_transfer_engine.libs'))")
if [[ ! -d "$LIBS_DIR" ]]; then
    echo "smoke: FAILED - $LIBS_DIR missing (build-mooncake did not install mooncake wheel?)" >&2
    exit 1
fi
if [[ ! -f "$LIBS_DIR/libcuda.so.1" || ! -f "$LIBS_DIR/libcudart.so.12" ]]; then
    if ! command -v gcc >/dev/null 2>&1; then
        apt-get install -qq -y --no-install-recommends gcc >/dev/null 2>&1
    fi
    cat > /tmp/mooncake_cuda_stub.c <<'CUEOF'
/* Stub libcuda.so.1 + libcudart.so.12 for mooncake.store on NPU. Functions listed cover
 * store.so's DT_NEEDED UND symbols (objdump -T | awk '/UND/'); store.so never actually
 * calls them (USE_ASCEND_DIRECT=ON swaps transport to ADXL/HIXL). All return 0
 * (CUDA_SUCCESS); output params receive non-NULL dummy addresses to prevent caller
 * dereference crashes. */
#include <stddef.h>
/* libcuda.so.1 (driver API) */
int cuInit(unsigned f) { return 0; }
int cuDeviceGet(int *d, int o) { if (d) *d = 0; return 0; }
int cuDeviceGetAttribute(int *v, int a, int d) { if (v) *v = 0; return 0; }
int cuDeviceGetCount(int *c) { if (c) *c = 0; return 0; }
int cuDeviceGetName(char *n, int l, int d) { return 0; }
int cuDeviceComputeCapability(int *m, int *n, int d) { if (m) *m = 0; if (n) *n = 0; return 0; }
int cuDevicePrimaryCtxRetain(void **c, int d) { if (c) *c = (void*)0x1; return 0; }
int cuDevicePrimaryCtxRelease_v2(int d) { return 0; }
int cuCtxCreate(void **c, unsigned f, int d) { if (c) *c = (void*)0x2; return 0; }
int cuCtxCreate_v2(void **c, unsigned f, int d) { if (c) *c = (void*)0x2; return 0; }
int cuCtxDestroy(void *c) { return 0; }
int cuCtxSetCurrent(void *c) { return 0; }
int cuCtxGetCurrent(void **c) { if (c) *c = (void*)0x2; return 0; }
int cuCtxGetDevice(int *d) { if (d) *d = 0; return 0; }
int cuMemAlloc(void **p, size_t s) { if (p) *p = (void*)0x1000; return 0; }
int cuMemAllocHost(void **p, size_t s, unsigned f) { if (p) *p = (void*)0x2000; return 0; }
int cuMemFree(void *p) { return 0; }
int cuMemFreeHost(void *p) { return 0; }
int cuMemcpyHtoD(void *d, const void *s, size_t n) { return 0; }
int cuMemcpyDtoH(void *d, const void *s, size_t n) { return 0; }
int cuMemcpyDtoD(void *d, const void *s, size_t n) { return 0; }
int cuMemcpyHtoDAsync(void *d, const void *s, size_t n, void *st) { return 0; }
int cuMemcpyDtoHAsync(void *d, const void *s, size_t n, void *st) { return 0; }
int cuMemcpyDtoDAsync(void *d, const void *s, size_t n, void *st) { return 0; }
int cuMemsetD8(void *d, unsigned char v, size_t n) { return 0; }
int cuMemsetD16(void *d, unsigned short v, size_t n) { return 0; }
int cuMemsetD32(void *d, unsigned int v, size_t n) { return 0; }
int cuStreamCreate(void **s, unsigned f) { if (s) *s = (void*)0x3000; return 0; }
int cuStreamDestroy(void *s) { return 0; }
int cuStreamSynchronize(void *s) { return 0; }
int cuStreamQuery(void *s) { return 0; }
int cuEventCreate(void **e, unsigned f) { if (e) *e = (void*)0x4000; return 0; }
int cuEventDestroy(void *e) { return 0; }
int cuEventRecord(void *e, void *s) { return 0; }
int cuEventSynchronize(void *e) { return 0; }
int cuEventQuery(void *e) { return 0; }
int cuMemAddressReserve(void **p, size_t s, size_t a, void *b, unsigned long long f) { if (p) *p = (void*)0x5000; return 0; }
int cuMemAddressFree(void *p, size_t s) { return 0; }
int cuMemCreate(void **h, size_t s, void *p, unsigned long long f) { if (h) *h = (void*)0x6000; return 0; }
int cuMemRelease(void *h) { return 0; }
int cuMemMap(void *p, size_t s, size_t o, void *h) { return 0; }
int cuMemUnmap(void *p, size_t s) { return 0; }
int cuMemSetAccess(void *p, size_t s, void *d, int c) { return 0; }
int cuMemGetAccess(unsigned long long *f, void *l, void *p) { if (f) *f = 0; return 0; }
int cuMemGetAllocationGranularity(size_t *g, void *p, int o) { if (g) *g = 1; return 0; }
int cuMemGetAddressRange_v2(void **b, size_t *s, size_t p) { if (b) *b = (void*)0x7000; if (s) *s = 0; return 0; }
int cuMemRetainAllocationHandle(void **h, void *p) { if (h) *h = (void*)0x8000; return 0; }
int cuMemGetHandleForAddressRange(void *h, void *p, size_t s, int o, unsigned long long f) { return 0; }
int cuMemExportToShareableHandle(void *o, void *h, int t, unsigned long long f) { return 0; }
int cuMemImportFromShareableHandle(void **h, void *i, int t) { if (h) *h = (void*)0xB000; return 0; }
int cuPointerGetAttribute(void *d, int a, void *p) { return 0; }
int cuPointerSetAttribute(void *d, int a, void *p) { return 0; }
int cuGetErrorName(int e, const char **n) { if (n) *n = "stub"; return 0; }
int cuGetErrorString(int e, const char **s) { if (s) *s = "stub"; return 0; }
/* libcudart.so.12 (runtime API) */
int cudaMalloc(void **p, size_t s) { if (p) *p = (void*)0x1000; return 0; }
int cudaMallocHost(void **p, size_t s, unsigned f) { if (p) *p = (void*)0x2000; return 0; }
int cudaMallocManaged(void **p, size_t s, unsigned f) { if (p) *p = (void*)0x1000; return 0; }
int cudaFree(void *p) { return 0; }
int cudaFreeHost(void *p) { return 0; }
int cudaMemcpy(void *d, const void *s, size_t n, int k) { return 0; }
int cudaMemcpyAsync(void *d, const void *s, size_t n, int k, void *st) { return 0; }
int cudaMemcpyBatchAsync(void *d, void **s, size_t *n, size_t c, void *a, void *st) { return 0; }
int cudaMemset(void *p, int v, size_t n) { return 0; }
int cudaMemsetAsync(void *p, int v, size_t n, void *st) { return 0; }
int cudaStreamCreateWithFlags(void **s, unsigned f) { if (s) *s = (void*)0x3000; return 0; }
int cudaStreamDestroy(void *s) { return 0; }
int cudaStreamQuery(void *s) { return 0; }
int cudaStreamSynchronize(void *s) { return 0; }
int cudaEventCreateWithFlags(void **e, unsigned f) { if (e) *e = (void*)0x4000; return 0; }
int cudaEventDestroy(void *e) { return 0; }
int cudaEventRecord(void *e, void *s) { return 0; }
int cudaEventSynchronize(void *e) { return 0; }
int cudaEventQuery(void *e) { return 0; }
int cudaGetDevice(int *d) { if (d) *d = 0; return 0; }
int cudaGetDeviceCount(int *c) { if (c) *c = 0; return 0; }
int cudaSetDevice(int d) { return 0; }
int cudaGetDeviceProperties_v2(void *p, int d) { return 0; }
int cudaDeviceCanAccessPeer(int *c, int d, int p) { if (c) *c = 0; return 0; }
int cudaDeviceEnablePeerAccess(int p, unsigned f) { return 0; }
int cudaDeviceDisablePeerAccess(int p) { return 0; }
int cudaDeviceGetPCIBusId(char *b, int l, int d) { return 0; }
int cudaHostAlloc(void **p, size_t s, unsigned f) { if (p) *p = (void*)0x2000; return 0; }
int cudaHostGetDevicePointer(void **d, void *h, unsigned f) { if (d) *d = (void*)0x1000; return 0; }
int cudaHostRegister(void *p, size_t s, unsigned f) { return 0; }
int cudaHostUnregister(void *p) { return 0; }
int cudaGetLastError(void) { return 0; }
int cudaPeekAtLastError(void) { return 0; }
int cudaGetErrorString(int e, const char **s) { if (s) *s = "stub"; return 0; }
int cudaGetErrorName(int e, const char **n) { if (n) *n = "stub"; return 0; }
int cudaPointerGetAttributes(void *a, void *p) { return 0; }
int cudaIpcGetMemHandle(void *h, void *d) { return 0; }
int cudaIpcOpenMemHandle(void **d, void *h, unsigned f) { if (d) *d = (void*)0x1000; return 0; }
int cudaIpcCloseMemHandle(void *d) { return 0; }
int cudaLaunchHostFunc(void *s, void *f, void *a) { return 0; }
CUEOF
    gcc -shared -fPIC -Wl,-soname,libcuda.so.1 \
        -o "$LIBS_DIR/libcuda.so.1" /tmp/mooncake_cuda_stub.c 2>/tmp/smoke-stub.err \
        || { echo "smoke: FAILED - libcuda stub compile failed:" >&2
             tail -20 /tmp/smoke-stub.err >&2 || true
             exit 1; }
    gcc -shared -fPIC -Wl,-soname,libcudart.so.12 \
        -o "$LIBS_DIR/libcudart.so.12" /tmp/mooncake_cuda_stub.c 2>/tmp/smoke-stub.err \
        || { echo "smoke: FAILED - libcudart stub compile failed:" >&2
             tail -20 /tmp/smoke-stub.err >&2 || true
             exit 1; }
fi
# install stubs 后 mooncake.store 必须能 import（store.so 的 UND 符号将来扩展时 stub 这里也得跟）。
if ! python -c 'import mooncake.store' 2>/tmp/smoke-stub.err; then
    echo "smoke: FAILED - mooncake.store still fails after CUDA stub install:" >&2
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

### 跑 specforge 训练（1 步）

```shell #test id="smoke-train" load="model_path>>MODEL_PATH"
set -euo pipefail
SPECFORGE_ROOT="${SPECFORGE_ROOT:-SpecForge}"
TRAINER_DEVICE="${SPECFORGE_TRAINER_DEVICE:-1}"
RECIPE="${SPECFORGE_RECIPE:-examples/configs/online/disaggregated/external/qwen3.5-4b-dflash-online-npu.yaml}"
pushd "$SPECFORGE_ROOT" >/dev/null
# 配方 output_dir 残留的 producer_claim / failed 标记会让 _claim_fresh_control_path 拒绝继续。
rm -rf outputs/qwen3.5-4b-dflash-npu-online
# recipe 默认 tracking.report_to=tensorboard，没装就 ValueError；smoke 1 步不开 tracker。
# Qwen3.5-4B 是多模态 Qwen3_5ForConditionalGeneration，权重前缀 model.language_model.*，
# 显式覆盖 embedding_key 到多模态 layout。
ASCEND_RT_VISIBLE_DEVICES=$TRAINER_DEVICE \
HCCL_CONNECT_TIMEOUT=7200 HCCL_EXEC_TIMEOUT=7200 \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
specforge train -c "$RECIPE" \
    training.max_steps=1 \
    training.batch_size=1 \
    training.accumulation_steps=1 \
    data.max_length=512 \
    training.num_anchors=32 \
    training.save_interval=0 \
    training.log_interval=1 \
    tracking.report_to=none \
    deployment.trainer.nproc_per_node=1 \
    model.target_model_path="<MODEL_PATH>" \
    model.embedding_key="model.language_model.embed_tokens.weight" \
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

```shell #test-result id="smoke-train" fuzzy='...'
smoke: OK - 1-step training completed (exit=0)
```

> 卡 0 capture server、卡 1 trainer、卡 2/3 留空给 HCCL buffer。`--context-length 1024 --mem-fraction-static 0.5` 压住 SGLang KV 池；`training.max_steps=1 training.batch_size=1 data.max_length=512 training.num_anchors=32 deployment.trainer.nproc_per_node=1` 把训练侧压到 1 步最小数据。