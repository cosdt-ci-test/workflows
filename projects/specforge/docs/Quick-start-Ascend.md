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
uv pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi torch_npu==2.11.0
# specforge 的依赖（无 CUDA 哨兵，正常装）
uv pip install transformers==5.8.1 datasets tqdm accelerate huggingface-hub numpy openai-harmony pydantic psutil pyyaml safetensors requests tensorboard typing-extensions wandb yunchang fastapi uvicorn aiohttp pyzmq python-multipart
# sglang 0.5.14 上游 requires_dist 里非 CUDA-only 项；里头 quack-kernels 自己带 nvidia-cutlass-dsl<0 哨兵，
# torch / numpy / pydantic / 等基础 dep 已经装好，整批 --no-deps 装
uv pip install --no-deps orjson anthropic apache-tvm-ffi av blobfile build compressed-tensors decord2 distro easydict einops gguf interegular IPython kernels llguidance mistral_common msgspec ninja openai outlines packaging partial_json_parser pillow prometheus-client py-spy pybase64 quack-kernels scipy sentencepiece setproctitle sgl-deep-gemm starlette triton torchvision
# sglang wheel 本身 --no-deps 装（cluster 镜像把它的 Requires-Dist cuda-python 改成 <0 哨兵，绕开解析）
uv pip install --no-deps --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi sglang==0.5.14
```

检查 torch / torch_npu / sglang 是否装好且 NPU 设备可用：

```shell #test id="check-torch"
python -c "import torch, torch_npu, sglang; print('torch=', torch.__version__); print('torch_npu=', torch_npu.__version__); print('sglang', sglang.__version__); print('is_available:', torch.npu.is_available()); print('count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-torch"
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
# mooncake-transfer-engine 是 specforge 在线训练里 specforge runtime 的 client 端。
# master server 二进制（mooncake_master）从仓库 release tarball 拿；smoke 直接跑仓内构建好的二进制。
uv pip install --no-deps 'mooncake-transfer-engine @ https://github.com/kvcache-ai/Mooncake/archive/refs/heads/main.tar.gz#subdirectory=mooncake-transfer-engine'
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
git clone --depth 1 --branch <ref> https://github.com/sgl-project/SpecForge.git SpecForge
cd SpecForge
uv pip install --no-deps .
specforge --version
```

\<ref> 为最新的 release 分支名。

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
...
{train,export,benchmark}
```

`specforge train --help` 展示 typed run config 入口：

```shell #test id="specforge-train-help"
specforge train --help
```

输出结果类似如下：

```shell #test-result id="specforge-train-help"
usage: specforge train [-h] -c CONFIG [--role {auto,all,producer,consumer,both}] [--node-rank NODE_RANK] [--plan] [overrides ...]
...
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
cleanup() {
    echo "smoke: cleanup"
    pkill -9 -f sglang.launch_server 2>/dev/null || true
    pkill -9 -f mooncake_master 2>/dev/null || true
    pkill -9 -f "specforge train" 2>/dev/null || true
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
ASCEND_PATCH=$(ls patches/sglang/*/spec-capture-ascend-mount.patch 2>/dev/null | head -1 || true)
if [[ -n "$ASCEND_PATCH" ]]; then
    SGLANG_DIR=$(python -c "import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))")
    echo "smoke: applying ascend companion patch ($ASCEND_PATCH)"
    pushd "$SGLANG_DIR" >/dev/null
    git apply "$OLDPWD/$ASCEND_PATCH" 2>&1 || echo "smoke: ascend patch already applied (ok)"
    popd >/dev/null
fi
popd >/dev/null

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
for _ in $(seq 1 30); do
    if nc -z 127.0.0.1 "$MOONCAKE_RPC_PORT" 2>/dev/null; then
        echo "smoke: mooncake ready (rpc $MOONCAKE_RPC_PORT, pid=$MOONCAKE_PID)"
        break
    fi
    sleep 1
done
if ! nc -z 127.0.0.1 "$MOONCAKE_RPC_PORT" 2>/dev/null; then
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
    --max-model-len 1024 \
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

> 卡 0 跑 capture server，卡 1 跑 trainer，卡 2/3 空闲给 HCCL buffer。Smoke 的 `--max-model-len 1024 --mem-fraction-static 0.5` 把 SGLang KV池压住，`training.max_steps=1 training.batch_size=1 training.max_length=512 training.num_anchors=32 deployment.trainer.nproc_per_node=1` 把训练侧压到 1 步最小数据。