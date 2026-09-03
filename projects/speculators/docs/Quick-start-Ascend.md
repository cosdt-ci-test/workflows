# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑 [Speculators](https://github.com/vllm-project/speculators) 端到端：转换 DFlash draft → 抽训练数据 → torchrun 训 draft → `vllm serve` 挂载 draft 做推理 smoke。

## 前置条件

### 硬件

Atlas 900 A2 / A3 或 Ascend 950 系列 NPU，至少 1 卡。

### 基础软件

- Python 3.12 环境
- [CANN 9.1.0](https://ascend.github.io/docs/sources/ascend/quick_install.html)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

### 本文档示例使用的版本

**配套机器**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1），Ubuntu 22.04

**配套镜像**：

swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/vllm-ascend/vllm-ascend:v0.23.0

镜像预装 vllm 0.23.0 + vllm-ascend 0.23.0 + triton-ascend 3.2.2 + torch 2.10.0+cpu + torch_npu 2.10.0.post4 + torchvision 0.25.0+cpu + torchaudio 2.10.0+cpu + transformers 5.5.4 + modelscope 1.39.1 + CANN 9.1.0 + Python 3.12。

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.10.0+cpu（镜像预装） |
| torch_npu | 2.10.0.post4（镜像预装） |
| torchvision | 0.25.0+cpu（镜像预装，vllm-ascend Required-by） |
| torchaudio | 2.10.0+cpu（镜像预装） |
| vllm | 0.23.0（镜像预装） |
| vllm-ascend | 0.23.0（镜像预装） |
| triton-ascend | 3.2.2（镜像预装） |
| triton | 3.5.0（镜像预装） |
| transformers | 5.5.4（镜像预装；speculators 透传范围 >=4.56.1,<5.15.0） |
| modelscope | 1.39.1（镜像预装；下面步骤会钉到 1.37.0） |
| speculators | 最新 release |
| draft 模型 | z-lab/Qwen3-8B-DFlash-b16 |
| verifier | Qwen/Qwen3-8B |

### 前置安装

确认 NPU 设备可见：

```shell
npu-smi info
```

> `npu-smi` 不存在就回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 装驱动。

检查 Python 版本（应输出 3.12.x）：

```shell #test id="check-py"
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

确认镜像预装的 torch / torch_npu stack 不被改过（防御性 rollback：万一之前有人 `pip install torch` 升到 2.12 撞 ABI，这里强制拉回 2.10.0+cpu）：

```shell #test-setup id="check-torch-stack"
pip install --force-reinstall --no-deps \
  -f https://mirrors.aliyun.com/pytorch-wheels/cpu \
  'torch==2.10.0+cpu'
```

> ⚠ **Step 3a（launch_vllm）和 Step 4（vllm serve）启动前必须设 `TORCHDYNAMO_DISABLE=1` + `--enforce-eager`。**
>
> flex_attention 在 dynamo capture 阶段会撞 torch_npu 不支持的 HOP / UB，trace 时直接抛 `Unsupported: Import failure`；关掉 dynamo + cudagraph，让 vllm 走纯 eager 路径就稳。`TORCHDYNAMO_DISABLE=1` 必须 export 在 shell 里、不能塞进 vllm 进程命令行（vllm 自己 fork 之后才会去读 env）。
>
> **torch / torch_npu 不要 pip install 升级**：vllm_ascend_C.so 是按 torch 2.10 的 `at::Tag` namespace 编的；torch 2.11 / 2.12 公网没有任何镜像能下 torch_npu post 版本（Aliyun / Tsinghua / Huawei 全 404），所以连「升 torch + 升 torch_npu」这条路都走不通。

加载 CANN env 并验证镜像预装的 vllm-ascend 栈（含 torch / torch_npu / torchvision / torchaudio / transformers / vllm / vllm-ascend / triton*）：

```shell #test id="verify-vllm-stack"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -c "
import torch, torch_npu, torchvision, torchaudio, transformers
print(f'torch={torch.__version__}')
print(f'torch_npu={torch_npu.__version__}')
print(f'torchvision={torchvision.__version__}')
print(f'torchaudio={torchaudio.__version__}')
print(f'transformers={transformers.__version__}')
print('is_available:', torch.npu.is_available())
print('npu_count:', torch.npu.device_count())
"

python -c "import importlib.metadata; print(f'vllm={importlib.metadata.version(\"vllm\")}')"
python -c "import importlib.metadata; print(f'vllm_ascend={importlib.metadata.version(\"vllm-ascend\")}')"
python -c "import importlib.metadata; print(f'triton_ascend={importlib.metadata.version(\"triton-ascend\")}')"
python -c "import importlib.metadata; print(f'triton={importlib.metadata.version(\"triton\")}')"
```

输出结果如下：

```shell #test-result id="verify-vllm-stack" fuzzy='xxx'
torch=2.10.0+cpu
torch_npu=2.10.0.post4
torchvision=0.25.0+cpu
torchaudio=2.10.0+cpu
transformers=xxx
is_available: True
npu_count: xxx
vllm=0.23.0+empty
vllm_ascend=0.23.0
triton_ascend=3.2.2
triton=3.5.0
```

装 modelscope：

```shell #test id="install-deps"
uv pip install 'modelscope==1.37.0'
python -c "import modelscope; print(f'modelscope={modelscope.__version__}')"
```

输出结果如下：

```shell #test-result id="install-deps"
modelscope=1.37.0
```

## 安装 Speculators

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆 speculators 上游 release tag 源码到 `/root/speculators/` 并 editable 安装（`<ref>` 由工作流注入最新 release tag）：

```shell #test id="speculators-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/vllm-project/speculators.git /root/speculators
cd /root/speculators
uv pip install -e .
speculators --version
python -c "from importlib.metadata import version; print('speculators', version('speculators'))"
```

输出结果如下：

```shell #test-result id="speculators-install-source" fuzzy='xxx'
speculators version: xxx
speculators xxx
```

## 端到端：convert → 训练数据生成 → 训练 → 部署

### 前置：下载 draft 与 verifier

从 ModelScope 拉 draft 模型 z-lab/Qwen3-8B-DFlash-b16：

```shell #test-setup store="draft_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('z-lab/Qwen3-8B-DFlash-b16'))" | tail -n 1
```

从 ModelScope 拉 verifier 模型 Qwen/Qwen3-8B：

```shell #test-setup store="verifier_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B'))" | tail -n 1
```

### Step 1：convert（DFlash 算法）

`speculators convert` 把本地 draft + verifier 读进来按 DFlash 算法重映射权重、写到 `/root/dflash-qwen3-8b-converted/`。CLI 的 `--algorithm` 不支持 `dflash`，走 Python API：

```shell #test-setup store="dflash_path" load="draft_path>>draft_path" load="verifier_path>>verifier_path"
python << 'PY'
from speculators.convert import convert_model

convert_model(
    model="<draft_path>",
    verifier="<verifier_path>",
    algorithm="dflash",
    output_path="/root/dflash-qwen3-8b-converted",
)
PY
test -f /root/dflash-qwen3-8b-converted/config.json
test -f /root/dflash-qwen3-8b-converted/model.safetensors
echo "/root/dflash-qwen3-8b-converted"
```

确认 convert 产物存在（`config.json` + `model.safetensors`）：

```shell #test id="pipeline-step1-convert" load="dflash_path>>dflash_path"
ls -1 <dflash_path>/config.json <dflash_path>/model.safetensors
echo <dflash_path>
```

输出结果如下：

```shell #test-result id="pipeline-step1-convert"
/root/dflash-qwen3-8b-converted/config.json
/root/dflash-qwen3-8b-converted/model.safetensors
/root/dflash-qwen3-8b-converted
```

### Step 2：训练数据预处理

用上游 `scripts/prepare_data.py` 把 JSONL chat 数据 tokenize 写到 `/root/dflash-train-data/`（HF arrow 数据集）：

```shell #test-setup store="data_path" load="verifier_path>>verifier_path"
set -euo pipefail
DATA_DIR=/root/dflash-train-data
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

# 顶层 key 必须是 "conversations"（不是 "messages"）；每条 user + assistant 都要填
cat > /tmp/prompts.jsonl << 'JSONL'
{"conversations":[{"role":"user","content":"Briefly describe AI topic #0."},{"role":"assistant","content":"AI is a field of computer science."}]}
{"conversations":[{"role":"user","content":"Briefly describe AI topic #1."},{"role":"assistant","content":"AI is a field of computer science."}]}
{"conversations":[{"role":"user","content":"Briefly describe AI topic #2."},{"role":"assistant","content":"AI is a field of computer science."}]}
{"conversations":[{"role":"user","content":"Briefly describe AI topic #3."},{"role":"assistant","content":"AI is a field of computer science."}]}
{"conversations":[{"role":"user","content":"Briefly describe AI topic #4."},{"role":"assistant","content":"AI is a field of computer science."}]}
{"conversations":[{"role":"user","content":"Briefly describe AI topic #5."},{"role":"assistant","content":"AI is a field of computer science."}]}
{"conversations":[{"role":"user","content":"Briefly describe AI topic #6."},{"role":"assistant","content":"AI is a field of computer science."}]}
{"conversations":[{"role":"user","content":"Briefly describe AI topic #7."},{"role":"assistant","content":"AI is a field of computer science."}]}
{"conversations":[{"role":"user","content":"Briefly describe AI topic #8."},{"role":"assistant","content":"AI is a field of computer science."}]}
{"conversations":[{"role":"user","content":"Briefly describe AI topic #9."},{"role":"assistant","content":"AI is a field of computer science."}]}
JSONL

cd /root/speculators

python scripts/prepare_data.py \
  --model "<verifier_path>" \
  --data /tmp/prompts.jsonl \
  --output "$DATA_DIR" \
  --max-samples 10 \
  --seq-length 8192 \
  --overwrite

echo "$DATA_DIR"
```

确认数据集已落盘（行数 + `token_freq.pt` 存在）：

```shell #test id="pipeline-step2-extract" load="data_path>>data_path"
echo <data_path>
python -c "from datasets import load_from_disk; print(len(load_from_disk('<data_path>')))"
python -c "
import torch
from pathlib import Path
p = Path('<data_path>') / 'token_freq.pt'
print('exists:', p.exists(), 'size:', p.stat().st_size if p.exists() else 0)
freq = torch.load(p, weights_only=True)
print('token_freq keys:', list(freq.keys()) if isinstance(freq, dict) else type(freq).__name__)
print('len:', len(freq))
"
```

输出结果如下：

```shell #test-result id="pipeline-step2-extract" fuzzy='xxx'
/root/dflash-train-data
10
exists: True size: xxx
token_freq keys: xxx
len: xxx
```

### Step 3：训练（单卡 torchrun）

单卡 64 GB NPU 装不下「vllm 16 GB 权重 + KV + train draft 模型 + optimizer 激活」并发跑，所以拆成两步：先生成 hidden_states 缓存，再离线训。

起 vllm 一次性 generate 10 条 hidden_states 写到 `/tmp/hs-train/`（train.py 的 FileBackend 直接读这个目录；生成完杀 vllm 释放全部 NPU 给后续 train 留 64 GB 完整空间）：

```shell #test-setup store="hs_dir" load="data_path>>data_path" load="verifier_path>>verifier_path"
set -euo pipefail

cd /root/speculators

HS_DIR=/tmp/hs-train
rm -rf "$HS_DIR"
mkdir -p "$HS_DIR"

# 关 dynamo + --enforce-eager：见 install-torch 步骤的 ⚠ 说明 —— 避开
# dynamo capture 把 enable_custom_op() 的 fallback 吃掉触雷
export TORCHDYNAMO_DISABLE=1
rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true

# CANN env：vllm 进程 fork 出 EngineCore 后会动态 load torch_npu 的 atb
# extension (`/usr/local/.../libop_plugin_atb.so`)，缺 LD_LIBRARY_PATH 里的
# `libatb.so` 直接 `OSError: libatb.so: cannot open shared object file`。
# ascend-toolkit/set_env.sh 不含 nnal 的 atb 路径，必须单独 source。
#
# atb/set_env.sh 是 zsh 写的，里面有两处在 bash + `set -u` 下报错：
#   line 12 `until [[ -z "$1" ]]`（$1 没传时 unbound）
#   line 43 `if [[ -n "$ZSH_VERSION" ]]`（ZSH_VERSION 没设时 unbound）
# CI test framework 会自动 prepend `set -euo pipefail`，所以 source 前
# 必须 `set +u` 关掉 nounset，source 完再 `set -u` 恢复 —— 单 pre-export
# ZSH_VERSION 顶不住 $1 那个坑
set +u
export ZSH_VERSION="${ZSH_VERSION:-}"
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

setsid nohup python scripts/launch_vllm.py "<verifier_path>" \
  --target-layer-ids 2 18 34 \
  --hidden-states-path "$HS_DIR" \
  -- \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --enforce-eager \
  > /tmp/vllm-gen.log 2>&1 < /dev/null &
VLLM_GEN_PID=$!
VLLM_GEN_PGID=$(ps -o pgid= -p "$VLLM_GEN_PID" | tr -d ' ')
cleanup_vllm_gen() {
  kill -- -"$VLLM_GEN_PGID" 2>/dev/null || true
  for _pid in $(pgrep -P "$VLLM_GEN_PID" 2>/dev/null) "$VLLM_GEN_PID"; do
    kill -9 "$_pid" 2>/dev/null || true
  done
  pkill -9 -x "vllm" 2>/dev/null || true
}
trap cleanup_vllm_gen EXIT

# 等 /health 200（最长 6 min）
VLLM_READY=0
for i in {1..180}; do
  if curl -sf http://127.0.0.1:8000/health > /dev/null; then
    VLLM_READY=1
    break
  fi
  sleep 2
done
if [ "$VLLM_READY" != "1" ]; then
  echo "vllm server failed to come up within 6 min; tail of vllm-gen.log:" >&2
  tail -80 /tmp/vllm-gen.log >&2
  cleanup_vllm_gen
  exit 1
fi

python scripts/data_generation_offline.py \
  --model "<verifier_path>" \
  --preprocessed-data "<data_path>" \
  --output "$HS_DIR" \
  --max-samples 10 \
  --concurrency 4 \
  --validate-outputs >/tmp/hs-gen.log 2>&1 || HS_RC=$?
HS_RC=${HS_RC:-0}
# 日志走 stderr，避免污染 store 捕获的 stdout（store 把最后一行的值当变量）
tail -30 /tmp/hs-gen.log >&2

HS_COUNT=$(ls -1 "$HS_DIR"/hs_*.safetensors 2>/dev/null | wc -l)
if [ "$HS_RC" -ne 0 ] || [ "$HS_COUNT" -ne 10 ]; then
  echo "=== data_generation_offline.py failed (rc=$HS_RC, hs_count=$HS_COUNT/10); full log ===" >&2
  cat /tmp/hs-gen.log >&2
  cleanup_vllm_gen
  exit 1
fi

cleanup_vllm_gen
sleep 5

echo "$HS_DIR"
```

用上游 `scripts/train.py` 单卡 torchrun 训 1 epoch × 10 sample（smoke 验证管线通，不指望 loss 真下降）：

```shell #test-setup store="checkpoint_path" load="hs_dir>>hs_dir" load="data_path>>data_path" load="verifier_path>>verifier_path"
set -euo pipefail
CHECKPOINT_DIR=/root/dflash-trained
rm -rf "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

cd /root/speculators

# 关 dynamo + ASCEND_LAUNCH_BLOCKING=1：见 install-torch 步骤的 ⚠ 说明。
# ASCEND_LAUNCH_BLOCKING=1 让 NPU kernel 错误同步上抛（不设的话 CANN 错误是
# silent kill，下次失败看不到 traceback）
export TORCHDYNAMO_DISABLE=1
export ASCEND_LAUNCH_BLOCKING=1

# CANN env：torch_npu 的 atb extension 需要 `libatb.so`；Step 3a 已 source，
# 同一个 shell 状态如果跨 step 中断了就要再 source 一次（bash 子 shell 不继承）。
# set +u / set -u：见 Step 3a 注释（atb 是 zsh 写的，bash + set -u 会撞）
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# --hidden-states-dtype float32：spec 把 LN.weight 写死 fp32，autocast 不会
# cast LN；而 dflash.forward 复用的 V 来自 hidden_states 缓存（默认 bf16），
# Q/K fp32 + V bf16 会撞 flex_attention 的 dtype check。把 V 也 cast fp32
# 让 Q/K/V 对齐。A2 64 GB 装得下 1.9 GB fp32 缓存。备注：CUDA 跑同一份
# train.py 也会有这个 dtype 不匹配，spec 是先有 CUDA 路径后迁到 NPU。
# --on-missing raise 强制走 FileBackend 读 <hs_dir> 缓存；不带 --vllm-endpoint
# 让 dataloader 不去问不存在的 server
#
# --max-anchors 32 --draft-attn-impl sdpa：spec 默认是 `--max-anchors 3072
# --draft-attn-impl simple_flex_attention`（CUDA 路径）；torch 2.10 的
# `flex_attention._validate_device` 只放行 CUDA / CPU / HPU，NPU device
# 直接 ValueError，不是 OOM 是 hard reject。64 GB NPU 也必须切 sdpa，不
# 是只 32 GB 才需要。sdpa 不走 DFlash 的 anchor-block 稀疏 mask，用各
# draft layer 的 sliding-window（window=2048）；smoke 只验管线通，
# 真训练需要补一个 NPU 能跑的 flex_attention 后端。sdpa + 32 anchor
# 实测 val/loss=6.688，3.5 GB 落盘，~3 min/epoch on 910B4。
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --verifier-name-or-path "<verifier_path>" \
  --data-path "<data_path>" \
  --hidden-states-path "<hs_dir>" \
  --hidden-states-dtype float32 \
  --save-path "$CHECKPOINT_DIR" \
  --draft-vocab-size 32000 \
  --epochs 1 \
  --lr 3e-4 \
  --speculator-type dflash \
  --block-size 8 \
  --max-anchors 32 \
  --draft-attn-impl sdpa \
  --num-layers 5 \
  --target-layer-ids 2 18 34 \
  --on-missing raise >/tmp/train.log 2>&1 || TRAIN_RC=$?
TRAIN_RC=${TRAIN_RC:-0}

if [ "$TRAIN_RC" -ne 0 ]; then
  echo "=== train.py failed (rc=$TRAIN_RC); full train.log follows ===" >&2
  cat /tmp/train.log >&2
  exit 1
fi
# trainer 把 checkpoint 写到 "$CHECKPOINT_DIR/<step>/" 子目录；Step 4 用
# <checkpoint_path> 当 draft_model 路径，需要直接读 config.json /
# model.safetensors，把最新子目录的内容拷到根，然后清掉 trainer 的
# 元数据（optimizer/scheduler state、run.yaml、training_state.json、
# val_metrics.json、checkpoint_best/epoch0_end symlinks、checkpoint
# 子目录），只留 config.json + model.safetensors 给下游 pipeline-step3-train
# test 做精确 ls 对账。
#
# LATEST_CKPT 只看数字子目录（[0-9]*），不吸 symlink-to-dir：
# `ls -1d "$CHECKPOINT_DIR"/*/` 会把 checkpoint_best -> 0 / epoch0_end
# -> 0 这类 symlink-to-dir 也列出来，sort -V 还会把 epoch0_end 排
# 在 0 后面，结果拿到一个 symlink path 让 cp 间接跟链；改用 [0-9]*
# glob 直接锁 trainer 自己创建的 step 子目录
LATEST_CKPT=$(ls -1d "$CHECKPOINT_DIR"/[0-9]*/ 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CKPT" ] && [ "$LATEST_CKPT" != "$CHECKPOINT_DIR/" ]; then
  cp -af "$LATEST_CKPT"/. "$CHECKPOINT_DIR"/
fi
if ! test -f "$CHECKPOINT_DIR/config.json" || ! test -f "$CHECKPOINT_DIR/model.safetensors"; then
  echo "=== train.py rc=0 但 checkpoint 缺失 (looked under $CHECKPOINT_DIR/) ===" >&2
  cat /tmp/train.log >&2
  exit 1
fi
# 清 trainer 元数据：symlinks / 子目录 / optimizer & scheduler state /
# run.yaml / training_state.json / val_metrics.json / config.py
rm -f "$CHECKPOINT_DIR"/checkpoint_best "$CHECKPOINT_DIR"/epoch0_end
rm -rf "$CHECKPOINT_DIR"/[0-9]*/
rm -f "$CHECKPOINT_DIR"/optimizer_state_dict.pt \
      "$CHECKPOINT_DIR"/scheduler_state_dict.pt \
      "$CHECKPOINT_DIR"/run.yaml \
      "$CHECKPOINT_DIR"/training_state.json \
      "$CHECKPOINT_DIR"/val_metrics.json \
      "$CHECKPOINT_DIR"/train_command.txt \
      "$CHECKPOINT_DIR"/config.py

echo "$CHECKPOINT_DIR"
```

确认训练产物（checkpoint 目录 + `config.json` + `model.safetensors`）：

```shell #test id="pipeline-step3-train" load="checkpoint_path>>checkpoint_path"
echo <checkpoint_path>
ls -1 <checkpoint_path>
```

输出结果如下：

```shell #test-result id="pipeline-step3-train"
/root/dflash-trained
config.json
model.safetensors
```

### Step 4：`vllm serve` 挂 draft 做推理

起 vllm-ascend serve 把训好的 draft 挂上做 chat completion smoke（8 token completion）：

```shell #test id="pipeline-step4-serve" load="checkpoint_path>>draft_model" load="verifier_path>>verifier_path"
# 关 dynamo + --enforce-eager：见 install-torch 步骤的 ⚠ 说明
export TORCHDYNAMO_DISABLE=1
rm -rf /root/.cache/vllm/torch_compile_cache 2>/dev/null || true

# CANN env：见 Step 3a 注释 —— `libatb.so` 不在 ascend-toolkit 默认 LD path 里
# set +u / set -u：见 Step 3a 注释（atb 是 zsh 写的）
set +u
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

# num_speculative_tokens=5：vllm-ascend 限制 (num_speculative_tokens + 1) ≤ 15
nohup vllm serve "<verifier_path>" \
  --host 127.0.0.1 --port 8000 \
  --served-model-name Qwen/Qwen3-8B \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --speculative-config '{"method":"dflash","model":"<draft_model>","num_speculative_tokens":5}' \
  > /tmp/vllm-serve.log 2>&1 &
VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null" EXIT

# 等 /health 200（最长 6 min）
for i in {1..180}; do
  curl -sf http://127.0.0.1:8000/health > /dev/null && break
  sleep 2
done

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}'

kill "$VLLM_PID" 2>/dev/null || true
```

输出结果如下：

```shell #test-result id="pipeline-step4-serve" fuzzy='xxx'
{"id":"chatcmpl-xxx","object":"chat.completion","created":xxx,"model":"Qwen/Qwen3-8B","choices":[{"index":0,"message":{"role":"assistant","content":"xxx","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":null},"logprobs":null,"finish_reason":"length","stop_reason":null,"token_ids":null,"routed_experts":null}],"service_tier":null,"system_fingerprint":"vllm-xxx","usage":{"prompt_tokens":xxx,"total_tokens":xxx,"completion_tokens":xxx,"prompt_tokens_details":null,"completion_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"prompt_text":null,"kv_transfer_params":null}
```

### 编程式入口：SpeculatorsConfig / TokenProposalConfig

验证 `SpeculatorsConfig` / `TokenProposalConfig` 可以在 NPU 环境 import + 实例化（不依赖 GPU 计算）：

```shell #test id="config-import" load="verifier_path>>verifier_path"
python << 'PY'
from speculators import VerifierConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

verifier = VerifierConfig(
    name_or_path="<verifier_path>",
    architectures=["Qwen3ForCausalLM"],
)
proposal = GreedyTokenProposalConfig(
    proposal_type="greedy",
    speculative_tokens=5,
    verifier_accept_k=1,
    accept_tolerance=0.0,
)
print("verifier:", verifier.name_or_path)
print("verifier architectures:", verifier.architectures)
print("proposal type:", proposal.proposal_type)
print("proposal speculative_tokens:", proposal.speculative_tokens)
PY
```

输出结果如下：
```shell #test-result id="config-import"
verifier: /root/.cache/modelscope/hub/models/Qwen/Qwen3-8B
verifier architectures: ['Qwen3ForCausalLM']
proposal type: greedy
proposal speculative_tokens: 5
```
