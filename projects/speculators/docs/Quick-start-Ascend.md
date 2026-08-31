# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上跑 [Speculators](https://github.com/vllm-project/speculators) 端到端：转换 DFlash draft → 抽训练数据 → torchrun 训 draft → `vllm serve` 挂载 draft 做推理 smoke。配套 vllm-ascend 0.23.0 + vLLM 0.23.0。

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

镜像预装 vllm==0.23.0 + vllm-ascend==0.23.0 + triton-ascend==3.2.2 + torch 2.10.0+cpu + torch_npu 2.10.0.post4 + CANN 9.1.0 + Python 3.12。

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post4 |
| vllm | 0.23.0（镜像预装） |
| vllm-ascend | 0.23.0（镜像预装） |
| triton-ascend | 3.2.2（镜像预装） |
| triton | 3.5.0（镜像预装） |
| transformers | 由 `speculators` 透传拉入（>=4.56.1,<5.15.0） |
| modelscope | 1.37.0 |
| speculators | 最新 release |
| draft 模型 | z-lab/Qwen3-8B-DFlash-b16 |
| verifier | Qwen/Qwen3-8B |

### 前置安装

确认 NPU 可见：

```shell
npu-smi info
```

> `npu-smi` 不存在就回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 装驱动。

检查 Python：

```shell #test id="check-py"
python --version
```
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

验证镜像预装的 vllm-ascend 栈：

```shell #test id="verify-vllm-stack"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -c "import torch, torch_npu; print(f'torch={torch.__version__}'); print(f'torch_npu={torch_npu.__version__}'); print('is_available:', torch.npu.is_available()); print('npu_count:', torch.npu.device_count())"

python -c "import importlib.metadata; print(f'vllm={importlib.metadata.version(\"vllm\")}')"
python -c "import importlib.metadata; print(f'vllm_ascend={importlib.metadata.version(\"vllm-ascend\")}')"
python -c "import importlib.metadata; print(f'triton_ascend={importlib.metadata.version(\"triton-ascend\")}')"
python -c "import importlib.metadata; print(f'triton={importlib.metadata.version(\"triton\")}')"
```

```shell #test-result id="verify-vllm-stack" fuzzy='xxx'
torch=2.10.0+cpu
torch_npu=2.10.0.post4
is_available: True
npu_count: xxx
vllm=0.23.0+empty
vllm_ascend=0.23.0
triton_ascend=3.2.2
triton=3.5.0
```

安装 modelscope：

```shell #test-setup
uv pip install 'modelscope==1.37.0'
```

```shell #test id="install-deps"
python -c "import modelscope; print(f'modelscope={modelscope.__version__}')"
```

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

```shell #test id="speculators-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/vllm-project/speculators.git /root/speculators
cd /root/speculators
uv pip install -e .
speculators --version
python -c "from importlib.metadata import version; print('speculators', version('speculators'))"
```

`\<ref>` 由工作流注入最新 release tag。

```shell #test-result id="speculators-install-source" fuzzy='xxx'
speculators version: xxx
speculators xxx
```

## 端到端：convert → 训练数据生成 → 训练 → 部署

### 前置：下载 draft 与 verifier

```shell #test-setup store="draft_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('z-lab/Qwen3-8B-DFlash-b16'))" | tail -n 1
```

```shell #test-setup store="verifier_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B'))" | tail -n 1
```

### Step 1：convert（DFlash 算法）

CLI 的 `--algorithm` 不支持 `dflash`，走 Python API：

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

```shell #test id="pipeline-step1-convert" load="dflash_path>>dflash_path"
ls -1 <dflash_path>/config.json <dflash_path>/model.safetensors
echo <dflash_path>
```

```shell #test-result id="pipeline-step1-convert"
/root/dflash-qwen3-8b-converted/config.json
/root/dflash-qwen3-8b-converted/model.safetensors
/root/dflash-qwen3-8b-converted
```

### Step 2：训练数据预处理

用上游 `scripts/prepare_data.py` 把 JSONL chat 数据 tokenize 写成 HF arrow 数据集。

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

```shell #test-result id="pipeline-step2-extract" fuzzy='xxx'
/root/dflash-train-data
10
exists: True size: xxx
token_freq keys: xxx
len: xxx
```

### Step 3：训练（单卡 torchrun）

离线模式：先起 vllm 一次性 generate hidden_states 落盘，杀 vllm 释放 NPU，再 train.py 离线读 cache。

```shell #test-setup store="hs_dir" load="data_path>>data_path" load="verifier_path>>verifier_path"
set -euo pipefail

cd /root/speculators

HS_DIR=/tmp/hs-train
rm -rf "$HS_DIR"
mkdir -p "$HS_DIR"

setsid nohup python scripts/launch_vllm.py "<verifier_path>" \
  --target-layer-ids 2 18 34 \
  --hidden-states-path "$HS_DIR" \
  -- \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
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
  echo "vllm server failed to come up within 6 min; tail of vllm-gen.log:"
  tail -80 /tmp/vllm-gen.log
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
tail -30 /tmp/hs-gen.log

HS_COUNT=$(ls -1 "$HS_DIR"/hs_*.safetensors 2>/dev/null | wc -l)
if [ "$HS_RC" -ne 0 ] || [ "$HS_COUNT" -ne 10 ]; then
  echo "=== data_generation_offline.py failed (rc=$HS_RC, hs_count=$HS_COUNT/10); full log ==="
  cat /tmp/hs-gen.log
  cleanup_vllm_gen
  exit 1
fi

cleanup_vllm_gen
sleep 5

echo "$HS_DIR"
```

```shell #test-setup store="checkpoint_path" load="hs_dir>>hs_dir" load="data_path>>data_path" load="verifier_path>>verifier_path"
set -euo pipefail
CHECKPOINT_DIR=/root/dflash-trained
rm -rf "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

cd /root/speculators

# --on-missing raise 强制走 FileBackend 读 <hs_dir> 缓存；不带 --vllm-endpoint 让
# dataloader 不会去问不存在的 server
torchrun --standalone --nproc_per_node=1 scripts/train.py \
  --verifier-name-or-path "<verifier_path>" \
  --data-path "<data_path>" \
  --hidden-states-path "<hs_dir>" \
  --save-path "$CHECKPOINT_DIR" \
  --draft-vocab-size 32000 \
  --epochs 1 \
  --lr 3e-4 \
  --speculator-type dflash \
  --block-size 8 \
  --max-anchors 3072 \
  --num-layers 5 \
  --target-layer-ids 2 18 34 \
  --on-missing raise >/tmp/train.log 2>&1 || TRAIN_RC=$?
TRAIN_RC=${TRAIN_RC:-0}

if [ "$TRAIN_RC" -ne 0 ]; then
  echo "=== train.py failed (rc=$TRAIN_RC); full train.log follows ===" >&2
  cat /tmp/train.log >&2
  exit 1
fi
if ! test -f "$CHECKPOINT_DIR/config.json" || ! test -f "$CHECKPOINT_DIR/model.safetensors"; then
  echo "=== train.py rc=0 但 checkpoint 缺失 ===" >&2
  cat /tmp/train.log >&2
  exit 1
fi

echo "$CHECKPOINT_DIR"
```

```shell #test id="pipeline-step3-train" load="checkpoint_path>>checkpoint_path"
echo <checkpoint_path>
ls -1 <checkpoint_path>
```

```shell #test-result id="pipeline-step3-train"
/root/dflash-trained
config.json
model.safetensors
```

### Step 4：`vllm serve` 挂 draft 做推理

```shell #test id="pipeline-step4-serve" load="checkpoint_path>>draft_model" load="verifier_path>>verifier_path"
# num_speculative_tokens=5：vllm-ascend 限制 (num_speculative_tokens + 1) ≤ 15
nohup vllm serve "<verifier_path>" \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.85 \
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

```shell #test-result id="pipeline-step4-serve" fuzzy='xxx'
{"id":"chatcmpl-xxx","object":"chat.completion","created":xxx,"model":"Qwen/Qwen3-8B","choices":[{"index":0,"message":{"role":"assistant","content":"xxx"},"finish_reason":"length"}]}
```

### 编程式入口：SpeculatorsConfig / TokenProposalConfig

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

```shell #test-result id="config-import"
verifier: /root/.cache/modelscope/hub/models/Qwen/Qwen3-8B
verifier architectures: ['Qwen3ForCausalLM']
proposal type: greedy
proposal speculative_tokens: 5
```
