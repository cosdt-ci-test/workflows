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

镜像预装 vllm==0.23.0 + vllm-ascend==0.23.0 + triton-ascend==3.2.2 + torch 2.10.0+cpu + torch_npu 2.10.0.post4 + CANN 9.1.0 + Python 3.12；torch + torch_npu 由 `### 前置安装` 的 `#test-setup install-torch` 步骤当场升到 2.13.0+cpu / 2.13.0rc1（仅升 torch 栈，base 镜像本身不变）。

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.13.0+cpu（`#test-setup install-torch` 从镜像预装的 2.10.0+cpu 升上来） |
| torch_npu | 2.13.0rc1（`#test-setup install-torch` 从镜像预装的 2.10.0.post4 升上来） |
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

确认 NPU 设备可见：

```shell
npu-smi info
```

> `npu-smi` 不存在就回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 装驱动。

检查 Python 版本（应输出 3.12.x）：

```shell #test id="check-py"
python --version
```
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

升级 torch 栈到 2.13（CPU-only build + torch_npu 2.13.0rc1），保留镜像的 CANN 9.1.0 不动：

```shell #test-setup id="install-torch"
# torch 2.13.0+cpu（CPU-only build，跟当前 2.10.0+cpu 同形态）：阿里云主 pypi
# 只发 torch-2.13.0（CUDA build），不发 +cpu 变体；显式走 PyTorch 官方 CPU 索引
# 拉 +cpu wheel，避免给镜像拽入 CUDA 库。--force-reinstall 因为镜像预装的
# 2.10.0+cpu 是 PEP 660 不可变缓存的 wheel，--upgrade 在版本跨度大的时候不替换。
uv pip install --index-url https://download.pytorch.org/whl/cpu \
  --upgrade --force-reinstall 'torch==2.13.0+cpu'

# torch_npu 2.13.0rc1 修了 flex_attention HOP 在 AutocastPrivateUse1 dispatch
# key 上没注册 kernel 的问题（torch_npu/utils/patch_flexattention.py::
# _register_npu_flex_attention_autocast），是这次 smoke 失败的根治版。
# --no-deps 因为 torch 已经在上一步固定好，且 torch_npu 的依赖声明走 find-links
# 会跨索引解析冲突（aliyun 索引没有 torch_npu 元数据），用 --no-deps 隔离避免
# 误判。⚠ vllm-ascend 0.23.0 可能 pin 了 torch_npu 版本约束，Step 2/4 启动 vllm
# 时如果 import torch_npu 撞版本不兼容，要单独处理（升 vllm-ascend 或同款隔离）。
uv pip install \
  --find-links https://mirrors.aliyun.com/pypi/simple/torch-npu/ \
  --find-links https://mirrors.huaweicloud.com/ascend/repos/pypi/torch-npu/ \
  --no-deps --upgrade --force-reinstall 'torch_npu==2.13.0rc1'
```

加载 CANN env 并验证镜像预装的 vllm-ascend 栈（应输出下表的版本号）：

```shell #test id="verify-vllm-stack"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -c "import torch, torch_npu; print(f'torch={torch.__version__}'); print(f'torch_npu={torch_npu.__version__}'); print('is_available:', torch.npu.is_available()); print('npu_count:', torch.npu.device_count())"

python -c "import importlib.metadata; print(f'vllm={importlib.metadata.version(\"vllm\")}')"
python -c "import importlib.metadata; print(f'vllm_ascend={importlib.metadata.version(\"vllm-ascend\")}')"
python -c "import importlib.metadata; print(f'triton_ascend={importlib.metadata.version(\"triton-ascend\")}')"
python -c "import importlib.metadata; print(f'triton={importlib.metadata.version(\"triton\")}')"
```

```shell #test-result id="verify-vllm-stack" fuzzy='xxx'
torch=2.13.0+cpu
torch_npu=2.13.0.rc1
is_available: True
npu_count: xxx
vllm=0.23.0+empty
vllm_ascend=0.23.0
triton_ascend=3.2.2
triton=3.5.0
```

装 modelscope（用来从 ModelScope 拉权重；镜像不含）：

```shell #test-setup
uv pip install 'modelscope==1.37.0'
```

确认 modelscope 版本：

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

克隆 speculators 上游 release tag 源码到 `/root/speculators/` 并 editable 安装（`<ref>` 由工作流注入最新 release tag）：

```shell #test id="speculators-install-source" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/vllm-project/speculators.git /root/speculators
cd /root/speculators
uv pip install -e .
speculators --version
python -c "from importlib.metadata import version; print('speculators', version('speculators'))"
```

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
# tail 必须走 stderr —— vllm client 在 hs-gen.log 里打 "INFO HTTP Request: ...
# HTTP/1.1 200 OK"，30 行 ≈ 2.5 KB。store="hs_dir" 全量捕获 stdout（含最终
# echo "$HS_DIR"），下游 load="h_dir>>h_dir" 把 <h_dir> 塞进 bash 注释，注释
# 里的换行把 # ... 截断，bash 当命令执行 "HTTP/1.1 200 OK" 就 rc=127 挂掉。
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

# 关 dynamo —— speculators v0.7.0 src/speculators/models/dflash/core.py:30 有
# 模块级 _compiled_create_block_mask = torch.compile(create_block_mask)，无
# 守门。NPU 上首次 forward 触发 inductor fused triton kernel 编译，CANN BiShengIR
# 报 "ub overflow, requires 3014656 bits while 1572864 bits available"（376 KB
# > 192 KB UB），kernel 拒绝编译 → train.py ERR99999 → torchrun ChildFailed。
# 关 dynamo 后 torch.compile() 返回原函数，BiShengIR 完全不被叫到。10-sample
# smoke 不差这点 fused kernel 加速。
# 2.13.0rc1 torch_npu/_inductor/select_algorithm.py 引入了 flex_attention 的
# inductor 选择算法，理论上可开 dynamo；但 BiShengIR UB overflow 是否修了要
# 单独验证，留待下次跑通后单独验。
export TORCHDYNAMO_DISABLE=1

# ASCEND_LAUNCH_BLOCKING=1 — 让 NPU kernel 错误同步上浮为 Python 异常；不加的话
# CANN ERR99999 是 silent kill，Python 进程被 signal 干掉后没有 traceback 进
# /tmp/train.log，下次失败没法定位是哪个 op 炸的
export ASCEND_LAUNCH_BLOCKING=1

# --hidden-states-dtype float32：spec v0.7.0 schema 写死 "Model master weights
# are always kept in fp32"，dflash.forward 内 LN.weight 是 fp32；torch.autocast
# policy 表里 nn.LayerNorm / aten::layer_norm 在 cuda/cpu/npu 全标 _cast_no_op
# —— 不管 autocast 走哪条，LN(weight=fp32, input=bf16) 输出 fp32。dflash.forward
# 的 V 不是从 Linear 算出来的，是直接复用 dataloader 来的 hidden_states（默认
# bf16），所以 Q/K fp32, V bf16 → flex_attention dtype check 挂
# (torch/nn/attention/flex_attention.py:1473)。
# 把 V 也 cast 成 fp32 让 Q/K/V 对齐。schema 允许 float32 选项
# (train.py:548 getattr(torch, args.hidden_states_dtype))。A2 64 GB 装得下
# 1.9 GB 的 fp32 hidden_states cache（bf16 是 0.5 GB）。
# 备注：CUDA 上跑同一份 train.py 也会挂同款 dtype mismatch，除非先
# model.bfloat16() 把 LN.weight 也 cast bf16。spec 跳过了这一步，所以 dtype
# 对不齐是 spec 架构问题，CUDA 那边只是被 .bfloat16() 绕开了。
# --on-missing raise 强制走 FileBackend 读 <hs_dir> 缓存；不带 --vllm-endpoint 让
# dataloader 不会去问不存在的 server
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

确认训练产物（checkpoint 目录 + `config.json` + `model.safetensors`）：

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

起 vllm-ascend serve 把训好的 draft 挂上做 chat completion smoke（8 token completion）：

```shell #test id="pipeline-step4-serve" load="checkpoint_path>>draft_model" load="verifier_path>>verifier_path"
# num_speculative_tokens=5：vllm-ascend 限制 (num_speculative_tokens + 1) ≤ 15
nohup vllm serve "<verifier_path>" \
  --host 127.0.0.1 --port 8000 \
  --served-model-name Qwen/Qwen3-8B \
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

```shell #test-result id="config-import"
verifier: /root/.cache/modelscope/hub/models/Qwen/Qwen3-8B
verifier architectures: ['Qwen3ForCausalLM']
proposal type: greedy
proposal speculative_tokens: 5
```
