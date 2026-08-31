# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上用 vllm-ascend v0.23.0（配套 vLLM v0.23.0）跑 [Speculators](https://github.com/vllm-project/speculators) 的完整端到端链路：把第三方 speculative decoding draft 模型（DFlash）转换成标准 `speculators` 格式、用 vllm-ascend 抽训练数据、torchrun 训 draft 模型、最后 `vllm serve` 把训好的 draft 挂上做推理 smoke。。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的 `torch` + `torch_npu`，且 `torch` 能正常 `import` 并 `torch.npu.is_available() == True`（参考 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch），按 torch ↔ torch_npu ↔ CANN 三方兼容矩阵选择版本）
- 可用的 `uv`（本文档安装步骤全部走 `uv pip install`）。如果机器上没有：`pip install uv` 或参照 [uv 官方安装指南](https://docs.astral.sh/uv/getting-started/installation/)

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/vllm-ascend/vllm-ascend:vllm-ascend:v0.23.0

> 该镜像**预装** vllm==0.23.0 + vllm-ascend==0.23.0 + triton-ascend==3.2.2 + triton 3.5.0 + torch 2.10.0+cpu + torch_npu 2.10.0.post4 + CANN 9.1.0 + Python 3.12。本文「前置安装」一节只验证版本 + 装 modelscope，不再源码 build vllm、不再拉 vllm-ascend NPU variant wheel。

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.10.0+cpu |
| torch_npu | 2.10.0.post4 |
| transformers | 由 `speculators` 透传拉入（>=4.56.1,<5.15.0） |
| vllm | 0.23.0（镜像预装） |
| triton-ascend | 3.2.2（镜像预装）|
| triton | 3.5.0（镜像预装） |
| vllm-ascend | 0.23.0（镜像预装，NPU variant wheel） |
| modelscope | 1.37.0（镜像不含，`前置安装` 一节装） |
| speculators | 最新 release 的源码/二进制 |
| draft 模型 | [z-lab/Qwen3-8B-DFlash-b16] |
| verifier | [Qwen/Qwen3-8B] |

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出类似：

```
+------------------------------------------------------------------------------------------------+
| npu-smi 25.5.2                   Version: 25.5.2                                               |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 5     910B4               | OK            | 89.9        39                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          2922 / 32768         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| No running processes found in NPU 5                                                            |
+===========================+===============+====================================================+
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。

检查 Python 版本：

```shell #test id="check-py"
python --version
```
输出结果如下：
```shell #test-result id="check-py" fuzzy='xxx'
Python 3.12.xxx
```

#### 验证 vllm-ascend 镜像预装

> [配套镜像](#配套镜像) 已预装 vllm==0.23.0 + vllm-ascend==0.23.0 + triton-ascend==3.2.2 + torch 2.10.0+cpu + torch_npu 2.10.0.post4 + CANN 9.1.0 + Python 3.12。下面只做版本验证 —— **不** 源码 build vllm，**不** `pip install` vllm-ascend NPU variant wheel（bare-CANN 镜像时代那些 25-35 min 的冷启开销在新镜像里都没了）。

加载 CANN env 并验证版本：

```shell #test id="verify-vllm-stack"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -c "import torch, torch_npu; print(f'torch={torch.__version__}'); print(f'torch_npu={torch_npu.__version__}'); print('is_available:', torch.npu.is_available()); print('npu_count:', torch.npu.device_count())"

python -c "import importlib.metadata; print(f'vllm={importlib.metadata.version(\"vllm\")}')"
python -c "import importlib.metadata; print(f'vllm_ascend={importlib.metadata.version(\"vllm-ascend\")}')"
python -c "import importlib.metadata; print(f'triton_ascend={importlib.metadata.version(\"triton-ascend\")}')"
python -c "import importlib.metadata; print(f'triton={importlib.metadata.version(\"triton\")}')"
```

输出结果如下：

```shell #test-result id="verify-vllm-stack" fuzzy='xxx'
torch=2.10.0+cpu
torch_npu=2.10.0.post4
is_available: True
npu_count: xxx
vllm=0.23.0
vllm_ascend=0.23.0
triton_ascend=3.2.2
triton=3.5.0
```

> 如果某个版本对不上或 `npu_count` 不是 ≥ 1，重新 `docker pull` 一次该镜像；本文档不补救版本漂移——镜像里就该是这些版本。

#### 安装 modelscope

```shell #test-setup
uv pip install 'modelscope==1.37.0'
```

打印安装版本：
```shell #test id="install-deps"
python -c "import modelscope; print(f'modelscope={modelscope.__version__}')"
```

输出结果如下：

```shell #test-result id="install-deps"
modelscope=1.37.0
```

## 安装 Speculators

### 使用 uv 进行安装

```shell #test id="speculators-install-binary"
uv pip install speculators
speculators --version
python -c "from importlib.metadata import version; print('speculators', version('speculators'))"
```

输出结果类似如下：

```shell #test-result id="speculators-install-binary" fuzzy='xxx'
speculators version: xxx
speculators xxx
```
- xxx 表示最新的版本号
<!--
```shell #test-setup
uv pip uninstall speculators
```
-->

### 从源码安装

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag，安装并且验证：

```shell #test id="speculators-install-source" load="upstream_ref>>ref"
# 显式 clone 到绝对路径：测试进程 cwd 是 workflows/projects/speculators（engine
# working-directory 钉死），相对 clone 会落到 <cwd>/speculators；Step 16 #test-setup
# 的 `cd /root/speculators` 拿不到这个目录、整个 train.py 链就静默失败
git clone --depth 1 --branch <ref> https://github.com/vllm-project/speculators.git /root/speculators
cd /root/speculators
uv pip install -e .
speculators --version
python -c "from importlib.metadata import version; print('speculators', version('speculators'))"
```

\<ref> 为安装的最新的 release tag

输出结果类似如下：

```shell #test-result id="speculators-install-source" fuzzy='xxx'
speculators version: xxx
speculators xxx
```
- xxx 表示最新的版本号

## 完整链路：convert → 训练数据生成 → 训练 → 部署（4 个核心场景端到端示例）

### 前置：下载 draft 与 verifier

默认使用 **ModelScope** 进行模型下载（draft + verifier 都在 ModelScope 上有完整镜像）。持久缓存中可能残留之前中断下载产生的残缺权重文件，测试框架会在下载前做 safetensors 完整性校验，损坏的模型目录会被整体清除并重新下载。

```shell #test-setup store="draft_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('z-lab/Qwen3-8B-DFlash-b16'))" | tail -n 1
```

输出类似：

```
/root/.cache/modelscope/hub/models/z-lab/Qwen3-8B-DFlash-b16
```

```shell #test-setup store="verifier_path"
python -c "from modelscope import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B'))" | tail -n 1
```

输出类似：

```
/root/.cache/modelscope/hub/models/Qwen/Qwen3-8B
```

### Step 1 Standardized, Extensible Format（convert）

`speculators convert` 把本地 draft 目录 + verifier 目录读进来，按 DFlash 算法重映射权重、写入 `speculators_config`，输出到一个新目录。CLI 的 `--algorithm` 选项只接受 `eagle` / `eagle3` / `mtp` 三个值（见上游 `src/speculators/__main__.py:99` 的 `click.Choice(["eagle", "eagle3", "mtp"])`），DFlash 不在 CLI 白名单里——DFlash 只在 Python API `convert_model(algorithm="dflash", ...)` 里支持（见 `convert/entrypoints.py:32` 的 `Literal["eagle3", "mtp", "dflash"]`），所以本节走 Python API：

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

> `python | grep "Saved to"` 看似省事，但 convert 抛异常时 traceback 也走 stderr 进 grep、grep 找不到退出 1，bash 没 `set -e / pipefail` 所以 `dflash_path` 仍被捕获、framework 看不到失败。改成：python 的 stderr 自由流出（异常 traceback 命中 `ERROR_MARKERS`），两个 `test -f` 显式断言 config.json / model.safetensors 存在，最后 `echo` 单写一行纯路径让 store 拿到干净字符串。下面的 `<dflash_path>` 是测试框架占位符（`load="dflash_path>>dflash_path"`），不要写 `$dflash_path`（shell 变量跨 #test 不保留）。

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

### Step 2 — 场景 1：Training Data Preprocessing

用上游 `scripts/prepare_data.py` 把 chat-formatted JSONL tokenize + chat template → HF arrow 数据集写到 `$DATA_DIR`（同 upstream canonical `examples/train/dflash_qwen3_8b_sharegpt_online_5k.sh` 的 Step 1）。train.py 的 `ArrowDataset.load_from_disk($DATA_DIR)` 直接吃。

之前这步走的是 vllm-ascend 的 `extract_hidden_states` 离线 API 写单文件 .safetensors —— 但 `load_from_disk` 不吃 safetensors、只看 `dataset.save_to_disk()` 写出来的 `.arrow` shards + `dataset_info.json`（CI 33166102161 之后暴露的第二个 bug），同时文件命名 `0-b124ee50.safetensors` 也不对 FileBackend 期望的 `hs_<idx>.safetensors`。干脆切到上游 prepare_data.py + 在线模式（Step 16 起 vllm server 拉 hidden states）。

```shell #test-setup store="data_path" load="verifier_path>>verifier_path"
set -euo pipefail
DATA_DIR=/root/dflash-train-data
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR"

# 10 条 chat samples：每条 user + assistant 都填，否则 loss_mask 全 0、prepare_data.py
# 默认会因 assistant token 不足 raise。smoke 不指望 loss 真下降，只要 chat template +
# tokenizer 跑通 + arrow + token_freq.pt 都写出来即可
# 顶层 key 必须是 "conversations"（不是 "messages"），load_and_preprocess_dataset
# 直接 examples.get("conversations", [])，messages 字段会全部被 silently drop，
# 末尾 raise "No samples remain after preprocessing"（CI 33172655874 教训）
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

# 复用 source install 那步 clone 的仓库（cwd 不跨 #test 块，需重新 cd）
cd /root/speculators

# prepare_data.py 走 chat template + tokenizer，输出 HF arrow 数据集 + token_freq.pt；
# --seq-length 8192 与 train.py default 对齐
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

输出结果如下：

```shell #test-result id="pipeline-step2-extract" fuzzy='xxx'
/root/dflash-train-data
10
exists: True size: xxx
token_freq keys: xxx
len: xxx
```

### Step 3 — 场景 2：Draft Model Training Support

用上游 `scripts/train.py` + 单卡 `torchrun --nproc_per_node=1` 训 1 epoch × 10 sample（**smoke 验证管线通，不指望 loss 真下降**）：

```shell #test-setup store="hs_dir" load="data_path>>data_path" load="verifier_path>>verifier_path"
set -euo pipefail

# 复用 source install 那步 clone 的 speculators 仓库（cwd 不跨 #test 块，需重新 cd）
cd /root/speculators

# 离线模式（时间分片替代物理拆 GPU）：先起 vllm 一次性 generate 10 条 hidden_states
# 落 $HS_DIR/hs_<idx>.safetensors，杀 vllm 释放 64 GB NPU，再 train.py 离线读 cache。
# 上游 canonical `examples/train/dflash_qwen3_8b_sharegpt_online_5k.sh` 是 4x H100
# 用 CUDA_VISIBLE_DEVICES 把 vllm 2 GPU + train 2 GPU 物理拆开；我们单 NPU 64 GB
# 装不下 vllm 16 GB 权重 + KV + train draft 模型 + optimizer 32x padding 激活并发跑
# （CI 33177074202 → 33193836422 三轮调参 0.9/0.3/0.5 都翻车：OOM / KV cache 不够 /
# 同样 KV cache 不够）。杀 vllm 后 train 拿满 64 GB 完全够。
HS_DIR=/tmp/hs-train
rm -rf "$HS_DIR"
mkdir -p "$HS_DIR"

# setsid 让 vllm 跑在独立 session + process group，cleanup 用 kill -- -$PGID 整组杀
setsid nohup python scripts/launch_vllm.py "<verifier_path>" \
  --target-layer-ids 2 18 34 \
  --hidden-states-path "$HS_DIR" \
  -- \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  > /tmp/vllm-gen.log 2>&1 < /dev/null &
VLLM_GEN_PID=$!
VLLM_GEN_PGID=$(ps -o pgid= -p "$VLLM_GEN_PID" | tr -d ' ')
# 注意：不能用 `pkill -f "scripts/launch_vllm.py"` ——bash 子进程 cmdline 也含这串、
# 会把 bash 一起 -9 自杀（CI 33196117621 教训：cleanup 调 pkill → bash 收 SIGKILL →
# rc=-9 + stderr=0B，看不到任何诊断）。所有 cleanup 必须走 $VLLM_GEN_PID 或更
# 具体的固定字符串（不含本脚本 cmdline 内容）
cleanup_vllm_gen() {
  # 先 SIGTERM 整 vllm 进程组（包括 vllm fork 的 worker 子进程）
  kill -- -"$VLLM_GEN_PGID" 2>/dev/null || true
  # 兜底 SIGKILL launcher + engine 子进程（用 launcher 实际 PID，绕开 cmdline 匹配）
  for _pid in $(pgrep -P "$VLLM_GEN_PID" 2>/dev/null) "$VLLM_GEN_PID"; do
    kill -9 "$_pid" 2>/dev/null || true
  done
  # 最后兜底用绝对固定字符串（vllm 框架内部 module 路径，bash cmdline 不会有）
  pkill -9 -x "vllm" 2>/dev/null || true
}
trap cleanup_vllm_gen EXIT

# 等 /health 200（最长 6 min，裸 vllm-ascend load Qwen3-8B 实测 3-4 min）
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

# 用上游 data_generation_offline.py 把 10 条 hidden_states 写 $HS_DIR（hs_<idx>.safetensors
# 命名正好对 FileBackend cache 契约），vllm 释放全部 NPU 后给 train 留出 64 GB 完整空间
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

# 杀 vllm 释放 NPU 内存给 train.py
cleanup_vllm_gen
sleep 5  # 给 vllm worker 完全退出 + NPU 释放

echo "$HS_DIR"
```

```shell #test-setup store="checkpoint_path" load="hs_dir>>hs_dir" load="data_path>>data_path" load="verifier_path>>verifier_path"
set -euo pipefail
CHECKPOINT_DIR=/root/dflash-trained
rm -rf "$CHECKPOINT_DIR"
mkdir -p "$CHECKPOINT_DIR"

cd /root/speculators

# 离线训练：--on-missing raise 强制走 FileBackend 读 <hs_dir> 缓存，不再起 vllm endpoint；
# （train.py 的 argparse choices 是 generate/skip/warn/raise，没有 error；CI 33201184782 教训）
# 显式不带 --vllm-endpoint，避免 dataloader 误以为有 server 可问。
# train.py stdout+stderr 全写到 /tmp/train.log，失败时把整 log cat 到 bash stderr——
# framework 的 ERROR_MARKERS 看到 'Traceback (most recent call last' 会触发 ≤256 KB
# 全量 dump（CI 33240618205 / 33239422249 历史教训：之前 `2>&1 | tee` 把 traceback
# 灌到 bash stdout，framework 只 head/tail 2000 字符，train.py 真因被截到 `ker:
# There appear to be %d '` 这种 partial format string，根本看不出哪儿炸的）。
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
  echo "=== train.py rc=0 但 checkpoint 缺失（config.json / model.safetensors）; full train.log ===" >&2
  cat /tmp/train.log >&2
  exit 1
fi

echo "$CHECKPOINT_DIR"
```

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

### Step 4 — 场景 4：Seamless vLLM-Ascend Integration

把 Step 3 训出的 checkpoint 喂给 `vllm-ascend serve --speculative-config`，做一次 chat completion smoke：

```shell #test id="pipeline-step4-serve" load="checkpoint_path>>draft_model" load="verifier_path>>verifier_path"
# num_speculative_tokens=5：vllm-ascend 限制 (num_speculative_tokens + 1) ≤ 15（受
# npu_fused_infer_attention_score 算子约束），5 留余量
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

# 8-token completion smoke（JSON 不可逐字预测，用 fuzzy）
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}'

kill "$VLLM_PID" 2>/dev/null || true
```

> vllm-ascend 启动时同时 load verifier（~16 GB）+ draft（训出的几 MB-几 GB）+ 编译 DFlash proposer NPU graph，比裸 vllm 慢；6 min 是 1-card A2 上 vllm-ascend v0.23.0 + Qwen3-8B 的实测上界（实际通常 3-4 min）。

输出结果如下（精确字符串不可预测，用 fuzzy 容忍 UUID / 内容）：

```shell #test-result id="pipeline-step4-serve" fuzzy='xxx'
{"id":"chatcmpl-xxx","object":"chat.completion","created":xxx,"model":"Qwen/Qwen3-8B","choices":[{"index":0,"message":{"role":"assistant","content":"xxx"},"finish_reason":"length"}]}
```

### 编程式入口：`SpeculatorsConfig` / `VerifierConfig` / `TokenProposalConfig`

`speculators` 同时暴露 Python API，对应上游 Quicktour 的「config / model / proposals」分层。本节用最小代码验证 `config.py` 与 `proposals/greedy.py` 可在 NPU 环境中正常 import 并实例化（不依赖任何 GPU 计算）：

```shell #test id="config-import" load="verifier_path>>verifier_path"
python << 'PY'
from speculators import VerifierConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig

# VerifierConfig.architectures 是 pydantic 必填字段，直接 VerifierConfig(name_or_path=...)
# 会触发 ValidationError；显式传 architectures 或调 VerifierConfig.from_pretrained(...)
verifier = VerifierConfig(
    name_or_path="<verifier_path>",
    architectures=["Qwen3ForCausalLM"],
)
# TokenProposalConfig 是 pydantic registry 基类，唯一已注册的 proposal_type 是
# greedy；DFlash convert_model 路径里也是用 greedy
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
