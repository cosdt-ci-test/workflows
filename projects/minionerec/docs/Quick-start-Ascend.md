# 快速开始：在昇腾 NPU 上运行 MiniOneRec 生成式推荐模型

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**：在单卡 NPU 上完成 MiniOneRec 的 SID 监督微调（SFT）、约束解码评估与推荐导向 RL 三个阶段。

[MiniOneRec](https://github.com/AkaliKong/MiniOneRec) 是一个全开源的**生成式推荐**框架，覆盖 SID 构建、SFT 与推荐导向 RL 全流程。三个阶段都跑在单卡上，规模压到首次跑通的量级：SFT 与评估不改源码，RL 只做三处单行替换（原因见第 10 节）。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**，需支持 bf16；910A 不支持 bf16 不在验证范围）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh`（验证基线：CANN 9.1.0） |
| Python | 3.10+（验证基线：3.12） |
| 编译工具 | git |

---

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

---

## 2. 检查环境是否就绪

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
npu-smi info
python --version
```

**预期**：`ASCEND_HOME_PATH` 非空；`npu-smi` 打印设备列表且 Health 为 OK。

---

## 3. 克隆 MiniOneRec

将 `<UPSTREAM_REF>` 换成目标**分支、tag 或 commit**（上游默认分支为 `main`）。

```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```

```shell #test-setup id="clone" load="upstream_ref>>UPSTREAM_REF"
# 只取最新快照并重试；关掉 HTTP/2，大文件连续流在跨境链路上容易被代理切断
for i in 1 2 3 4 5; do
    rm -rf MiniOneRec
    if GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1 \
        git clone --depth 1 https://github.com/AkaliKong/MiniOneRec.git; then
        break
    fi
    echo "clone attempt $i failed, retrying"
    sleep $((i * 10))
done
test -d MiniOneRec || { echo "clone failed after 5 attempts"; exit 1; }
cd MiniOneRec
git checkout <UPSTREAM_REF>
```

---

## 4. 安装 NPU 依赖

SFT 只需要下列包。`bitsandbytes` 单独钉在 aarch64 有预编译包的 0.48.1：`sft.py` / `evaluate.py` 顶部的 `import bitsandbytes as bnb` 从未被实际使用，但不改源码就绕不开这行 import，必须让它能导入。RL 阶段另需 `trl`（`rl.py` 从它取 `GRPOConfig`）、`scikit-learn`（`ndcg_score`）与 `wandb`（`sasrec.py` 无条件导入）。`sft.py` 的 `TrainingArguments` 默认 `report_to="all"`，装好 wandb 后会尝试 `wandb.init()` 并因缺少 API key 失败，所以第 8 节导出 `WANDB_MODE=disabled`。

torch 栈与 CANN 9.1.0 配套（torch 2.9.0 + torch_npu 2.9.0.post2）：

```shell #test-setup id="install"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install torch==2.9.0 torch-npu==2.9.0.post2
pip install modelscope transformers==4.57.1 accelerate==1.10.1 datasets==4.2.0 \
    tokenizers==0.22.1 fire==0.7.1 pandas==2.2.2 tqdm bitsandbytes==0.48.1 \
    trl==0.24.0 scikit-learn==1.5.2 wandb==0.22.2
```

---

## 5. 观察默认行为：两个入口的设备走向不一样

MiniOneRec 的两个入口对「有没有 GPU」的依赖并不相同。`sft.py` 把设备选择交给 `TrainingArguments`，而 transformers 的设备链里 NPU 分支排在 CUDA 之前，零注入也能拿到 `npu:0`；`evaluate.py` 则在模块级就把 `device` 固化成 `"cpu"`。下面这段命令能同时看到这两种结果：

```shell #test id="probe-default"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd MiniOneRec
python -c "
import torch, torch_npu
from transformers import TrainingArguments
print('npu available:', torch.npu.is_available())
print('cuda mapped:', torch.cuda.is_available())
print('evaluate.py device:', 'cuda' if torch.cuda.is_available() else 'cpu')
print('trainer device:', TrainingArguments(output_dir='/tmp/probe-default').device)
"
```

输出结果如下：

```shell #test-result id="probe-default"
npu available: True
cuda mapped: False
evaluate.py device: cpu
trainer device: npu:0
```

---

## 6. 只在评估命令里启用 NPU 映射

SFT 不需要任何注入；评估与 RL 需要 `torch.cuda` 有值。为了让注入只作用于需要它的命令，把 `sitecustomize.py` 放进独立目录、通过 `PYTHONPATH` 挂到命令上——不写进 site-packages，其他进程不受影响。

```shell #test-setup id="npu-shim"
mkdir -p /tmp/minionerec-npu-shim
cat > /tmp/minionerec-npu-shim/sitecustomize.py <<'INS'
import torch_npu.contrib.transfer_to_npu  # noqa: F401
INS
```

确认作用域只覆盖挂了 `PYTHONPATH` 的那条命令：

```shell #test id="check-scoped"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd MiniOneRec
python -c "import torch; print('default process:', 'cuda' if torch.cuda.is_available() else 'cpu')"
PYTHONPATH=/tmp/minionerec-npu-shim python -c "
import torch
print('with shim:', 'cuda' if torch.cuda.is_available() else 'cpu')
print('tensor landed on:', torch.zeros(1, device='cuda').device.type)
"
```

输出结果如下（代码里写的是 `cuda`，张量实际落在 NPU）：

```shell #test-result id="check-scoped"
default process: cpu
with shim: cuda
tensor landed on: npu
```

确认仓库源码没被动过（前面的检查与注入都不落在仓库里）：

```shell #test id="no-source-change"
cd MiniOneRec
test -z "$(git status --porcelain)" && echo "source untouched"
```

输出结果如下：

```shell #test-result id="no-source-change"
source untouched
```

---

## 7. 下载基座模型与预处理数据

基座模型走 ModelScope（Qwen2.5-**base**，规避 README 公告的 Instruct 模型约束解码失效问题）；官方预处理的 Amazon Industrial_and_Scientific 数据只在 HuggingFace 侧，经 hf-mirror 拉取（排除 .npy 嵌入文件，仅 RQ/SID 构建阶段需要）。`info/` 下的 SID-条目清单 txt 是评估阶段构建约束 trie 与对齐评测目标的输入，一并拉取。

```shell #test-setup id="download"
cd MiniOneRec
modelscope download --model Qwen/Qwen2.5-0.5B --local_dir ./Qwen2.5-0.5B
export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='kkknight/MiniOneRec', allow_patterns=['Amazon/index/Industrial_and_Scientific.index.json','Amazon/index/Industrial_and_Scientific.item.json','Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv','Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv','Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv','Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt'], local_dir='data')"
```

校验资产落位：

```shell #test id="verify-assets"
cd MiniOneRec
test -f Qwen2.5-0.5B/config.json && echo "base model ok"
test -f data/Amazon/index/Industrial_and_Scientific.index.json && echo "sid index ok"
test -f data/Amazon/index/Industrial_and_Scientific.item.json && echo "item meta ok"
test -f data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv && echo "train csv ok"
test -f data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv && echo "valid csv ok"
test -f data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt && echo "info txt ok"
```

输出结果如下：

```shell #test-result id="verify-assets"
base model ok
sid index ok
item meta ok
train csv ok
valid csv ok
info txt ok
```

---

## 8. SFT 训练

单卡跑 1 epoch：`sample=600` 限制每个子数据集（SID 序列 / SID-条目对齐 / 融合序列推荐）各取 600 条；`cutoff_len=256` 截断序列。完整训练（8 卡 A100、全量数据）见上游 `sft.sh`。

```shell #test-setup id="sft-run"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
cd MiniOneRec
python sft.py \
    --base_model ./Qwen2.5-0.5B \
    --train_file ./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --output_dir ./smoke_out \
    --batch_size 8 \
    --micro_batch_size 2 \
    --num_epochs 1 \
    --cutoff_len 256 \
    --sample 600 \
    --category Industrial_and_Scientific \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    --freeze_LLM False \
    --train_from_scratch False
```

训练完成后校验产物（`model.safetensors` + `config.json` + tokenizer 配置）：

```shell #test id="verify-sft"
cd MiniOneRec
test -f smoke_out/final_checkpoint/model.safetensors && echo "model saved"
test -f smoke_out/final_checkpoint/config.json && echo "config saved"
test -f smoke_out/final_checkpoint/tokenizer_config.json && echo "tokenizer saved"
grep -o '"model_type": "[a-z0-9_.-]*"' smoke_out/final_checkpoint/config.json
```

输出结果如下：

```shell #test-result id="verify-sft"
model saved
config saved
tokenizer saved
"model_type": "qwen2"
```

---

## 9. 约束解码评估

用上一节得到的 `final_checkpoint` 在 test 切片上做 beam search 评估，再算 HR/NDCG/CC。**CC（无效 SID 生成数）必须为 0**——`ConstrainedLogitsProcessor` 用 trie 约束每步只允许合法 SID 前缀 token，这是约束解码在 NPU 上生效的关键信号。本节用 `num_beams=20`、`batch_size=2` 在 test 切片上评估。

切出 test 子集：

```shell #test-setup id="prep-eval"
cd MiniOneRec
head -n 200 data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv > test_small.csv
wc -l test_small.csv
```

运行评估（进度条与 transformers 警告走 stderr，stdout 只有 category 名）：

```shell #test id="run-eval"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
cd MiniOneRec
PYTHONPATH=/tmp/minionerec-npu-shim python evaluate.py \
    --base_model ./smoke_out/final_checkpoint \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --test_data_path test_small.csv \
    --result_json_data ./eval_result.json \
    --category Industrial_and_Scientific \
    --num_beams 20 --batch_size 2
```

输出结果如下（进度条走 stderr）：

```shell #test-result id="run-eval"
industrial and scientific items
...
```

校验预测产物：

```shell #test id="verify-eval"
cd MiniOneRec
python -c "import json; d=json.load(open('eval_result.json')); print(len(d)); print('predict' in d[0])"
```

输出结果如下：

```shell #test-result id="verify-eval"
199
True
```

计算 HR/NDCG/CC。HR/NDCG 随训练状态浮动，判断约束解码是否生效看 CC 是否为 0：

```shell #test id="calc-metrics"
cd MiniOneRec
python calc.py \
    --path ./eval_result.json \
    --item_path ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

输出结果如下：

```shell #test-result id="calc-metrics"
20
[1, 3, 5, 10, 20]
NDCG:...
HR...
0
```

---

## 10. 推荐导向 RL

RL 在 SFT 产物上再做一轮 GRPO：每个 prompt 采样多条候选、组内归一化奖励、并用 KL 约束贴近参考模型；动作空间是有限的 SID 列表，候选由约束 beam search 生成。

`rl.py` 在 NPU 上会自动选用 torch 原生的 `adamw_torch`，无需修改源码。三个数据集的采样数由 `--sample` 控制；训练集只有 40 行，不传该参数会超出样本池，所以下面的命令显式传 `--sample 40`。

把训练集切到 40 行（`pandas` 允许取样数等于总体数）：

```shell #test id="prep-rl"
cd MiniOneRec
python - <<'PY'
import pandas as pd

df = pd.read_csv('data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv')
df.head(40).to_csv('data/rl_train_40.csv', index=False)
print('rl train rows:', len(pd.read_csv('data/rl_train_40.csv')))
PY
```

输出结果如下：

```shell #test-result id="prep-rl"
rl train rows: 40
```

在 0 号卡上跑 1 epoch。`train_batch_size` 必须能被 `num_generations` 整除；`num_generations` 同时是 beam 宽度，本节取 2。

```shell #test id="run-rl"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
cd MiniOneRec
PYTHONPATH=/tmp/minionerec-npu-shim python rl.py \
    --model_path ./smoke_out/final_checkpoint \
    --train_file ./data/rl_train_40.csv \
    --sample 40 \
    --eval_file ./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv \
    --info_file ./data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt \
    --category Industrial_and_Scientific \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --num_generations 2 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --beta 1e-3 \
    --beam_search True \
    --reward_type ranking \
    --sync_ref_model True \
    --test_during_training False \
    --eval_step 1.0 \
    --output_dir ./rl_out \
    --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
```

输出结果如下（数据集对象与进度条走 stderr）：

```shell #test-result id="run-rl"
Industrial_and_Scientific
...
```

日志里的 `reward`、`reward_std`、`loss`、`grad_norm`、`kl` 会一直是 0，这不是故障。`rule_reward` 要求生成的 SID 与目标完全相等；上一节只训 1 epoch，精确命中率接近 0，组内两条候选都命不中，`ndcg_rule_reward` 还会把整组补 0，GRPO 的组内优势因此无从计算，梯度自然为 0。到这里三个阶段都已跑通；要看到非零的训练信号，需要更充分的 SFT。

校验 `rl.py` 结尾的两步落盘（`trainer.save_model` 与 `model.save_pretrained`）：

```shell #test id="verify-rl"
cd MiniOneRec
test -f rl_out/final_checkpoint/model.safetensors && echo "rl model saved"
test -f rl_out/final_checkpoint/config.json && echo "rl config saved"
test -f rl_out/final_checkpoint/tokenizer_config.json && echo "rl tokenizer saved"
```

输出结果如下：

```shell #test-result id="verify-rl"
rl model saved
rl config saved
rl tokenizer saved
```

---

## 相关链接

- 上游仓库：[AkaliKong/MiniOneRec](https://github.com/AkaliKong/MiniOneRec)
- 基座模型：[Qwen2.5-0.5B](https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B)
- 预处理数据与 SID 索引：[kkknight/MiniOneRec](https://huggingface.co/kkknight/MiniOneRec)
- 昇腾 PyTorch 安装与兼容性：[Ascend/pytorch](https://gitcode.com/Ascend/pytorch)
