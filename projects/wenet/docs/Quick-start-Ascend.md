# 快速开始：在昇腾 NPU 上用 WeNet 训练语音识别模型

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**：基于 [aishell-1](https://www.openslr.org/33) 真实数据集，按 WeNet 官方 [NPU 实验脚本](https://github.com/wenet-e2e/wenet/blob/main/examples/aishell/s0/run_npu.sh)（[官方教程](https://wenet.org.cn/wenet/tutorial_aishell.html#)）完成数据下载、数据准备、训练、推理与导出全流程。

[WeNet](https://github.com/wenet-e2e/wenet) 是一个生产级端到端语音识别工具包，支持流式和非流式识别。昇腾侧通过 `torch-npu` 将计算调度到 NPU。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

与官方安装指南（`examples/aishell/s0`）的版本要求保持一致：

| 类别 | 最低版本 | 推荐版本 |
| --- | --- | --- |
| CANN | 8.0.RC2.alpha003 | latest |
| Python | 3.10 | 3.10 |
| torch | 2.1.0 | 2.2.0 |
| torch-npu | 2.1.0 | 2.2.0 |
| torchaudio | 2.1.0 | 2.2.0 |
| deepspeed | 0.13.2 | latest |

> **注意**：CANN 最低版本为 8.0.rc1，安装 CANN 时请同时安装 Kernel 算子包。本文配套镜像为
> `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.0.0-910b-ubuntu22.04-py3.10`（CANN 8.0.0 + Python 3.10，满足最低要求），
> torch / torch-npu / torchaudio 使用推荐版本 **2.2.0**。deepspeed 最低要求 0.13.2，但 0.16+ 在
> torch 2.2.0 下会触发 `torch.library.custom_op` `AttributeError`，本文显式钉在 **0.14.4**。

编译工具 git、音频工具 sox 需要提前就绪。

---

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

---

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

**预期**：命令退出码为 0，并打印设备列表。

### 2.2 确认 CANN 与 Python

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
python --version
```

**预期**：`ASCEND_HOME_PATH` 非空；`npu-smi` 与 `python` 均能打印出版本信息。

---

## 3. 克隆 WeNet 并安装依赖

将 `<UPSTREAM_REF>` 换成目标**分支、tag 或 commit**（上游默认分支为 `main`）。

```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```

```shell #test-setup id="clone" load="upstream_ref>>UPSTREAM_REF"
git clone https://github.com/wenet-e2e/wenet.git
cd wenet
git checkout <UPSTREAM_REF>
```

安装 WeNet 及其 NPU 依赖（与官方安装指南一致：`[torch-npu]` extra 已将 torch / torch-npu / torchaudio 钉在推荐版本 2.2.0，并附带 `numpy<2`；`requirements.txt` 只约束 `deepspeed>=0.14.0`，为避免 pip 解析到与 torch 2.2.0 不兼容的 0.16+，安装后显式回钉 0.14.4）：

```shell #test-setup id="install"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd wenet
pip install -e .[torch-npu]
pip install -r requirements.txt
pip install "deepspeed==0.14.4"
```

安装 sox：

```shell #test-setup
pip install sox
apt-get update && apt-get install -y sox libsox-dev || true
```

---

## 4. 验证 torch_npu 安装

```shell #test id="check-npu"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('npu available:', torch.npu.is_available()); print('npu count:', torch.npu.device_count())"
```

输出结果如下：

```shell #test-result id="check-npu" fuzzy='xxx'
torch: xxx
torch_npu: xxx
npu available: True
npu count: 1
```

---

## 5. 下载数据（stage -1）

stage -1 阶段将 aishell-1 数据下载到本地路径 `$data`（主包 `data_aishell.tgz` 约 15.6 GB；`$data` 必须为**绝对路径**，且下载脚本要求目录已存在，需提前 `mkdir -p` 创建。如果已下载数据，把 `--data` 换成实际数据集存放的绝对路径即可）。下载可能因网络波动失败，脚本支持断点续传，可安全重试：

```shell #test-setup id="download-data"
mkdir -p /root/asr-data/OpenSLR/33
cd wenet/examples/aishell/s0
MAX_RETRIES=3
for i in $(seq 1 $MAX_RETRIES); do
  echo "Attempt $i of $MAX_RETRIES..."
  bash run_npu.sh --stage -1 --stop_stage -1 --data /root/asr-data/OpenSLR/33
  if [ -f /root/asr-data/OpenSLR/33/data_aishell/.complete ] && [ -f /root/asr-data/OpenSLR/33/resource_aishell/.complete ]; then
    echo "Download completed successfully"
    break
  fi
  echo "Attempt $i failed, retrying..."
  sleep 5
done
# 验证下载是否成功：检查 .complete 标记文件
ls -la /root/asr-data/OpenSLR/33/data_aishell/.complete /root/asr-data/OpenSLR/33/resource_aishell/.complete || {
  echo "ERROR: Download failed after $MAX_RETRIES attempts"
  exit 1
}
```

验证 `data_aishell` 与 `resource_aishell` 两个数据包均下载解压完成（脚本以 `.complete` 标记断点，重跑 stage -1 会跳过已完成部分）：

```shell #test id="verify-download"
echo "=== 检查下载标记文件 ==="
ls -la /root/asr-data/OpenSLR/33/data_aishell/.complete /root/asr-data/OpenSLR/33/resource_aishell/.complete
echo "=== 检查数据目录内容 ==="
ls /root/asr-data/OpenSLR/33/data_aishell/ | head -5
ls /root/asr-data/OpenSLR/33/resource_aishell/ | head -5
```

输出结果如下：

```shell #test-result id="verify-download"
=== 检查下载标记文件 ===
... /root/asr-data/OpenSLR/33/data_aishell/.complete
... /root/asr-data/OpenSLR/33/resource_aishell/.complete
=== 检查数据目录内容 ===
...
...
...
...
...
...
...
...
```

---

## 6. 准备训练数据（stage 0）

stage 0 阶段为训练数据准备阶段，将使用 `local/aishell_data_prep.sh` 脚本将训练数据重新组织为 `wav.scp` 和 `text` 两部分。`wav.scp` 每行记录两个制表符分隔的列：`wav_id` 和 `wav_path`；`text` 每行记录 `wav_id` 和 `text_label`：

```shell #test-setup id="prep-data"
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 0 --stop_stage 0 --data /root/asr-data/OpenSLR/33
```

验证训练 / 验证 / 测试集划分（aishell-1 官方固定划分为 120098 / 14326 / 7176 条）：

```shell #test id="verify-prep"
cd wenet/examples/aishell/s0
wc -l data/train/wav.scp data/train/text data/dev/wav.scp data/test/wav.scp | awk '{print $1, $2}'
```

输出结果如下：

```shell #test-result id="verify-prep"
120098 data/train/wav.scp
120098 data/train/text
14326 data/dev/wav.scp
7176 data/test/wav.scp
261698 total
```

---

## 7. 提取最佳 cmvn 特征（stage 1）

stage 1 阶段从训练数据中提取 cmvn 特征，本阶段为可选阶段，设置 `cmvn=false` 可跳过本阶段。`tools/compute_cmvn_stats.py` 用于提取全局 cmvn（倒谱均值和方差归一化）统计数据，用来归一化声学特征：

```shell #test-setup id="cmvn"
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 1 --stop_stage 1
```

---

## 8. 生成 token 字典（stage 2）

stage 2 阶段生成训练所需 token 字典，用于 CTC 解码阶段查询，将输出转换为文字：

```shell #test-setup id="dict"
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 2 --stop_stage 2 --data /root/asr-data/OpenSLR/33
```

---

## 9. 准备 WeNet 数据格式（stage 3）

stage 3 阶段生成 WeNet 所需格式的文件 `data.list`，每一行都是 json 格式，包含关键词 `key`（文件名称）、语音文件地址 `wav` 和对应文本内容 `txt` 三个关键数据：

```shell #test-setup id="data-list"
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 3 --stop_stage 3 --data /root/asr-data/OpenSLR/33
```

验证生成的文件（字典前三行为固定特殊符号，其后按字频排序；`data.list` 首行为真实语音的 json 记录）：

```shell #test id="verify-data"
cd wenet/examples/aishell/s0
head -5 data/dict/lang_char.txt
head -1 data/train/data.list
```

输出结果如下：

```shell #test-result id="verify-data" fuzzy='xxx'
<blank> 0
<unk> 1
<sos/eos> 2
xxx
xxx
xxx
```

---

## 10. 模型训练（stage 4）

`run_npu.sh` 脚本中实现了 NPU 卡号的自动获取和相关环境变量设置，可直接启动昇腾 NPU 上的模型训练。为控制时长，将 `max_epoch` 从 240 缩短到 5（其余参数全部保持脚本默认值）：

> **注意**：训练产物校验放在命令尾部，快速失败：

```shell #test-setup id="train"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd wenet/examples/aishell/s0
cp conf/train_conformer.yaml conf/train_conformer_5ep.yaml
sed -i 's/max_epoch: .*/max_epoch: 5/' conf/train_conformer_5ep.yaml
bash run_npu.sh --stage 4 --stop_stage 4 --train_config conf/train_conformer_5ep.yaml
test -f exp/conformer/train.yaml || { echo "train failed, check stderr above"; exit 1; }
```

训练完成后检查输出：

```shell #test id="verify-train"
cd wenet/examples/aishell/s0
ls -la exp/conformer/train.yaml
ls exp/conformer/*.pt | head -5
```

输出结果如下：

```shell #test-result id="verify-train"
... exp/conformer/train.yaml
exp/conformer/epoch_0.pt
exp/conformer/epoch_1.pt
exp/conformer/epoch_2.pt
exp/conformer/epoch_3.pt
exp/conformer/epoch_4.pt
```

---

## 11. 测试推理（stage 5）

stage 5 为模型测试推理阶段，将测试集中语音文件识别为文本。此外，stage 5 还提供平均模型的功能：当 `${average_checkpoint}` 为 `true`（脚本默认值）时，将交叉验证集上最佳的 `${average_num}` 个模型平均，生成增强模型 `avg_5.pt`，供解码与导出使用：

```shell #test-setup id="infer"
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 5 --stop_stage 5 --average_num 5
```

验证推理结果（测试集 7176 条全部识别完成，并抽样打印前两条识别文本）：

```shell #test id="verify-infer"
cd wenet/examples/aishell/s0
test -f exp/conformer/avg_5.pt && echo "avg_5.pt ok"
wc -l exp/conformer/ctc_greedy_search/text | awk '{print $1}'
head -2 exp/conformer/ctc_greedy_search/text
```

输出结果如下（xxx 为识别文本，随模型收敛情况变化）：

```shell #test-result id="verify-infer" fuzzy='xxx'
avg_5.pt ok
7176
xxx
xxx
```

---

## 12. 导出训练好的模型（stage 6）

stage 6 为模型导出阶段，`wenet/bin/export_jit.py` 使用 `Libtorch` 导出以上训练好的模型（基于 stage 5 生成的 `avg_5.pt`），导出的模型可用于其他编程语言（如 C++）的推理：

```shell #test-setup id="export"
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 6 --stop_stage 6 --average_num 5
```

验证导出产物：

```shell #test id="verify-export"
cd wenet/examples/aishell/s0
test -f exp/conformer/final.zip && echo "final.zip ok"
test -f exp/conformer/final_quant.zip && echo "final_quant.zip ok"
```

输出结果如下：

```shell #test-result id="verify-export"
final.zip ok
final_quant.zip ok
```

---

## 13. 验证完整流程

确认所有关键文件均已生成：

```shell #test id="verify-all"
cd wenet/examples/aishell/s0
echo "=== 数据文件 ==="
ls data/dict/lang_char.txt data/train/data.list data/dev/data.list data/test/data.list | sort
echo "=== 训练输出 ==="
ls exp/conformer/train.yaml exp/conformer/final.pt | sort
echo "=== 推理输出 ==="
ls exp/conformer/avg_5.pt exp/conformer/ctc_greedy_search/text exp/conformer/ctc_prefix_beam_search/text | sort
echo "=== 导出输出 ==="
ls exp/conformer/final.zip exp/conformer/final_quant.zip | sort
echo "=== 流程完成 ==="
```

输出结果如下：

```shell #test-result id="verify-all"
=== 数据文件 ===
data/dev/data.list
data/dict/lang_char.txt
data/test/data.list
data/train/data.list
=== 训练输出 ===
exp/conformer/final.pt
exp/conformer/train.yaml
=== 推理输出 ===
exp/conformer/avg_5.pt
exp/conformer/ctc_greedy_search/text
exp/conformer/ctc_prefix_beam_search/text
=== 导出输出 ===
exp/conformer/final.zip
exp/conformer/final_quant.zip
=== 流程完成 ===
```

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `npu-smi` 找不到 | 未 `source set_env.sh`，或 `npu-smi` 不在 `PATH` | 重做第 1-2 节 |
| `import torch_npu` 失败 | torch/torch_npu 版本不匹配 | 检查 [兼容矩阵](https://gitcode.com/Ascend/pytorch) |
| `npu available: False` | NPU 设备未挂载或驱动问题 | 检查 `/dev/davinci0` 是否存在 |
| `no such directory $data` | 数据目录未创建 | `mkdir -p` 创建绝对路径数据目录（stage -1 的前置要求） |
| 数据下载慢 / 失败 | openslr 出口带宽波动 | 重跑 stage -1，脚本按 `.complete` 断点续传；已内置 3 次自动重试 |
| 训练报错 OOM | batch_size 过大 | 减小 batch_size 或使用真实数据 |
| 训练段错误（worker fork 后） | torch_npu 2.2.0 + CANN 8.0.0 下 DataLoader 多进程 worker 的 fork-safety 问题（上游 PR #2563 验证栈可正常，属栈版本行为漂移） | 回退 `--num_workers 0`，并把 `wenet/utils/train_utils.py` 中硬编码的 `persistent_workers=True` / `prefetch_factor=args.prefetch` 改为与 `num_workers > 0` 条件兼容 |
| `sox` 命令失败 | 未安装 sox | `apt-get install sox libsox-dev` |
