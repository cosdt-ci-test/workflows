# 快速开始：在昇腾 NPU 上用 WeNet 训练语音识别模型

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**：在单卡 NPU 上完成 WeNet 的数据准备、训练和推理全流程。

[WeNet](https://github.com/wenet-e2e/wenet) 是一个生产级端到端语音识别工具包，支持流式和非流式识别。昇腾侧通过 `torch-npu` 将计算调度到 NPU。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| Python | 3.10+ |
| 编译工具 | git |
| 音频工具 | sox |

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

安装 WeNet 及其 NPU 依赖：

```shell #test-setup id="install"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd wenet
pip install -e .
pip install torch==2.2.0 torch-npu==2.2.0.post2
pip install torchaudio==0.17.0 --index-url https://download.pytorch.org/whl/cpu
pip install "deepspeed<0.19.6" tensorboardX
pip install "numpy<2"
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

通过 ModelScope 下载 aishell-1 数据集（~15.6 GB），然后创建软链接适配 weNet 脚本期望的目录结构：

```shell #test-setup id="download-data"
pip install modelscope -q
mkdir -p /root/asr-data/OpenSLR/33
# 从 ModelScope 下载 AISHELL-1 数据集
python -c "
from modelscope import snapshot_download
snapshot_download('OmniData/AISHELL-1', local_dir='/root/asr-data/AISHELL-1', repo_type='dataset')
"
# 创建 weNet 期望的目录结构（软链接）
ln -sf /root/asr-data/AISHELL-1/data_aishell /root/asr-data/OpenSLR/33/data_aishell
ln -sf /root/asr-data/AISHELL-1/resource_aishell /root/asr-data/OpenSLR/33/resource_aishell
# 创建 .complete 标记文件
touch /root/asr-data/OpenSLR/33/data_aishell/.complete
touch /root/asr-data/OpenSLR/33/resource_aishell/.complete
```

验证 `data_aishell` 与 `resource_aishell` 两个数据包均下载完成：

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

stage 0 阶段为训练数据准备阶段，将使用 `local/aishell_data_prep.sh` 脚本将训练数据重新组织为 `wav.scp` 和 `text` 两部分：

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

## 7. 训练 5 epochs（stage 4）

创建自定义配置文件，将 `max_epoch` 从 240 缩短到 5：

```shell #test-setup id="train"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd wenet/examples/aishell/s0
cp conf/train_conformer.yaml conf/train_conformer_5ep.yaml
sed -i 's/max_epoch: .*/max_epoch: 5/' conf/train_conformer_5ep.yaml
bash run_npu.sh --stage 4 --stop_stage 4 --train_config conf/train_conformer_5ep.yaml --data /root/asr-data/OpenSLR/33
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

## 8. 测试推理（stage 5）

使用训练好的模型对测试数据进行推理验证：

```shell #test-setup id="infer"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 5 --stop_stage 5 --average_num 5 --data /root/asr-data/OpenSLR/33
```

验证推理结果：

```shell #test id="verify-infer"
cd wenet/examples/aishell/s0
ls -la exp/conformer/ctc_greedy_search/text
head -3 exp/conformer/ctc_greedy_search/text
```

输出结果如下：

```shell #test-result id="verify-infer"
...
BAC009S0002W001 ...
BAC009S0002W002 ...
BAC009S0002W003 ...
```

---

## 9. 验证完整流程

确认所有关键文件均已生成：

```shell #test id="verify-all"
cd wenet/examples/aishell/s0
echo "=== 数据文件 ==="
ls data/dict/lang_char.txt data/train/data.list data/dev/data.list data/test/data.list
echo "=== 训练输出 ==="
ls exp/conformer/train.yaml exp/conformer/final.pt
echo "=== 推理输出 ==="
ls exp/conformer/ctc_greedy_search/text exp/conformer/ctc_prefix_beam_search/text
echo "=== 流程完成 ==="
```

输出结果如下：

```shell #test-result id="verify-all"
=== 数据文件 ===
data/dict/lang_char.txt
data/train/data.list
data/dev/data.list
data/test/data.list
=== 训练输出 ===
exp/conformer/train.yaml
exp/conformer/final.pt
=== 推理输出 ===
exp/conformer/ctc_greedy_search/text
exp/conformer/ctc_prefix_beam_search/text
=== 流程完成 ===
```

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `npu-smi` 找不到 | 未 `source set_env.sh`，或 `npu-smi` 不在 `PATH` | 重做第 1-2 节 |
| `import torch_npu` 失败 | torch/torch_npu 版本不匹配 | 检查 [兼容矩阵](https://gitcode.com/Ascend/pytorch) |
| `npu available: False` | NPU 设备未挂载或驱动问题 | 检查 `/dev/davinci0` 是否存在 |
| 数据下载慢 / 失败 | ModelScope CDN 波动 | 重跑 stage -1，脚本按 `.complete` 断点续传 |
| `sox` 命令失败 | 未安装 sox | `apt-get install sox libsox-dev` |
