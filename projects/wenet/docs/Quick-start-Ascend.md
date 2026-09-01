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
pip install torch==2.10.0 torch-npu==2.10.0.post4
pip install torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

安装 sox：

```shell #test-setup
pip install sox
apt-get update && apt-get install -y sox libsox-dev || true
```

修复 WeNet 与 PyTorch 2.10.0 的兼容性问题（`Union` 导入）：

```shell #test-setup id="fix-wenet-compat"
cd wenet
sed -i 's/from torch.nn.modules.conv import _ConvNd, _size_2_t, Union, _pair, Tensor, Optional/from typing import Optional, Union\nfrom torch import Tensor\nfrom torch.nn.common_types import _size_2_t\nfrom torch.nn.modules.conv import _ConvNd\nfrom torch.nn.modules.utils import _pair/' wenet/models/squeezeformer/conv2d.py
```

---

## 4. 验证 torch_npu 安装

```shell #test id="check-npu"
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

## 5. 准备训练数据

本节在本地生成最小化的测试数据，用于快速验证 WeNet 数据准备流程。

创建目录结构：

```shell #test-setup id="create-dirs"
mkdir -p /tmp/wenet-mock/data_aishell/wav/train/S0001
mkdir -p /tmp/wenet-mock/data_aishell/transcript
mkdir -p /tmp/wenet-mock/data/{train,dev,test}
```

用 sox 生成 3 条合成音频（16kHz, 1秒）：

```shell #test id="gen-audio"
sox -n -r 16000 -c 1 /tmp/wenet-mock/data_aishell/wav/train/S0001/BAC009S0002W001.wav trim 0.0 1.0
sox -n -r 16000 -c 1 /tmp/wenet-mock/data_aishell/wav/train/S0001/BAC009S0002W002.wav trim 0.0 1.0
sox -n -r 16000 -c 1 /tmp/wenet-mock/data_aishell/wav/train/S0001/BAC009S0002W003.wav trim 0.0 1.0
ls -la /tmp/wenet-mock/data_aishell/wav/train/S0001/
```

输出结果如下：

```shell #test-result id="gen-audio"
total 200
drwxr-xr-x 2 root root  4096 ...
drwxr-xr-x 3 root root  4096 ...
-rw-r--r-- 1 root root 64080 ...
-rw-r--r-- 1 root root 64080 ...
-rw-r--r-- 1 root root 64080 ...
```

创建 wav.scp 和 text 文件：

```shell #test id="create-scp"
cat > /tmp/wenet-mock/data/train/wav.scp << 'EOF'
BAC009S0002W001 /tmp/wenet-mock/data_aishell/wav/train/S0001/BAC009S0002W001.wav
BAC009S0002W002 /tmp/wenet-mock/data_aishell/wav/train/S0001/BAC009S0002W002.wav
BAC009S0002W003 /tmp/wenet-mock/data_aishell/wav/train/S0001/BAC009S0002W003.wav
EOF

cat > /tmp/wenet-mock/data/train/text << 'EOF'
BAC009S0002W001 今天天气真好
BAC009S0002W002 我喜欢编程
BAC009S0002W003 语音识别很有意思
EOF

cp /tmp/wenet-mock/data/train/wav.scp /tmp/wenet-mock/data/dev/
cp /tmp/wenet-mock/data/train/text /tmp/wenet-mock/data/dev/
cp /tmp/wenet-mock/data/train/wav.scp /tmp/wenet-mock/data/test/
cp /tmp/wenet-mock/data/train/text /tmp/wenet-mock/data/test/

cat /tmp/wenet-mock/data/train/wav.scp
cat /tmp/wenet-mock/data/train/text
```

输出结果如下：

```shell #test-result id="create-scp"
BAC009S0002W001 /tmp/wenet-mock/data_aishell/wav/train/S0001/BAC009S0002W001.wav
BAC009S0002W002 /tmp/wenet-mock/data_aishell/wav/train/S0001/BAC009S0002W002.wav
BAC009S0002W003 /tmp/wenet-mock/data_aishell/wav/train/S0001/BAC009S0002W003.wav
BAC009S0002W001 今天天气真好
BAC009S0002W002 我喜欢编程
BAC009S0002W003 语音识别很有意思
```

---

## 6. 准备 WeNet 数据格式（stage 2-3）

进入 aishell/s0 目录，将 mock 数据链接到 WeNet 期望的位置：

```shell #test id="link-data"
cd wenet/examples/aishell/s0
mkdir -p data
ln -sf /tmp/wenet-mock/data/train data/train
ln -sf /tmp/wenet-mock/data/dev data/dev
ln -sf /tmp/wenet-mock/data/test data/test
ls -la data/
```

输出结果如下：

```shell #test-result id="link-data"
total 8
drwxr-xr-x 2 root root 4096 ...
drwxr-xr-x 5 root root 4096 ...
lrwxrwxrwx 1 root root   24 ... dev -> /tmp/wenet-mock/data/dev
lrwxrwxrwx 1 root root   25 ... test -> /tmp/wenet-mock/data/test
lrwxrwxrwx 1 root root   26 ... train -> /tmp/wenet-mock/data/train
```

运行 stage 2（生成字典）和 stage 3（生成 data.list）：

```shell #test-setup id="prepare-data"
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 2 --stop_stage 3
```

验证生成的文件：

```shell #test id="verify-data"
cd wenet/examples/aishell/s0
ls -la data/dict/lang_char.txt
head -5 data/dict/lang_char.txt
ls -la data/train/data.list
head -1 data/train/data.list
```

输出结果如下：

```shell #test-result id="verify-data"
...
<blank> 0
<unk> 1
<sos/eos> 2
...
```

---

## 7. 训练 5 epochs（stage 4）

创建自定义配置文件，将 `max_epoch` 从 240 缩短到 5：

```shell #test-setup id="train"
cd wenet/examples/aishell/s0
cp conf/train_conformer.yaml conf/train_conformer_5ep.yaml
sed -i 's/max_epoch: .*/max_epoch: 5/' conf/train_conformer_5ep.yaml
bash run_npu.sh --stage 4 --stop_stage 4 --train_config conf/train_conformer_5ep.yaml
```

训练完成后检查输出：

```shell #test id="verify-train"
cd wenet/examples/aishell/s0
ls -la exp/conformer/train.yaml
ls exp/conformer/*.pt | head -5
```

输出结果如下：

```shell #test-result id="verify-train"
exp/conformer/train.yaml
exp/conformer/epoch_1.pt
exp/conformer/epoch_2.pt
exp/conformer/epoch_3.pt
exp/conformer/epoch_4.pt
exp/conformer/epoch_5.pt
```

---

## 8. 测试推理（stage 5）

使用训练好的模型对测试数据进行推理验证：

```shell #test-setup id="infer"
cd wenet/examples/aishell/s0
bash run_npu.sh --stage 5 --stop_stage 5
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
| 训练报错 OOM | 数据量太小，batch_size 过大 | 减小 batch_size 或使用真实数据 |
| `sox` 命令失败 | 未安装 sox | `apt-get install sox libsox-dev` |
