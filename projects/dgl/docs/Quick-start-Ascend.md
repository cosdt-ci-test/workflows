# DGL-Ascend 快速入门指南（Ascend NPU）

欢迎使用 DGL-Ascend！本指南帮助你在单卡昇腾 NPU 上安装 DGL-Ascend 并跑通一个图神经网络训练示例。

## 系统要求

- **硬件**：Atlas 900 A2 系列（Ascend 910B4），单卡 32 GB HBM 可跑本文全部内容
- **操作系统**：Linux（Ubuntu 22.04）
- **存储**：至少 20 GB 可用空间（含源码编译与示例数据）

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| CANN | ≥ 8.5.1 | 环境搭建安装（CI 以 9.1.0 验证） |
| Python | ≥ 3.10 | 自备环境（CI 镜像为 3.12） |
| torch / torch_npu | 2.9.0 / 2.9.0.post2 | 下方安装 |
| DGL-Ascend | 源码（master） | 下方编译安装 |

> 也可用带 CANN 的昇腾镜像（如 [ascendhub cann 镜像](https://www.hiascend.com/developer/ascendhub)）跳过 CANN 安装，其余步骤相同。

## 环境搭建

**安装 CANN**（≥ 8.5.1，与驱动配套，[快速安装脚本](https://ascend.github.io/docs/sources/ascend/quick_install.html)会自动识别卡型），安装完成后：

```shell
source ~/Ascend/ascend-toolkit/set_env.sh
```

**准备 Python 环境**：Python ≥ 3.10。

**安装 torch + torch_npu**：

```shell #test-setup id="dgl-install-torch"
pip install uv
uv pip install "torch==2.9.0" "torch_npu==2.9.0.post2"
```

## 安装 DGL-Ascend

**获取源码并编译安装**（DGL-Ascend 提供昇腾 NPU 算子，需从源码编译）：

```shell #test-setup id="dgl-install"
git clone https://github.com/BUPT-GAMMA/dgl-ascend.git
cd dgl-ascend
git submodule update --init --recursive
bash script/build_dgl_ascend.sh
cd python
pip install -e .
```

**验证安装**（全部就位时输出 `dgl: xxx` 与 `npu available: True`）：

```shell #test id="dgl-install-verify"
python -c "
import dgl
import torch
import torch_npu
print('dgl:', dgl.__version__)
print('torch:', torch.__version__)
print('npu available:', torch.npu.is_available())
"
```

```shell #test-result id="dgl-install-verify" fuzzy='xxx' fuzzy='...'
...dgl: xxx
torch: 2.9.xxx
npu available: True
```

## 运行 LightGCN

在 gowalla 数据集上训练一版 LightGCN 图推荐模型（单卡、1 轮到）：

```shell #test id="dgl-lightgcn-smoke"
cd dgl-ascend/examples/pytorch/lightgcn
wget https://s3.us-west-2.amazonaws.com/dgl-data/dataset/gowalla.zip
unzip gowalla.zip
# 图里存在孤立节点，关闭 GraphConv 的 0 入度检查
sed -i 's/weight=False, bias=False)/weight=False, bias=False, allow_zero_in_degree=True)/' model.py
python main.py --dataset gowalla --batch 2048 --recdim 64 --epochs 1 --device npu
```

```shell #test-result id="dgl-lightgcn-smoke" fuzzy='...'
...Average BPR Loss: ...
```

> **注意**：如新开终端执行，先 `source ~/Ascend/ascend-toolkit/set_env.sh`。更多模型与示例见 [DGL-Ascend 仓库](https://github.com/BUPT-GAMMA/dgl-ascend)。