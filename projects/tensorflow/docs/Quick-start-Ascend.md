# Quick Start（TensorFlow 1.15 + Ascend NPU）

本文在单张昇腾 NPU 上安装 TensorFlow 1.15 和 TF Adapter 9.1.0，并运行
官方 `NpuOptimizer` 加法样例。TensorFlow 1.15 已停止演进，因此这里看护的
是一个**固定兼容基线**，不是 TensorFlow 最新 2.x release。

安装和迁移逻辑来自 TF Adapter 9.1.0 的三份官方文档：

- [安装 TensorFlow 1.15](https://gitcode.com/cann/tensorflow/blob/9.1.0/docs/zh/tfadapter_1/installation/tensorflow-1-15_install.md)
- [安装 TF Adapter](https://gitcode.com/cann/tensorflow/blob/9.1.0/docs/zh/tfadapter_1/installation/tfadapter_install.md)
- [TF Adapter 1.x 快速入门](https://gitcode.com/cann/tensorflow/blob/9.1.0/docs/zh/tfadapter_1/quick_start.md)

也可以从昇腾社区文档中心查看对应的[在线最新文档](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/latest/migration/tfmigr1/docs/zh/tfadapter_1/quick_start.md)。

## 前置条件

### 硬件

- Atlas 900 A2 / A3 训练系列产品；
- 至少一张可用的昇腾 NPU；
- 驱动和容器运行环境已经配置完成，`npu-smi info` 能正常显示设备。

本文样例只创建一个 TensorFlow Session，不使用 HCCL 或多卡并行，所以
`linux-aarch64-a2-1` 的一张 NPU 足够。CI Runner 会自动提供设备和驱动，
无需在工作流中重复填写 `--device` 或驱动目录挂载。

### 基础软件

在运行本文前，需要先安装可用的 CANN 9.1.0。安装方式参考
[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。
CI 使用与 ms-swift 相同来源的 AscendHub CANN 镜像，Python 包默认经过
集群 PyPI 缓存，并把华为云昇腾 PyPI 作为额外源。

### 本文档验证的固定配套

**配套机器**：Atlas 900 A2 PODc（Ascend 910B，单卡），Linux aarch64，
Ubuntu 22.04。

**配套镜像**：

`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12-devel`

镜像自带的 Python 3.12 用来运行文档测试框架；TensorFlow 1.15 运行在隔离的
Python 3.7.10 环境中。`devel` 变体只用于提供编译 h5py 2.8.0 所需的工具，
CANN 版本和设备环境仍与 ms-swift 的 9.1.0 基线一致。

| 组件 | 版本 |
| --- | --- |
| Python | 3.7.10 |
| TensorFlow | 1.15.0 (`v1.15.0`) |
| CANN | 9.1.0 |
| TF Adapter branch | 9.1.0 |
| TF Adapter wheel release | `tfa_v0.0.49_9.1.0` |
| npu_bridge | 1.15.0 |
| HDF5 | 1.10.5 |
| h5py | 2.8.0 |
| NPU | Ascend 910B × 1 |

## 准备 Python 3.7 环境

官方文档要求 TensorFlow 1.15 使用 Python 3.7.x；这里固定为仍处于官方范围
内的 Python 3.7.10。为了不替换 CANN 镜像的系统 Python，使用固定版本的
micromamba 创建隔离环境，并把下载内容保存到当前容器的工作缓存目录。

```shell #test-setup
set -euo pipefail
TF_CACHE=/root/.cache/tensorflow
TF_ENV="$TF_CACHE/envs/tf115"
MICROMAMBA="$TF_CACHE/bin/micromamba"
MICROMAMBA_ARCHIVE="$TF_CACHE/micromamba-2.3.2-linux-aarch64.tar.bz2"
mkdir -p "$TF_CACHE/bin" "$TF_CACHE/envs"

if [ ! -x "$MICROMAMBA" ]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 300 \
    https://micro.mamba.pm/api/micromamba/linux-aarch64/2.3.2 \
    -o "$MICROMAMBA_ARCHIVE.tmp"
  echo "7ded447a291cd1a05efe42c895a43f11fa3446011957cffe899aeabda8c3ee25  $MICROMAMBA_ARCHIVE.tmp" \
    | sha256sum -c -
  mv "$MICROMAMBA_ARCHIVE.tmp" "$MICROMAMBA_ARCHIVE"
  tar -xjf "$MICROMAMBA_ARCHIVE" -C "$TF_CACHE" bin/micromamba
fi

export MAMBA_ROOT_PREFIX="$TF_CACHE/mamba-root"
if [ -x "$TF_ENV/bin/python" ]; then
  "$MICROMAMBA" install -y -p "$TF_ENV" -c conda-forge \
    'python=3.7.10' 'pip=23.1.2' 'setuptools=59.8.0' 'wheel=0.37.1' \
    'hdf5=1.10.5' 'numpy=1.19.5'
else
  "$MICROMAMBA" create -y -p "$TF_ENV" -c conda-forge \
    'python=3.7.10' 'pip=23.1.2' 'setuptools=59.8.0' 'wheel=0.37.1' \
    'hdf5=1.10.5' 'numpy=1.19.5'
fi
```

检查 Python 与 HDF5 版本：

```shell #test id="check-python"
TF_CACHE=/root/.cache/tensorflow
TF_ENV="$TF_CACHE/envs/tf115"
"$TF_ENV/bin/python" - <<'PY'
import re
import subprocess
import sys
from pathlib import Path

print("Python", sys.version.split()[0])
h5_config = subprocess.check_output(
    [str(Path(sys.prefix) / "bin" / "h5cc"), "-showconfig"],
    text=True,
)
match = re.search(r"HDF5 Version:\s+(\S+)", h5_config)
assert match is not None, h5_config
print("HDF5", match.group(1))
PY
```

```shell #test-result id="check-python"
Python 3.7.10
HDF5 1.10.5
```

## 安装 TensorFlow 1.15

官方说明指出 PyPI 没有 Linux aarch64 的 TensorFlow 1.15 wheel。这里使用
Ascend 官方镜像项目公开的 aarch64 wheel，并校验 SHA-256；HDF5 1.10.5
由上一步安装，h5py 2.8.0 按官方要求从源码编译。

```shell #test-setup
set -euo pipefail
TF_CACHE=/root/.cache/tensorflow
TF_ENV="$TF_CACHE/envs/tf115"
TF_PYTHON="$TF_ENV/bin/python"
TF_WHEEL="$TF_CACHE/wheels/tensorflow-1.15.0-cp37-cp37m-manylinux2014_aarch64.whl"
mkdir -p "$TF_CACHE/wheels"

if [ ! -s "$TF_WHEEL" ]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 900 \
    https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/python/packages/tensorflow-1.15.0-cp37-cp37m-manylinux2014_aarch64.whl \
    -o "$TF_WHEEL.tmp"
  mv "$TF_WHEEL.tmp" "$TF_WHEEL"
fi
echo "c2d6df0930f6558ec9bb741c219cb84f90f906cc9e2c28c6561960a1404dec39  $TF_WHEEL" \
  | sha256sum -c - >/dev/null

"$TF_PYTHON" -m pip install --upgrade \
  'Cython==0.29.36' 'pkgconfig==1.5.5' 'protobuf==3.20.3'
HDF5_DIR="$TF_ENV" "$TF_PYTHON" -m pip install --no-build-isolation 'h5py==2.8.0'
"$TF_PYTHON" -m pip install --upgrade "$TF_WHEEL"
```

验证 TensorFlow 和 h5py 的实际安装版本：

```shell #test id="install-tensorflow"
TF_PYTHON=/root/.cache/tensorflow/envs/tf115/bin/python
TF_CPP_MIN_LOG_LEVEL=2 "$TF_PYTHON" - <<'PY'
import h5py
import tensorflow as tf

print("tensorflow", tf.__version__)
print("h5py", h5py.__version__)
PY
```

```shell #test-result id="install-tensorflow"
tensorflow 1.15.0
h5py 2.8.0
```

## 安装 TF Adapter

从与 CANN 9.1.0 对应的官方发布 `tfa_v0.0.49_9.1.0` 获取
`npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl`，并按照官方文档用
`-t "$TFPLUGIN_INSTALL_PATH"` 安装到独立目录。

```shell #test-setup
set -euo pipefail
TF_CACHE=/root/.cache/tensorflow
TF_ENV="$TF_CACHE/envs/tf115"
TF_PYTHON="$TF_ENV/bin/python"
TFPLUGIN_INSTALL_PATH="$TF_CACHE/tfplugin-9.1.0"
ADAPTER_WHEEL="$TF_CACHE/wheels/npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl"
mkdir -p "$TF_CACHE/wheels" "$TFPLUGIN_INSTALL_PATH"

if [ ! -s "$ADAPTER_WHEEL" ]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 600 \
    https://gitcode.com/cann/tensorflow/releases/download/tfa_v0.0.49_9.1.0/npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl \
    -o "$ADAPTER_WHEEL.tmp"
  mv "$ADAPTER_WHEEL.tmp" "$ADAPTER_WHEEL"
fi
echo "faac580f9a732e86b6ad2a49150eb450757ee18c173ec099044d855d364d4d98  $ADAPTER_WHEEL" \
  | sha256sum -c - >/dev/null

"$TF_PYTHON" -m pip install --no-deps --upgrade --force-reinstall \
  -t "$TFPLUGIN_INSTALL_PATH" "$ADAPTER_WHEEL"
```

验证插件安装版本：

```shell #test id="install-tf-adapter"
TF_CACHE=/root/.cache/tensorflow
TF_ENV="$TF_CACHE/envs/tf115"
TF_PYTHON="$TF_ENV/bin/python"
TFPLUGIN_INSTALL_PATH="$TF_CACHE/tfplugin-9.1.0"
PYTHONPATH="$TFPLUGIN_INSTALL_PATH${PYTHONPATH:+:$PYTHONPATH}" \
  TF_CPP_MIN_LOG_LEVEL=2 "$TF_PYTHON" - <<'PY'
import pkg_resources
import npu_bridge

print("npu_bridge", pkg_resources.get_distribution("npu-bridge").version)
PY
```

```shell #test-result id="install-tf-adapter"
npu_bridge 1.15.0
```

## 配置运行环境

下面的变量与官方 Quick Start 一致。若 CANN 安装在非默认目录，请替换
`set_env.sh` 路径；每个终端会话都需要重新执行这些 `export`。

```shell #test-setup
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
  source /usr/local/Ascend/cann/set_env.sh
else
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export TFPLUGIN_INSTALL_PATH=/root/.cache/tensorflow/tfplugin-9.1.0
export PYTHONPATH="${TFPLUGIN_INSTALL_PATH}:${PYTHONPATH:-}"
export JOB_ID=tensorflow-quick-start
export ASCEND_DEVICE_ID=0
```

## 运行 NpuOptimizer 样例

以下逻辑保持官方迁移步骤：导入 `npu_bridge`、向 Session 注册
`NpuOptimizer`，并关闭会与 NPU 图优化冲突的 remapping 和内存优化。为让
CI 输出稳定，输入由官方示例中的随机张量改成占位符和固定数据。

```shell #test id="run-npu-optimizer"
set -euo pipefail
TF_CACHE=/root/.cache/tensorflow
TF_PYTHON="$TF_CACHE/envs/tf115/bin/python"
TFPLUGIN_INSTALL_PATH="$TF_CACHE/tfplugin-9.1.0"
PYTHONPATH="$TFPLUGIN_INSTALL_PATH${PYTHONPATH:+:$PYTHONPATH}" \
JOB_ID=tensorflow-quick-start \
ASCEND_DEVICE_ID=0 \
TF_CPP_MIN_LOG_LEVEL=2 \
"$TF_PYTHON" - <<'PY'
import tensorflow as tf
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

a = tf.placeholder(tf.float32, shape=[2, 3], name="a")
b = tf.placeholder(tf.float32, shape=[2, 3], name="b")
c = tf.add(a, b, name="sum")

config = tf.ConfigProto(allow_soft_placement=True)
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

with tf.Session(config=config) as session:
    result = session.run(
        c,
        feed_dict={
            a: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            b: [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        },
    )

expected = [[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]]
assert result.tolist() == expected, result
print("optimizer", custom_op.name)
print("result", result.tolist())
print("TensorFlow Ascend Quick Start PASSED")
PY
```

```shell #test-result id="run-npu-optimizer" fuzzy="..."
...
optimizer NpuOptimizer
result [[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]]
TensorFlow Ascend Quick Start PASSED
...
```
