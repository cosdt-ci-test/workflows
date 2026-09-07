# Quick Start（TensorFlow 2.6.5 + Ascend NPU）

本文在单张昇腾 NPU 上安装 TensorFlow 2.6.5 和 TF Adapter 9.1.0，并运行
TF Adapter 2.x 的基本加法样例。TensorFlow 2.6.5 已停止演进，本文使用固定
兼容版本组合，不适用于 TensorFlow 最新 release。

安装和迁移逻辑来自 TF Adapter 9.1.0 的官方文档：

- [安装 TensorFlow 2.6.5](https://gitcode.com/cann/tensorflow/blob/9.1.0/docs/zh/tfadapter_2/installation/tensorflow-2-6-5_install.md)
- [安装 TF Adapter](https://gitcode.com/cann/tensorflow/blob/9.1.0/docs/zh/tfadapter_2/installation/tfadapter_install.md)
- [TensorFlow 2.6.5 手工迁移](https://gitcode.com/cann/tensorflow/blob/9.1.0/docs/zh/tfadapter_2/migration/script_migration/manual_porting.md)
- [npu.open API](https://gitcode.com/cann/tensorflow/blob/9.1.0/docs/zh/tfadapter_2/apiref/npu-open.md)

## 前置条件

### 硬件

- Atlas 900 A2 / A3 训练系列产品；
- 至少一张可用的昇腾 NPU；
- 驱动和容器运行环境已经配置完成，`npu-smi info` 能正常显示设备。

本文样例只打开一个 NPU 自定义设备，不使用 HCCL 或多卡并行，所以一张 NPU
即可。运行前请确认当前环境能正常访问该设备。

### 基础软件

在运行本文前，需要先安装可用的 CANN 9.1.0。安装方式参考
[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

### 本文档验证的固定配套

**配套机器**：Atlas 900 A2 PODc（Ascend 910B，单卡），Linux aarch64，
Ubuntu 22.04。

**配套镜像**：

`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12-devel`

保留镜像自带的 Python 3.12，另外将 Python 3.9.25 安装到
`/usr/local/python3.9.25`，只用于 TensorFlow 2.6.5。`devel` 变体提供
编译 Python、HDF5 和 h5py 所需的工具。

| 组件 | 版本 |
| --- | --- |
| Python | 3.9.25 |
| TensorFlow | 2.6.5 (`v2.6.5`) |
| CANN | 9.1.0 |
| TF Adapter branch | 9.1.0 |
| TF Adapter wheel release | `tfa_v0.0.49_9.1.0` |
| npu_device | 2.6.5 |
| HDF5 | 1.10.5 |
| h5py | 3.1.0 |
| numpy | 1.19.5 |
| protobuf | 3.19.6 |
| NPU | Ascend 910B × 1 |

## 安装前准备

官方文档支持 Python 3.7.x、3.8.x 和 3.9.x；这里使用与公开 aarch64 wheel
匹配的 Python 3.9.25。通过 `make altinstall` 安装到独立前缀，不修改
CANN 镜像已有的 Python 3.12。

### 安装编译依赖

如果已经配置了可用的 Ubuntu 软件源，可以删除下面的 `sed` 命令；在国内
环境中可将默认 `ports.ubuntu.com` 切换到阿里云 `ubuntu-ports` 镜像。

```shell #test-setup
set -euo pipefail
sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list
echo "Updating package indexes..."
apt-get update
echo "Package indexes updated."
```

```shell #test-setup
set -euo pipefail
echo "Installing build dependencies..."
apt-get install -y --no-install-recommends \
  build-essential ca-certificates curl \
  libbz2-dev libffi-dev libgdbm-dev liblzma-dev libncurses5-dev \
  libreadline-dev libsqlite3-dev libssl-dev tk-dev uuid-dev zlib1g-dev
echo "Build dependencies installed."
```

### 安装 Python 3.9.25

```shell #test-setup
set -euo pipefail
echo "Preparing Python 3.9.25..."
PYTHON_PREFIX=/usr/local/python3.9.25
PYTHON_ARCHIVE=/tmp/Python-3.9.25.tgz

if [ ! -x "$PYTHON_PREFIX/bin/python3.9" ]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 600 \
    https://repo.huaweicloud.com/python/3.9.25/Python-3.9.25.tgz \
    -o "$PYTHON_ARCHIVE"
  echo "a7438eabd3a48139f42d4e058096af8d880b0bb6e8fb8c78838892e4ce5583f2  $PYTHON_ARCHIVE" \
    | sha256sum -c -
  tar -xzf "$PYTHON_ARCHIVE" -C /tmp
  cd /tmp/Python-3.9.25
  ./configure --prefix=/usr/local/python3.9.25 --with-ensurepip=install
  make -j"$(nproc)"
  make altinstall
fi
ln -sfn "$PYTHON_PREFIX/bin/python3.9" "$PYTHON_PREFIX/bin/python3"
ln -sfn "$PYTHON_PREFIX/bin/pip3.9" "$PYTHON_PREFIX/bin/pip3"
export PATH="$PYTHON_PREFIX/bin:$PATH"
echo "Python 3.9.25 is ready."
```

### 编译安装 HDF5 1.10.5

对于 Linux aarch64，需要先编译安装 HDF5 1.10.5。

#### 下载 HDF5 源码包

```shell #test-setup
set -euo pipefail
echo "Downloading HDF5 1.10.5 if needed..."
HDF5_ARCHIVE=/tmp/hdf5-1.10.5.tar.gz
if [ ! -s "$HDF5_ARCHIVE" ]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 600 \
    https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz \
    -o "$HDF5_ARCHIVE"
  echo "6d4ce8bf902a97b050f6f491f4268634e252a63dadd6656a1a9be5b7b7726fa8  $HDF5_ARCHIVE" \
    | sha256sum -c -
fi
echo "HDF5 1.10.5 source is ready."
```

#### 解压 HDF5 源码包

```shell #test-setup
set -euo pipefail
tar -zxvf /tmp/hdf5-1.10.5.tar.gz -C /tmp
```

#### 配置、编译和安装 HDF5

```shell #test-setup
set -euo pipefail
echo "Building HDF5 1.10.5 if needed..."
HDF5_PREFIX=/usr/local/hdf5
if [ ! -x "$HDF5_PREFIX/bin/h5cc" ]; then
  cd /tmp/hdf5-1.10.5
  ./configure --prefix=/usr/local/hdf5
  make -j16 && make install
fi
echo "HDF5 1.10.5 is ready."
```

#### 配置 HDF5 环境变量

```shell #test-setup
export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-}
```

### 安装 h5py

#### 安装 h5py 依赖包

```shell #test-setup
set -euo pipefail
echo "Installing h5py dependencies..."
export PATH=/usr/local/python3.9.25/bin:$PATH
export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-}
pip3 install "setuptools==59.8.0"
pip3 install "Cython<3"
pip3 install wheel
pip3 install "numpy==1.19.5"
echo "h5py dependencies installed."
```

#### 安装 h5py 3.1.0

```shell #test-setup
set -euo pipefail
echo "Installing h5py 3.1.0..."
export PATH=/usr/local/python3.9.25/bin:$PATH
export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-}
HDF5_DIR=/usr/local/hdf5 pip3 install "h5py==3.1.0"
echo "h5py 3.1.0 is installed."
```

检查 Python、HDF5、numpy 和 h5py 版本：

```shell #test id="check-python"
export PATH=/usr/local/python3.9.25/bin:$PATH
python3 - <<'PY'
import re
import subprocess
import sys

import h5py
import numpy

print("Python", sys.version.split()[0])
h5_config = subprocess.check_output(
    ["/usr/local/hdf5/bin/h5cc", "-showconfig"],
    text=True,
)
match = re.search(r"HDF5 Version:\s+(\S+)", h5_config)
assert match is not None, h5_config
print("HDF5", match.group(1))
print("numpy", numpy.__version__)
print("h5py", h5py.__version__)
PY
```

```shell #test-result id="check-python"
Python 3.9.25
HDF5 1.10.5
numpy 1.19.5
h5py 3.1.0
```

## 安装 TensorFlow 2.6.5

PyPI 没有 Linux aarch64 的 TensorFlow 2.6.5 wheel。这里使用 Ascend 官方
镜像项目公开的 Python 3.9 aarch64 wheel，并校验 SHA-256。

```shell #test-setup
set -euo pipefail
echo "Installing TensorFlow 2.6.5..."
TF_CACHE=/root/.cache/tensorflow
TF_WHEEL="$TF_CACHE/wheels/tensorflow-2.6.5-cp39-cp39-linux_aarch64.whl"
mkdir -p "$TF_CACHE/wheels"

if [ ! -s "$TF_WHEEL" ]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 900 \
    https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindX/OpenSource/packages/tensorflow-2.6.5-cp39-cp39-linux_aarch64.whl \
    -o "$TF_WHEEL.tmp"
  mv "$TF_WHEEL.tmp" "$TF_WHEEL"
fi
echo "be1c8f52d6a72cc0db5826605f61c196777f5939441b7e87442688a5d1866bd0  $TF_WHEEL" \
  | sha256sum -c - >/dev/null

export PATH=/usr/local/python3.9.25/bin:$PATH
pip3 install "protobuf==3.19.6"
pip3 install "$TF_WHEEL"
echo "TensorFlow 2.6.5 is installed."
```

验证 TensorFlow 版本和 Eager 模式：

```shell #test id="install-tensorflow"
export PATH=/usr/local/python3.9.25/bin:$PATH
LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-} \
  TF_CPP_MIN_LOG_LEVEL=2 python3 - <<'PY'
import tensorflow as tf

print("tensorflow", tf.__version__)
print("eager", tf.executing_eagerly())
PY
```

```shell #test-result id="install-tensorflow"
tensorflow 2.6.5
eager True
```

## 安装框架插件包 TF Adapter

TF Adapter 2.x 通过 `npu_device` 将 NPU 注册为 TensorFlow 自定义设备。

### 安装插件包

#### 1. 获取 TF Adapter 安装包

从 TF Adapter GitCode 仓选择与 CANN 9.1.0 配套的发布
`tfa_v0.0.49_9.1.0`，获取
`npu_device-2.6.5-py3-none-manylinux2014_aarch64.whl`。

```shell #test-setup
set -euo pipefail
echo "Downloading TF Adapter 9.1.0 if needed..."
PACKAGE_DIR=/home/package
ADAPTER_WHEEL="$PACKAGE_DIR/npu_device-2.6.5-py3-none-manylinux2014_aarch64.whl"
mkdir -p "$PACKAGE_DIR"

if [ ! -s "$ADAPTER_WHEEL" ]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 600 \
    https://gitcode.com/cann/tensorflow/releases/download/tfa_v0.0.49_9.1.0/npu_device-2.6.5-py3-none-manylinux2014_aarch64.whl \
    -o "$ADAPTER_WHEEL.tmp"
  mv "$ADAPTER_WHEEL.tmp" "$ADAPTER_WHEEL"
fi
echo "68a14762b24ebfafe554c2a29406be2932b82a1950938d1de97a2cc0909d73fc  $ADAPTER_WHEEL" \
  | sha256sum -c - >/dev/null
echo "TF Adapter wheel is ready."
```

#### 2. 安装 TF Adapter

```shell #test-setup
set -euo pipefail
echo "Installing npu_device 2.6.5..."
export PATH=/usr/local/python3.9.25/bin:$PATH
TFPLUGIN_INSTALL_PATH=$HOME/Ascend/tfplugin
ADAPTER_WHEEL=/home/package/npu_device-2.6.5-py3-none-manylinux2014_aarch64.whl
mkdir -p "$TFPLUGIN_INSTALL_PATH"
pip3 install "$ADAPTER_WHEEL" --force-reinstall -t "$TFPLUGIN_INSTALL_PATH"
echo "npu_device 2.6.5 is installed."
```

- `--force-reinstall`：强制重新安装插件包。
- `-t`：指定 TF Adapter 的安装路径。

#### 3. 设置 TF Adapter 环境变量

```shell #test-setup
TFPLUGIN_INSTALL_PATH=$HOME/Ascend/tfplugin
export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH
```

`TFPLUGIN_INSTALL_PATH` 为 TF Adapter 软件包的安装路径。

验证插件安装版本：

```shell #test id="install-tf-adapter"
export PATH=/usr/local/python3.9.25/bin:$PATH
TFPLUGIN_INSTALL_PATH=$HOME/Ascend/tfplugin
PYTHONPATH="$TFPLUGIN_INSTALL_PATH${PYTHONPATH:+:$PYTHONPATH}" \
  LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-} \
  TF_CPP_MIN_LOG_LEVEL=2 python3 - <<'PY'
from importlib.metadata import version

import npu_device

print("npu_device", version("npu-device"))
PY
```

```shell #test-result id="install-tf-adapter"
npu_device 2.6.5
```

## 配置运行环境

若 CANN 安装在非默认目录，请替换 `set_env.sh` 路径；每个终端会话都需要
重新设置这些环境变量。

```shell #test-setup
if [ -f /usr/local/Ascend/cann/set_env.sh ]; then
  source /usr/local/Ascend/cann/set_env.sh
else
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export PATH=/usr/local/python3.9.25/bin:$PATH
export TFPLUGIN_INSTALL_PATH=$HOME/Ascend/tfplugin
export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH
export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-}
export JOB_ID=tensorflow-quick-start
export ASCEND_DEVICE_ID=0
```

## 运行 npu_device 样例

下面的调用方式来自 TF Adapter 2.x 的 `examples/basic_tests.py` 和手工迁移
文档：通过 `npu.open().as_default()` 设置默认 NPU 设备，并用
`@tf.function` 编译 TensorFlow 运算。

```shell #test id="run-npu-device"
set -euo pipefail
export PATH=/usr/local/python3.9.25/bin:$PATH
TFPLUGIN_INSTALL_PATH=$HOME/Ascend/tfplugin
PYTHONPATH="$TFPLUGIN_INSTALL_PATH${PYTHONPATH:+:$PYTHONPATH}" \
LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-} \
JOB_ID=tensorflow-quick-start \
ASCEND_DEVICE_ID=0 \
TF_CPP_MIN_LOG_LEVEL=2 \
python3 - <<'PY'
import tensorflow as tf
import npu_device as npu

npu_context = npu.open().as_default()

@tf.function
def add(left, right):
    return tf.add(left, right)

left = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
right = tf.constant([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
result = add(left, right)
expected = [[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]]

assert result.numpy().tolist() == expected, result
print("device", npu_context.name())
print("result", result.numpy().tolist())
print("TensorFlow 2.6.5 Ascend Quick Start PASSED")
PY
```

```shell #test-result id="run-npu-device" fuzzy="..."
...
device ...NPU:0
result [[11.0, 22.0, 33.0], [44.0, 55.0, 66.0]]
TensorFlow 2.6.5 Ascend Quick Start PASSED
...
```
