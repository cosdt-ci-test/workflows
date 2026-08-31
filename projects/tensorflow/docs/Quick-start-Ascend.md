# Quick Start（TensorFlow 1.15 + Ascend NPU）

本文在单张昇腾 NPU 上安装 TensorFlow 1.15 和 TF Adapter 9.1.0，并运行
官方 `NpuOptimizer` 加法样例。TensorFlow 1.15 已停止演进，本文使用固定的
兼容版本组合，不适用于 TensorFlow 最新 2.x release。

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
一张 NPU 即可。运行前请确认当前环境能正常访问该设备。

### 基础软件

在运行本文前，需要先安装可用的 CANN 9.1.0。安装方式参考
[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

### 本文档验证的固定配套

**配套机器**：Atlas 900 A2 PODc（Ascend 910B，单卡），Linux aarch64，
Ubuntu 22.04。

**配套镜像**：

`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12-devel`

保留镜像自带的 Python 3.12，另外将 Python 3.7.10 安装到
`/usr/local/python3.7.10`，只用于 TensorFlow 1.15。`devel` 变体提供编译
Python、HDF5 和 h5py 所需的工具。

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
内的 Python 3.7.10。使用 `make altinstall` 安装到独立前缀，不创建 conda
或 venv，也不会修改 CANN 镜像已有的 Python 3.12。

### 安装编译依赖

如果已经配置了可用的 Ubuntu 软件源，可以删除下面的 `sed` 命令；在国内
环境中可将默认 `ports.ubuntu.com` 切换到阿里云 `ubuntu-ports` 镜像。

```shell #test-setup
set -euo pipefail
sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  build-essential ca-certificates curl \
  libbz2-dev libffi-dev libgdbm-dev liblzma-dev libncurses5-dev \
  libreadline-dev libsqlite3-dev libssl-dev tk-dev uuid-dev zlib1g-dev
```

### 安装 Python 3.7.10

```shell #test-setup
set -euo pipefail
PYTHON_PREFIX=/usr/local/python3.7.10
PYTHON_ARCHIVE=/tmp/Python-3.7.10.tgz

if [ ! -x "$PYTHON_PREFIX/bin/python3.7" ]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 600 \
    https://repo.huaweicloud.com/python/3.7.10/Python-3.7.10.tgz \
    -o "$PYTHON_ARCHIVE"
  echo "c9649ad84dc3a434c8637df6963100b2e5608697f9ba56d82e3809e4148e0975  $PYTHON_ARCHIVE" \
    | sha256sum -c -
  tar -xzf "$PYTHON_ARCHIVE" -C /tmp
  cd /tmp/Python-3.7.10
  ./configure --prefix=/usr/local/python3.7.10 --with-ensurepip=install
  make -j"$(nproc)"
  make altinstall
fi
```

### 安装 uv

```shell #test-setup
set -euo pipefail
python -m pip install -q uv
```

uv 可以通过 `--python` 将包安装到指定的 Python 3.7，无需替换系统 Python
或创建虚拟环境。`UV_PYTHON_DOWNLOADS=never` 用于禁止自动下载其他解释器。

### 编译安装 HDF5 1.10.5

```shell #test-setup
set -euo pipefail
HDF5_PREFIX=/usr/local/hdf5
HDF5_ARCHIVE=/tmp/hdf5-1.10.5.tar.gz

if [ ! -x "$HDF5_PREFIX/bin/h5cc" ]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 600 \
    https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz \
    -o "$HDF5_ARCHIVE"
  echo "6d4ce8bf902a97b050f6f491f4268634e252a63dadd6656a1a9be5b7b7726fa8  $HDF5_ARCHIVE" \
    | sha256sum -c -
  tar -xzf "$HDF5_ARCHIVE" -C /tmp
  cd /tmp/hdf5-1.10.5
  ./configure --prefix=/usr/local/hdf5
  make -j16 && make install
fi

export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-}
```

### 安装 h5py 2.8.0

```shell #test-setup
set -euo pipefail
TF_PYTHON=/usr/local/python3.7.10/bin/python3.7
HDF5_PREFIX=/usr/local/hdf5

UV_PYTHON_DOWNLOADS=never uv pip install --python "$TF_PYTHON" \
  'setuptools==59.8.0' 'Cython==0.29.36' \
  'wheel==0.37.1' 'numpy==1.19.5' 'pkgconfig==1.5.5'
CPATH="$HDF5_PREFIX/include/:$HDF5_PREFIX/lib/" \
LD_LIBRARY_PATH="$HDF5_PREFIX/lib/:${LD_LIBRARY_PATH:-}" \
HDF5_DIR="$HDF5_PREFIX" UV_PYTHON_DOWNLOADS=never \
  uv pip install --python "$TF_PYTHON" --no-build-isolation 'h5py==2.8.0'
```

检查 Python、HDF5 和 h5py 版本：

```shell #test id="check-python"
TF_PYTHON=/usr/local/python3.7.10/bin/python3.7
"$TF_PYTHON" - <<'PY'
import re
import subprocess
import sys
import h5py

print("Python", sys.version.split()[0])
h5_config = subprocess.check_output(
    ["/usr/local/hdf5/bin/h5cc", "-showconfig"],
    text=True,
)
match = re.search(r"HDF5 Version:\s+(\S+)", h5_config)
assert match is not None, h5_config
print("HDF5", match.group(1))
print("h5py", h5py.__version__)
PY
```

```shell #test-result id="check-python"
Python 3.7.10
HDF5 1.10.5
h5py 2.8.0
```

## 安装 TensorFlow 1.15

官方说明指出 PyPI 没有 Linux aarch64 的 TensorFlow 1.15 wheel。这里使用
Ascend 官方镜像项目公开的 aarch64 wheel，并校验 SHA-256。

```shell #test-setup
set -euo pipefail
TF_CACHE=/root/.cache/tensorflow
TF_PYTHON=/usr/local/python3.7.10/bin/python3.7
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

UV_PYTHON_DOWNLOADS=never \
  uv pip install --python "$TF_PYTHON" 'protobuf==3.20.3'
UV_PYTHON_DOWNLOADS=never \
  uv pip install --python "$TF_PYTHON" "$TF_WHEEL"
```

验证 TensorFlow 的实际安装版本：

```shell #test id="install-tensorflow"
TF_PYTHON=/usr/local/python3.7.10/bin/python3.7
LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-} \
  TF_CPP_MIN_LOG_LEVEL=2 "$TF_PYTHON" - <<'PY'
import tensorflow as tf

print("tensorflow", tf.__version__)
PY
```

```shell #test-result id="install-tensorflow"
tensorflow 1.15.0
```

## 安装 TF Adapter

从与 CANN 9.1.0 对应的官方发布 `tfa_v0.0.49_9.1.0` 获取
`npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl`，并按照官方文档用
`-t "$TFPLUGIN_INSTALL_PATH"` 安装到独立目录。

```shell #test-setup
set -euo pipefail
TF_CACHE=/root/.cache/tensorflow
TF_PYTHON=/usr/local/python3.7.10/bin/python3.7
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

UV_PYTHON_DOWNLOADS=never \
  uv pip install --python "$TF_PYTHON" --no-deps --reinstall \
  --target "$TFPLUGIN_INSTALL_PATH" "$ADAPTER_WHEEL"
```

验证插件安装版本：

```shell #test id="install-tf-adapter"
TF_CACHE=/root/.cache/tensorflow
TF_PYTHON=/usr/local/python3.7.10/bin/python3.7
TFPLUGIN_INSTALL_PATH="$TF_CACHE/tfplugin-9.1.0"
PYTHONPATH="$TFPLUGIN_INSTALL_PATH${PYTHONPATH:+:$PYTHONPATH}" \
  LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-} \
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
export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-}
export JOB_ID=tensorflow-quick-start
export ASCEND_DEVICE_ID=0
```

## 运行 NpuOptimizer 样例

以下逻辑保持官方迁移步骤：导入 `npu_bridge`、向 Session 注册
`NpuOptimizer`，并关闭会与 NPU 图优化冲突的 remapping 和内存优化。为了
得到可复现的结果，输入由随机张量改成占位符和固定数据。

```shell #test id="run-npu-optimizer"
set -euo pipefail
TF_CACHE=/root/.cache/tensorflow
TF_PYTHON=/usr/local/python3.7.10/bin/python3.7
TFPLUGIN_INSTALL_PATH="$TF_CACHE/tfplugin-9.1.0"
PYTHONPATH="$TFPLUGIN_INSTALL_PATH${PYTHONPATH:+:$PYTHONPATH}" \
LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-} \
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
