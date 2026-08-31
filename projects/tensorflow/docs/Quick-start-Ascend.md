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
| GCC | linux_gcc7.3.0 |
| NPU | Ascend 910B × 1 |

## 安装前准备

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
  build-essential ca-certificates curl git openjdk-8-jdk-headless unzip zip \
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
ln -sfn "$PYTHON_PREFIX/bin/python3.7" "$PYTHON_PREFIX/bin/python3"
ln -sfn "$PYTHON_PREFIX/bin/pip3.7" "$PYTHON_PREFIX/bin/pip3"
export PATH="$PYTHON_PREFIX/bin:$PATH"
```

### 编译安装 HDF5 1.10.5

对于 Linux aarch64，需要先编译安装 HDF5 1.10.5。

#### 下载 HDF5 源码包

```shell #test-setup
set -euo pipefail
HDF5_ARCHIVE=/tmp/hdf5-1.10.5.tar.gz
if [ ! -s "$HDF5_ARCHIVE" ]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 600 \
    https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz \
    -o "$HDF5_ARCHIVE"
  echo "6d4ce8bf902a97b050f6f491f4268634e252a63dadd6656a1a9be5b7b7726fa8  $HDF5_ARCHIVE" \
    | sha256sum -c -
fi
```

#### 解压 HDF5 源码包

```shell #test-setup
set -euo pipefail
tar -zxvf /tmp/hdf5-1.10.5.tar.gz -C /tmp
```

#### 配置、编译和安装 HDF5

```shell #test-setup
set -euo pipefail
HDF5_PREFIX=/usr/local/hdf5
if [ ! -x "$HDF5_PREFIX/bin/h5cc" ]; then
  cd /tmp/hdf5-1.10.5
  ./configure --prefix=/usr/local/hdf5
  make -j16 && make install
fi
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
export PATH=/usr/local/python3.7.10/bin:$PATH
export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-}
pip3 install "Cython<3"
pip3 install wheel
pip3 install numpy
```

#### 安装 h5py 2.8.0

```shell #test-setup
set -euo pipefail
export PATH=/usr/local/python3.7.10/bin:$PATH
export CPATH=/usr/local/hdf5/include/:/usr/local/hdf5/lib/
export LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-}
pip3 install h5py==2.8.0
```

检查 Python、HDF5 和 h5py 版本：

```shell #test id="check-python"
export PATH=/usr/local/python3.7.10/bin:$PATH
python3 - <<'PY'
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

## 安装 TensorFlow

Linux aarch64 的 pip 源未提供 TensorFlow 1.15 wheel，需要从源码编译。

编译前确认使用 `linux_gcc7.3.0`：

```shell #test-setup
set -euo pipefail
GCC_VERSION=$(gcc -dumpfullversion -dumpversion)
if [[ "$GCC_VERSION" != 7.3* ]]; then
  echo "TensorFlow 1.15 requires linux_gcc7.3.0, found gcc $GCC_VERSION" >&2
  exit 1
fi
```

### 安装 Bazel 0.26.1

```shell #test-setup
set -euo pipefail
BAZEL_VERSION=0.26.1
BAZEL_ROOT=/tmp/bazel-${BAZEL_VERSION}
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-arm64
export PATH="$JAVA_HOME/bin:$PATH"
if ! bazel version 2>/dev/null | grep -q "Build label: ${BAZEL_VERSION}"; then
  mkdir -p "$BAZEL_ROOT"
  curl -fL --retry 3 --connect-timeout 30 --max-time 600 \
    https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-dist.zip \
    -o "$BAZEL_ROOT/bazel-0.26.1-dist.zip"
  cd "$BAZEL_ROOT"
  unzip -q bazel-0.26.1-dist.zip
  env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh
  install -m 0755 output/bazel /usr/local/bin/bazel
fi
bazel version
```

### 获取 TensorFlow v1.15.0 源码

```shell #test-setup
set -euo pipefail
TF_SOURCE=/tmp/tensorflow-v1.15.0
if [ ! -d "$TF_SOURCE/.git" ]; then
  git clone --depth 1 --branch v1.15.0 \
    https://github.com/tensorflow/tensorflow.git "$TF_SOURCE"
fi
```

### 1. 下载 nsync 1.22.0

```shell #test-setup
set -euo pipefail
curl -fL --retry 3 --connect-timeout 30 --max-time 600 \
  https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/nsync/archive/1.22.0.tar.gz \
  -o /tmp/nsync-1.22.0.original.tar.gz
tar -xzf /tmp/nsync-1.22.0.original.tar.gz -C /tmp
```

### 2. 修改 nsync 1.22.0

按照安装文档编辑 `nsync-1.22.0/platform/c++11/atomic.h`：

```shell #test-setup
set -euo pipefail
python - <<'PY'
from pathlib import Path

path = Path('/tmp/nsync-1.22.0/platform/c++11/atomic.h')
text = path.read_text(encoding='utf-8')
anchor = 'NSYNC_CPP_START_\n'
if '#define ATM_CB_() __sync_synchronize()' not in text:
    text = text.replace(
        anchor,
        anchor + '\n#define ATM_CB_() __sync_synchronize()\n',
        1,
    )

for name in (
    'atm_cas_nomb_u32_',
    'atm_cas_acq_u32_',
    'atm_cas_rel_u32_',
    'atm_cas_relacq_u32_',
):
    start = text.index(f'static INLINE int {name}')
    body_start = text.index('{', start) + 1
    body_end = text.index('\n}', body_start)
    statement = text[body_start:body_end].strip()
    if 'ATM_CB_();' in statement:
        continue
    if not statement.startswith('return (') or not statement.endswith(');'):
        raise RuntimeError(f'unexpected {name} body: {statement}')
    expression = statement[len('return '):]
    replacement = (
        f'\n    int result = {expression}\n'
        '    ATM_CB_();\n'
        '    return result;'
    )
    text = text[:body_start] + replacement + text[body_end:]

path.write_text(text, encoding='utf-8')
PY
```

### 3. 重新压缩 nsync 1.22.0

```shell #test-setup
set -euo pipefail
tar -czf /tmp/nsync-1.22.0.tar.gz -C /tmp nsync-1.22.0
```

### 4. 生成 sha256sum 校验码

```shell #test-setup
set -euo pipefail
sha256sum /tmp/nsync-1.22.0.tar.gz
```

### 5. 修改 sha256sum 和 urls

将新校验码和本地 `file://` 地址写入 `tensorflow/workspace.bzl`：

```shell #test-setup
set -euo pipefail
python - <<'PY'
import hashlib
import re
from pathlib import Path

archive = Path('/tmp/nsync-1.22.0.tar.gz')
workspace = Path('/tmp/tensorflow-v1.15.0/tensorflow/workspace.bzl')
sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
text = workspace.read_text(encoding='utf-8')
name_pos = text.index('name = "nsync"')
start = text.rfind('    tf_http_archive(', 0, name_pos)
end = text.index('\n    )', name_pos) + len('\n    )')
block = text[start:end]
block, count = re.subn(
    r'sha256 = "[0-9a-f]+"',
    f'sha256 = "{sha256}"',
    block,
    count=1,
)
if count != 1:
    raise RuntimeError('failed to update nsync sha256')
local_url = '            "file:///tmp/nsync-1.22.0.tar.gz",\n'
if local_url not in block:
    block = block.replace('        urls = [\n', '        urls = [\n' + local_url, 1)
workspace.write_text(text[:start] + block + text[end:], encoding='utf-8')
PY
```

### 6. 编译 TensorFlow

#### 配置编译选项

```shell #test-setup
set -euo pipefail
cd /tmp/tensorflow-v1.15.0
export PATH=/usr/local/python3.7.10/bin:$PATH
export PYTHON_BIN_PATH=/usr/local/python3.7.10/bin/python3.7
PYTHON_LIB_PATH=$(python3 -c 'import site; print(site.getsitepackages()[0])')
export PYTHON_LIB_PATH
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_COMPUTECPP=0
export TF_NEED_OPENCL=0
export TF_ENABLE_XLA=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_NEED_MPI=0
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0
export CC_OPT_FLAGS='-march=native'
./configure
```

#### 编译并生成 TensorFlow wheel

TensorFlow 与 TF Adapter 的 C++ ABI 都配置为 `0`：

```shell #test-setup
set -euo pipefail
cd /tmp/tensorflow-v1.15.0
bazel build --config=opt \
  --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 \
  //tensorflow/tools/pip_package:build_pip_package
mkdir -p /tmp/tensorflow-package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
  /tmp/tensorflow-package
```

### 7. 安装编译好的 TensorFlow

```shell #test-setup
set -euo pipefail
export PATH=/usr/local/python3.7.10/bin:$PATH
cd /tmp/tensorflow-package
pip3 install tensorflow-1.15.0-*.whl
```

### 8. 验证 TensorFlow

执行以下命令验证安装效果：

```shell #test id="install-tensorflow"
export PATH=/usr/local/python3.7.10/bin:$PATH
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

```shell #test-result id="install-tensorflow" fuzzy="xxx"
Tensor("Sum:0", shape=(), dtype=float32)
```

## 安装框架插件包 TF Adapter

TF Adapter 用于在 NPU 上执行 TensorFlow 网络的训练或在线推理。

### 安装插件包

#### 1. 获取 TF Adapter 安装包

从 TF Adapter GitCode 仓选择与 CANN 9.1.0 配套的发布
`tfa_v0.0.49_9.1.0`，获取
`npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl`。

```shell #test-setup
set -euo pipefail
PACKAGE_DIR=/home/package
ADAPTER_WHEEL="$PACKAGE_DIR/npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl"
mkdir -p "$PACKAGE_DIR"

if [ ! -s "$ADAPTER_WHEEL" ]; then
  curl -fL --retry 5 --retry-all-errors --connect-timeout 30 --max-time 600 \
    https://gitcode.com/cann/tensorflow/releases/download/tfa_v0.0.49_9.1.0/npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl \
    -o "$ADAPTER_WHEEL.tmp"
  mv "$ADAPTER_WHEEL.tmp" "$ADAPTER_WHEEL"
fi
echo "faac580f9a732e86b6ad2a49150eb450757ee18c173ec099044d855d364d4d98  $ADAPTER_WHEEL" \
  | sha256sum -c - >/dev/null
```

#### 2. 安装 TF Adapter

```shell #test-setup
set -euo pipefail
export PATH=/usr/local/python3.7.10/bin:$PATH
TFPLUGIN_INSTALL_PATH=$HOME/Ascend/tfplugin
ADAPTER_WHEEL=/home/package/npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl
mkdir -p "$TFPLUGIN_INSTALL_PATH"
pip3 install "$ADAPTER_WHEEL" --force-reinstall -t "$TFPLUGIN_INSTALL_PATH"
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
export PATH=/usr/local/python3.7.10/bin:$PATH
TFPLUGIN_INSTALL_PATH=$HOME/Ascend/tfplugin
PYTHONPATH="$TFPLUGIN_INSTALL_PATH${PYTHONPATH:+:$PYTHONPATH}" \
  LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-} \
  TF_CPP_MIN_LOG_LEVEL=2 python3 - <<'PY'
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
export TFPLUGIN_INSTALL_PATH=$HOME/Ascend/tfplugin
export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH
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
export PATH=/usr/local/python3.7.10/bin:$PATH
TFPLUGIN_INSTALL_PATH=$HOME/Ascend/tfplugin
PYTHONPATH="$TFPLUGIN_INSTALL_PATH${PYTHONPATH:+:$PYTHONPATH}" \
LD_LIBRARY_PATH=/usr/local/hdf5/lib/:${LD_LIBRARY_PATH:-} \
JOB_ID=tensorflow-quick-start \
ASCEND_DEVICE_ID=0 \
TF_CPP_MIN_LOG_LEVEL=2 \
python3 - <<'PY'
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
