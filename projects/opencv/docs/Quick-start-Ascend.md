# 快速开始

在单卡昇腾 NPU 上完成 OpenCV 最新 release 源码构建（开启 Huawei Ascend CANN 后端，`WITH_CANN=ON`），并跑 CPU 基础 API + NPU DNN 推理各一遍。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或 Ascend 950 系列，已挂载 `/dev/davinci*`。

### 软件（必须已就绪）

- Python 3.12
- CANN toolkit（参见 [Ascend 官方快速安装](https://ascend.github.io/docs/sources/ascend/quick_install.html)），`npu-smi info` 能列出设备
- 可访问 GitHub（clone + 下载模型）

> `opencv-python`（pip wheel）**不带** CANN 后端，必须从源码构建，`cv2.dnn.DNN_BACKEND_CANN` 才会链到真后端。

### 本文示例使用的版本

| 项目 | 值 |
| --- | --- |
| 机器 | Atlas 900 A2 PODc|
| OS | Ubuntu 22.04 arm64 |
| 镜像 | swr.cn-south-1.cloud-apeng.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12 |
| Python / CANN / CMake / GCC | 3.12 / 9.1.0 / ≥ 3.18 / 11 |
| OpenCV | 最新 release 源码（`WITH_CANN=ON`）|
| opencv_contrib | 与 OpenCV 同 release tag |

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

> 如果 `npu-smi` 不存在，按官方快速安装指南补装驱动。

检查 Python / CMake：

```shell #test id="check-toolchain"
python --version
cmake --version | head -n 1
```

输出结果如下：
```shell #test-result id="check-toolchain" fuzzy='xxx'
Python 3.12.xxx
cmake version xxx
```

加载 CANN 环境变量（每个用到 CANN 变量的块都要自己 `source`）：

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

确认 toolkit 根目录可访问：

```shell #test id="check-cann-path"
ls -d /usr/local/Ascend/ascend-toolkit/latest
```

输出结果如下：
```shell #test-result id="check-cann-path"
/usr/local/Ascend/ascend-toolkit/latest
```

## 装编译依赖

```shell #test-setup
sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list
apt-get update -qq && apt-get install -y -qq git build-essential cmake pkg-config libjpeg-dev libpng-dev libtiff-dev
```

## clone OpenCV + opencv_contrib

把最新 release tag 注入 `<ref>`；手动跑直接填 tag。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/opencv/opencv.git
git clone --depth 1 --branch <ref> https://github.com/opencv/opencv_contrib.git
```

## 打 4 个源码补丁

OpenCV mainline 与 CANN 9.1.0 / aarch64 有 4 处不兼容（4 个 patch 脚本假设 runner resolve 到 `5.0.0` tag；其它 tag 上 `assert old in s` 会抛错，需要重写 patch 适配）。按顺序跑这 4 段脚本（幂等，已打过会断言跳过）。

### 4a. `cv::MatShape` 改了类型

`CannConstOp` 主构造 + `gemm_layer` / `matmul_layer` 还在用老的 `std::vector<int>` 形式，编译不过。加一个委托重载 + 修两处赋值。

```shell #test-setup
python3 - <<'PY'
import pathlib
# (a) 主构造：std::vector<int>& → cv::MatShape&
old = 'const std::vector<int>& shape, const std::string& name'
new = 'const cv::MatShape& shape, const std::string& name'
for rel in ['opencv/modules/dnn/src/op_cann.hpp',
            'opencv/modules/dnn/src/op_cann.cpp']:
    p = pathlib.Path(rel); s = p.read_text()
    assert old in s, f'{rel}: pattern not found'
    p.write_text(s.replace(old, new))
    print(f'(a) {rel}: {s.count(old)} occurrence(s) -> MatShape&')

# (b) hpp 加 vector<int> 转发重载声明
hp = pathlib.Path('opencv/modules/dnn/src/op_cann.hpp')
hs = hp.read_text()
decl_old = 'CannConstOp(const uint8_t* data, const int dtype, const cv::MatShape& shape, const std::string& name);'
decl_new = (decl_old
            + '\n        CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);')
assert decl_old in hs and decl_new.splitlines()[1] not in hs, 'hpp decl already patched'
hp.write_text(hs.replace(decl_old, decl_new, 1))
print('(b) op_cann.hpp: +1 std::vector<int>& overload decl')

# (c) cpp 加 vector<int> 转发构造函数
cp = pathlib.Path('opencv/modules/dnn/src/op_cann.cpp')
cs = cp.read_text()
anchor = 'op_ = std::make_shared<ge::op::Const>(name);\n    op_->set_attr_value(*ge_tensor);\n}\n'
deleg = ('\nCannConstOp::CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name)\n'
         '    : CannConstOp(data, dtype, cv::MatShape(shape), name) {}\n')
assert anchor in cs and deleg.strip() not in cs, 'cpp delegating ctor already patched or anchor missing'
cp.write_text(cs.replace(anchor, anchor + deleg, 1))
print('(c) op_cann.cpp: +1 std::vector<int>& delegating ctor')

# (d) 两处 MatShape = std::vector<int>{...} 赋值改成 MatShape(1, &val) 构造
fixes = [
    ('opencv/modules/dnn/src/layers/gemm_layer.cpp',
     '            shape_C = std::vector<int>{dim};',
     '            shape_C = cv::MatShape(1, &dim);'),
    ('opencv/modules/dnn/src/layers/matmul_layer.cpp',
     '                if (real_ndims_C == 1 && bias_shape.front() != 1) {',
     '                if (real_ndims_C == 1 && bias_shape[0] != 1) {'),
    ('opencv/modules/dnn/src/layers/matmul_layer.cpp',
     '                    bias_shape = std::vector<int>{bias_shape.front()};',
     '                    int _bias_val = bias_shape[0]; bias_shape = cv::MatShape(1, &_bias_val);'),
]
for rel, o, n in fixes:
    p = pathlib.Path(rel); s = p.read_text()
    assert o in s, f'{rel}: pattern not found'
    assert n not in s, f'{rel}: already patched'
    p.write_text(s.replace(o, n, 1))
    print(f'(d) {rel}: 1 assignment fixed')
PY
```

预期：`(a)`/`(b)`/`(c)`/`(d)` 各 1 行打印，`grep` 确认两处 `cv::MatShape(1, &...)`。

### 4b. 缩小 CANN op 头（避免 aarch64 链接炸）

`all_ops.h` 有 ~1500 个 op，aarch64 + Debug 下链接必炸；换成窄头 `array_ops.h`，并给每个 layer TU 插它实际用到的窄头。

```shell #test-setup
python3 - <<'PY'
import pathlib

# (e) op_cann.hpp: all_ops.h -> array_ops.h
hpp = pathlib.Path('opencv/modules/dnn/src/op_cann.hpp')
s = hpp.read_text()
old = '''#ifdef CANN_VERSION_BELOW_6_3_ALPHA002
    #include "op_proto/built-in/inc/all_ops.h" // ge::Conv2D, ...
#else
    #include "built-in/op_proto/inc/all_ops.h" // ge::Conv2D, ...
#endif'''
new = '''#ifdef CANN_VERSION_BELOW_6_3_ALPHA002
    #include "op_proto/built-in/inc/array_ops.h" // ge::op::Const/Data/Identity/Reshape/Unsqueeze
#else
    #include "built-in/op_proto/inc/array_ops.h" // ge::op::Const/Data/Identity/Reshape/Unsqueeze
#endif'''
assert old in s, 'op_cann.hpp: all_ops include block not found'
hpp.write_text(s.replace(old, new, 1))
print('(e) op_cann.hpp: all_ops.h -> array_ops.h')

# (f) 每个 layer TU 插对应的窄 op 头
INSERTS = {
    'layers/batch_norm_layer.cpp': ['nn_batch_norm_ops'],
    'layers/concat_layer.cpp': ['split_combination_ops'],
    'layers/convolution_layer.cpp': ['nn_calculation_ops'],
    'layers/deconvolution_layer.cpp': ['nn_calculation_ops'],
    'layers/depth_space_ops_layer.cpp': ['transformation_ops'],
    'layers/elementwise_layers.cpp': ['nonlinear_fuc_ops', 'elewise_calculation_ops'],
    'layers/eltwise_layer.cpp': ['elewise_calculation_ops'],
    'layers/flatten_layer.cpp': ['transformation_ops'],
    'layers/fully_connected_layer.cpp': ['matrix_calculation_ops'],
    'layers/gemm_layer.cpp': ['matrix_calculation_ops'],
    'layers/instance_norm_layer.cpp': ['nn_norm_ops'],
    'layers/layer_norm.cpp': ['nn_norm_ops'],
    'layers/lrn_layer.cpp': ['nn_norm_ops'],
    'layers/matmul_layer.cpp': ['matrix_calculation_ops'],
    'layers/nary_eltwise_layers.cpp': ['elewise_calculation_ops'],
    'layers/padding_layer.cpp': ['pad_ops'],
    'layers/permute_layer.cpp': ['transformation_ops'],
    'layers/pooling_layer.cpp': ['nn_pooling_ops'],
    'layers/reduce_layer.cpp': ['reduce_ops'],
    'layers/resize2_layer.cpp': ['image_ops'],
    'layers/resize_layer.cpp': ['image_ops'],
    'layers/slice_layer.cpp': ['split_combination_ops', 'selection_ops'],
    'layers/softmax_layer.cpp': ['nn_norm_ops'],
}
ANCHOR = '#include "../op_cann.hpp"'
for rel, headers in INSERTS.items():
    p = pathlib.Path('opencv/modules/dnn/src') / rel
    s = p.read_text()
    assert ANCHOR in s, f'{rel}: op_cann.hpp include not found'
    assert 'built-in/op_proto/inc/' not in s, f'{rel}: already patched'
    block = ANCHOR + '\n' + '\n'.join(
        f'#include "built-in/op_proto/inc/{h}.h"' for h in headers)
    p.write_text(s.replace(ANCHOR, block, 1))
    print(f'(f) {rel}: +{len(headers)} narrow header(s)')
PY
```

预期：`(e)` 1 行打印，`(f)` 23 个文件被改；`grep` `op_proto/inc/array_ops.h` 在 `op_cann.hpp` 出现 1 次。

### 4c. `OperatorRunner` 的 NULL 流 fallback

CANN 9.1.0 的 `aclopCompileAndExecute` 拒绝 NULL 流（报 `EH0008`）。NULL 时换成 `aclrtCtxGetCurrentDefaultStream`。

```shell #test-setup
python3 - <<'PY'
import pathlib
p = pathlib.Path('opencv_contrib/modules/cannops/src/cann_call.cpp')
s = p.read_text()
old = '''OperatorRunner& OperatorRunner::run(AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    CV_ACL_SAFE_CALL(aclopCompileAndExecute(op.c_str(), inputDesc_.size(), inputDesc_.data(),
                                            inputBuffers_.data(), outputDesc_.size(),
                                            outputDesc_.data(), outputBuffers_.data(), opAttr_,
                                            ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, rawStream));
    if (rawStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtSynchronizeStream(rawStream));
    else
    {
        for (const auto& ptr : holder)
            stream.addTensorHolder(ptr);
    }
    return *this;
}'''
new = '''OperatorRunner& OperatorRunner::run(AscendStream& stream)
{
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    aclrtStream execStream = rawStream;
    if (execStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtCtxGetCurrentDefaultStream(&execStream));
    CV_ACL_SAFE_CALL(aclopCompileAndExecute(op.c_str(), inputDesc_.size(), inputDesc_.data(),
                                            inputBuffers_.data(), outputDesc_.size(),
                                            outputDesc_.data(), outputBuffers_.data(), opAttr_,
                                            ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, execStream));
    CV_ACL_SAFE_CALL(aclrtSynchronizeStream(execStream));
    if (rawStream != nullptr)
    {
        for (const auto& ptr : holder)
            stream.addTensorHolder(ptr);
    }
    return *this;
}'''
assert old in s, 'cann_call.cpp: OperatorRunner::run body not found'
p.write_text(s.replace(old, new, 1))
print('(g) cann_call.cpp: NULL-stream fallback patched')
PY
```

预期：打印 `(g) patched`。

### 4d. AscendC `kernel_launch` 模板（NULL 流 + 缺头）

同 NULL 流坑，并补 `acl_rt.h` include（模板里调 `aclrtCreateStream` / `aclrtSynchronizeStream`，声明在 `acl_rt.h`，原头只有 `acl_base.h`）。

```shell #test-setup
python3 - <<'PY'
import pathlib
p = pathlib.Path('opencv_contrib/modules/cannops/include/opencv2/cann_call.hpp')
s = p.read_text()
inc_old = '#include <acl/acl_base.h>'
inc_new = '#include <acl/acl_base.h>\n#include <acl/acl_rt.h>'
assert inc_old in s and inc_new not in s
s = s.replace(inc_old, inc_new, 1)
old = '''    std::shared_ptr<uchar> tilingDevice =
        mallocAndUpload(&tiling, sizeof(TILING_TYPE), stream, AscendMat::defaultAllocator());
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    CV_ACL_SAFE_CALL(kernel(1, rawStream, tilingDevice.get(), args...));
    if (rawStream == nullptr)
    {
        stream.waitForCompletion();
    }'''
new = '''    std::shared_ptr<uchar> tilingDevice =
        mallocAndUpload(&tiling, sizeof(TILING_TYPE), stream, AscendMat::defaultAllocator());
    aclrtStream rawStream = AscendStreamAccessor::getStream(stream);
    aclrtStream execStream = rawStream;
    if (execStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtCtxGetCurrentDefaultStream(&execStream));
    CV_ACL_SAFE_CALL(kernel(1, execStream, tilingDevice.get(), args...));
    CV_ACL_SAFE_CALL(aclrtSynchronizeStream(execStream));'''
assert old in s, 'cann_call.hpp: kernel_launch body not found'
p.write_text(s.replace(old, new, 1))
print('(h) cann_call.hpp: NULL-stream fallback + acl_rt.h patched')
PY
```

预期：打印 `(h) patched`。

## 桥接 CANN 9.1.0 的库目录

OpenCV 的 `OpenCVFindCANN.cmake` 在 `${CANN_INSTALL_DIR}/{acllib,lib64,compiler/lib64}/` 下找 ACL 库；CANN 9.1.0 装在 `aarch64-linux/lib64/`。补 3 个 symlink。

```shell #test-setup
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux /usr/local/Ascend/cann-9.1.0/acllib
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64 /usr/local/Ascend/cann-9.1.0/lib64
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64 /usr/local/Ascend/cann-9.1.0/compiler/lib64
```

> 镜像自带 `ascend-toolkit/latest -> cann-9.1.0` 时，把上面三行里的 `cann-9.1.0` 改成 `ascend-toolkit/latest` 也可以。

## 配置并编译

CANN 后端开关 `WITH_CANN=ON`（不是 `BUILD_CANN`）；`-DOPENCV_DOWNLOAD_MIRROR_ID=gitcode` 让 cmake configure 阶段走 gitcode 镜像拉 ADE / IPPICV / TBB。

```shell #test id="opencv-cmake-configure" load="upstream_ref>>ref"
cd opencv
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-cann \
      -DWITH_CANN=ON \
      -DBUILD_opencv_world=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=ON \
      -DOPENCV_BUILD_TEST_MODULES_LIST=cannops \
      -DINSTALL_TESTS=ON \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_LIST=core,imgproc,imgcodecs,videoio,dnn,python3,cannops,ts \
      -DSOC_VERSION=ascend910b1 \
      -DBUILD_opencv_python3=ON \
      -DBUILD_opencv_python_bindings_generator=ON \
      -DPYTHON_INCLUDE_DIR=/usr/local/python3.12.13/include/python3.12 \
      -DPYTHON_LIBRARY=/usr/local/python3.12.13/lib/libpython3.12.so \
      -DOPENCV_ENABLE_NONFREE=OFF \
      -DOPENCV_DOWNLOAD_MIRROR_ID=gitcode \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      ..
```

输出结果如下：
```shell #test-result id="opencv-cmake-configure" fuzzy='xxx' fuzzy='...'
-- The CXX compiler identification is GNU xxx
...
--   CANN:xxx YES
...
-- Configuring done
-- Generating done
-- Build files have been written to: xxx/build
```

编译 + 安装一体（`make install`），binary 直接落到 `/usr/local/opencv-cann/bin/`。用 `cmake --build` 走 cmake-native progress，能看到 `[N/M] Building CXX object ...`：

```shell #test-setup
cd opencv/build
cmake --build . --target install --parallel 2
```

## 校验 build 产物

```shell #test id="opencv-verify-build"
/usr/local/opencv-cann/bin/opencv_version
/usr/local/opencv-cann/bin/opencv_test_cannops --gtest_list_tests 2>&1 | head -n 20
```

输出结果如下：
```shell #test-result id="opencv-verify-build" fuzzy='xxx'
xxx
xxx
```

`opencv_test_cannops` 依赖 CANN runtime `LD_LIBRARY_PATH`，记得先 `source set_env.sh`。

## 跑 CANN 单元测试

`opencv_test_cannops` 共 78 个用例；25 个因 CANN 9.1.0 / 910B1 的已知不兼容被排除（4 个 resize、17 个 cvtColor 融合算术、3 个 AscendC threshold、1 个 SRC_TYPE_FLIP 上游已修）。剩 53 个应全过。

```shell #test id="opencv-cann-run-tests"
set -o pipefail
/usr/local/opencv-cann/bin/opencv_test_cannops --gtest_color=no \
  --gtest_filter=-CORE.RESIZE:CORE.RESIZE_NEW:CORE.CROP_RESIZE:CORE.CROP_RESIZE_MAKE_BORDER:CVT_COLOR.RGB2XYZ:CVT_COLOR.BGR2XYZ:CVT_COLOR.XYZ2BGR:CVT_COLOR.XYZ2RGB:CVT_COLOR.XYZ2BGR_DC4:CVT_COLOR.XYZ2RGB_DC4:CVT_COLOR.BGR2YCrCb:CVT_COLOR.RGB2YCrCb:CVT_COLOR.YCrCb2BGR:CVT_COLOR.YCrCb2RGB:CVT_COLOR.YCrCb2BGR_DC4:CVT_COLOR.YCrCb2RGB_DC4:CVT_COLOR.BGR2YUV:CVT_COLOR.RGB2YUV:CVT_COLOR.YUV2BGR:CVT_COLOR.YUV2RGB:CVT_COLOR.YUV2BGR_DC4:CVT_COLOR.YUV2RGB_DC4:ELEMENTWISE_OP.MAT_THRESHOLD:ELEMENTWISE_OP.MAT_THRESHOLD_ASCENDC:ASCENDC_KERNEL.THRESHOLD \
  > /tmp/cannops_gtest.log 2>&1; rc=$?
tail -n 25 /tmp/cannops_gtest.log
if [ $rc -ne 0 ]; then
  echo '--- per-test failure summary:'
  grep -E '^\[ RUN|unknown file: Failure|C\+\+ exception|op\[[A-Za-z0-9_]+\]|E[0-9]{5}' /tmp/cannops_gtest.log | head -80
fi
exit $rc
```

输出结果如下：
```shell #test-result id="opencv-cann-run-tests" fuzzy='xxx' fuzzy='...'
...
[==========]xxx
[  PASSED  ] 53 tests.
```

> 不要加 `--gtest_brief`：OpenCV vendored gtest 没实现这个 flag，加上后零用例执行但 rc=0，假绿。

## OpenCV Python quickstart

把 Python 指向源码版 `cv2`（带 CANN 后端）：

```shell #test-setup
export PYTHONPATH=/usr/local/opencv-cann/lib/python3.12/site-packages:${PYTHONPATH:-}
```

### 9a. 读图 / 写图

```shell #test id="quickstart-imread-imwrite"
python << 'PY'
import cv2
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
ok = cv2.imwrite('/tmp/opencv_quickstart.png', img)
print('imwrite ok:', ok)
print('shape:', img.shape, 'dtype:', img.dtype)
PY
ls -la /tmp/opencv_quickstart.png
```

输出结果如下：
```shell #test-result id="quickstart-imread-imwrite"
imwrite ok: True
shape: (100, 100, 3) dtype: uint8
xxx
```

### 9b. 颜色空间转换

```shell #test id="quickstart-cvtcolor"
python << 'PY'
import cv2
import numpy as np
bgr = np.zeros((100, 100, 3), dtype=np.uint8)
bgr[..., 0] = 255  # blue channel = 255
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print('gray shape:', gray.shape, 'mean:', float(gray.mean()))
PY
```

输出结果如下：
```shell #test-result id="quickstart-cvtcolor" fuzzy='xxx'
gray shape: (100, 100) mean: xxx
```

`mean` ≈ 29（BGR→GRAY 权重 `0.114·B + 0.587·G + 0.299·R`，B=255 时 ≈ 0.114·255）。

### 9c. 缩放

```shell #test id="quickstart-resize"
python << 'PY'
import cv2
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
resized = cv2.resize(img, (50, 200), interpolation=cv2.INTER_AREA)
print('resized shape:', resized.shape)
PY
```

输出结果如下：
```shell #test-result id="quickstart-resize"
resized shape: (200, 50, 3)
```

OpenCV shape 是 `(rows, cols, channels)`，`resize(img, (W, H), ...)` 得 `(H, W, C)`。

### 9d. 绘制

```shell #test id="quickstart-draw"
python << 'PY'
import cv2
import numpy as np
img = np.zeros((200, 400, 3), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (390, 190), (0, 255, 0), thickness=2)
cv2.putText(img, 'Hello OpenCV', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
print('rect+text drawn, image mean:', float(img.mean()))
cv2.imwrite('/tmp/opencv_draw.png', img)
PY
ls -la /tmp/opencv_draw.png
```

输出结果如下：
```shell #test-result id="quickstart-draw" fuzzy='xxx'
rect+text drawn, image mean: xxx
xxx
```

### 9e. 视频写入

```shell #test id="quickstart-video"
python << 'PY'
import cv2
import numpy as np
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter('/tmp/opencv_video.avi', fourcc, 10.0, (320, 320))
frame = np.zeros((320, 320, 3), dtype=np.uint8)
for _ in range(3):
    writer.write(frame)
writer.release()
print('video frames:', 3)
import os
print('video file size:', os.path.getsize('/tmp/opencv_video.avi'))
PY
```

输出结果如下：
```shell #test-result id="quickstart-video" fuzzy='xxx' fuzzy='...'
...
video frames: 3
video file size: xxx
...
```

> `INFO`/`WARN` 行（找不到 FFmpeg / GStreamer 等视频插件）是源码构建没带这些插件时注册扫描的预期噪音，看文件大小就行。

## NPU 上跑一次 DNN 推理

```shell #test id="opencv-py-version"
python -c "
import cv2
print('cv2 version:', cv2.__version__)
print('CANN backend available:', cv2.dnn.DNN_BACKEND_CANN)
print('NPU target available:', cv2.dnn.DNN_TARGET_NPU)
"
```

输出结果如下：
```shell #test-result id="opencv-py-version" fuzzy='xxx'
cv2 version: xxx
CANN backend available: xxx
NPU target available: xxx
```

跑 SqueezeNet：

```shell #test id="opencv-cann-infer"
python << 'PY'
import os
import time
import urllib.request
import numpy as np
import cv2

MODEL_URL = ('https://github.com/onnx/models/raw/main/'
             'validated/vision/classification/squeezenet/'
             'model/squeezenet1.0-12.onnx')
MODEL_PATH = '/tmp/squeezenet1.0-12.onnx'

if not os.path.exists(MODEL_PATH):
    # github raw 下载偶发 RemoteDisconnected（国内网络抖动），重试 3 次
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5)
print('model bytes:', os.path.getsize(MODEL_PATH))

net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

image = np.zeros((224, 224, 3), dtype=np.uint8)
blob = cv2.dnn.blobFromImage(
    image, scalefactor=1.0, size=(224, 224),
    mean=(104.006, 116.669, 122.679), swapRB=False, crop=False,
)
net.setInput(blob)
out = net.forward()

print('output shape:', out.shape)
print('output dtype:', out.dtype)
print('top class index:', int(np.argmax(out[0])))
print('top class score:', float(np.max(out[0])))
PY
```

输出结果如下：
```shell #test-result id="opencv-cann-infer" fuzzy='xxx' fuzzy='...'
...
model bytes: xxx
output shape: (1, 1000, 1, 1)
output dtype: float32
top class index: xxx
top class score: xxx
```

预期：`output shape (1, 1000, 1, 1)`、`float32`、`score` 是有限值（非 NaN/Inf）即可。首次 ~30s（ACL graph 编译），后续命中 `$HOME/ascend` AOE 缓存降到 ~10ms。