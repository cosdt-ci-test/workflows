# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上完成 OpenCV 5.0.0 源码构建（开启 Huawei Ascend CANN 后端，`WITH_CANN=ON`），并用 CPU 侧基础 API + NPU 侧 DNN 推理各跑一遍。

本文档带你完成 5 件事：

1. 验证 CANN/NPU 工具链可用
2. clone OpenCV + opencv_contrib，针对 CANN 9.1.0 / aarch64 打 4 个源码补丁，并用 symlink 把 CANN 的库接到 OpenCV CMake 期望的位置
3. cmake 配置（`WITH_CANN=ON`，Debug build）并编译
4. 跑单元测试（53/53 通过）和 OpenCV Python quickstart
5. 通过 CANN 后端在 NPU 上跑一次 DNN 推理

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或 Ascend 950 系列。

### 软件（必须已就绪）

- 可用的 Python 3.12 环境
- CANN toolkit（参见 [Ascend 官方快速安装](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与 CANN 匹配的驱动/固件（`npu-smi info` 能列出 NPU 设备）
- 可访问 GitHub（SSH 或 HTTPS）—— clone OpenCV + opencv_contrib 和下载 SqueezeNet ONNX 都要用

> `opencv-python`（pip wheel）**不带** CANN 后端，必须从源码构建，`DNN_BACKEND_CANN` 这个 enum 才会真正链到能跑的后端。

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 PODc（Ascend 910B4，64 GB × 1）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| CMake | ≥ 3.18 |
| GCC | 11 |
| OpenCV | 5.0.0 源码（`WITH_CANN=ON`）|
| opencv_contrib | 与 OpenCV 同 release tag |

CANN 9.1.0 把 ACL / AOE / GE 库装在 `${CANN_INSTALL_DIR}/aarch64-linux/lib64/`，但 OpenCV 5.0.0 的 `OpenCVFindCANN.cmake` 还在 `${CANN_INSTALL_DIR}/{acllib,lib64,compiler/lib64}/` 下找。下面的「桥接 CANN 9.1.0 的库目录布局」一节用 3 个 symlink 补上这个差异。

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

> 如果 `npu-smi` 不存在，按 Ascend 官方快速安装指南补装驱动。

检查 Python / CMake 版本：

```shell #test id="check-toolchain"
python --version
cmake --version | head -n 1
```

```shell #test-result id="check-toolchain" fuzzy='xxx'
Python 3.12.xxx
cmake version xxx
```

加载 CANN 环境变量：

每个 `#test` 块是独立的 `subprocess.run`，所以每个用到 CANN 变量的块都要自己 `source`（cmake / make / python 各 source 一次——`prepare_environment` 不会重复 source）。

```shell #test id="load-cann"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

输出结果如下：
```shell #test-result id="load-cann"
...
```

确认 toolkit 根目录可访问：

```shell #test id="check-cann-path"
ls -d /usr/local/Ascend/ascend-toolkit/latest
```

输出结果如下：
```shell #test-result id="check-cann-path" fuzzy='xxx'
/usr/local/Ascend/ascend-toolkit/xxx
```

## 装编译依赖

镜像基于原生 `ubuntu:22.04`（arm64），`/etc/apt/sources.list` 指向 Canonical 的海外 ports 镜像（国内极慢）。先换成阿里云 arm64 ports，再装图像编解码 dev 包（镜像里 `git` / `build-essential` / `cmake` 镜像构建时已装）。

```shell #test-setup
sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list
apt-get update -qq && apt-get install -y -qq git build-essential cmake pkg-config libjpeg-dev libpng-dev libtiff-dev
```

## clone OpenCV + opencv_contrib

工作流 runner 通过下面隐藏的 `#test-setup` 把最新 release tag 注入 `<ref>`；手动跑的话直接填 tag 即可（今天就是 `5.0.0`）。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/opencv/opencv.git
git clone --depth 1 --branch <ref> https://github.com/opencv/opencv_contrib.git
```

> `<ref>` 是 OpenCV 最新 release tag。两个仓库必须用同一个 tag。

## 打 4 个源码补丁

OpenCV 5.0.0 mainline 与 CANN 9.1.0 / aarch64 有 4 个已知不兼容。每个 patch 脚本都是幂等的——断言旧字符串存在 → 替换 → 每改一处打印一行。按顺序执行即可；任何脚本抛 `AssertionError` 都说明源码已经偏离文档预期，需要先排查。

### 4a. `cv::MatShape` 现在是 struct

OpenCV 5.0.0 把 `cv::MatShape` 从 `std::vector<int>` 别名改成了 struct。`CannConstOp` 构造函数和两个 layer 赋值（`gemm_layer`、`matmul_layer`）还在用老的 `std::vector<int>` 形式，编译不过。加一个委托重载 + 修两处赋值。

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

# (c) cpp 加 vector<int> 转发构造函数（委托给 MatShape 主构造）
cp = pathlib.Path('opencv/modules/dnn/src/op_cann.cpp')
cs = cp.read_text()
anchor = 'op_ = std::make_shared<ge::op::Const>(name);\n    op_->set_attr_value(*ge_tensor);\n}\n'
deleg = ('\nCannConstOp::CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name)\n'
         '    : CannConstOp(data, dtype, cv::MatShape(shape), name) {}\n')
assert anchor in cs and deleg.strip() not in cs, 'cpp delegating ctor already patched or anchor missing'
cp.write_text(cs.replace(anchor, anchor + deleg, 1))
print('(c) op_cann.cpp: +1 std::vector<int>& delegating ctor')

# (d) 两处 MatShape = std::vector<int>{...} 赋值改成 MatShape(1, &val) raw-array 构造
fixes = [
    ('opencv/modules/dnn/src/layers/gemm_layer.cpp',
     '            shape_C = std::vector<int>{dim};',
     '            shape_C = cv::MatShape(1, &dim);'),
    # matmul_layer.cpp:515 也用了 bias_shape.front()（在 if 条件里 != 1），先
    # 把 .front() 全部换成 [0]，再修赋值那一行。
    ('opencv/modules/dnn/src/layers/matmul_layer.cpp',
     '                if (real_ndims_C == 1 && bias_shape.front() != 1) {',
     '                if (real_ndims_C == 1 && bias_shape[0] != 1) {'),
    ('opencv/modules/dnn/src/layers/matmul_layer.cpp',
     '                    bias_shape = std::vector<int>{bias_shape.front()};',
     '                    int _bias_val = bias_shape[0]; bias_shape = cv::MatShape(1, &_bias_val);'),
]
for rel, o, n in fixes:
    p = pathlib.Path(rel); s = p.read_text()
    assert o in s, f'{rel}: pattern not found: {o!r}'
    assert n not in s, f'{rel}: already patched'
    p.write_text(s.replace(o, n, 1))
    print(f'(d) {rel}: 1 assignment fixed')
PY
echo '---grep verify:'
grep -n 'CannConstOp(const uint8_t\* data, const int dtype,' \
  opencv/modules/dnn/src/op_cann.hpp \
  opencv/modules/dnn/src/op_cann.cpp
grep -n 'cv::MatShape(1, &\|cv::MatShape(1, &_bias_front)' \
  opencv/modules/dnn/src/layers/gemm_layer.cpp \
  opencv/modules/dnn/src/layers/matmul_layer.cpp
```

预期：`(a)` 两条 `1 occurrence(s) -> MatShape&`，`(b)` +1 声明，`(c)` +1 转发构造体，`(d)` 两条 `1 assignment fixed`。最后 `grep` 在 hpp 看到 2 行构造声明、cpp 看到 2 个构造函数定义，在 `gemm_layer.cpp:534` 和 `matmul_layer.cpp:516` 各 1 行 raw-array 构造。

### 4b. aarch64 上把 `all_ops.h`（~1500 个 op）换成 `array_ops.h`（68 个 op）

`op_cann.hpp` include 了 CANN 全量 op 头 `all_ops.h`（~1500 个 `REG_OP` 宏）。`-O0` 下每个实例展开成 ~134 KB 的 file-local 注册 lambda，链接器无法跨 TU 去重；200 个 dnn TU × 1500 个 op 把 `.text` 撑爆，超过 aarch64 `BL` 指令 ±128 MB 的跳转范围，dnn 链接必炸。换成窄头 `array_ops.h`，并给每个 layer TU 插它实际实例化用到的 op 类对应的窄头。

```shell #test-setup
python3 - <<'PY'
import pathlib

# (e) op_cann.hpp: all_ops.h (~1500 ops) -> array_ops.h (68 ops)
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

# (f) 每个 layer TU 按需插入窄 op 头（只含该文件实例化的 op 类）
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
echo '---grep verify:'
grep -rn 'op_proto/inc/array_ops.h' opencv/modules/dnn/src/op_cann.hpp
grep -rc 'built-in/op_proto/inc/' opencv/modules/dnn/src/layers/*.cpp | grep -v ':0' | wc -l
```

预期：`(e)` 1 行替换；`(f)` 23 个文件被改（其中 2 个文件各 +2 个头）。最后 `grep` 在 `op_cann.hpp` 看到 1 行 `array_ops.h`，计数为 `23`。

### 4c. `cannops::OperatorRunner` 的 NULL 流 fallback

CANN 9.1.0 的 `aclopCompileAndExecute` 拒绝老 CANN 的 NULL "默认流" 语义，会报 `EH0008`（找不到 allocator）。当入参是 NULL 时，把流换成 `aclrtCtxGetCurrentDefaultStream`。

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
    // CANN 9.1.0: aclopCompileAndExecute's GE path calls aclrtAllocatorGetByStream,
    // which rejects NULL (legacy default-stream) pointers with EH0008. Swap in the
    // context's registered default stream - it carries an allocator registration.
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
print('(g) cann_call.cpp: OperatorRunner::run NULL-stream fallback patched')
PY
grep -n 'CtxGetCurrentDefaultStream' opencv_contrib/modules/cannops/src/cann_call.cpp | head -2
```

预期：打印 `(g) ... patched`，`grep` 在 `cann_call.cpp` 看到 1 行 `CtxGetCurrentDefaultStream`。

### 4d. AscendC `kernel_launch` 模板的 NULL 流 + 缺头补丁

AscendC kernel 启动路径同 NULL 流坑，在 910B1 上触发 AI Core 越界。给 `kernel_launch` 打 fallback，同时补上 `acl_rt.h`（模板里直接调 `aclrtCreateStream` / `aclrtSynchronizeStream` / `aclrtDestroyStream`，这些声明在 `acl_rt.h`，原头只 include 了 `acl_base.h`）。

```shell #test-setup
python3 - <<'PY'
import pathlib
p = pathlib.Path('opencv_contrib/modules/cannops/include/opencv2/cann_call.hpp')
s = p.read_text()
# kernel_launch 调的 aclrtCreateStream / aclrtSynchronizeStream / aclrtDestroyStream
# 声明在 acl_rt.h；该头原来只 include 了 acl_base.h，template 里直接调会报
# "no arguments ... depend on a template parameter"（两阶段名字查找）。
inc_old = '#include <acl/acl_base.h>'
inc_new = '#include <acl/acl_base.h>\n#include <acl/acl_rt.h>'
assert inc_old in s and inc_new not in s, 'acl include line not found / already patched'
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
    // CANN 9.1.0: AscendC kernel launch + aclrtSynchronizeStream reject NULL
    // (legacy default) streams via the allocator path. Use the context's
    // registered default stream instead of a bare aclrtCreateStream.
    aclrtStream execStream = rawStream;
    if (execStream == nullptr)
        CV_ACL_SAFE_CALL(aclrtCtxGetCurrentDefaultStream(&execStream));
    CV_ACL_SAFE_CALL(kernel(1, execStream, tilingDevice.get(), args...));
    CV_ACL_SAFE_CALL(aclrtSynchronizeStream(execStream));'''
assert old in s, 'cann_call.hpp: kernel_launch body not found'
p.write_text(s.replace(old, new, 1))
print('(h) cann_call.hpp: kernel_launch NULL-stream fallback + acl_rt.h include patched')
PY
grep -n 'CtxGetCurrentDefaultStream\|acl_rt' opencv_contrib/modules/cannops/include/opencv2/cann_call.hpp | head -4
```

预期：打印 `(h) ... patched`，`grep` 在 `cann_call.hpp` 看到 1 行 `acl_rt.h` include + 1 行 `CtxGetCurrentDefaultStream`。

## 桥接 CANN 9.1.0 的库目录布局

OpenCV 的 `OpenCVFindCANN.cmake` 在 `${CANN_INSTALL_DIR}/{acllib,lib64,compiler/lib64}/` 下找 ACL / AOE / GE 库，CANN 9.1.0 实际装在 `${CANN_INSTALL_DIR}/aarch64-linux/lib64/`。补三个 symlink 让 CMake 能找到。

```shell #test-setup
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux /usr/local/Ascend/cann-9.1.0/acllib
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64 /usr/local/Ascend/cann-9.1.0/lib64
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64 /usr/local/Ascend/cann-9.1.0/compiler/lib64
```

> 镜像自带 `ascend-toolkit/latest -> cann-9.1.0` 时，把上面三行里的 `cann-9.1.0` 改成 `ascend-toolkit/latest` 也可以。

## 配置并编译

mainline 5.0.0 里 CANN 后端的开关变量是 `WITH_CANN=ON`（没有 `BUILD_CANN`），用环境变量 `ASCEND_TOOLKIT_HOME` 告诉 CMake `ascend-toolkit` 装在哪（或直接 `-DCANN_INSTALL_DIR=...` 覆盖）。

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

```shell #test-result id="opencv-cmake-configure" fuzzy='xxx' fuzzy='...'
...
--   CANN:xxx YES
...
```

> `OPENCV_DOWNLOAD_MIRROR_ID=gitcode` 让 cmake configure 阶段拉 ADE / IPPICV / TBB / 字体 / wechat_qrcode 模型时走 gitcode.net 镜像（国内可达）。

`make install` 编译 + 安装是一体的，最终 binary 直接落到 `/usr/local/opencv-cann/bin/`，不要拆成两步（拆开会让中间状态 binary 路径错位）。用 `cmake --build` 走 cmake-native progress 输出，能看到 `[N/M] Building CXX object ...`，比裸 `make -j2` 的"安静"更友好（裸 make 只在 target 完成才打一行，配合 -j 容易以为卡住）。

```shell #test-setup
cd opencv/build
cmake --build . --target install --parallel 2
```

## 校验 build 产物

```shell #test id="opencv-verify-build"
/usr/local/opencv-cann/bin/opencv_version
/usr/local/opencv-cann/bin/opencv_test_cannops --gtest_list_tests 2>&1 | head -n 20
```

```shell #test-result id="opencv-verify-build" fuzzy='xxx'
5.0.0
xxx
```

`opencv_version` 报源码版本。`opencv_test_cannops --gtest_list_tests` 列出可用的 gtest 套件——应该看到 `CORE.` / `CVT_COLOR.` / `ELEMENTWISE_OP.` / `AscendMat.` / `ASCENDC_KERNEL.`（**没有** `CANN` 前缀）。`opencv_test_cannops` 依赖 CANN runtime 库的 `LD_LIBRARY_PATH`，记得先 `source set_env.sh`。

## 跑 CANN 单元测试

`opencv_test_cannops` 一共 78 个用例，其中 25 个因 CANN 9.1.0 / 910B1 的 3 类已知不兼容被排除：

- `CORE.RESIZE` / `CORE.CROP_RESIZE*`（4 个）—— `ResizeArea` op 的 shape 推断不认 CANN 9.1.0 的输入 layout
- `CVT_COLOR.*XYZ / *YCrCb / *YUV`（17 个）—— 带融合算术的 Cast+Mul 颜色转换在 GE 的 `BuildSingleOpModel` 阶段失败
- `ELEMENTWISE_OP.MAT_THRESHOLD*` / `ASCENDC_KERNEL.THRESHOLD`（3 个）—— AscendC threshold kernel 在 910B1 上触发 AI Core OOB

（再之前还曾有 `SRC_TYPE_FLIP`，上游已修。）其余 53 个应当全过。

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

```shell #test-result id="opencv-cann-run-tests" fuzzy='xxx' fuzzy='...'
...
[==========]xxx
[  PASSED  ] 53 tests.
```

> 不要加 `--gtest_brief`：OpenCV 5.0.0 vendored 的 gtest 没实现这个 flag，无法识别的 flag 会直接打 help 文本、**零用例执行**、退出码还是 0（看起来假绿）。脚本里 `set -o pipefail` 是为了拦住这种假绿。

## OpenCV Python quickstart

把 Python 指向源码版 `cv2`（带 CANN 后端的那份）。下面所有 Python 块 import 的都是它。

```shell #test-setup
export PYTHONPATH=/usr/local/opencv-cann/lib/python3.12/site-packages:${PYTHONPATH:-}
```

OpenCV quickstart 的 5 类基础 API：imread/imwrite、cvtColor、resize、draw、video writer。

### 9a. 读图 / 写图

```shell #test id="quickstart-imread-imwrite"
python << 'PY'
import cv2
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)  # 100x100 black BGR
ok = cv2.imwrite('/tmp/opencv_quickstart.png', img)
print('imwrite ok:', ok)
print('shape:', img.shape, 'dtype:', img.dtype)
PY
ls -la /tmp/opencv_quickstart.png
```

```shell #test-result id="quickstart-imread-imwrite" fuzzy='xxx'
imwrite ok: True
shape: (100, 100, 3) dtype: uint8xxx
```

预期：`imwrite` 返回 `True`，文件被创建，dtype `uint8`。

### 9b. 颜色空间转换（BGR → GRAY）

```shell #test id="quickstart-cvtcolor"
python << 'PY'
import cv2
import numpy as np
bgr = np.zeros((100, 100, 3), dtype=np.uint8)
bgr[..., 0] = 255  # set blue channel to 255
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print('gray shape:', gray.shape, 'mean:', float(gray.mean()))
PY
```

```shell #test-result id="quickstart-cvtcolor" fuzzy='xxx'
gray shape: (100, 100) mean: xxx
```

`mean` 约等于 29（BGR→GRAY 权重 `0.114·B + 0.587·G + 0.299·R`，B=255 时为 0.114·255 ≈ 29）。

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

```shell #test-result id="quickstart-resize" fuzzy='xxx'
resized shape: (200, 50, 3)xxx
```

OpenCV 的 shape 是 `(rows, cols, channels)`，`resize(img, (50, 200), ...)` 实际得到 `(200, 50, 3)`（参数 `(width, height)` 映射到 `(cols, rows)`）。

### 9d. 绘制（矩形 + 文字）

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

```shell #test-result id="quickstart-draw" fuzzy='xxx'
rect+text drawn, image mean: xxx
```

`mean` ≈ 5–10（全黑底 + 一条细绿矩形 + 一行白字 → 平均亮度很低）。

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

```shell #test-result id="quickstart-video" fuzzy='xxx' fuzzy='...'
...
video frames: 3
video file size: xxx
...
```

`INFO` / `WARN` 行（找不到 FFmpeg / GStreamer / INTEL_MFX 等视频插件）是源码构建没带这些插件时注册扫描的预期噪音——忽略，看文件大小就行。

## NPU 上跑一次 DNN 推理

确认 `cv2` 已经链上 CANN 后端，然后对一张零图跑 SqueezeNet（ImageNet 分类器，约 5 MB），检查输出是有限值。

```shell #test id="opencv-py-version"
python -c "
import cv2
print('cv2 version:', cv2.__version__)
print('CANN backend available:', cv2.dnn.DNN_BACKEND_CANN)
print('NPU target available:', cv2.dnn.DNN_TARGET_NPU)
"
```

```shell #test-result id="opencv-py-version" fuzzy='xxx'
cv2 version: xxx
CANN backend available: xxx
NPU target available: xxx
```

预期：`cv2 version: 5.0.0`；两个 enum 都是非零整数。

```shell #test id="opencv-cann-infer"
python << 'PY'
import os
import time
import urllib.request
import numpy as np
import cv2

# SqueezeNet 1.0 ONNX from onnx/models — stable raw URL on github.
# ~5 MB; known to be supported by OpenCV's CANN backend
# (the standard sample in the OpenCV Huawei-CANN-Backend wiki).
# opset 12 build; the ancient squeezenet1.0-1.onnx was removed
# upstream (404), -12 is the current canonical file.
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

# SqueezeNet expects 224x224 BGR, ImageNet mean subtract.
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

```shell #test-result id="opencv-cann-infer" fuzzy='xxx' fuzzy='...'
...
model bytes: xxx
output shape: (1, 1000xxx
output dtype: float32
top class index: xxx
top class score: xxx
```

预期：

- `output shape: (1, 1000, 1, 1)` —— 1000 个 ImageNet logits，尾部维度由 graph 编译保留
- `output dtype: float32`
- `top class score` ∈ [0, 1]（任意有限值都行，不要 NaN / Inf）—— 能拿到有限值就说明 ACL graph 编译成功
- 首次跑 ~30 秒（ACL graph compile）；后续命中 `$HOME/ascend` 下的 AOE 缓存，降到 ~10 ms