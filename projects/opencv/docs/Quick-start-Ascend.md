# Quick Start (Ascend NPU)

在单卡昇腾 NPU 上完成 OpenCV 安装与 CANN (Huawei Ascend) 后端 DNN 推理：先用 pip 装官方 wheel 走完官方 [quickstart 流程](https://opencv-opencv.mintlify.app/quickstart) 的 read/write/draw/color/resize 五类基础 API；再从源码构建带 `WITH_CANN=ON` 的 OpenCV（mainline 5.0.0，opencv_contrib 提供 `cannops` 模块；参考 [OpenCV Huawei CANN Backend wiki](https://github.com/opencv/opencv/wiki/Huawei-CANN-Backend) 的模块对照，镜像统一到 CANN 9.1.0 体系），用 [SqueezeNet ONNX](https://opencv-opencv.mintlify.app/installation#python) 跑一次 NPU forward。

## 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或者 Ascend 950 系列产品，并按需完成物理机或容器内的设备挂载（`/dev/davinci*` 等）。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- 可用的 Python 环境
- 可用的 CANN（参考 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 与上面 CANN 匹配的驱动 / 固件，且 `npu-smi info` 能正常列出 NPU 设备
- 可访问 GitHub（克隆 opencv + opencv_contrib 源码），以及从 ONNX Model Zoo 下载 SqueezeNet 模型

OpenCV 通过 DNN 模块内置 Huawei CANN 后端：构建阶段把 OpenCV 链接到 `ascend-toolkit` 提供的 ACL/AOE/GE 库，运行时通过 `cv2.dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)` 把推理图下沉到 NPU。pip 发布的 `opencv-python` wheel 默认**未启用** `WITH_CANN`，必须走源码构建路径才能在 NPU 上跑 DNN。

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
| 驱动 / 固件 | 与 CANN 9.1.0 匹配（镜像内置） |
| CMake | >= 3.18 |
| GCC | 11 (Ubuntu 22.04 自带) |
| opencv | 5.0.0（源码，启用 CANN） |
| opencv_contrib | 与 opencv 同 release tag |

> CANN 9.1.0 体系对应的 ascend-toolkit 默认安装路径为 `/usr/local/Ascend/ascend-toolkit/latest`，其下 `include/acllite`、`lib64/libascendcl.so` 等 ACL 头 / 库为构建 OpenCV CANN 后端的必要依赖。

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

> 如果 `npu-smi` 不存在，请回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 补装驱动。

检查 Python / CMake 版本：

```shell #test id="check-toolchain"
python --version
cmake --version | head -n 1
```

```shell #test-result id="check-toolchain" fuzzy='xxx'
Python 3.12.xxx
cmake version xxx
```

加载 CANN 环境变量（`set_env.sh` 把 `ASCEND_HOME` / `LD_LIBRARY_PATH` / `PATH` 等灌进当前 shell；每个 `#test` 块都是独立 `subprocess.run`，所以这步只在 load-cann 这一块里 source 一次，下面 cmake configure / make / python 块各自再 source 一次以拿到 CANN 变量——`prepare_environment` 不再做 source）：

```shell #test id="load-cann"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

```shell #test-result id="load-cann"
...
```

预期：source 退出码为 0（`set_env.sh` 自身无输出），`export PATH` 也退出 0。`PATH=/usr/local/sbin:$PATH` 仅在本块生效——`prepare_environment` 不再追加这个目录，每一块自己保证。

检查 CANN 安装路径：

```shell #test id="check-cann-path"
ls -d /usr/local/Ascend/ascend-toolkit/latest
```

```shell #test-result id="check-cann-path" fuzzy='xxx'
/usr/local/Ascend/ascend-toolkit/xxx
```

> 如果路径不存在或 `latest` 软链指向其他版本，回到 [Ascend 官方快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html) 校验 CANN 安装。

## 安装 OpenCV（源码，启用 CANN 后端）

pip 发布的 `opencv-python` / `opencv-python-headless` wheel **不带** CANN 后端（enum 常量在但 backend 实现未链接），必须从源码构建：克隆 `opencv` + `opencv_contrib`，用 CMake 把 `WITH_CANN=ON` 打开后编译。安装路径用 `/usr/local/opencv-cann`，与默认 `python -c "import cv2"` 解析路径完全隔离。

#### 安装基础编译依赖

镜像基于原生 `ubuntu:22.04`（arm64），`/etc/apt/sources.list` 指向 Canonical 的 `ports.ubuntu.com`（海外源），国内 runner 上 `apt-get update` 拉索引极慢甚至超时。先把源换成阿里云的 arm64 ports 归档（必须是 `ubuntu-ports`，不能用 x86 的 `ubuntu/` 归档），再装依赖：

```shell #test-setup
sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list
apt-get update -qq && apt-get install -y -qq git build-essential cmake pkg-config libjpeg-dev libpng-dev libtiff-dev
```

> 测试容器每次 run 重建，sed 只影响本次容器，不污染 runner 或镜像。镜像构建时已装过 git / build-essential / cmake（构建历史可见），这条命令真正要下载的只有 libjpeg / libpng / libtiff 三个图像编解码 dev 包，换源后秒级完成。

#### 从源码克隆 opencv + opencv_contrib

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获并注入到下方 clone 命令中；markdown 渲染器会丢掉注释里全部内容，读者看不到这段代码，但 runner 仍然执行它并 store="upstream_ref" -->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆上游仓库并 checkout 到工作流注入的最新 release tag；opencv 与 opencv_contrib 版本必须一致：

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/opencv/opencv.git
git clone --depth 1 --branch <ref> https://github.com/opencv/opencv_contrib.git
```

> `<ref>` 为工作流注入的最新 release tag。

#### 修补 OpenCV 5.0.0 CANN backend 的 MatShape 签名不匹配

OpenCV 5.0.0 把 `cv::MatShape` 从 `std::vector<int>` 的别名改成了一个独立 struct（mat.hpp:106，加了 layout / dims 等字段），但 `modules/dnn/src/op_cann.{hpp,cpp}` 里的 `CannConstOp` 构造函数仍按老签名 `const std::vector<int>& shape` 写，导致 `dnn` 模板里 `std::make_shared<CannConstOp>(..., shape(w_mat), ...)`（convolution_layer.cpp:703 等 20+ 处）编译报 `no known conversion for argument 3 from 'cv::MatShape' to 'const std::vector<int>&'`。

把构造函数第三参数改成 `const cv::MatShape&` 即可——`MatShape` 自身提供了 `begin()` / `end()`（mat.hpp:142-145），`.cpp` 里 `std::vector<int64_t> shape_{shape.begin(), shape.end()};` 不需要改。

但只改一个签名还不够——`batch_norm_layer.cpp:383` / `elementwise_layers.cpp:652,2815` / `slice_layer.cpp:691,739` 等几处用本地 `std::vector<int> shape_{...}` 然后传给 `CannConstOp`，会再报反向错误 `no known conversion for argument 3 from 'std::vector<int>' to 'const cv::MatShape&'`。最干净的修复是加一个转发构造函数：第二签名接受 `const std::vector<int>&`，在初始化列表里构造 `cv::MatShape(shape)` 后委托给主构造，避免改 4 个 layer 文件里的局部变量类型。

还不够——`gemm_layer.cpp:534` 和 `matmul_layer.cpp:516` 各有一处 `xxx_shape = std::vector<int>{...}` 赋值给 `MatShape` 局部变量（不是传给函数，是 `operator=`），这条路径转发重载救不了（重载只匹配函数调用），得改这两行用 `MatShape` 的 raw-array 构造 `cv::MatShape(1, &val)`（mat.hpp:111）。改完这俩 `make` 才能彻底过 dnn。

等 5.0.1 / 主仓把 CANN backend 重命名到 contrib 后这步可删。

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

预期：步骤 (a) 两行各 `1 occurrence(s) -> MatShape&`；(b) hpp +1 decl；(c) cpp +1 delegating ctor；(d) 2 个 assignment 各修复。最后 `grep` 第一组在 hpp 看到 2 行构造声明、cpp 看到 2 个构造函数定义；第二组在 gemm_layer.cpp:534 和 matmul_layer.cpp:516 各 1 行。Python 而非 sed：`&` 在 sed 替换串里代表匹配文本，转义 `\&` 在 GNU/BSD sed 行为不一致，换 Python 避免踩坑。

#### 修补 all_ops.h 头文件爆炸（aarch64 链接 R_AARCH64_CALL26 溢出）

CI 33244401262 在 `make` 编完 dnn 全部 252 个 TU 后死在最后一步链接：`libopencv_dnn.so` 报 `relocation truncated to fit: R_AARCH64_CALL26 against symbol google::protobuf::Arena::...`。根因不在 protobuf：`modules/dnn/src/op_cann.hpp` include 了 CANN 全量 op 头 `built-in/op_proto/inc/all_ops.h`（~1500 个 op 类），而这个头经 `net_impl.hpp` 被 200+ 个 dnn TU 引入。每个 `REG_OP` 宏在**每个**包含它的 TU 里展开成 file-local 注册 lambda（局部符号，链接器无法跨 TU 去重）加 ~134KB 静态初始化代码——`-O0` 不做死代码消除，200 TU × 1500 op × ~1.7KB ≈ 500MB 不可去重的 .text（实测 `ld -r` 合并后 516MB；同一份源码不带 CANN 头只有 21MB）。aarch64 `BL` 指令跳转范围 ±128MB，protobuf 弱内联函数与静态库成员在如此巨大的 .text 里间距轻松超限，链接必炸。上游 CI 用 Release(-O3) 构建，未用到的注册 lambda 被优化器全部消除，从未踩到；Debug(-O0) + aarch64 是本看护独有的组合。

修法：`op_cann.hpp` 把 `all_ops.h` 换成 `array_ops.h`（只定义 op_cann.hpp 自身用到的 `ge::op::Const/Data/Identity/Reshape/Unsqueeze`，68 个 op），再给 23 个真正实例化 op 类的 layer TU 按需插窄头——每个文件只包含自己用到的那类 op（conv → `nn_calculation_ops`、激活 → `nonlinear_fuc_ops`、pooling → `nn_pooling_ops`……op→头文件的映射用 `grep -l "REG_OP(\<op\)" /usr/local/Ascend/cann-9.1.0/opp/built-in/op_proto/inc/*.h` 枚举，同一个 op 挑 op 数最少的头）。实测 dnn .text 从 516MB 降到 47MB（仍有 20 万个 op 注册符号，运行时注册完整），链接通过；每个 TU 预处理体积缩水约八成，cc1plus 内存压力同步下降——graph_fusion 大 TU 不再是 OOM 高危。这个补丁值得报给 opencv 上游（op_cann.hpp 不该 include all_ops.h），等上游修复后可删。

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

预期：(e) 1 行替换；(f) 23 个文件各 +1~2 个窄头。最后 grep 第一组在 op_cann.hpp 看到 1 行 array_ops；第二组输出 23（插入窄头的 layer 文件数）。`net_cann.cpp` 用的 `ge::op::Data`、`op_cann.cpp` 用的 `ge::op::Const`、blank/const/reshape layer 用的 `Identity/Const/Reshape/Unsqueeze` 都已由 op_cann.hpp 里的 array_ops.h 覆盖，不用单独加。

#### 桥接 OpenCV 5.0.0 与 CANN 9.1.0 的 layout 差

mainline 5.0.0 的 `OpenCVFindCANN.cmake` 假设库在 `${CANN_INSTALL_DIR}/{acllib,lib64,compiler/lib64}/` 下，但 CANN 9.1.0 实际把 ACL/AOE/GE 库都装在 `${CANN_INSTALL_DIR}/aarch64-linux/lib64/`。补三个软链让 cmake 找得到：

```shell #test-setup
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux /usr/local/Ascend/cann-9.1.0/acllib
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64 /usr/local/Ascend/cann-9.1.0/lib64
ln -sfn /usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64 /usr/local/Ascend/cann-9.1.0/compiler/lib64
```

> 后续版本（OpenCV 5.0.1+ / CANN 9.2+）若 `OpenCVFindCANN.cmake` 把搜索路径加进 `aarch64-linux/lib64/`，这步可以删。镜像自带 `ascend-toolkit/latest -> cann-9.1.0` 时，把上面三行里的 `cann-9.1.0` 改成 `ascend-toolkit/latest` 也可以。

#### 修补 cannops 默认流（NULL stream）与 CANN 9.1.0 的不兼容

cannops 的默认流是 NULL 指针（`cann_call.cpp` 的 `DefaultDeviceInitializer` 直接 `aclrtStream stream = nullptr`——ACL 的 legacy"默认流"语义，老 CANN 上 `aclopCompileAndExecute(..., NULL)` 合法）。但 CANN 9.1.0 的 `aclopCompileAndExecute` GE 执行路径内部会调 `aclrtAllocatorGetByStream(stream)`，这个 API **不接受 NULL**，直接报 `Invalid_Argument_Null_Pointer(EH0008): stream cannot be a NULL pointer`——于是测试里每个 op 执行（ConcatD/SplitD/TransposeD/ReverseV2/...）全灭，72 个用例同一死法（CI 33265462625 诊断输出实锤）。修法：patch `OperatorRunner::run`，默认流时临时创建一个真实 stream 跑 op、同步后销毁：

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
    // CANN 9.1.0: aclopCompileAndExecute's GE path calls aclrtAllocatorGetByStream
    // internally, which rejects NULL (legacy default-stream) pointers with EH0008.
    aclrtStream execStream = rawStream;
    bool ownExecStream = false;
    if (execStream == nullptr)
    {
        CV_ACL_SAFE_CALL(aclrtCreateStream(&execStream));
        ownExecStream = true;
    }
    CV_ACL_SAFE_CALL(aclopCompileAndExecute(op.c_str(), inputDesc_.size(), inputDesc_.data(),
                                            inputBuffers_.data(), outputDesc_.size(),
                                            outputDesc_.data(), outputBuffers_.data(), opAttr_,
                                            ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, execStream));
    CV_ACL_SAFE_CALL(aclrtSynchronizeStream(execStream));
    if (ownExecStream)
        CV_ACL_SAFE_CALL(aclrtDestroyStream(execStream));
    else
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
grep -n 'ownExecStream' opencv_contrib/modules/cannops/src/cann_call.cpp | head -4
```

预期：打印 `(g) ... patched`，grep 看到 4 行 `ownExecStream`。原逻辑对 NULL 流走 `aclrtSynchronizeStream(NULL)`（同样依赖 legacy 语义），新逻辑显式建流 + 同步 + 销毁，行为等价且不依赖 NULL stream 兼容性。这个补丁值得报给 opencv_contrib 上游。

同样的 NULL 流坑还有第二处：AscendC kernel 的启动路径 `kernel_launch`（`cann_call.hpp` 里的 inline template，threshold 系列 kernel 走这里）直接把 NULL 传给 kernel 启动函数、NULL 时再调 `stream.waitForCompletion()` → `aclrtSynchronizeStream(NULL)`——CANN 9.1.0 下前者引发 AI Core 越界错误（kernel 没被正确提交）、后者报 `EH0012: The stream is not registered with any allocator`。补 patch (h)：

```shell #test-setup
python3 - <<'PY'
import pathlib
p = pathlib.Path('opencv_contrib/modules/cannops/include/opencv2/cann_call.hpp')
s = p.read_text()
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
    // CANN 9.1.0: AscendC kernel launch + aclrtSynchronizeStream both reject
    // NULL (legacy default) streams via the allocator path. Create a real one.
    aclrtStream execStream = rawStream;
    bool ownExecStream = false;
    if (execStream == nullptr)
    {
        CV_ACL_SAFE_CALL(aclrtCreateStream(&execStream));
        ownExecStream = true;
    }
    CV_ACL_SAFE_CALL(kernel(1, execStream, tilingDevice.get(), args...));
    CV_ACL_SAFE_CALL(aclrtSynchronizeStream(execStream));
    if (ownExecStream)
        CV_ACL_SAFE_CALL(aclrtDestroyStream(execStream));'''
assert old in s, 'cann_call.hpp: kernel_launch body not found'
p.write_text(s.replace(old, new, 1))
print('(h) cann_call.hpp: kernel_launch NULL-stream fallback patched')
PY
grep -n 'ownExecStream' opencv_contrib/modules/cannops/include/opencv2/cann_call.hpp | head -4
```

预期：打印 `(h) ... patched`，grep 看到 4 行 `ownExecStream`。

#### CMake 配置：开启 WITH_CANN

mainline 5.0.0 里 CANN 后端的开关变量是 `WITH_CANN`（不是 `BUILD_CANN`，后者不存在），通过环境变量 `ASCEND_TOOLKIT_HOME` 指向 `ascend-toolkit` 安装根目录（也可用 `-DCANN_INSTALL_DIR=...` 直接覆盖）——这一步与 [OpenCV Huawei CANN Backend wiki](https://github.com/opencv/opencv/wiki/Huawei-CANN-Backend) 的 Step 3 一致：

> CI runner 内存极紧，`-O3` 模板优化阶段 cc1plus 内存 spike 会把 dnn 大 TU OOM kill，所以走 `CMAKE_BUILD_TYPE=Debug`（`-O0 -g`）砍优化内存，再配合下面 `-j2`——Debug 编译是为了 CI 通过，不是为生产性能。

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

预期：`CANN: ... YES` 这一行出现在 cmake summary 段——这正是 wiki 上 Step 4（Verification）所要求的"先看 CMake 报告"步骤。`OPENCV_DOWNLOAD_MIRROR_ID=gitcode` 让主仓 cmake configure 阶段拉 ADE / IPPICV / TBB / xfeatures2d 数据 / 字体 / wechat_qrcode 模型时走 gitcode.net 镜像（中国大陆可达）。

`BUILD_LIST` 把全模块 + ~35 个 contrib 模块砍到 8 个——quickstart 只用到 imwrite/imread/resize/cvtColor/putText/VideoWriter（core+imgproc+imgcodecs+videoio）、dnn、python 绑定和 cannops 测试。注意 `ts` 必须在列表里：它是 `opencv_test_cannops` 的测试框架依赖，被 BUILD_LIST 白名单排除时 cannops 的 `ocv_add_accuracy_tests` 建不出测试目标，`opencv_contrib/modules/cannops/CMakeLists.txt:23` 的 `ocv_target_link_libraries(opencv_test_cannops ...)` 直接报 `invalid target`。`ts` 不进列表时 xfeatures2d / face 等 contrib 模块自然被剔除，不再需要单独的 `BUILD_opencv_xfeatures2d=OFF`。

`PYTHON_INCLUDE_DIR` / `PYTHON_LIBRARY` 这对参数是必须的：镜像的 Python 3.12.13 是源码装在 `/usr/local/python3.12.13` 的，cmake 老式 `find_package(PythonLibs)` 不搜这里，不传这对参数 python3 模块会**静默**落进 summary 的 `Unavailable: ... python3 ...` 一行——装出来的 `/usr/local/opencv-cann` 下没有 `lib/python3.12/site-packages`，后面 quickstart 的 `import cv2` 必挂。别只盯 `CANN: YES` 那一行。

`SOC_VERSION=ascend910b1` 必须与实际卡型匹配：contrib `cannops/ascendc_kernels/CMakeLists.txt` 默认编给 `ascend310p3`（cache 变量），AscendC kernel 的 host stub 在加载时用 `AscendCheckSoCVersion` 对比编译 SoC 与 `aclrtGetSocName` 返回的运行时 SoC——不匹配 kernel 不注册，`ASCENDC_KERNEL.*` / `MAT_THRESHOLD_ASCENDC` 等用例直接失败（CI 33262664977：910B1 的 runner 编 310P3 内核，73 用例全红）。`ascend910b1` 是 CANN `host_config.cmake` 的 `ascend910b_list` 合法值；换 runner 卡型时同步改这里。

两个测试相关开关：`OPENCV_BUILD_TEST_MODULES_LIST=cannops` 让 `BUILD_TESTS=ON` 只构建 cannops 一个模块的测试二进制（否则全仓 ~15 个模块、每个几十个 test TU 的 accuracy tests 都会进默认构建目标，`make` 多花 ~40 分钟）；`INSTALL_TESTS=ON` 把测试二进制装进 `CMAKE_INSTALL_PREFIX/bin`（OpenCV 默认**不**安装 opencv_test_*，不打开这个开关 `/usr/local/opencv-cann/bin/opencv_test_cannops` 不会存在）。

#### 编译并安装到 /usr/local/opencv-cann

> **必须用一条 `make install -j2` 走完构建+安装，不要先 `make` 再 `make install` 分两次跑。** cannops 的 AscendC kernel 走 CANN `ascendc.cmake` 的 ExternalProject（`BUILD_ALWAYS TRUE`），其中 `merge_obj_text.sh` 对 m200 内核 `.o` 做**原地** ld.lld 合并（`-o` 与输入同路径）且无防重入保护：第一次跑把 bisheng 产出的 REL 转成 EXEC，第二次再喂给 `ld.lld` 就报 `unknown file type`。`make` 和 `make install` 各触发一次 EP build（自定义目标永远视为过期），两次必炸——本仓在 CANN 镜像里实测 `make -j2 && make install` 100% 复现 `ld.lld: ...m200_obj... unknown file type`，单次 `make install -j2` 则一遍过。
>
> CI runner 内存极紧，dnn 模板大 TU（matmul / dft / reshape2 / slice2 / pad2 / padding / resize / reduce / recurrent2 / permute / group_norm / nary_eltwise / if / shape / split2 / transpose layer）`cc1plus` 在 `-O3` 下会被 OOM kill，所以 cmake 走 `Debug`（`-O0 -g`）砍优化内存、再 `-j2` 限制并行度（上面 all_ops.h 修补之后每个 dnn TU 预处理体积缩水约八成，内存压力进一步下降）。`BUILD_LIST` 砍模块 + 窄头修补双管齐下后，`-j2` 全量构建+安装约 15 分钟（本地 CANN 镜像 arm64 实测 607s build + ~200s install；CI runner 核慢一些预计 40–70 分钟）。

```shell #test-setup
cd opencv/build
make install -j2
```

#### 校验二进制 + CANN 后端可用性

跑 OpenCV 自带的 versioninfo 和 cannops 测试列举，确认二进制可执行且测试用例在列：

```shell #test id="opencv-verify-build"
/usr/local/opencv-cann/bin/opencv_version
/usr/local/opencv-cann/bin/opencv_test_cannops --gtest_list_tests 2>&1 | head -n 20
```

```shell #test-result id="opencv-verify-build" fuzzy='xxx'
5.0.0
xxx
```

预期：versioninfo 第一行打印 `5.0.0`；`opencv_test_cannops --gtest_list_tests` 列出 cannops 模块的 gtest 套件（`CORE.` / `CVT_COLOR.` / `ELEMENTWISE_OP.` / `AscendMat.` / `ASCENDC_KERNEL.`，来自 contrib `modules/cannops/test/`）。注意套件名**没有** `CANN` 前缀——不要按直觉写成 `CANNxxx`（mainline 5.0.0 的 CANN 后端单元测试在这个 `opencv_test_cannops` 二进制里，主仓 `opencv_test_dnn` 已不带 `*HUAWEI*` 用例）。

#### 跑一遍 CANN 单元测试

`opencv_test_cannops` 内置了 CANN 后端的小型模型测例，会调用 ACL 把模型图下沉到 NPU：

```shell #test id="opencv-cann-run-tests"
set -o pipefail
/usr/local/opencv-cann/bin/opencv_test_cannops --gtest_color=no > /tmp/cannops_gtest.log 2>&1; rc=$?
tail -n 25 /tmp/cannops_gtest.log
if [ $rc -ne 0 ]; then
  echo '--- failure assertion blocks:'
  grep -A 10 'unknown file: Failure' /tmp/cannops_gtest.log | head -120
fi
exit $rc
```

```shell #test-result id="opencv-cann-run-tests" fuzzy='...' fuzzy='xxx'
[==========]xxx
...
[  PASSED  ] xxx
```

> 不要加 `--gtest_brief`：OpenCV 5.0.0 vendored 的这份 gtest（`modules/ts/src/ts_gtest.cpp`）没有实现该 flag（只认 list_tests / color / filter / print_time / output / repeat 等），带 `--gtest_` 前缀但无法解析的参数会走"unrecognized Google Test flag"路径——直接打印 flag 帮助文本、**一个测试都不跑、退出码还是 0**（CI 33259147288 实测：958B stdout 全是 help 文本，`set -o pipefail` 拦不住 rc=0 的假绿）。`--gtest_list_tests` 是实现了的，所以上一步能正常列举。


## 使用样例：OpenCV 官方 quickstart

完整跑一遍 [OpenCV Quickstart 文档](https://opencv-opencv.mintlify.app/quickstart) 的五类基础 API：读图 → 保存 → 灰度转换 → 缩放 → 在图上画矩形 + 文字。每步都用一行 Python 写完，输出预期可校验。本节使用上面源码构建的 `cv2`（带 CANN 后端），先把它加入 `PYTHONPATH`，下面所有 Python 块 import 的就是源码版：

```shell #test-setup
export PYTHONPATH=/usr/local/opencv-cann/lib/python3.12/site-packages:${PYTHONPATH:-}
```

### 读图 / 写图（imread / imwrite）

构造一张 100×100 的 BGR 零图像（`np.zeros` 模拟 `cv2.imread` 的输入），`cv2.imwrite` 写到 `/tmp`：

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

预期：`imwrite` 返回 `True`、文件被创建，dtype 是 `uint8`（OpenCV 默认 BGR 通道序）。

### 颜色空间转换（cvtColor）

`cv2.cvtColor` 把 BGR 转成 GRAY（单通道）。这一步验证 image-processing 基础管线：

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

预期：`gray shape: (100, 100)`（单通道），`mean` 在 0~255 之间（Blue=255 经过 BGR→GRAY 权重后约为 29.85）。

### 缩放（resize）

`cv2.resize` 把 100×100 缩到 50×200，用 `cv2.INTER_AREA` 插值（缩小时抗锯齿效果最好）：

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

预期：输出形状 `(200, 50, 3)`——OpenCV 的 shape 是 `(rows, cols, channels)`，所以 `(50, 200)` 的 `(width, height)` 参数实际对应 `resize` 后 rows=200 / cols=50。

### 绘制（rectangle / putText）

在图上画一个绿色矩形和一行白字。`cv2.rectangle` 接受 `(image, pt1, pt2, color, thickness)`，颜色是 BGR 三元组（这里是 `(0, 255, 0)` = 纯绿）；`cv2.putText` 接受 `(image, text, org, fontFace, fontScale, color)`：

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

预期：`image mean` 在 0~30 之间（全黑底图 + 一条绿色矩形 + 一行白字后，平均亮度很低）。`/tmp/opencv_draw.png` 被创建。

### 视频写入（VideoWriter）

`cv2.VideoCapture` 既能从摄像头/文件读，也能接受一个「整数」打开一个合成视频源（`0` 通常是默认摄像头；CI 上跳过）；这里反过来用 `cv2.VideoWriter` 验证 video 模块基础管线可用：

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

```shell #test-result id="quickstart-video" fuzzy='xxx'
video frames: 3
video file size: xxx
```

预期：3 帧写入完成，`/tmp/opencv_video.avi` 被创建（MJPG 编码后即使是 3 帧黑帧也有几 KB）。

## 使用样例：用 CANN 后端跑一次 DNN 推理

`PYTHONPATH` 已切到 `/usr/local/opencv-cann/lib/python3.12/site-packages`（quickstart 节那段 `#test-setup`，以及 runner 的 `prepare_environment` 在 `os.environ` 里预设了一份），下面 `python` 块 import 的是上面源码构建的 `cv2`，带 CANN 后端：

校验 cv2 版本与 CANN 后端 enum：

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

预期：`CANN backend available` / `NPU target available` 都打印一个非零整数（cv2 把 backend/target 编码为 enum）。

跑一个最小 DNN 推理：从 ONNX Model Zoo 拉一份 SqueezeNet (~5MB)，构造一张零图像，`cv2.dnn.Net.forward()` 在 NPU 上跑前向：

```shell #test id="opencv-cann-infer"
python << 'PY'
import os
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
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
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

```shell #test-result id="opencv-cann-infer" fuzzy='xxx'
model bytes: xxx
output shape: (1, 1000)xxx
output dtype: float32
top class index: xxx
top class score: xxx
```

预期：

- `output shape: (1, 1000)`：SqueezeNet 输出是 1×1000 的 ImageNet 1000-class logits。
- `top class score` 在 `0~1` 之间：随机初始化输入对应全 0 图像，输出是数据集 bias，对 **某一** 类的得分会高于其他 999 类；如果输出是 NaN / Inf 或全是同一数值，说明 ACL graph 编译失败。
- 第一次跑会触发 ACL graph 编译（耗时数十秒），日志里有 `acl graph compile ...` 字样；第二次起命中 `$HOME/ascend` 下的 AOE 缓存，forward 耗时降到 10 ms 量级。

> 这里只验证 CANN 后端的"链接 + ACL graph 编译 + forward 输出"链路完整。`opencv-cann-run-tests` 那一步才是对 CANN 后端 **opencv 自带测试用例** 的完整验证，两段合起来覆盖 wiki 上的"模型推理 sample"和"单元测试验证"两路样本。

## 小贴士

- 源码构建把 `cv2` 装到 `/usr/local/opencv-cann/lib/python3.12/site-packages`，独立于系统的 `python -c "import cv2"` 解析路径；想恢复系统版本只需 `unset PYTHONPATH` 或把 `prepare_environment` 加的路径从 `os.environ` 里去掉。runner 已经把 `PYTHONPATH` 切到源码版，本地手动跑时也要 export 同样的值才能 import 到带 CANN 后端的 cv2。
- CANN 9.1.0 与 CANN 8.x 在 `ascend-toolkit` 安装路径上行为一致（都是 `/usr/local/Ascend/ascend-toolkit/latest`），但 ACL 接口在 8.x → 9.x 间做过一次 breaking change（新增 `acltdt` 通道、`aclop` 接口签名变更）。如果 OpenCV 是用 CANN 8.x 的 `acllib` 编译的，运行时挂在 9.1.0 上会报 `ACL_ERROR_INVALID_PARAM`；本文档的 `ASCEND_TOOLKIT_HOME` 直接指向当前镜像的 9.1.0 toolkit，从源头避免这个问题。
- 想验证 NPU 推理真生效：把上面 `opencv-cann-infer` 块的 `np.zeros` 换成真实图片（如 `cv2.imread('/tmp/photo.jpg')`），第一次 forward 会触发 ACL graph 编译（`aoe` 缓存落在 `$HOME/ascend` 下），第二次起命中缓存，性能才会贴近 NPU 真实吞吐。
- 多卡推理：`cv2.dnn` 自身不管理 device 亲和性，多张 NPU 的负载分配需要在进程级别串行或起多进程（每进程 `setPreferableTarget` 到不同 NPU ID）；CANN 9.1.0 上 `ASCEND_RT_VISIBLE_DEVICES=0,1` 限定本进程可见设备。
- 想跑官方 OpenCV samples 里更复杂的 sample（如 `object_detection.py`、`segmentation.py`）：这些 sample 接受 `--backend CANN` 命令行参数，把 OpenCV 编译成带 CANN 即可直接复用。
- 清理：
  ```bash
  rm -rf opencv opencv_contrib /usr/local/opencv-cann \
         /tmp/opencv_quickstart.png /tmp/opencv_draw.png \
         /tmp/opencv_video.avi /tmp/squeezenet1.0-12.onnx
  ```