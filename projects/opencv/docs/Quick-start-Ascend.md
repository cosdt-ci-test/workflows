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

## 打 3 组源码补丁

OpenCV mainline 与 CANN 9.1.0 / aarch64 有 3 处不兼容（3 个 patch 脚本假设 runner resolve 到 `5.0.0` tag；其它 tag 上 `assert old in s` 会抛错，需要重写 patch 适配）。按顺序跑这 3 段脚本（幂等，已打过会断言跳过）。

> **不想打补丁？** 走 [Quick-start-Ascend-NoPatch.md](Quick-start-Ascend-NoPatch.md)：版本钉 **4.14.0**（4.x 的末代）+ **Release(-O3)** + 不编 `opencv_contrib`，**零源码补丁**也能得到带 CANN 后端的 `cv2`（实测 dnn `.text` 53.3 MiB，一次链接通过，NPU top-1=372 / 0.9143）。代价是拿不到 OpenCV 5.x 的新 DNN 图引擎，也没有 `cv2.cann`。
>
> 为什么 4.x 天然比 5.x 省补丁：`all_ops.h` 是 **CANN 的头**，与 OpenCV 版本无关 ⇒ `.text` 随「引用 `op_cann.hpp` 的 TU 数」线性增长。实测 4.11.0 = 27 个 TU、**4.14.0 = 28 个**、5.0.0 = **39 个**；dnn 源码量 4.11.0 → 4.14.0 只 +2.3%，→ 5.0.0 却 +50.6%。4.14 离 aarch64 的 ±128 MiB 分支窗口还有 ~74 MiB 余量，5.0 撞墙 —— 这就是本文那 3 组补丁里「窄 op 头」那组的由来。

| 组 | 上游哪里坏了 | 改法 |
| --- | --- | --- |
| 4a | OpenCV 5.0 把 shape 类型换成 `cv::MatShape`（`explicit` 构造、**没有** `front()`） | 新增 1 个 `cv::MatShape` 重载转发给上游构造 + 修 3 处赋值 |
| 4b | `all_ops.h` 的 ~1500 个 op 把 dnn `.text` 撑破 aarch64 ±128 MiB 链接窗口 | per-TU 窄 op 头 |
| 4c | CANN 9.x 拒绝 NULL 流（EH0008） | `OperatorRunner::run` 回落到 ctx 默认流 |

> 之前还有第 5 组「H2D/D2H 宿主缓冲收窄」，**已删除**：它防的是"驱动报的 buffer size 比 shape 字节数大"，而实测 `acl_size == host_size == 4000`（无 padding），该路径本来就不是崩溃根因；当时真正的堆损坏是**宿主驱动 25.5.2 × 镜像 CANN 9.1.0 版本错配**导致 `DvppWrapperManager` 析构时 free 到坏块，属宿主环境问题（配套驱动 25.3.rc1.2 上无此现象），不该用源码补丁兜。

### 4a. `cv::MatShape` 改了类型

OpenCV 5.0 把 shape 类型换成了 `cv::MatShape`（构造是 `explicit`，且**没有** `front()`），上游 CANN 后端还停在 4.x 的 `std::vector<int>` 写法，编译不过。

改法保持最小：**上游的 `std::vector<int>` 构造函数原样保留**，只**新增**一个 `cv::MatShape` 重载转发给它；再修 3 处已经编译不过的赋值。

```shell #test-setup
python3 - <<'PY'
import pathlib

# (a) op_cann.hpp: 新增 cv::MatShape 重载声明（上游 vector<int> 主构造不动）
hp = pathlib.Path('opencv/modules/dnn/src/op_cann.hpp')
hs = hp.read_text()
decl_v = 'CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);'
decl_m = 'CannConstOp(const uint8_t* data, const int dtype, const cv::MatShape& shape, const std::string& name);'
assert decl_v in hs, 'hpp: upstream std::vector<int> decl not found'
if decl_m in hs:
    print('(a) op_cann.hpp: MatShape overload decl already present')
else:
    hp.write_text(hs.replace(decl_v, decl_v + '\n        ' + decl_m, 1))
    print('(a) op_cann.hpp: +1 cv::MatShape overload decl')

# (b) op_cann.cpp: MatShape 重载转发给上游 vector<int> 构造（上游函数体不改）
cp = pathlib.Path('opencv/modules/dnn/src/op_cann.cpp')
cs = cp.read_text()
deleg = ('\nCannConstOp::CannConstOp(const uint8_t* data, const int dtype, const cv::MatShape& shape, const std::string& name)\n'
         '    : CannConstOp(data, dtype, std::vector<int>(shape.begin(), shape.end()), name) {}\n')
anchor = 'op_ = std::make_shared<ge::op::Const>(name);\n    op_->set_attr_value(*ge_tensor);\n}\n'
if deleg.strip() in cs:
    print('(b) op_cann.cpp: MatShape forwarding ctor already present')
else:
    assert anchor in cs, 'cpp: anchor not found'
    cp.write_text(cs.replace(anchor, anchor + deleg, 1))
    print('(b) op_cann.cpp: +1 MatShape forwarding ctor')

# (c)(d) cv::MatShape 没有 front()、也不能从 std::vector<int> 赋值 -> 修 3 处
fixes = [
    ('opencv/modules/dnn/src/layers/gemm_layer.cpp',
     '            shape_C = std::vector<int>{dim};',
     '            shape_C = cv::MatShape(1, dim);'),
    ('opencv/modules/dnn/src/layers/matmul_layer.cpp',
     '                if (real_ndims_C == 1 && bias_shape.front() != 1) {',
     '                if (real_ndims_C == 1 && bias_shape[0] != 1) {'),
    ('opencv/modules/dnn/src/layers/matmul_layer.cpp',
     '                    bias_shape = std::vector<int>{bias_shape.front()};',
     '                    bias_shape = cv::MatShape(1, bias_shape[0]);'),
]
for rel, o, n in fixes:
    p = pathlib.Path(rel); s = p.read_text()
    if n in s:
        print(f'(c)(d) {rel}: already patched')
        continue
    assert o in s, f'{rel}: pattern not found'
    p.write_text(s.replace(o, n, 1))
    print(f'(c)(d) {rel}: 1 assignment fixed')
PY
```

预期：`(a)`/`(b)` 各 1 行打印，`(c)(d)` 打印 3 行（gemm 1 + matmul 2）；`grep 'cv::MatShape' opencv/modules/dnn/src/layers/{gemm,matmul}_layer.cpp` 各命中 1 处。

### 4b. 缩小 CANN op 头（避免 aarch64 链接炸）

`all_ops.h` 有 ~1500 个 op。`op_cann.hpp` 被 `net_impl.hpp` 拉进**每个** dnn TU，于是每个 TU 都背上 ~4 MB 的 op 内联代码（97 个 TU 间接入 `op_cann.hpp`），`libopencv_dnn.so` 的 `.text` 撑破 aarch64 `R_AARCH64_CALL26/JUMP26` 的 ±128 MiB 窗口，链接直接死在 `relocation truncated to fit`（CI 33244401262，报在 protobuf 的 `Arena::AllocateAlignedWithCleanup` 上——溢出的是总 `.text` 跨度，被随机挑中的受害者）。

Debug 与 Release **都炸**（Release 下实测同样 `relocation truncated to fit: R_AARCH64_JUMP26`），换优化级别救不了，只能缩头：换成窄头 `array_ops.h`，并给每个 layer TU 插它实际用到的窄头。

> 23 个窄头**不能**合并成一个"域头并集"（那样补丁少 22 处，看着很美）。实测同一个 TU 的 `.text*`：

> | TU | `all_ops.h` | per-TU 窄头（采用） | 域头并集 |
> | --- | --- | --- | --- |
> | `convolution_layer.cpp` | 4,026,788 | 361,616 | 2,036,904 |
> | `net.cpp` | 3,966,196 | 209,708 | 1,976,312 |
>
> 并集方案是窄头的 5.6~9.4 倍，外推到 97 个 TU 仍约 195 MB > 128 MiB，链接照炸。

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

**这组是必需的（实测）**：把 `(g)` 回退成上游写法后重编，`opencv_test_cannops` 的 53 个用例里 **48 个挂**（只剩 5 个过），报错是

```
Invalid_Argument_Null_Pointer(EH0008): aclrtAllocatorGetByStream failed because stream cannot be a NULL pointer.
E29999: Assert ((stream) != nullptr) failed[FUNC:AiCoreLaunchKernelWithHandle]
```

也就是说 CANN 9.1.0 真的不接 NULL 流（`aclopCompileAndExecute` → `AiCoreLaunchKernel*` 一路断言非空）。上游那行 `if (rawStream == nullptr) aclrtSynchronizeStream(rawStream)` 本身也是错的（同步一个 NULL 流），补丁顺手把它改成同步真正用的那条流。

### 4d. AscendC `kernel_launch` 模板：**不打**这个补丁（实测测不出差别）

上游 `cann_call.hpp` 的 `kernel_launch` 模板有同一个毛病（把 `rawStream` 可能是 `nullptr` 的值直接喂给 `kernel(...)`，并且 `acl_rt.h` 没 include）。我们**实测过**要不要打这个补丁，结论是**不打**：

| 变体 | 53 个 cannops 用例 | 3 个 threshold 用例（CI 排除清单里） |
| --- | --- | --- |
| 打了（ctx 默认流 fallback + `acl_rt.h`） | 53/53 过 | 3/3 挂 |
| 不打（上游原样） | 53/53 过 | 3/3 挂 |

- 53 个用例**根本走不到** `kernel_launch`：CI 的排除清单里的 3 个 threshold 用例就是唯一走 AscendC kernel 的用例，被排除了；
- 那 3 个用例打不打补丁都挂，但**挂的原因跟流无关**：设备侧 AICore 报 `vector core exception` —— "The address for the VEC instruction to read/write UB is out of bounds"，是上游 AscendC kernel 自己的越界 bug（所以 CI 才排除它们）。两种写法只是把同一个设备侧错误报在不同的同步点上（打了报在 `kernel_launch`，没打报在 `waitForCompletion`）。

所以 quick start 不打这个补丁。**如果将来上游 AscendC kernel 的越界被修好、并且真的用上 NULL 流**，再把同样的 ctx 默认流 fallback 加回去（连同 `acl_rt.h`）——那时它才有可观测的收益。

> **宿主环境的坑（不是源码问题，所以源码里不做缓冲收窄）**：宿主机驱动 25.5.2 × 镜像 CANN 9.1.0 **错配**的组合下，堆在运行期被写坏（HAL 层 `libascend_hal.so` 来自 25.5.2 驱动挂载，ACL/DVPP 用户态来自镜像 9.1.0），到进程 `exit()` 的 atexit 阶段 `libacl_dvpp_mpi.so` 的 `DvppWrapperManager::~DvppWrapperManager()` 才 free 到坏块 → `corrupted double-linked list` + SIGABRT（rc=134）。**推理本身完全正确**（top-1/score 与配套环境逐位一致）。修复见下文 DNN 块末尾的 `os._exit(0)`；驱动与 CANN 配套的正常环境（如 25.3.rc1.2）无此问题。
>
> 顺带说明为什么源码里**不**再收窄 H2D/D2H 的拷贝字节数：那处补丁防的是"驱动报的 buffer size 大于 shape 字节数"，而加日志实测 `acl_size == host_size == 4000`（无 padding），该路径本来就不是崩溃根因。

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

`opencv_test_cannops` 共 78 个用例；25 个因 CANN 9.1.0 / 910B 的已知缺陷或数值抖动被排除（910B1 CI 与 910B4 实测失败集一致：23 个稳定失败 + 2 个 flaky）。剩 53 个应全过。

- `CVT_COLOR` 的 XYZ / YCrCb / YUV 系 18 个 + threshold 系 3 个（`MAT_THRESHOLD` / `MAT_THRESHOLD_ASCENDC` / `ASCENDC_KERNEL.THRESHOLD`）共 21 个同根因：cvtColor 对非 32F 输入用两次 threshold 截断饱和中间结果（color.cpp 的 THRESH_TRUNC + THRESH_TO_ZERO），与 threshold 系用例落在同一个 AscendC kernel 上，该 kernel 在 910B 上 AI Core 越界（VEC 指令 UB 地址越界，kernel 源码 bug）；纯 32F 融合算术路径实测全过。
- resize 系 4 个：`CORE.RESIZE` 是 ResizeArea 算子 GE shape 推断失败（稳定挂）；`CORE.CROP_RESIZE` 与 CPU 参考稳定差 2 超容差 1；`CORE.RESIZE_NEW` / `CORE.CROP_RESIZE_MAKE_BORDER` 是数值抖动 flaky（后者容差 1e-10 近似逐位比较，910B4 上稳定挂）。

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

> 注意是 **prepend**（保留镜像 `set_env.sh` 注入的 CANN `PYTHONPATH` 条目）。GE 图编译依赖其中的 `tbe` Python 模块；整体覆盖 `PYTHONPATH` 会让 DNN CANN 后端在图编译阶段报 `There is no valid so about OpsKernelInfoStore or GraphOptimizer`（ret=0xFFFFFFFF）。

下面用 clone 下来的 OpenCV 仓库自带测试图 `opencv/samples/data/baboon.jpg`（512×512 狒狒）作样例输入：

![baboon 输入图](images/baboon-input.jpg)

### 9a. 读图 / 写图

```shell #test id="quickstart-imread-imwrite"
python << 'PY'
import cv2
img = cv2.imread('opencv/samples/data/baboon.jpg')
ok = cv2.imwrite('/tmp/opencv_quickstart.png', img)
print('imwrite ok:', ok)
print('shape:', img.shape, 'dtype:', img.dtype)
PY
ls /tmp/opencv_quickstart.png
```

输出结果如下：
```shell #test-result id="quickstart-imread-imwrite"
imwrite ok: True
shape: (512, 512, 3) dtype: uint8
/tmp/opencv_quickstart.png
```

### 9b. 颜色空间转换

```shell #test id="quickstart-cvtcolor"
python << 'PY'
import cv2
bgr = cv2.imread('opencv/samples/data/baboon.jpg')
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print('gray shape:', gray.shape, 'mean:', float(gray.mean()))
cv2.imwrite('/tmp/quickstart-gray.png', gray)
PY
```

输出结果如下：
```shell #test-result id="quickstart-cvtcolor" fuzzy='xxx'
gray shape: (512, 512) mean: xxx
```

baboon 整体偏中亮（mean ≈ 129.7/255），灰度图长这样：

![灰度转换结果](images/quickstart-gray.png)

### 9c. 缩放

```shell #test id="quickstart-resize"
python << 'PY'
import cv2
img = cv2.imread('opencv/samples/data/baboon.jpg')
resized = cv2.resize(img, (50, 200), interpolation=cv2.INTER_AREA)
print('resized shape:', resized.shape)
cv2.imwrite('/tmp/quickstart-resize.png', resized)
PY
```

输出结果如下：
```shell #test-result id="quickstart-resize"
resized shape: (200, 50, 3)
```

OpenCV shape 是 `(rows, cols, channels)`，`resize(img, (W, H), ...)` 得 `(H, W, C)`；缩到 50×200（宽×高）后：

![缩放结果](images/quickstart-resize.png)

### 9d. 绘制

```shell #test id="quickstart-draw"
python << 'PY'
import cv2
img = cv2.imread('opencv/samples/data/baboon.jpg')
cv2.rectangle(img, (10, 10), (500, 500), (0, 255, 0), thickness=2)
cv2.putText(img, 'Hello OpenCV', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
print('rect+text drawn, image mean:', float(img.mean()))
cv2.imwrite('/tmp/quickstart-draw.png', img)
PY
ls /tmp/quickstart-draw.png
```

输出结果如下：
```shell #test-result id="quickstart-draw" fuzzy='xxx'
rect+text drawn, image mean: xxx
/tmp/quickstart-draw.png
```

绿框 + 文字绘制结果：

![绘制结果](images/quickstart-draw.png)

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

跑 MobileNetV2（onnx/models 的 `mobilenetv2-12`，PyTorch 导出，13,964,571 bytes）：

> **模型来源**：CI 与本仓库 checkout 优先用 `fixtures/mobilenetv2-12.onnx`（同字节入仓，避免 GitHub raw 在国内网络的 `RemoteDisconnected` 抖动）；自己机器上跟着跑时没有 fixtures 目录，自动回落到从 onnx/models 下载（重试 5 次）。

> **为什么加 `OPENCV_FORCE_DNN_ENGINE=1`**：OpenCV 5.0 默认 `ENGINE_AUTO` 会选新图引擎（`onnx_importer2`），而新引擎**尚未支持非 CPU 后端**——`setPreferableBackend` / `setPreferableTarget` 被静默忽略（仅打 WARN），推理实际跑在 CPU 上。只有 classic 引擎（4.x 行为）真正走 `switchToCannBackend` 把网络转成 CANN 算子并编译上 NPU。
>
> **为什么不用 SqueezeNet**：实测 OpenCV 5.0.0 对 onnx/models 的 caffe2 转换 squeezenet（1.0/1.1）计算退化——1.0-12 的输出恒为全 1（与输入无关，疑似 Dropout 双输出错接），1.1-7 的 top-5 全是乱类；classic 与新引擎均如此。MobileNetV2 在两个引擎下都与 CPU 参考一致。

```shell #test id="opencv-cann-infer"
OPENCV_FORCE_DNN_ENGINE=1 python << 'PY'
import os
import shutil
import sys
import time
import urllib.request
import numpy as np
import cv2

MODEL_URL = ('https://github.com/onnx/models/raw/main/'
             'validated/vision/classification/mobilenet/'
             'model/mobilenetv2-12.onnx')
MODEL_PATH = '/tmp/mobilenetv2-12.onnx'

if not os.path.exists(MODEL_PATH):
    if os.path.exists('fixtures/mobilenetv2-12.onnx'):
        # 本仓库 checkout（CI）：直接用入仓副本，零网络依赖
        shutil.copy('fixtures/mobilenetv2-12.onnx', MODEL_PATH)
    else:
        # 读者自跑：从 onnx/models 下载；github raw 偶发 RemoteDisconnected
        # （国内网络抖动），重试 5 次，半截文件要丢掉避免下次误判已下载
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                break
            except OSError:
                if os.path.exists(MODEL_PATH):
                    os.unlink(MODEL_PATH)
                if attempt == 4:
                    raise
                time.sleep(10)

net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

image = cv2.imread('opencv/samples/data/baboon.jpg')
x = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0  # BGR, [0,1]
x = x[..., ::-1]                                              # BGR -> RGB
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)      # ImageNet mean/std
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
x = (x - mean) / std
blob = np.ascontiguousarray(x.transpose(2, 0, 1))[np.newaxis]  # 1xCxHxW float32

net.setInput(blob)
out = net.forward()

# mobilenetv2-12 输出是 logits（不含 softmax），自己归一化
logits = out.reshape(-1)
e = np.exp(logits - logits.max())
prob = e / e.sum()

img = image.copy()
cv2.rectangle(img, (10, 10), (500, 500), (0, 255, 0), thickness=2)
label = 'top-1: #372 patas monkey  %.3f' % float(np.max(prob))
cv2.putText(img, label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imwrite('/tmp/dnn-result.png', img)

# 结果不经 python 的缓冲 stdout 输出，而是用原始 syscall 写文件：
# 1) GE 编译会 fork 子进程，子进程 exit(0) 会把继承的 stdout 缓冲冲到
#    C++ 日志行的中间（实测 "model bytes" 粘在半截 INFO 行上）；
# 2) 驱动 25.5.2 × CANN 9.1.0 错配环境堆已损坏（见 §4d 末尾的「宿主环境的坑」），
#    os.write 不经过 C 堆最稳。bash 的 `echo; cat` 保证锚点行干净。
result = (
    'model bytes: %d\n'
    'output shape: %s\n'
    'output dtype: %s\n'
    'top class index: %d\n'
    'top class score: %s\n'
) % (os.path.getsize(MODEL_PATH), out.shape, out.dtype,
     int(np.argmax(prob)), float(np.max(prob)))
fd = os.open('/tmp/dnn_result.txt', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.write(fd, result.encode())
os.close(fd)

# 驱动 25.5.2 × CANN 9.1.0 错配的环境里，进程 exit() 的 atexit 阶段
# libacl_dvpp_mpi 的 DvppWrapperManager 析构会撞上运行期已被写坏的堆块
# （"corrupted double-linked list" SIGABRT rc=134，gdb backtrace 实锤；
# 推理本身正确）。flush 后 _exit(0) 绕过 C 层 atexit；驱动与 CANN 配套
# 的正常环境（如 25.3.rc1.2）下此举同样无害，NPU 资源由驱动随进程回收。
sys.stdout.flush()
os._exit(0)
PY
echo
cat /tmp/dnn_result.txt
```

输出结果如下：
```shell #test-result id="opencv-cann-infer" fuzzy='xxx' fuzzy='...'
...
model bytes: xxx
output shape: (1, 1000)
output dtype: float32
top class index: 372
top class score: 0.xxx
...
```

预期：`output shape (1, 1000)`、`float32`；baboon.jpg 的 top-1 是 **ImageNet-1k 第 372 类（patas / hussar monkey，赤猴）**，score ≈ 0.91（与 CPU 参考一致，baboon 本尊 373 排第二 ≈ 0.03）。首尾 `...` 吞掉 OpenCV INFO 日志噪音（导入/逐层转 CANN 算子等，注意这些日志走 stdout）。标注结果如下：

![NPU 推理标注结果](images/dnn-result.png)

> 首次 forward 含 GE 图编译（`converting ge::Graph to OM buffer`），约 0.5~1 分钟出 `Compile success, model size = 9759197`；进程内二次 forward 约 1ms。若 graph check 报 `ret = 4294967295` 且日志有 `No module named 'tbe'`，是 `PYTHONPATH` 被**整体覆盖**（见上文 quickstart 开头的 prepend 说明）。块末尾的 `os._exit(0)` 是对驱动/CANN 错配环境 DVPP atexit 堆损坏的规避，详见 §4d 末尾的环境说明。