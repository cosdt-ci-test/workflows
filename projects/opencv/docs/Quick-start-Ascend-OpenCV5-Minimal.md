# 快速开始（OpenCV 5.x 最小补丁版）

一句话结论先摆出来：**OpenCV 5.x 没有「零补丁」路线**。原始 5.0.0 源码在 `WITH_CANN=ON` 下**编译就过不去**；把 `opencv_contrib` 排除掉之后，最小补丁集是 **2 组**（`cv::MatShape` 适配 + 窄 op 头），共 **25 个文件 / +34 行 / -5 行**。本文给出实测证据、两段可直接复制的补丁脚本，以及一套在 910B4 上跑通的 DNN NPU 验证。

本文是 [Quick-start-Ascend-NoPatch.md](Quick-start-Ascend-NoPatch.md)（4.14.0 零补丁）的 **5.x 对照版**：那篇证明「4.x 可以一行补丁都不打」，本文证明「5.x 不行，最少要多少」。

### 三段实测（同一台机、同一套参数，只有补丁在变）

实测环境：**Atlas 900 A2 PODc（910B4）/ Ubuntu 22.04.5 arm64 / CANN 9.2.0 / gcc 11.4.0 / Python 3.12.15 / CMake 3.22.1**，`CMAKE_BUILD_TYPE=Release`、`WITH_CANN=ON`、`BUILD_LIST=core,imgproc,imgcodecs,videoio,dnn,python3`、**不编 contrib**。

| 阶段 | 源码状态 | 结果 |
| --- | --- | --- |
| ① | **原始 5.0.0，零补丁** | ❌ **编译失败**：`cv::MatShape` 与上游 `CannConstOp(const uint8_t*, int, const std::vector<int>&, ...)` 接不上（`gemm_layer.cpp:517` / `:534`）；构建退出码 **2**，`libopencv_dnn.so` 产不出来 |
| ② | ① + 第 1 组（MatShape 适配，4 文件 / +7 -3） | ❌ **编译期全过、链接期炸**：`relocation truncated to fit: R_AARCH64_JUMP26`，6 个 protobuf 的 `.pb.cc` 全被截断 |
| ③ | ② + 第 2 组（窄 op 头，24 文件 / +27 -2） | ✅ **构建通过**（`.text` = 15,455,920 B）；**NPU 推理通过**：124 层转 CANN 算子、GE 图编译 1 次、top-1 372、score 0.9142662、**1 ms vs CPU 31 ms（40×）** |

三个阶段的源码树逐字节可核：阶段 ③ 的 25 个文件与已知可用的 5.0.0+补丁树**完全一致**（`diff` 全过），变更规模 `25 files changed, 34 insertions(+), 5 deletions(-)`。

## 0. 与同仓其它 OpenCV 文档的关系

| 文档 | OpenCV 版本 | 源码补丁 | 编 contrib | NPU 能覆盖到 |
| --- | --- | --- | --- | --- |
| [Quick-start-Ascend-NoPatch.md](Quick-start-Ascend-NoPatch.md) | 4.14.0 | **0 组** | 否 | 只有 `cv2.dnn` 的 CANN 后端 |
| **本文** | **5.0.0** | **2 组** | 否 | 同上（但**必须强制经典引擎**，见 §10.1） |
| [Quick-start-Ascend.md](Quick-start-Ascend.md) / [Quick-start-Ascend-OpenCV5.md](Quick-start-Ascend-OpenCV5.md) | 5.0.0 | 3 组 | 是 | DNN + `cv2.cann.*` 经典算子（23 个算子） |

本文比主文档少的**那一组**补丁（`OperatorRunner` 的 NULL 流 fallback）在 `opencv_contrib/modules/cannops/` 里 —— 本文**不编 contrib**，所以那一组用不上。换句话说：

> **5.x 只要不上 `cv2.cann` 经典算子，最小补丁就是 2 组；要上 `cv2.cann` 就是 3 组。**

选本文的理由：

- 你要用 **5.x 的 DNN**（新版 ONNX 覆盖、算子融合、经典引擎的行为差异），但不想维护 contrib 那侧的补丁；
- 你能接受「2 组、34 行」的本地改动，并且希望它们是**幂等**的（脚本重复跑会打印 `already patched` 并跳过）。

不选本文的理由：

- 你要的是「一行补丁都不打」→ 去 [4.14.0 那篇](Quick-start-Ascend-NoPatch.md)；
- 你要用 `cv2.cann.*` 加速经典算子 → 去主文档（3 组补丁 + 编 contrib）。

## 1. 结论一：原始 5.0.0 零补丁 = 编译失败（实测）

### 1.1 根因：上游 CANN 后端的 shape 签名在 4.11 → 5.0.0 之间一行没改

| | 4.14.0 | 5.0.0 |
| --- | --- | --- |
| `modules/dnn/src/op_cann.hpp` 的 md5 | `4034857fe5da06aad27b8ae264a04fc7` | `4034857fe5da06aad27b8ae264a04fc7` |
| 字节数 | 4446 | 4446 |
| `CannConstOp` 构造函数签名 | `CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name)` | **完全相同** |

**逐字节相同**（上游 CANN 后端这 5 个版本没动过一行）。而 OpenCV 5.0 把 dnn 层的 shape 换成了 `cv::MatShape`：

```cpp
// modules/core/include/opencv2/core/mat.hpp（5.0.0）
struct CV_EXPORTS_W_SIMPLE MatShape
{
    MatShape();
    explicit MatShape(size_t dims, const int* sizes=nullptr, DataLayout layout=DATA_LAYOUT_UNKNOWN, int C=0);
    explicit MatShape(size_t dims, int value, DataLayout layout=DATA_LAYOUT_UNKNOWN);
    explicit MatShape(int dims, int value, DataLayout layout=DATA_LAYOUT_UNKNOWN);
    explicit MatShape(const std::vector<int>& shape, DataLayout layout=DATA_LAYOUT_UNKNOWN, int C=0);
    ...
};
```

`cv::MatShape` 是 **`explicit`** 构造、**没有 `front()`**、也不能从 `std::vector<int>` 赋值 —— 于是 5.0 的 dnn 层代码（`shape(mat)` 现在返回 `cv::MatShape`）与上游 CANN 后端的 `std::vector<int>` 构造**接不上**。

> 4.14.0 也有 `cv::MatShape`（4.x 就有），但 4.x 的 dnn 层代码里 `shape(mat)` 返回的还是 `std::vector<int>`，所以两边正好错开、编译得过。**这不是 5.0 引入的 bug，是上游 CANN 后端从 4.11 起就没跟上 OpenCV 的类型演进。**

### 1.2 编译错误原文（实测）

对**未经任何修改**的 5.0.0 源码跑 `cmake --build . --target opencv_dnn`：

```
modules/dnn/src/layers/gemm_layer.cpp:534:43: error: no match for 'operator=' (operand types are 'cv::MatShape' and 'std::vector<int>')
  118 |     MatShape& operator = (const MatShape& shape);
      |     no known conversion for argument 1 from 'std::vector<int>' to 'const cv::MatShape&'
/usr/include/c++/11/ext/new_allocator.h:162:11: error: no matching function for call to 'cv::dnn::CannConstOp::CannConstOp(unsigned char*&, int, cv::MatShape, std::__cxx11::basic_string<char>)'
modules/dnn/src/layers/../op_cann.hpp:81:9: note: candidate: 'cv::dnn::CannConstOp::CannConstOp(const uint8_t*, int, const std::vector<int>&, const string&)'
      |     no known conversion for argument 3 from 'cv::MatShape' to 'const std::vector<int>&'
modules/dnn/src/layers/gemm_layer.cpp:517:60:   required from here
```

make 的第一个失败点：

```
gmake[3]: *** [modules/dnn/CMakeFiles/opencv_dnn.dir/build.make:1071: modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/convolution_layer.cpp.o] Error 1
gmake: *** [Makefile:576: opencv_dnn] Error 2
```

**构建退出码 = 2**，`libopencv_dnn.so` 根本没产出。

### 1.3 影响面：39 个引用 CANN 后端的 TU 里 5 个挂

把 5.0.0 里所有 `#include "op_cann.hpp"`（含 `net_impl.hpp` 间接引入）的翻译单元逐个单独语法检查，**5 个**文件因 `MatShape` 不匹配报错：

| 文件 | 报错点 |
| --- | --- |
| `modules/dnn/src/layers/convolution_layer.cpp` | `:703` 传 `cv::MatShape` 给 `CannConstOp` |
| `modules/dnn/src/layers/deconvolution_layer.cpp` | 同上 |
| `modules/dnn/src/layers/fully_connected_layer.cpp` | 同上 |
| `modules/dnn/src/layers/gemm_layer.cpp` | `:517` 传 `MatShape` + `:534` `MatShape = std::vector<int>` |
| `modules/dnn/src/layers/matmul_layer.cpp` | `:515`/`:516` `cv::MatShape` **没有 `front()`** |

其余 34 个 TU 不受影响 —— 它们要么不碰 `CannConstOp`，要么只用 `std::vector<int>` 的路径。

**所以第 1 组补丁只需要改 4 个文件**：`op_cann.hpp`（加重载声明）、`op_cann.cpp`（加重载转发）、`gemm_layer.cpp` / `matmul_layer.cpp`（修 2 处 `std::vector<int>` 赋值 + 2 处 `.front()`）。

## 2. 结论二：只打第 1 组（MatShape 适配）还不够

把第 1 组补丁打上、重新编译 —— **编译期全过了**（39 个 TU 一个不落），但**链接期炸**：

```
[ 95%] Linking CXX shared library ../../lib/libopencv_dnn.so
function.pb.cc:(.text.startup._GLOBAL__sub_I.00102_function.pb.cc+0x10): relocation truncated to fit: R_AARCH64_JUMP26 against symbol `google::protobuf::internal::AddDescriptorsRunner::AddDescriptorsRunner(google::protobuf::internal::DescriptorTable const*)' defined in .text._ZN6google8protobuf8internal20AddDescriptorsRunnerC2EPKNS1_15DescriptorTableE section in ../../3rdparty/lib/liblibprotobuf.a(generated_message_reflection.cc.o)
graph.pb.cc:... 同上
op_def.pb.cc:... 同上
tensor.pb.cc:... 同上
tensor_shape.pb.cc:... 同上
types.pb.cc:... 同上
versions.pb.cc:... 同上
collect2: error: ld returned 1 exit status
gmake: *** [Makefile:576: opencv_dnn] Error 2
```

**6 个 protobuf 的 `.pb.cc` 全被截断**，`libopencv_dnn.so` 产不出来（构建退出码 2）。

根因是 CANN 的 `all_ops.h`：它一次拉进 ~1500 个 op 的头，每个引用 `op_cann.hpp` 的 TU 都把这一整套内联函数展开一遍。aarch64 的 `R_AARCH64_CALL26/JUMP26` 分支指令只有 **±128 MiB** 的跨度，`libopencv_dnn.so` 的 `.text` 一旦超过这个窗口，链接器就报 `relocation truncated to fit`。

5.0.0 站在窗口外面很远的位置：

| | 4.14.0 | 5.0.0 |
| --- | --- | --- |
| dnn `layers/*.cpp`（不含 `cpu_kernels/`） | 62 | **116** |
| dnn `layers/cpu_kernels/*.cpp` | — | **14** |
| 引用 `op_cann.hpp` 的文件数（`.cpp` + `.hpp`） | **28** | **39** |
| dnn 源码量 | 3,068,750 B | **4,518,814 B（比 4.11.0 +50.6%，比 4.14.0 +47.3%）** |

`all_ops.h` 的体积是 **CANN 侧的常量**（与 OpenCV 版本无关），所以 `.text` 随「引用 TU 数 × 自身代码量」一起涨：4.14.0 站在「TU 只多 1 个、自身代码只多 2.3%」的位置，5.0.0 站在「TU 多 39%、自身代码多 47%」的位置 —— 后者把 `all_ops.h` 展开的体积顶出了窗口。

> **Release 救不了**：`CMAKE_BUILD_TYPE=Release`（`-O3`）只能把 `all_ops.h` 展开的代码压小，**压不进 ±128 MiB 窗口**。实测 5.0.0 用 Release + `all_ops.h` 一样链接失败（报错对象随链接顺序随机变 —— protobuf / arena 都可能是受害者，别顺着报错对象查根因）。

**唯一出路是缩头**：把 `op_cann.hpp` 里的 `all_ops.h` 换成窄头 `array_ops.h`，再给每个 layer TU 插它**实际用到**的那个窄 op 头。

## 3. 结论三：2 组补丁 = 通（实测）

| 补丁组 | 改法 | 涉及文件 | 行数 |
| --- | --- | --- | --- |
| 第 1 组 | `cv::MatShape` 适配 | 4 | +7 / -3 |
| 第 2 组 | 窄 op 头（`all_ops.h` → 按需窄头） | 24 | +27 / -2 |
| **合计** | | **25** | **+34 / -5** |

打完后：

```
[ 98%] Linking CXX shared library ../../lib/libopencv_dnn.so
-- Installing: /usr/local/opencv-cann-50/lib/libopencv_dnn.so.5.0.0
```

产物体积（910B4 / CANN 9.2.0 / gcc 11.4 / Release -O3）：

| | 4.14.0 零补丁（`all_ops.h`） | 5.0.0 最小补丁（窄 op 头） |
| --- | --- | --- |
| `.text` | 56,123,568 B（≈53.5 MiB） | **15,455,920 B（≈14.7 MiB）** |
| `Total` | 77,331,307 B | 21,863,824 B |
| 库文件大小 | 98,304,552 B | 28,953,192 B |

**5.0.0 用窄 op 头之后 `.text` 只有 4.14.0 零补丁的 1/3.6**（14.7 MiB vs 53.5 MiB）—— 因为窄头把 `all_ops.h` 那 ~1500 个 op 的内联代码整个省掉了，只留每个 TU 真正用到的 op。5.0.0 的 dnn 源码量比 4.14.0 大 47%，但**最终产物的 `.text` 反而更小**。

> 同一份 5.0.0 + 窄 op 头源码在 **Debug（-O0）** 下 `.text` = 31,641,392 B（≈30.2 MiB）、库文件 252,232,064 B —— Debug 的 `.text` 更大、调试信息更多，但**仍在 ±128 MiB 窗口内**。窄 op 头是 5.0 唯一能过链接这一关的路子，与优化级别无关。

## 4. 最小补丁集（两段脚本）

两段都是**幂等**的：重复跑会打印 `already patched` 并跳过，中途失败可以直接重跑。

### 组 1. `cv::MatShape` 适配

保持最小：**上游的 `std::vector<int>` 构造函数原样保留**，只**新增**一个 `cv::MatShape` 重载转发给它；再修 3 处已经编译不过的赋值。

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

预期：`(a)`/`(b)` 各 1 行打印，`(c)(d)` 打印 3 行（gemm 1 + matmul 2）。

### 组 2. 缩小 CANN op 头（避免 aarch64 链接炸）

`op_cann.hpp` 换成窄头 `array_ops.h`，再给每个 layer TU 插它**实际用到**的那个窄 op 头。

> **23 个窄头不能合并成一个「域头并集」**（那样补丁少 22 处，看着很美）：同一个 TU 的 `.text*` 实测 `all_ops.h` 是 361 KB~4 MB，域头并集是窄头的 5.6~9.4 倍，外推到全部 TU 仍远大于 128 MiB，链接照炸。

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
if new in s:
    print('(e) op_cann.hpp: array_ops.h already in place')
else:
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
    if all(f'built-in/op_proto/inc/{h}.h' in s for h in headers):
        print(f'(f) {rel}: already patched')
        continue
    assert ANCHOR in s, f'{rel}: op_cann.hpp include not found'
    assert 'built-in/op_proto/inc/' not in s, f'{rel}: partially patched'
    block = ANCHOR + '\n' + '\n'.join(
        f'#include "built-in/op_proto/inc/{h}.h"' for h in headers)
    p.write_text(s.replace(ANCHOR, block, 1))
    print(f'(f) {rel}: +{len(headers)} narrow header(s)')
PY
```

预期：`(e)` 1 行打印，`(f)` 23 个文件被改。

### 补丁是否真的落到位

```shell #test id="ocv5m-patches-applied"
grep -c "cv::MatShape" opencv/modules/dnn/src/op_cann.hpp
grep -c "array_ops.h" opencv/modules/dnn/src/op_cann.hpp
grep -rl "built-in/op_proto/inc/" opencv/modules/dnn/src/layers/ | wc -l
```

输出结果如下：
```shell #test-result id="ocv5m-patches-applied"
1
2
23
```

> `grep -c "cv::MatShape"` 是 **1**：上游的 `std::vector<int>` 声明不动，只多一行重载声明。

## 5. 前置条件

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
| 机器 | Atlas 900 A2 PODc（910B4） |
| OS | Ubuntu 22.04.5 arm64 |
| CANN toolkit | **9.2.0**（`/usr/local/Ascend/ascend-toolkit/latest`）；CI 镜像里是 9.1.0，两者的库目录布局差异见「桥接 CANN 的库目录」 |
| Python / CMake / GCC | 3.12.15 / 3.22.1 / 11.4.0 |
| OpenCV | **5.0.0**（源码，`WITH_CANN=ON`，`CMAKE_BUILD_TYPE=Release`，**2 组补丁**） |
| opencv_contrib | **不编**（本文不覆盖 `cv2.cann`）|

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

> 如果 `npu-smi` 不存在，按官方快速安装指南补装驱动。

检查 Python / CMake：

```shell #test id="ocv5m-check-toolchain"
python --version
cmake --version | head -n 1
```

输出结果如下：
```shell #test-result id="ocv5m-check-toolchain" fuzzy='xxx'
Python 3.12.xxx
cmake version xxx
```

加载 CANN 环境变量（每个用到 CANN 变量的块都要自己 `source`）：

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

## 6. 装编译依赖 + clone 源码

```shell #test-setup
sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list
apt-get update -qq && apt-get install -y -qq git build-essential cmake pkg-config zlib1g-dev libjpeg-dev libpng-dev libtiff-dev
git clone --depth 1 --branch 5.0.0 https://github.com/opencv/opencv.git
```

## 7. 桥接 CANN 的库目录

OpenCV 的 `OpenCVFindCANN.cmake` 在 `${CANN_INSTALL_DIR}/{acllib,lib64,compiler/lib64}/` 下找 ACL 库：

| 找什么 | 期望位置 |
| --- | --- |
| `libascendcl.so` | `acllib/lib64/` |
| `libacl_op_compiler.so`、`libacl_dvpp_mpi.so` | `lib64/` |
| `libgraph.so`、`libge_compiler.so`、`libgraph_base.so` | `compiler/lib64/` |

这三个位置在 CANN 里都不叫这三个名字，所以要补软链。**统一指向「完整版 CANN 库目录」**：

- CANN 9.1.x：`<CANN>/lib64` 不存在，完整库目录是 `<CANN>/aarch64-linux/lib64`；
- CANN 9.2.x：`<CANN>/lib64` 已经是指向 `aarch64-linux/lib64` 的软链（同一批库）。

注意 **不要**指向 `<CANN>/aarch64-linux/devlib/linux/aarch64/` —— 那是裁剪过的设备侧运行时（`libgraph.so` 只有 198 KB，少了 `ge::AscendString` 的比较运算符，链接期报 `undefined reference to 'ge::AscendString::operator<(...)'`）。

先确认本机的 `<CANN>/lib64` 是完整版：

```shell #test id="ocv5m-cann-lib-layout"
CANN_DIR=/usr/local/Ascend/ascend-toolkit/latest
ls -l "$CANN_DIR/lib64/libascendcl.so" "$CANN_DIR/lib64/libgraph.so"
nm -D --defined-only "$CANN_DIR/lib64/libgraph.so" | grep -c '_ZNK2ge12AscendStringlt'
```

输出结果如下：
```shell #test-result id="ocv5m-cann-lib-layout" fuzzy='xxx'
xxx
xxx
1
```

补 symlink（`acllib` / `compiler` 要先是目录，所以先 `mkdir`）：

```shell #test-setup
CANN_DIR=/usr/local/Ascend/ascend-toolkit/latest
# CANN 9.1 没有 <CANN>/lib64，补一条指向 aarch64-linux/lib64 的软链；
# CANN 9.2 已经有了（本身就是 aarch64-linux/lib64 的软链），不动它。
[ -e "$CANN_DIR/lib64" ] || ln -sfn "$CANN_DIR/aarch64-linux/lib64" "$CANN_DIR/lib64"
mkdir -p "$CANN_DIR/acllib" "$CANN_DIR/compiler"
ln -sfn "$CANN_DIR/lib64" "$CANN_DIR/acllib/lib64"
ln -sfn "$CANN_DIR/lib64" "$CANN_DIR/compiler/lib64"
ls -l "$CANN_DIR/acllib/lib64/libascendcl.so" "$CANN_DIR/lib64/libacl_op_compiler.so" "$CANN_DIR/compiler/lib64/libgraph.so"
```

> `ln -sfn` 到已存在的软链是「改指向」而不是「往目录里塞链接」，所以重复执行是安全的。

## 8. 配置并编译

CANN 后端开关 `WITH_CANN=ON`（不是 `BUILD_CANN`）；`-DOPENCV_DOWNLOAD_MIRROR_ID=gitcode` 让 cmake configure 阶段走 gitcode 镜像拉 ADE / IPPICV / TBB。

**三个和 4.14 零补丁版不同的地方**：

- `CMAKE_BUILD_TYPE=Release` —— 5.0 用窄 op 头之后 `.text` 已经压到 14.7 MiB，Release 进一步把编译时间也压下来（实测 dnn 模块从 ~50 分钟降到 ~5 分钟）；
- `BUILD_LIST` 里**没有** `cannops` —— 不编 `opencv_contrib`，所以没有 contrib 侧的 NULL 流补丁（那组补丁是第 3 组，本文用不上）；
- `-DBUILD_opencv_apps=OFF` —— OpenCV 自带的 `apps/model-diagnostics` 是个**可执行文件**，它要直接链 CANN 的 `libge_compiler.so` / `libgraph.so` / `libgraph_base.so`，而这三个库自身还有一堆没被链进来的依赖（`ge::*`、`ascend_private::protobuf::*`、`absl::*`…）。共享库链接允许留未定义符号（所以 `libopencv_dnn.so` 没事），可执行文件链接不允许 → **链接期报几百条 `undefined reference` 并让整个 `all` 目标失败**。

安装前缀用 `/usr/local/opencv-cann-50`（4.14 零补丁版用 `/usr/local/opencv-cann-414`）：两套 OpenCV 可能落在同一台自托管 runner 上，装到同一个前缀会互相覆盖。

```shell #test id="ocv5m-opencv-cmake-configure"
cd opencv
mkdir -p build && cd build
set -o pipefail
PY_INC=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
PY_LIB=$(python3 -c 'import sysconfig,os,sys; print(os.path.join(sysconfig.get_config_var("LIBDIR"), "libpython%d.%d.so" % sys.version_info[:2]))')
test -f "$PY_INC/Python.h" || { echo "Python.h not found under $PY_INC"; exit 1; }
test -f "$PY_LIB" || { echo "libpython not found at $PY_LIB"; exit 1; }
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-cann-50 \
      -DWITH_CANN=ON \
      -DCANN_INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest \
      -DBUILD_opencv_world=OFF \
      -DBUILD_opencv_apps=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_LIST=core,imgproc,imgcodecs,videoio,dnn,python3 \
      -DSOC_VERSION=ascend910b1 \
      -DBUILD_opencv_python3=ON \
      -DBUILD_opencv_python_bindings_generator=ON \
      -DPYTHON_INCLUDE_DIR="$PY_INC" \
      -DPYTHON_LIBRARY="$PY_LIB" \
      -DOPENCV_ENABLE_NONFREE=OFF \
      -DOPENCV_DOWNLOAD_MIRROR_ID=gitcode \
      .. | tee /tmp/opencv-configure.log
```

> Python 路径用 `sysconfig` 现场取，不写死版本号目录：CI 镜像是 `/usr/local/python3.12.13`，验证机是 `/usr/local/python3.12.15`，两条路径都由 `sysconfig` 给出。

输出结果如下：
```shell #test-result id="ocv5m-opencv-cmake-configure" fuzzy='xxx' fuzzy='...'
-- The CXX compiler identification is GNU xxx
...
--   CANN:xxx YES
...
-- Configuring done
-- Generating done
-- Build files have been written to: xxx/build
```

编译 + 安装一体（`--target install`）。**并行度别开高**：窄 op 头省掉了 `all_ops.h` 那 ~4 MB/TU 的内联展开，但 39 个 TU 同时编译仍会吃掉可观内存（容器 cgroup 配额 32 GiB 时 `-j` 超过 6 会被 OOM killer 干掉 `cc1plus`）。并行度做成可覆盖的环境变量：默认 `--parallel 2`（保守），CI 机器内存更富裕时可以调高。

```shell #test-setup
cd opencv/build
cmake --build . --target install --parallel "${OPENCV_BUILD_JOBS:-2}"
```

> 只想「一次跑完」的话也可以把 `OPENCV_BUILD_JOBS` 设成 `$(nproc)`；但先看一眼容器的内存配额（`cat /sys/fs/cgroup/memory/memory.limit_in_bytes`，v2 是 `/sys/fs/cgroup/memory.max`），超配额就必然 OOM。

## 9. 校验 build 产物

```shell #test id="ocv5m-opencv-verify-build"
ls -l /usr/local/opencv-cann-50/lib/libopencv_core.so.5.0.0 /usr/local/opencv-cann-50/lib/libopencv_dnn.so.5.0.0
```

输出结果如下：
```shell #test-result id="ocv5m-opencv-verify-build" fuzzy='xxx'
xxx
xxx
```

看一眼 dnn 库的 `.text` —— 这是「窄 op 头」那组补丁的成果：

> 实测（5.0.0 + CANN 9.2.0 / 910B4 / gcc 11.4 / Release -O3，2 组补丁）：`libopencv_dnn.so.5.0.0` 的 `.text` = **15,455,920 B（≈14.7 MiB）**。对照 4.14.0 零补丁（`all_ops.h`）的 56,123,568 B（53.5 MiB）—— **窄 op 头让 5.0 的 `.text` 只有 4.14 零补丁的 1/3.6**，离 ±128 MiB 的链接窗口有 ~113 MiB 余量。

```shell #test id="ocv5m-opencv-dnn-text-size"
size -A /usr/local/opencv-cann-50/lib/libopencv_dnn.so.5.0.0 | grep -E '^\.text|^Total'
```

输出结果如下：
```shell #test-result id="ocv5m-opencv-dnn-text-size" fuzzy='xxx' fuzzy='...'
.text xxx
...
Total xxx
```

## 10. NPU 上跑一次 DNN 推理（**必须强制经典引擎**）

### 10.1 5.0 的默认引擎不上 NPU —— 这是 5.x 相对 4.x 最大的行为差异

OpenCV 5.0 重写了 DNN 引擎（带类型的算子图、形状推断、算子融合、动态 shape）。新引擎**不支持任何非 CPU 后端**：这时再调 `setPreferableBackend(CANN)` / `setPreferableTarget(NPU)` **既不报错也不生效**，只打 2 条 WARN 就把模型留在 CPU：

```
[ WARN:0] global net_impl_backend.cpp:297 Back-ends are not supported by the new graph engine for now
[ WARN:0] global net_impl_backend.cpp:345 Targets are not supported by the new graph engine for now
```

实测这一路：**0 次 GE 图编译、0 层转 CANN 算子**，跑出来的 score 是 CPU 的值（0.9148015），而经典引擎 + CANN 是 0.9142662 —— **两个分数不同，就说明模型根本没上卡**。

所以本文统一用环境变量 `OPENCV_FORCE_DNN_ENGINE=1`（对进程里所有 net 生效）；等价的写法是 `cv2.dnn.readNetFromONNX(path, cv2.dnn.ENGINE_CLASSIC)`。

### 10.2 跑起来

```shell #test-setup
cat > /tmp/ocv5m_infer.py <<'PY'
import ctypes, os, shutil, sys, time, urllib.request
import numpy as np
import cv2

# CANN 后端的关键进度日志全是 CV_LOG_INFO，而 Release 构建（带 NDEBUG）的
# 默认日志级别是 WARNING —— modules/core/src/logger.cpp 的
#   m_defaultUnconfiguredGlobalLevel = m_isDebugBuild ? LOG_LEVEL_INFO : LOG_LEVEL_WARNING
# 而 m_isDebugBuild 在 NDEBUG 下是 false。所以不显式抬到 INFO，
# 下面要断言的三行 CANN 标记一行都不会出现。
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_INFO)

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

# 两个 net 实例：NPU 那份走 CANN 后端，CPU 那份用默认引擎。
# 必须分开建 —— modules/dnn/src/net_cann.cpp 的 initBackend() 开头就是
#   if (netWasConverted && netWasAllocated) return;
# 同一个 net 上第二次 forward 直接复用已编译的 OM，不会再走一遍后端初始化。
net_npu = cv2.dnn.readNetFromONNX(MODEL_PATH)
net_npu.setPreferableBackend(cv2.dnn.DNN_BACKEND_CANN)
net_npu.setPreferableTarget(cv2.dnn.DNN_TARGET_NPU)

net_cpu = cv2.dnn.readNetFromONNX(MODEL_PATH)  # 不设 backend/target => CPU

image = cv2.imread('opencv/samples/data/baboon.jpg')
x = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0  # BGR, [0,1]
x = x[..., ::-1]                                              # BGR -> RGB
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)      # ImageNet mean/std
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
x = (x - mean) / std
blob = np.ascontiguousarray(x.transpose(2, 0, 1))[np.newaxis]  # 1xCxHxW float32

net_npu.setInput(blob)
net_cpu.setInput(blob)

# 捕获 OpenCV 的 INFO 日志。它们写进 C++ 的 std::cout，行首还带时间戳 +
# 绝对编译路径 + 函数名，没法直接拿来断言 —— 所以这里把 fd 1 临时改到文件，
# 跑完 forward 再刷一次缓冲取回来，最后只把「归一化后的标记行」写进结果文件。
LOG_PATH = '/tmp/cv_cann_log.txt'
saved_fd = os.dup(1)
log_fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(log_fd, 1)
try:
    out = net_npu.forward()          # 冷启：含 GE 图编译（0.5~1 分钟）
    t0 = time.perf_counter()
    out = net_npu.forward()          # 复用已编译的 OM —— 这才是纯推理耗时
    npu_ms = (time.perf_counter() - t0) * 1000.0
    out_cpu = net_cpu.forward()      # CPU 对照：先热身一次，让引擎把懒初始化做完
    t0 = time.perf_counter()
    out_cpu = net_cpu.forward()
    cpu_ms = (time.perf_counter() - t0) * 1000.0
finally:
    ctypes.CDLL(None).fflush(None)
    os.dup2(saved_fd, 1)
    os.close(log_fd)
    os.close(saved_fd)

cann_log = open(LOG_PATH, 'r', errors='replace').read()


def _fail(msg):
    sys.stderr.write(msg + '\n')
    fd = os.open('/tmp/dnn_result.txt', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(fd, (msg + '\n').encode())
    os.close(fd)
    sys.exit(1)


# 「这次真的上了 NPU」的硬证据 —— top-1=372 证明不了：同一份构建走默认
# CPU 引擎也是 372。层数也一起数：5.0 经典引擎实测 103 层转 CANN 算子。
CANN_MARKERS = (
    'DNN/CANN: converting ge::Graph to OM buffer',  # 逐层转完，开始编 OM
    'Compile success, model size = ',               # GE 图编译成功
    'DNN/CANN: OM buffer size = ',                  # OM 模型装进 CannNet
)
missing = [m for m in CANN_MARKERS if m not in cann_log]
if missing:
    sys.stderr.write('---- captured OpenCV log (tail 40) ----\n')
    sys.stderr.write('\n'.join(cann_log.splitlines()[-40:]) + '\n')
    _fail('CANN backend markers missing: %r' % (missing,))

n_layers = cann_log.count('converting layer')
if n_layers < 100:
    _fail('too few cann layers converted: %d (expected >= 100)' % n_layers)


def _top1(out_blob):
    logits = out_blob.reshape(-1)
    e = np.exp(logits - logits.max())
    prob = e / e.sum()
    return int(np.argmax(prob)), float(np.max(prob))


idx_npu, score_npu = _top1(out)
idx_cpu, score_cpu = _top1(out_cpu)

# NPU 与 CPU 必须给出同一个答案 —— 这是「加速器算对了」唯一的硬判据。
if idx_npu != idx_cpu:
    _fail('NPU vs CPU top-1 mismatch: npu=#%d cpu=#%d' % (idx_npu, idx_cpu))
if abs(score_npu - score_cpu) > 0.01:
    _fail('NPU vs CPU score mismatch: npu=%.6f cpu=%.6f (tol 0.01)'
          % (score_npu, score_cpu))
if out.shape != out_cpu.shape or out.dtype != out_cpu.dtype:
    _fail('NPU vs CPU output mismatch: %s/%s vs %s/%s'
          % (out.shape, out.dtype, out_cpu.shape, out_cpu.dtype))

img = image.copy()
cv2.rectangle(img, (10, 10), (500, 500), (0, 255, 0), thickness=2)
label = 'top-1: #%d patas monkey  %.3f' % (idx_npu, score_npu)
cv2.putText(img, label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imwrite('/tmp/dnn-result.png', img)

# 结果不经 python 的缓冲 stdout 输出，而是用原始 syscall 写文件：
# GE 图编译会 fork 子进程，子进程 exit(0) 会把继承的 stdout 缓冲冲到
# C++ 日志行的中间（实测 "model bytes" 粘在半截 INFO 行上）。
lines = ['cann marker: %s' % m for m in CANN_MARKERS]
lines += [
    'cann layers converted: %d' % n_layers,
    'model bytes: %d' % os.path.getsize(MODEL_PATH),
    'output shape: %s' % (out.shape,),
    'output dtype: %s' % out.dtype,
    'top class index: %d' % idx_npu,
    'top class score: %s' % score_npu,
    'cpu top class index: %d' % idx_cpu,
    'cpu top class score: %s' % score_cpu,
    'npu forward ms: %d' % round(npu_ms),
    'cpu forward ms: %d' % round(cpu_ms),
    'npu/cpu time ratio: %0.2f' % (cpu_ms / npu_ms if npu_ms > 0 else 0.0),
]
result = '\n'.join(lines) + '\n'
fd = os.open('/tmp/dnn_result.txt', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.write(fd, result.encode())
os.close(fd)
sys.stdout.flush()
os._exit(0)
PY
```

跑起来（`OPENCV_FORCE_DNN_ENGINE=1` 是 5.0 的必需项）：

```shell #test-setup
# PYTHONPATH 必须 prepend，不能覆盖 —— 整体覆盖掉 CANN 的 tbe 路径，
# DNN CANN 后端会在图编译阶段报 "There is no valid so about
# OpsKernelInfoStore or GraphOptimizer"（ret=0xFFFFFFFF）。
export PYTHONPATH=/usr/local/opencv-cann-50/lib/python3.12/site-packages:${PYTHONPATH:-}
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

```shell #test id="ocv5m-cann-infer"
OPENCV_FORCE_DNN_ENGINE=1 python /tmp/ocv5m_infer.py > /tmp/ocv5m_cann.log 2>&1 || { echo "classic-engine inference failed"; tail -n 30 /tmp/ocv5m_cann.log; exit 1; }
cat /tmp/dnn_result.txt
```

输出结果如下：
```shell #test-result id="ocv5m-cann-infer" fuzzy='xxx' fuzzy='...'
cann marker: DNN/CANN: converting ge::Graph to OM buffer
cann marker: Compile success, model size = xxx
cann marker: DNN/CANN: OM buffer size = xxx
cann layers converted: xxx
model bytes: 13964571
output shape: (1, 1000)
output dtype: float32
top class index: 372
top class score: xxx
cpu top class index: 372
cpu top class score: xxx
npu forward ms: xxx
cpu forward ms: xxx
npu/cpu time ratio: xxx
...
```

预期：`output shape (1, 1000)`、`float32`；baboon.jpg 的 top-1 是 **ImageNet-1k 第 372 类（patas / hussar monkey，赤猴）**。**实测（5.0.0 + 2 组补丁 + CANN 9.2.0 / 910B4 / 经典引擎 + CANN/NPU）**：

| | 实测值 |
| --- | --- |
| `top class score`（NPU） | **0.9142661690711975** |
| `cpu top class score`（同一次运行的 CPU 对照） | **0.9148014783859253** |
| `cann layers converted` | **124** |
| `npu forward ms`（热启） | **1 ms** |
| `cpu forward ms` | **31 ms** |
| `npu/cpu time ratio` | **40.04×** |
| 冷启 forward（含 GE 图编译） | **36.9 s** |

NPU 与 CPU 的 score 末位差 ~5e-4 —— 同一张图、同一份权重在两个引擎上的正常浮点差异，远小于脚本的 0.01 容差。

`cann layers converted` 是 5.0 经典引擎下的实测值（MobileNetV2-12 是 105 个 ONNX 节点，展开成 124 个 dnn 层），脚本断言「≥ 100」而不是硬编码，因为这个数随 CANN 小版本可能变。

> **首次 forward 含 GE 图编译**（`converting ge::Graph to OM buffer`），实测 36.9 s；进程内二次 forward 约 1ms。`npu forward ms` / `cpu forward ms` 量的是**热启** forward，脚本**不对它们设阈值**（CI runner 的 CPU 型号和核数不固定）。

### 10.3 怎么判断「这次真的上了 NPU」

三条硬证据，缺一不可：

1. `DNN/CANN: converting ge::Graph to OM buffer` 出现 **1** 次 —— GE 图编译真的发生了；
2. `converting layer` 出现 **≥ 100** 次（MobileNetV2-12 实测 **124**）—— 模型的每一层都被转成了 CANN 算子（不是「部分层在 NPU、部分在 CPU」）；
3. NPU 与 CPU 的 top-1 **相同**、score 差 ≤ 0.01 —— 加速器算对了。

光看「推理没报错」或「top-1 = 372」**证明不了上卡**：默认引擎下这两个现象和纯 CPU 完全一样（score 0.9148015）。

## 11. 已知边界

- **5.x 不存在零补丁路线**：原始 5.0.0 源码在 `WITH_CANN=ON` 下**编译失败**（§1），最少 2 组补丁 / 25 个文件 / +34 行。想要零补丁请用 [4.14.0 那篇](Quick-start-Ascend-NoPatch.md)。
- **5.0 的招牌特性上不了 NPU**：新图引擎、ONNX Runtime 后端、原生 LLM/VLM 在昇腾上目前都只能 CPU。要用 NPU 推理，必须 `OPENCV_FORCE_DNN_ENGINE=1` 走经典引擎（§10.1）。
- **经典引擎 + CANN 后端的能力边界**（与 4.x 相同，不是 5.0 引入的）：只吃 **fp32**（连 fp16 都不行）、**拒绝 int8 量化网**（5.0 还把 `Net quantize` 入口删了）、**不支持的层回落 CPU 而不是报错**（所以「上了 NPU」≠「全图在 NPU」）、后端/目标不可再切（`DNN: Can't switch backend from CANN to other`）。
- **本文不编 contrib**：`cv2.cann.*` 那 23 个经典算子（`cvtColor` / `threshold` / 算术 / `resize` / `crop` …）在本文的路线里**不存在**。要用它们请走主文档（3 组补丁）。
- **`all_ops.h` 链接窗口是硬约束**：aarch64 的 `R_AARCH64_CALL26/JUMP26` 只有 ±128 MiB 跨度，缩 `.text` 只是缓解、不是保证；5.0 上过这一关只有窄 op 头一条路。
- **补丁只对 5.0.x 成立**：`cv::MatShape` 是 5.0 引入的类型，4.x 源码树里根本没有它；上游一旦出 5.1/6.0，§4 的脚本会断言失败（这是设计好的失败方式，不是 bug）。
- **CANN 9.2 的裁剪版 `libgraph.so`**：把 CANN 软链指向 `<CANN>/lib64` 而不是 `<CANN>/aarch64-linux/devlib/linux/aarch64/`，否则链接期报 `undefined reference to 'ge::AscendString::operator<(...)'`。这条与 OpenCV 版本无关。
- **`videoio` 编了但没有编解码器**：源码构建默认不带 FFmpeg/GStreamer；本文未把视频写入列为必过项。
