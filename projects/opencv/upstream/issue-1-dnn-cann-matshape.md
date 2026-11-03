# 上游 issue 草稿 1/2：5.x 的 CANN 后端编译不过（MatShape 适配）

> 状态：**待提交**。目标仓库 `opencv/opencv`，base 分支 `5.x`。
> 关联 PR 草稿见 `pr-1-dnn-cann-matshape.md`（分支 `dnn_cann_matshape`，fork：
> `Devin-DC-Huang/opencv`）。
> 上游要求「一个 PR 对应一个 issue」，所以组 1（编译错）与组 2（链接错）分两个 issue。

---

## Title

```
dnn (CANN): the CANN backend does not compile with cv::MatShape
```

## Body

```markdown
### Problem

The CANN backend of the DNN module does not compile any more since 5.0.0.

Building `opencv_dnn` with `HAVE_CANN` defined stops in the first layer that
passes a `MatShape` to `CannConstOp`:

```
/usr/include/c++/11/ext/new_allocator.h:162:11: error: no matching function for call to 'cv::dnn::CannConstOp::CannConstOp(unsigned char* const&, int, cv::MatShape, std::__cxx11::basic_string<char>)'
  162 |         { ::new((void *)__p) _Up(std::forward<_Args>(__args)...); }
      |           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In file included from modules/dnn/src/layers/convolution_layer.cpp:50:
modules/dnn/src/op_cann.hpp:81:9: note: candidate: 'cv::dnn::CannConstOp::CannConstOp(const uint8_t*, int, const std::vector<int>&, const string&)'
   81 |         CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);
      |         ^~~~~~~~~~~
modules/dnn/src/op_cann.hpp:81:83: note:   no known conversion for argument 3 from 'cv::MatShape' to 'const std::vector<int>&'
   81 |         CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);
      |                                                           ~~~~~~~~~~~~~~~~~~~~~~~~^~~~~
modules/dnn/src/op_cann.hpp:78:11: note: candidate: 'cv::dnn::CannConstOp::CannConstOp(const cv::dnn::CannConstOp&)'
modules/dnn/src/op_cann.hpp:78:11: note: candidate expects 1 argument, 4 provided
modules/dnn/src/op_cann.hpp:78:11: note: candidate: 'cv::dnn::CannConstOp::CannConstOp(cv::dnn::CannConstOp&&)'
modules/dnn/src/op_cann.hpp:78:11: note: candidate expects 1 argument, 4 provided
gmake[3]: *** [modules/dnn/CMakeFiles/opencv_dnn.dir/build.make:1029: modules/dnn/CMakeFiles/opencv_dnn.dir/src/layers/convolution_layer.cpp.o] Error 1
```

(The `new_allocator.h` frame is where `std::make_shared` instantiates the
constructor; the callers are the `shape()` call sites listed below.)

Two more shapes of the same problem show up in the other affected layers:

```
modules/dnn/src/layers/gemm_layer.cpp:717:43: error: no match for 'operator=' (operand types are 'cv::MatShape' and 'std::vector<int>')
/usr/include/c++/11/ext/new_allocator.h:162:11: error: no matching function for call to 'cv::dnn::CannConstOp::CannConstOp(unsigned char*&, int, cv::MatShape, std::__cxx11::basic_string<char>)'
/usr/include/c++/11/ext/new_allocator.h:162:11: error: no matching function for call to 'cv::dnn::CannConstOp::CannConstOp(unsigned char*&, int, cv::MatShape&, std::__cxx11::basic_string<char>)'
```

and, in `matmul_layer.cpp`, `MatShape` has no `front()`:

```
modules/dnn/src/layers/matmul_layer.cpp:773:53: error: 'struct cv::MatShape' has no member named 'front'
  773 |                 if (real_ndims_C == 1 && bias_shape.front() != 1) {
      |                                                     ^~~~~
modules/dnn/src/layers/matmul_layer.cpp:774:62: error: 'struct cv::MatShape' has no member named 'front'
  774 |                     bias_shape = std::vector<int>{bias_shape.front()};
      |                                                              ^~~~~
modules/dnn/src/layers/matmul_layer.cpp:774:69: error: no matching function for call to 'std::vector<int>::vector(<brace-enclosed initializer list>)'
  774 |                     bias_shape = std::vector<int>{bias_shape.front()};
      |                                                                     ^
```

The 6 translation units that fail are `convolution_layer`, `deconvolution_layer`,
`fully_connected_layer`, `gemm_layer`, `layer_norm` and `matmul_layer`.

### The released 5.0.0 is affected as well

The `MatShape` structure was introduced by #26056 ("New dnn engine"), which
landed before the 5.0.0 release, so 5.0.0 already fails to build the CANN
backend. The failing sites in the 5.0.0 tag are these five:

| file | line | problem |
|---|---|---|
| `modules/dnn/src/layers/convolution_layer.cpp` | 703 | `shape(w_mat)` passed to `const std::vector<int>&` |
| `modules/dnn/src/layers/deconvolution_layer.cpp` | 1012 | `shape(mat_w)` passed to `const std::vector<int>&` |
| `modules/dnn/src/layers/fully_connected_layer.cpp` | 728 | `shape(weightsMat)` passed to `const std::vector<int>&` |
| `modules/dnn/src/layers/gemm_layer.cpp` | 534 | `shape_C = std::vector<int>{dim};` -- `shape_C` is a `MatShape` |
| `modules/dnn/src/layers/matmul_layer.cpp` | 515 | `bias_shape.front()` -- `MatShape` has no `front()` |

### Root cause

`cv::MatShape` used to be an alias of `std::vector<int>`:

```cpp
typedef std::vector<int> MatShape;
```

It is now a dedicated structure (introduced by #26056, "New dnn engine"), and
`Mat::shape()` returns it:

```cpp
CV_EXPORTS MatShape Mat::shape(int i=-1) const;
```

`MatShape` mimics the `std::vector<int>` API, but the two conversions the old
alias provided are gone:

* `MatShape(const std::vector<int>&, ...)` is `explicit`, so
  `std::vector<int>` does **not** convert to `MatShape` implicitly;
* there is no conversion `MatShape` -> `std::vector<int>` at all.

`CannConstOp` still takes the shape as `const std::vector<int>&`:

```cpp
CannConstOp(const uint8_t* data, const int dtype, const std::vector<int>& shape, const std::string& name);
```

while 7 of its call sites pass a `MatShape` returned by `shape()`:

| file | call |
|---|---|
| `modules/dnn/src/layers/convolution_layer.cpp` | `shape(w_mat)` |
| `modules/dnn/src/layers/deconvolution_layer.cpp` | `shape(mat_w)` |
| `modules/dnn/src/layers/fully_connected_layer.cpp` | `shape(weightsMat)` |
| `modules/dnn/src/layers/gemm_layer.cpp` | `shape(B)` |
| `modules/dnn/src/layers/layer_norm.cpp` | `shape(scale_mat)`, `shape(bias_mat)` |
| `modules/dnn/src/layers/matmul_layer.cpp` | `shape(B)` |

The other 25 call sites pass a shape local: 23 of them are a
`std::vector<int>`, and 2 are already a `MatShape` obtained from `shape()`
(`shape_C` in gemm, `bias_shape` in matmul). The 23 still match the current
signature, but they have to follow the parameter type once it changes.

Two more places break as soon as the shape type is unified:

* `modules/dnn/src/layers/gemm_layer.cpp` assigns a `std::vector<int>` to a
  `MatShape`:
  ```cpp
  auto shape_C = shape(mat_C);          // MatShape
  shape_C = std::vector<int>{dim};       // no implicit conversion -> error
  ```
* `modules/dnn/src/layers/matmul_layer.cpp` uses `front()`, which `MatShape`
  does not provide:
  ```cpp
  if (real_ndims_C == 1 && bias_shape.front() != 1) {
      bias_shape = std::vector<int>{bias_shape.front()};
  }
  ```

### How to reproduce

```
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CANN=ON \
      -DCANN_INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest \
      -DBUILD_LIST=core,imgproc,imgcodecs,videoio,dnn \
      -DSOC_VERSION=ascend910b1 ..
cmake --build . --target opencv_dnn
```

Tested with CANN 9.2.0 on an Atlas 900 A2 PODc (910B4), gcc 11.4.0.
`CannConstOp` is declared in `modules/dnn/src/op_cann.hpp`, which is pulled in
by every DNN translation unit that includes `net_impl.hpp` (the header itself
is guarded by `#ifdef HAVE_CANN`, so the CANN headers are only reached when
CANN is enabled). 104 translation units of the DNN module are affected, and
the build fails in the ones that use `CannConstOp`.

### Suggested fix

Take the shape as `const MatShape&` in `CannConstOp` and turn the shape locals
that feed it into `MatShape` as well:

* `modules/dnn/src/op_cann.hpp`, `modules/dnn/src/op_cann.cpp` — the
  declaration and the definition of `CannConstOp`;
* 13 layer files — the 15 shape locals plus 2 assignments, and
  `bias_shape.front()` -> `bias_shape[0]`.

On 5.x there is one more site that 5.0.0 does not have: `matmul_layer.cpp`
passes an inline `std::vector<int>{1}` to `CannConstOp`, which becomes
`MatShape{1}`.

`CannConstOp` is internal to the DNN module (every caller is in-tree), so
changing the parameter type is not an API break.

### Why CI does not catch this

`opencv/opencv` has no CANN job: `PR-5.x.yaml` runs on GitHub-hosted runners,
where the CANN toolkit cannot be installed, and the `ARM64` job does not pass
`-DWITH_CANN=ON`, so `HAVE_CANN` is never defined and the CANN sources are not
compiled at all. The same breakage was reported and fixed for 4.x in #26175 /
#26631 (fixed by #26632, "dnn: Fix CANN build"); the 5.x shape change needs the
same treatment.

This is the first of two independent problems in the CANN backend; the
aarch64 link failure caused by the CANN `all_ops.h` header is a separate
issue and will be filed separately.

There is also a third, independent problem, which this issue does not address:
on 5.x the CANN backend cannot be reached at runtime any more, because every
model is built on the new graph engine, and that engine rejects non-CPU
back-ends (`Back-ends are not supported by the new graph engine for now`).
The classic engine, which was the only way to reach the CANN backend, was
removed in #29341. That is worth a separate issue as well -- this one is only
about the build.
```

---

## 提交前需要补的实证数据

- [x] npu-4 上 baseline（上游 5.x HEAD）的**真实**编译报错原文 —— 已取到并写进正文
      （Atlas 900 A2 PODc 910B4 / CANN 9.2.0 / gcc 11.4.0 / OpenCV 5.x HEAD
      `3b79188917`）
- [x] npu-4 上打上补丁后整树 `opencv_dnn` **构建 + 链接成功**
      （`libopencv_dnn.so.5.1.0` 产出，0 编译错 / 0 重定位错）
- [x] 失败 TU 逐个确认：不是「一个 TU 报错」，而是 **6 个** TU 编译失败
      （`convolution_layer` / `deconvolution_layer` / `fully_connected_layer` /
      `gemm_layer` / `layer_norm` / `matmul_layer`），逐个单编复现，RC=2
- [x] 组 2（窄 op 头）在 5.x 上**不复现**：5.x 保留 `all_ops.h` 也能链接成功，
      所以它不是 5.x 的构建阻塞项，只是 5.0.0 上的链接问题 / 体积优化项
- [x] `net_cann.cpp` / `op_cann.cpp` / `op_cann.hpp` 在 5.0.0 与 5.x 之间
      **逐字节相同** —— CANN 后端逻辑两版一致，5.0.0 上的 NPU e2e 结论可迁移
- [x] 5.x 上再跑一次端到端 NPU 推理 —— **这条路走不通**，已查清并单独成 issue：
      5.x 的 ONNX 读取器建的是**新图引擎**（`onnx_importer2.cpp` 设
      `mainGraph`），新图引擎明确拒绝非 CPU 后端
      （`Back-ends are not supported by the new graph engine for now`，
      实测 61 个 op 全部 `-> CPU`，NPU/CPU 耗时比 1.00）；
      经典引擎在 #29341 之后已被删除 ⇒ 5.x 上任何模型都到不了 CANN。
      端到端 NPU 数字（124 层 / top-1 372）取自 **5.0.0**，
      那里 `ENGINE_CLASSIC` 还在，ONNX 走经典引擎，CANN 可达。
      ⇒ 第三个 issue 草稿：`issue-3-dnn-cann-unreachable-5x.md`
- [x] 确认上游 issue 里引用的 PR #26056 就是引入 MatShape 结构体、删掉
      `typedef std::vector<int> MatShape;` 的那一个（merge commit `3cd57ea09e`，
      PR 标题是「New dnn engine」，正文里已按标题措辞）
- [x] 确认 #26175 / #26631 / #26632 三个编号与标题一致：
      #26175「CANN build is broken」（issue）、
      #26631「code compile error with CANN on」（issue）、
      #26632「dnn: Fix CANN build」（PR）
