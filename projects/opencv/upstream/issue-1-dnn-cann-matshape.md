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

```
modules/dnn/src/op_cann.hpp:81:21: error: no matching function for call to
  'CannConstOp::CannConstOp(const unsigned char*, const int, const cv::MatShape&, const std::string&)'
modules/dnn/src/layers/fully_connected_layer.cpp:755:71: note: candidate: 'CannConstOp::CannConstOp(const unsigned char*, const int, const std::vector<int>&, const std::string&)'
```

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

while 7 of its call sites pass the result of `Mat::shape()`, i.e. a `MatShape`:

| file | call |
|---|---|
| `modules/dnn/src/layers/convolution_layer.cpp` | `shape(w_mat)` |
| `modules/dnn/src/layers/deconvolution_layer.cpp` | `shape(mat_w)` |
| `modules/dnn/src/layers/fully_connected_layer.cpp` | `shape(weightsMat)` |
| `modules/dnn/src/layers/gemm_layer.cpp` | `shape(B)` |
| `modules/dnn/src/layers/layer_norm.cpp` | `shape(scale_mat)`, `shape(bias_mat)` |
| `modules/dnn/src/layers/matmul_layer.cpp` | `shape(B)` |

The other 25 call sites pass a local `std::vector<int>`; those still match the
current signature, but they have to follow the parameter type once it changes.

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
The 39 translation units of the DNN module that are guarded by
`#ifdef HAVE_CANN` fail to compile.

### Suggested fix

Take the shape as `const MatShape&` in `CannConstOp` and turn the shape locals
that feed it into `MatShape` as well:

* `modules/dnn/src/op_cann.hpp`, `modules/dnn/src/op_cann.cpp` — the
  declaration and the definition of `CannConstOp`;
* 13 layer files — the 15 shape locals plus 2 assignments, and
  `bias_shape.front()` -> `bias_shape[0]`.

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
```

---

## 提交前需要补的实证数据

- [ ] npu-4 上 baseline（上游 5.x HEAD）的**真实**编译报错原文（`no matching
      function for call to CannConstOp::CannConstOp`）——用来替换上面手写的
      报错片段，务必逐字一致
- [ ] npu-4 上打上补丁后：39 个 CANN TU **全部编译通过**（`openv_dnn` 目标）
- [ ] （可选）叠加窄 op 头补丁后完整链接 + NPU 推理 top-1=372
- [ ] 确认上游 issue 里引用的 PR #26056 是「删掉 `typedef std::vector<int>
      MatShape;`」的那一个（需要 `gh` 复核 PR 标题与 diff）
- [ ] 确认 #26175 / #26631 / #26632 三个编号与标题一致（需要 `gh` 复核）
