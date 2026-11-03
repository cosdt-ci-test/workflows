# 上游 PR 草稿 1/2：组 1 —— cv::MatShape 适配

> 状态：**待提交**。目标仓库 `opencv/opencv`，base 分支 `5.x`。
> 分支：`dnn_cann_matshape`（fork `Devin-DC-Huang/opencv`），提交 `5ac8d03d24`
> `dnn(cann): fix build with cv::MatShape` —— 15 文件 / +21 / −21。
> 关联 issue 草稿见 `issue-1-dnn-cann-matshape.md`。
> 上游要求「一个 PR 对应一个 issue」，组 2（窄 op 头）单独走第二个 PR。

---

## Title

```
dnn(cann): fix build with cv::MatShape 🤖🤖🤖
```

> 🤖🤖🤖 是上游模板第 13 行的要求：自动化 agent 提交的 PR 标题必须以它结尾。

## Body

```markdown
### Description

`cv::MatShape` is no longer an alias of `std::vector<int>` since 5.0.0, and
`Mat::shape()` returns `MatShape`. `MatShape` mimics the vector API, but the
constructor from `std::vector<int>` is `explicit` and there is no conversion
back to `std::vector<int>`.

`CannConstOp` still takes the shape as `const std::vector<int>&`, while 7 of
its call sites pass the `MatShape` returned by `Mat::shape()`, so the CANN
backend does not compile:

```
modules/dnn/src/op_cann.hpp:81:21: error: no matching function for call to
  'CannConstOp::CannConstOp(const unsigned char*, const int, const cv::MatShape&, const std::string&)'
```

Take the shape as `const MatShape&` in `CannConstOp` and change the shape
locals that feed it into `MatShape` as well. `MatShape` has no `front()`, so
`bias_shape.front()` becomes `bias_shape[0]`.

`CannConstOp` is internal to the DNN module -- every caller is in-tree -- so
this is not an API break.

### Changes

| file | change |
|---|---|
| `modules/dnn/src/op_cann.hpp` | `CannConstOp` takes `const MatShape&` |
| `modules/dnn/src/op_cann.cpp` | matching definition |
| 13 layer files | 15 shape locals become `MatShape`, 2 assignments adapted, `front()` -> `[0]` |

The 15 shape locals are the ones feeding `CannConstOp`: `shape_`,
`bias_shape`, `shape_b`, `pads_shape`, `constant_values_shape`, `axes_shape`,
`shape_of_shape`, `shape_of_size_mat` (x2), `shape_size_splits`,
`split_dim_shape`, `shape_C`.

### Why the CANN backend is not covered by CI

`opencv/opencv` has no CANN job. `PR-5.x.yaml` builds on GitHub-hosted runners
where the CANN toolkit cannot be installed, and the `Ubuntu2404-ARM64` job does
not pass `-DWITH_CANN=ON`, so `HAVE_CANN` is undefined and the 39 CANN
translation units of the DNN module are never compiled. The equivalent 4.x
breakage was reported in #26175 / #26631 and fixed by #26632 ("dnn: Fix CANN
build").

### How to test

Built from source with the CANN backend enabled on an Atlas 900 A2 PODc
(910B4), CANN 9.2.0, gcc 11.4.0:

```
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CANN=ON \
      -DCANN_INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest \
      -DBUILD_LIST=core,imgproc,imgcodecs,videoio,dnn \
      -DSOC_VERSION=ascend910b1 ..
cmake --build . --target opencv_dnn
```

Before this patch the 39 CANN translation units fail to compile with
`no matching function for call to CannConstOp::CannConstOp(...)`; with the
patch all of them compile.

DNN inference on the NPU (MobileNetV2, `DNN_BACKEND_CANN` +
`DNN_TARGET_NPU`, classic engine): the CANN backend converts the graph and the
result matches the CPU reference.

### Checklist

- [x] I agree to contribute to the project under Apache 2 License.
- [x] To the best of my knowledge, the proposed patch is not based on a code under GPL or another license that is incompatible with OpenCV
- [x] The PR is proposed to the proper branch (`5.x`)
- [x] There is a reference to the original bug report and related work (#<issue>, and the 4.x precedent #26632)
- [x] There is accuracy test, performance test and test data in opencv_extra repository, if applicable — N/A, this is a build fix
- [x] The feature is well documented and sample code can be built with the project CMake — N/A, this is a build fix
```

---

## 提交前需要补的东西

- [ ] 把 `#<issue>` 换成真实 issue 编号（先提 issue，再回填 PR）
- [ ] 「How to test」里的报错原文改成 npu-4 实测的**逐字**输出（见 issue 草稿
      末尾的实证清单）
- [ ] 「the result matches the CPU reference」要换成实测数字（top-1 / score），
      不要写成模糊表述
- [ ] 标签：`category: dnn` + `category:dnn_cann`
- [ ] 分支名不能是 `2.4/3.4/4.x/5.x/master` —— `dnn_cann_matshape` 合规
- [ ] 提交前跑 `git diff --check` 与上游 pre-commit hook
