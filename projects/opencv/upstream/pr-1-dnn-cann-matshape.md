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
its call sites pass a `MatShape` returned by `shape()` and 2 more pass a
`MatShape` local, so the CANN backend does not compile:

```
modules/dnn/src/layers/gemm_layer.cpp:717:43: error: no match for 'operator=' (operand types are 'cv::MatShape' and 'std::vector<int>')
/usr/include/c++/11/ext/new_allocator.h:162:11: error: no matching function for call to 'cv::dnn::CannConstOp::CannConstOp(unsigned char*&, int, cv::MatShape, std::__cxx11::basic_string<char>)'
/usr/include/c++/11/ext/new_allocator.h:162:11: error: no matching function for call to 'cv::dnn::CannConstOp::CannConstOp(unsigned char*&, int, cv::MatShape&, std::__cxx11::basic_string<char>)'
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
| 13 layer files | 15 shape locals become `MatShape`, 2 assignments adapted, `front()` -> `[0]`, 1 inline literal |

The 15 shape locals are the ones feeding `CannConstOp`: `shape_`,
`bias_shape`, `shape_b`, `pads_shape`, `constant_values_shape`, `axes_shape`,
`shape_of_shape`, `shape_of_size_mat` (x2), `shape_size_splits`,
`split_dim_shape`, `shape_C`.

The 5.0.0 release is affected as well -- it has the same `MatShape`
structure -- so the fix is worth backporting to the next 5.x release.

### Why the CANN backend is not covered by CI

`opencv/opencv` has no CANN job. `PR-5.x.yaml` builds on GitHub-hosted runners
where the CANN toolkit cannot be installed, and the `Ubuntu2404-ARM64` job does
not pass `-DWITH_CANN=ON`, so `HAVE_CANN` is undefined and the 104 DNN
translation units that pull in `op_cann.hpp` are never compiled. The
equivalent 4.x breakage was reported in #26175 / #26631 and fixed by #26632
("dnn: Fix CANN build").

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

Before this patch the build fails in the CANN sources with the errors quoted
above; with the patch all CANN translation units of the DNN module compile.
Measured on the same machine, from the same tree, with only this patch applied:

| tree | `opencv_dnn` target | result |
|---|---|---|
| upstream `5.x` HEAD (`3b79188917`) | build fails | `Error 1` in `modules/dnn/CMakeFiles/opencv_dnn.dir/build.make:1029` (`layers/convolution_layer.cpp.o`); 6 translation units fail |
| upstream `5.x` HEAD + this patch | build succeeds | `libopencv_dnn.so.5.1.0` built and linked, no compile and no relocation errors |

The `MatShape` adaptation itself is verified end to end on the 5.0.0 release
(which has the same `MatShape` structure): with `DNN_BACKEND_CANN` and
`DNN_TARGET_NPU`, MobileNetV2 converts 124 layers to the NPU and the result
matches the CPU reference -- top-1 is 372 for both, the score is 0.91427 on the
NPU against 0.91480 on the CPU (difference 0.0005, tolerance 0.01), and the
NPU forward takes 1 ms against 34 ms on the CPU.

That measurement is done on 5.0.0 because it is the last release where the CANN
backend can still be reached at all: on 5.x every model is built on the new
graph engine, which rejects non-CPU back-ends, so `DNN_BACKEND_CANN` has no
effect there ("Back-ends are not supported by the new graph engine for now").
That is a separate problem and is not addressed by this patch -- see the
related issue. The patch itself is what makes `WITH_CANN=ON` buildable again
on 5.x.

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
- [x] 「How to test」里的报错原文 = npu-4 实测的**逐字**输出（已写入正文）
- [x] 「the result matches the CPU reference」已换成实测数字（124 层 / top-1 372 /
      NPU 0.91427 vs CPU 0.91480 / 1 ms vs 34 ms）
- [ ] 标签：`category: dnn` + `category:dnn_cann`
- [ ] 分支名不能是 `2.4/3.4/4.x/5.x/master` —— `dnn_cann_matshape` 合规
- [ ] 提交前跑 `git diff --check` 与上游 pre-commit hook
- [x] 提交后补上 npu-4 上「整树 `opencv_dnn` 构建通过」的结论 —— baseline `Error 1`
      （`convolution_layer.cpp.o`，共 6 个 TU 失败），打补丁后
      `libopencv_dnn.so.5.1.0` 构建 + 链接成功，0 编译错、0 重定位错
- [ ] 5.x 上跑一次端到端 NPU 推理（把 5.0.0 的 e2e 结论也变成 5.x 的实测）
