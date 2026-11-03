# 上游贡献工作区（opencv/opencv）

把 OpenCV 5.x 的 CANN（昇腾）后端修好，并按上游流程贡献回 `opencv/opencv`。

## 三个互相独立的问题

| | 问题 1：编译错 | 问题 2：链接错 | 问题 3：后端不可达 |
|---|---|---|---|
| 现象 | `WITH_CANN=ON` 时 CANN TU **编译不过**（`no matching function for call to CannConstOp::CannConstOp`），共 6 个 TU | 编译过了，但 aarch64 **链接**爆 `relocation truncated to fit: R_AARCH64_CALL26/JUMP26` | 编译链接都过了，但 `setPreferableBackend(DNN_BACKEND_CANN)` **被无视**，模型全跑 CPU |
| 根因 | `cv::MatShape` 在 5.0.0 从 `std::vector<int>` 的别名变成了独立结构体（#26056 "New dnn engine"），`CannConstOp` 的形参还是 `const std::vector<int>&` | CANN 的 `all_ops.h` 声明约 1500 个算子类，被 `#include` 进每个 TU，C++ 头文件是编译期文本展开 ⇒ aarch64 `BL` 的 26 位立即数（±128 MiB）装不下 | 5.x 所有模型都建在新图引擎上（`mainGraph` 非空），而新图引擎明确拒绝非 CPU 后端；经典引擎在 **#29341**（commit `49b8f7c03f`，2026-06-18）之后已被删除 |
| 修法 | `CannConstOp` 改收 `const MatShape&`，喂它的 15 个 shape 局部量一并改成 `MatShape`，`front()` → `[0]` | 把 `all_ops.h` 换成窄头（`array_ops.h` + 每个 TU 各自的算子头） | 上游决策：把 CANN 后端移植到新图引擎，或从 5.x 移除。本仓只出 issue，不出补丁 |
| 规模 | 15 文件 / +21 / −21 | 26 文件 / +27 / −2 | 0（无补丁） |
| 本仓对应物 | `projects/opencv/scripts/patch_opencv_cann.py` 的 (a)(b)(c)(d) | 同脚本的 (e)(f) | 无 |
| 上游状态 | 分支 `dnn_cann_matshape` 已提交并推送；issue + PR 草稿在本目录 | **未做**（用户指示「先做组 1」） | issue 草稿在本目录，**待提交** |

问题 1 与问题 2 互不依赖：1 修编译，2 修链接；2 单独存在时编译仍然是坏的。
问题 3 与前两个完全正交：即便 1、2 都修好，CANN 后端在 5.x 上也不会被启用。

## 文件

- `issue-1-dnn-cann-matshape.md` —— 问题 1 的上游 issue 草稿
- `pr-1-dnn-cann-matshape.md` —— 问题 1 的上游 PR 草稿
- `issue-3-dnn-cann-unreachable-5x.md` —— 问题 3 的上游 issue 草稿（无 PR）
- 问题 2（链接错）的 issue + PR 草稿待写

## 提交顺序：先 issue，后 PR

上游模板的 checklist 里有一条硬要求：**「There is a reference to the original bug report and related work」** ⇒ PR 正文必须写上 issue 编号。
所以流程是：先开 issue（拿到编号）→ 再把编号回填进 PR 正文 → 开 PR。
4.x 的同类断裂就是这个走法：issue #26175「CANN build is broken」+ #26631
「code compile error with CANN on」→ PR #26632「dnn: Fix CANN build」。

## 上游流程约束（已核对上游 wiki / 模板）

- base 分支 **`5.x`**；分支名不得是 `2.4` / `3.4` / `4.x` / `5.x` / `master`
- **一个 PR 对应一个 issue** ⇒ 三个问题各走一套
- 标题末尾加 `🤖🤖🤖`（模板第 13 行：自动化 agent 提交的 PR 必须加）
- 标签：`category: dnn`、`category:dnn_cann`
- 提交前 `git diff --check` + 上游 pre-commit hook；≤100 列、4 空格、无 tab、ASCII
- 相关 commit 风格：`dnn(cann): fix build with cv::MatShape`（上游最近大量使用 `dnn:` / `dnn(onnx):` 这种小写前缀风格）

## 上游 CI 对这两组补丁是**盲**的

`PR-5.x.yaml` 的 job 列表里没有 CANN；唯一的 ARM64 job
（`opencv-cn-lin-arm64`）**不传** `-DWITH_CANN=ON` ⇒ `HAVE_CANN` 从未定义
⇒ 那 104 个会拉进 `op_cann.hpp` 的 DNN TU 在上游任何一条流水线里**一次都没被编过**
（其中 39 个直接 include `op_cann.hpp`，82 个经 `net_impl.hpp` 间接引入）。

所以上游 CI 对这两组补丁唯一的信号是「你有没有把非 CANN 的构建搞坏」——
而组 2 的窄头 `#include` 目前是**无条件**插进 layer TU 的
（`#include "../op_cann.hpp"` 没有 `HAVE_CANN` 守卫），
**上游化之前必须先补 `#ifdef HAVE_CANN` 包裹**，否则所有非 CANN 构建都会挂。

## 历史

- 4.x 时代的同类断裂：issue #26175「CANN build is broken」+ #26631
  「code compile error with CANN on」，由 PR #26632「dnn: Fix CANN build」修掉
  （把 `initCannOp` 打成 `CV_Error(Error::StsNotImplemented, "")` 的桩）。
- 上游 5.0.0 到 5.x HEAD 之间，`op_cann.hpp` **一个字节都没改过**
  （4.14.0 与 5.0.0 的 md5 相同）⇒ CANN 被漏掉不是疏忽，是没人碰过。
- 后端**本身**也没变：`net_cann.cpp` / `op_cann.cpp` / `op_cann.hpp` 在
  5.0.0 与 5.x 之间逐字节相同 ⇒ 5.0.0 上量到的 NPU 行为可以直接搬到 5.x。
- 后端不可达是 #29341（2026-06-18）造成的，比 5.0.0 tag（2026-06-05）晚 13 天；
  同一个提交把 `ENGINE_CLASSIC` 从枚举里删掉，`readNetFromONNX()` 改成只调
  `readNetFromONNX2()`（新图引擎）。

## 实测记录（npu-4 / Atlas 900 A2 PODc 910B4 / CANN 9.2.0 / gcc 11.4.0）

| 树 | 结果 |
|---|---|
| 上游 5.x HEAD `3b79188917` | `opencv_dnn` 构建失败：`Error 1`（`convolution_layer.cpp.o`），6 个 TU 编译失败 |
| 上游 5.x HEAD + 问题 1 的补丁 | 构建 + 链接成功，`libopencv_dnn.so.5.1.0` 产出，0 编译错 / 0 重定位错 |
| 5.0.0 + 补丁（经典引擎） | 端到端 NPU 推理通过：124 层转 CANN 算子，top-1 372，NPU 0.91427 vs CPU 0.91480，1 ms vs 34 ms |
| 5.x + 补丁（ONNX） | 推理能跑，但全部 `-> CPU`，NPU/CPU 耗时比 1.00 ⇒ 后端没被启用（问题 3） |
