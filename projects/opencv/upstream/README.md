# 上游贡献工作区（opencv/opencv）

把 OpenCV 5.x 的 CANN（昇腾）后端修好，并按上游流程贡献回 `opencv/opencv`。

## 两个互相独立的缺陷

| | 组 1：编译错 | 组 2：链接错 |
|---|---|---|
| 现象 | `WITH_CANN=ON` 时 39 个 CANN TU **编译不过**（`no matching function for call to CannConstOp::CannConstOp`） | 编译过了，但 aarch64 **链接**爆 `relocation truncated to fit: R_AARCH64_CALL26/JUMP26` |
| 根因 | `cv::MatShape` 在 5.0.0 从 `std::vector<int>` 的别名变成了独立结构体（#26056 "New dnn engine"），`CannConstOp` 的形参还是 `const std::vector<int>&` | CANN 的 `all_ops.h` 声明约 1500 个算子类，被 `#include` 进每个 TU，C++ 头文件是编译期文本展开 ⇒ aarch64 `BL` 的 26 位立即数（±128 MiB）装不下 |
| 修法 | `CannConstOp` 改收 `const MatShape&`，喂它的 15 个 shape 局部量一并改成 `MatShape`，`front()` → `[0]` | 把 `all_ops.h` 换成窄头（`array_ops.h` + 每个 TU 各自的算子头） |
| 规模 | 15 文件 / +21 / −21 | 26 文件 / +27 / −2 |
| 本仓对应物 | `projects/opencv/scripts/patch_opencv_cann.py` 的 (a)(b)(c)(d) | 同脚本的 (e)(f) |
| 上游状态 | 分支 `dnn_cann_matshape` 已提交并推送；PR 草稿在本目录 | **未做**（用户指示「先做组 1」） |

两组补丁互不依赖：组 1 修编译，组 2 修链接；组 2 单独存在时编译仍然是坏的。

## 文件

- `issue-1-dnn-cann-matshape.md` —— 组 1 的上游 issue 草稿
- `pr-1-dnn-cann-matshape.md` —— 组 1 的上游 PR 草稿

## 上游流程约束（已核对上游 wiki / 模板）

- base 分支 **`5.x`**；分支名不得是 `2.4` / `3.4` / `4.x` / `5.x` / `master`
- **一个 PR 对应一个 issue** ⇒ 组 1、组 2 各走一套 issue + PR
- 标题末尾加 `🤖🤖🤖`（模板第 13 行：自动化 agent 提交的 PR 必须加）
- 标签：`category: dnn`、`category:dnn_cann`
- 提交前 `git diff --check` + 上游 pre-commit hook；≤100 列、4 空格、无 tab、ASCII
- 相关 commit 风格：`dnn(cann): fix build with cv::MatShape`（上游最近大量使用 `dnn:` / `dnn(onnx):` 这种小写前缀风格）

## 上游 CI 对这两组补丁是**盲**的

`PR-5.x.yaml` 的 job 列表里没有 CANN；唯一的 ARM64 job
（`opencv-cn-lin-arm64`）**不传** `-DWITH_CANN=ON` ⇒ `HAVE_CANN` 从未定义
⇒ 那 39 个 CANN TU 在上游任何一条流水线里**一次都没被编过**。

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
