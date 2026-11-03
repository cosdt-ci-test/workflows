# 上游贡献工作区（opencv/opencv）

把 OpenCV 5.x 的 CANN（昇腾）后端修好，并按上游流程贡献回 `opencv/opencv`。

## 三个互相独立的问题

| | 问题 1：编译错 | 问题 2：链接错 | 问题 3：后端不可达 |
|---|---|---|---|
| 现象 | `WITH_CANN=ON` 时 CANN TU **编译不过**（`no matching function for call to CannConstOp::CannConstOp`），共 6 个 TU | 编译过了，但 aarch64 **链接**爆 `relocation truncated to fit: R_AARCH64_CALL26/JUMP26` | 编译链接都过了，但 `setPreferableBackend(DNN_BACKEND_CANN)` **被无视**，模型全跑 CPU |
| 根因 | `cv::MatShape` 在 5.0.0 从 `std::vector<int>` 的别名变成了独立结构体（#26056 "New dnn engine"），`CannConstOp` 的形参还是 `const std::vector<int>&` | CANN 的 `all_ops.h` 声明约 1500 个算子类，被 `#include` 进每个 TU，C++ 头文件是编译期文本展开 ⇒ aarch64 `BL` 的 26 位立即数（±128 MiB）装不下 | 5.x 所有模型都建在新图引擎上（`mainGraph` 非空），而新图引擎明确拒绝非 CPU 后端；经典引擎在 **#29341**（commit `49b8f7c03f`，2026-06-18）之后已被删除 |
| 修法 | `CannConstOp` 改收 `const MatShape&`，喂它的 15 个 shape 局部量一并改成 `MatShape`，`front()` → `[0]` | 把 `all_ops.h` 换成窄头（`array_ops.h` + 每个 TU 各自的算子头） | 上游决策：把 CANN 后端移植到新图引擎（形态见下方 ORT 整模型旁路模板），或从 5.x 移除。本仓只出 issue，不出补丁 |
| 规模 | 15 文件 / +21 / −21 | 26 文件 / +27 / −2 | 0（无补丁） |
| 本仓对应物 | `projects/opencv/scripts/patch_opencv_cann.py` 的 (a)(b)(c)(d) | 同脚本的 (e)(f) | 无 |
| 上游状态 | 分支 `dnn_cann_matshape` 已提交并推送；issue + PR 草稿在本目录 | **未做**（用户指示「先做组 1」） | issue 草稿在本目录，**待提交** |

问题 1 与问题 2 互不依赖：1 修编译，2 修链接；2 单独存在时编译仍然是坏的。
问题 3 与前两个完全正交：即便 1、2 都修好，CANN 后端在 5.x 上也不会被启用。

### 5.x 上 DNN 还能跑到哪些加速器？（读 `net_impl_backend.cpp` 得到的结论）

新图引擎（`mainGraph` 非空，5.x 上恒成立）只放行 **两个后端 + 两个 target**：

| 放行 | 被拒（只打一条 warning，然后按 CPU 处理） |
|---|---|
| `DNN_BACKEND_OPENCV`、`DNN_BACKEND_CUDA` | `DNN_BACKEND_CANN`、`DNN_BACKEND_INFERENCE_ENGINE(_NGRAPH)`、`DNN_BACKEND_TIMVX`、`DNN_BACKEND_WEBNN`、`DNN_BACKEND_VKCOM` |
| `DNN_TARGET_CPU`、`DNN_TARGET_CUDA`/`CUDA_FP16` | `DNN_TARGET_NPU`、`DNN_TARGET_MYRIAD`、`DNN_TARGET_HDDL`、`DNN_TARGET_OPENCL(_FP16)`、`DNN_TARGET_VULKAN`、`DNN_TARGET_FPGA`、`DNN_TARGET_CPU_FP16` |

⇒ **5.x 上只有 CPU 和 CUDA(GPU) 能用**；昇腾 NPU、OpenVINO 的 MYRIAD/HDDL/NPU、
TimVX、OpenCL、Vulkan 全都进不去。经典引擎是这些后端唯一的挂载点，
而它在 #29341 之后已被删除。

npu-4 上实测（8 组 backend×target 组合，MobileNetV2 ONNX，120 个 op）：

| 组合 | 引擎实际采用 | 落在 CPU 的 op | 拒绝方 |
|---|---|---|---|
| CANN + NPU | b=8 t=0 | 120 | 后端被拒 |
| CANN + CPU | b=8 t=0 | 120 | 后端被拒 |
| OPENCV + OPENCL | b=3 t=0 | 120 | target 被拒 |
| OPENCV + VULKAN | b=3 t=0 | 120 | target 被拒 |
| OPENCV + CPU_FP16 | b=3 t=0 | 120 | target 被拒 |
| OPENVINO + MYRIAD | b=1000000 t=0 | 120 | 后端被拒 |
| CUDA + CUDA | b=5 t=0 | 120 | 后端被拒（本构建无 CUDA） |
| OPENCV + CPU | b=3 t=0 | 120 | —（正常） |

除纯 CPU 外，**所有加速器组合都塌回 CPU**（`t=0` 即 `DNN_TARGET_CPU`）。

## 为什么 4.x 还有、5.x 没了

CANN 后端是挂在**经典引擎**上的，而经典引擎正是 5.x 主线重构（#26056
「New dnn engine」起）要换掉的东西：

- **4.x**：只有 `net_impl.cpp`（没有 `net_impl2.cpp`），
  `net_impl_backend.cpp` 里没有 `if (mainGraph)` 分支 ⇒ `switchToCannBackend()`
  直接可达；Caffe / Darknet / TF importer 都还在；`op_cann.cpp` 是真实现
  （329 行，per-layer `initCannOp` 真的在搭 CANN GE 算子）。
  4.x 那次断裂只是 **GELU 少了个 `initCannOp`**，
  #26632「dnn/gelu_cann」用 9 行 `CV_Error(StsNotImplemented)` 桩掉。
  ⇒ 4.x 上 CANN **编译能过、运行时也能用**。
- **5.0.0**：新旧引擎并存，`EngineType` 还有 `ENGINE_CLASSIC` / `ENGINE_NEW` /
  `ENGINE_AUTO`，ONNX 仍走经典引擎 ⇒ CANN 可达（实测 124 层 / top-1 372）。
- **5.x**：经典引擎被整个删掉 ⇒ CANN 悬空。

时间线：

| 变更 | 日期 | 影响 |
|---|---|---|
| #26056「New dnn engine」 | 2024-10-16 | 引入新图引擎 + `MatShape` 结构体（组 1 的根因） |
| #26334「dnn_engine_change」 | 2024-11-08 | 引入 `if (mainGraph)` 分支，拒绝非 CPU 后端 |
| Darknet cleanup | 2026-04-30 | Darknet importer 删除 |
| #28678「caffe-importer-cleanup」 | 2026-06-02 | Caffe importer 删除 |
| **5.0.0 tag** | **2026-06-05** | 经典引擎还在，CANN 可达 |
| **#29341「disable_engine_classic」**（`49b8f7c03f`） | **2026-06-18** | 删掉 `ENGINE_CLASSIC` 与经典 ONNX importer ⇒ 经典引擎消失，CANN 失去唯一挂载点 |
| #29474「extra_parsers」 | 2026-10-22 | TF importer 删除；只剩 ONNX + TFLite，且都建新引擎 |

### 为什么「接回去」不是改一行代码

`NetImplCann` 这个 `Net::Impl` 子类与新引擎的全部接口就两个钩子，而两个钩子
在 5.x 里**只被经典引擎调用**：

| 钩子 | 作用 | 5.x 里的调用点 |
|---|---|---|
| `NetImplCann::initBackend()`（`net_cann.cpp:101`） | 逐层搭 CANN GE 图，编成一个 OM buffer（`compileCannGraph()`，`net_cann.cpp:292`），给每层挂 `CannBackendNode` | `net_impl.cpp:267`（经典引擎） |
| `NetImplCann::forwardLayer()`（`net_cann.cpp:254`） | **不做任何逐层计算**，只调一次 `cannNode->net->forward()` 启动整个 OM 图 | `net_impl.cpp:921 / 987 / 991`（经典引擎） |

新引擎的 `Net::Impl::forwardMainGraph()`（`net_impl2.cpp:917`）走的是
`finalize()`（选每 op 的执行器）+ `forwardGraph()`，**完全不碰这两个钩子**。
`NetImplOpenVINO`（`net_openvino.cpp`）结构一模一样 —— 所以 OpenVINO 后端
在 5.x 上同样失效（实测 `OPENVINO + MYRIAD` 被拒，120 个 op 全在 CPU）。

⇒ CANN 后端在 5.x 上不是「忘了打开一个开关」，而是**执行模型换代后没人接**。

### 但逐算子那一层是活的（重要更正）

5.x 把逐层钩子从 4.x 的 `initCannOp` 改名成 `initCann`
（`modules/dnn/include/opencv2/dnn/dnn.hpp:474`），**28 个 layer 类都实现了它**
（convolution / deconvolution / gemm / matmul / fc / pooling / softmax /
concat / batch_norm / elementwise / reduce / resize / slice …），
`net_cann.cpp:205` 仍在调用 `layer->initCann(...)`。
`op_cann.cpp`（329 行 GE 算子构造）与 `net_cann.cpp` 的建图 / 编 OM / 启动也都在。

⇒ **昇腾侧的算子知识不用重写**，缺的是**桥**：

| | 经典引擎（现在的 CANN 代码） | 新引擎 |
|---|---|---|
| blob 抽象 | `BackendWrapper`（`getHostMat()` / `copyToHost()` / `setHostDirty()`） | `Arg` / `Tensor` / `UMat` |
| 图遍历 | `Net::Impl::layers` + `LayerData`（`ld.inputBlobsWrappers` / `ld.backendNodes` / `ld.skip`） | `mainGraph->prog_` / `exec_` |

而且**没有通用插槽**：CUDA 是靠新增 UMat 版 `initCUDA` 重载 +
`Layer::forwardCUDA()` 接上的（`layer.cpp:67-73` 写着「new graph engine only
need to override initCUDA(context, inputs, outputs) with UMat」）。
TimVX / VkCom / WebNN / OpenVINO 都还停在 `BackendWrapper` 形态
⇒ 在 5.x 上一起失效。

### 两条移植路线，差一个量级

| 路线 | 形态 | 工作量参照 |
|---|---|---|
| **A 逐算子**（跟 CUDA 一样） | 加 UMat 版 `initCann` 重载 + 在 `finalizeGraph()` / `forwardGraph()` 里加 CANN 分支，28 个算子逐个改成新引擎语义 | 上游给 CUDA 做了 `modules/dnn/src/cuda4dnn/`：**103 个文件、46 个 primitive** —— 等于每个算子重写一遍。CANN 28 个算子同量级，**大工程** |
| **B 整模型旁路**（跟 ORT / PR #29726 一样） | `forwardMainGraph()` 开头整模型交给 CANN，编一次 OM、整体启动一次 | CANN 本来就是整模型语义，`op_cann.cpp` 与 28 个 `initCann` 的算子逻辑可保留，只换「谁遍历图、谁喂 blob、谁启动」⇒ 大概 `net_cann.cpp`（373 行）量级 |

### 但移植的模板已经存在

新引擎的 `forwardMainGraph()` 开头就有 ONNX Runtime 的**整模型旁路**：

```cpp
#ifdef HAVE_ONNXRUNTIME
    if (useOrtEngine && mainGraph && modelFormat == DNN_MODEL_ONNX && !modelFileName.empty())
        finalizeOrt();
    if (this->ort_session)
    {
        std::vector<Mat> ortOuts = runOrtSession(netInputLayer->blobs, std::vector<int>());
        ...
        return;
    }
#endif
```

进行中的 PR #29726「dnn: add AMD MIGraphX (ROCm) backend for ONNX models」
明说是「mirroring the ONNX Runtime whole-model bypass」。

而 CANN 后端**本身就已经是整模型语义**（编一次 OM、整体启动一次），
所以它天然适配这条旁路：编一次 OM → 当成一个 program 跑 → 失败回落默认引擎。

## 上游现状（2026-11-03 核对）

- **没有重复 issue**：GitHub 搜索 `repo:opencv/opencv` 标题含 CANN 的共 25 条，
  最新一条是 2025-06 的 #27402；搜 `disable_engine_classic` /
  `DNN_BACKEND_CANN` / 5.x 加速器相关关键词均无命中。
- 相关工作：
  - **#25025**「Introducing non-CPU HAL for OpenCV 5+」（2024-02 起，仍 open）
    —— 5.x 非 CPU 加速的总设计讨论
  - **#29726**「dnn: add AMD MIGraphX (ROCm) backend for ONNX models」（open）
    —— 在新引擎里加后端，抄的就是 ORT 整模型旁路
  - **#30439**「dnn(cuda): transformer layer coverage and CUDA Graph replay on
    the new engine」—— CUDA 正在被移植到新引擎

⇒ 上游的方向是**「把后端移植到新引擎」**，不是「保留经典引擎」。
CANN 要跟上，形态大概率是 #29726 那种整模型旁路。

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
  「code compile error with CANN on」，由 PR #26632「dnn/gelu_cann」修掉 ——
  它的改动只有 9 行：给 `GeluFunctor` 补一个 `initCannOp`，里面是
  `CV_Error(Error::StsNotImplemented, "")`（当时的 CANN toolkit 没有 GELU 算子）。
  **不是**把整个 CANN 后端打成桩。
- **4.x 上 CANN 是完整可用的**：4.x 只有经典引擎（没有 `net_impl2.cpp`），
  `net_impl_backend.cpp` 里也没有 `if (mainGraph)` 分支 ⇒ CANN 直接可达；
  Caffe / Darknet / TF importer 都还在；`op_cann.cpp` 是真实现（329 行，
  per-layer `initCannOp` 真的在调 CANN GE 算子）。
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
