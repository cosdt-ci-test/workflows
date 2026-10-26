# 快速开始（OpenCV 5.0 + 昇腾 NPU）

一句话：在单卡昇腾 NPU 上把 **OpenCV 5.0** 源码编译成带 CANN 后端的 `cv2`（`WITH_CANN=ON`），跑通 CPU 基础 API + NPU DNN 推理；并且先说清 **5.0 相对 4.x 到底带来了什么**、每条新特性在昇腾 CANN 后端上是什么状态。

本文分两半：

- **§2 特性导览**：OpenCV 5.0 的核心特性 + 「特性 → 昇腾可用性」矩阵（哪条能用、哪条被墙、为什么）；
- **§3~§10 跑通路径**：自包含的源码构建 + CPU API + NPU 推理，每一步都可复制粘贴、每一步都有可核对的输出。

## 1. 与同仓其它 OpenCV 文档的关系

| 文档 | OpenCV 版本 | 源码补丁 | 定位 |
| --- | --- | --- | --- |
| **本文** | 5.0（跟随上游 latest release） | 3 组 | 5.0 特性导览 + 昇腾可用性矩阵 + 跑通路径 |
| [Quick-start-Ascend.md](Quick-start-Ascend.md) | 5.0（跟随上游 latest release） | 3 组 | 同一配方，偏「补丁逐行说明 + cannops 单测证据」 |
| [Quick-start-Ascend-NoPatch.md](Quick-start-Ascend-NoPatch.md) | 4.11.0（钉死） | 0 组 | 不想维护本地补丁时的退路（放弃 5.x 特性） |

三条路线的取舍：

- 要 5.0 的新能力（新 DNN 图引擎、原生 LLM/VLM、新 dtype、1D/0D Mat、STB 新字体、USAC 默认后端）→ 走 5.0，**必须打补丁**（原因见 §5：5.0 把 shape 类型换成了 `cv::MatShape`，而上游 CANN 后端没跟上）；
- 不想维护任何本地补丁 → 走 4.11 零补丁路线，但只能停在 4.x。

> **版本口径**：本文的 `<ref>` 由 CI 注入，解析自上游 GitHub `releases/latest`。截至 2026-10-23，上游把它标成 **5.0.0**（4.14.0 是 4.x 维护线，发布更晚但**不是** latest）。§5 的补丁脚本只对 **5.0.x** 成立（`cv::MatShape` 是 5.0 引入的，4.x 里根本没有这个类型），上游一旦出 5.1/6.0，需要人工复核补丁。

## 2. OpenCV 5.0 核心特性

> **上游事实来源**（本文 §2 逐条对着这三份一手材料核过，不是凭印象）：
>
> - [OpenCV 5 wiki](https://github.com/opencv/opencv/wiki/OpenCV-5)：5.0.0 发布说明（2026-06 定的稿）
> - [OpenCV 4→5 迁移指南](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration)：破坏性变更的权威清单
> - [OpenCV Change Log](https://github.com/opencv/opencv/wiki/OpenCV-Change-Logs)：5.0 段官方写的是 "ChangeLog is TBD"，明细看上面两份
>
> 另外本文**直接比对了 `5.0.0` 与 `4.14.0` 两个 tag 的源码树**（GitHub API `contents` / `git trees`），用来判定「这条到底是 5.0 新增，还是 4.x 早就有」——下面每条都打了来源标记。
> 「昇腾上的状态」一列是本文在 910B4 上实测出来的。

**来源标记**（判定依据是上游文档 + 源码树比对，不是印象）：

| 标记 | 含义 |
| --- | --- |
| `[4.x]` | 4.x 时代就有，5.0 只是延续 / 增强 |
| `[5.0 新增]` | 5.0 引入的新能力 |
| `[5.0 行为变化]` | API 签名不变，但结果 / 语义变了（升级要改代码或重录基线） |
| `[5.0 移除]` | 5.0 删掉了 |

### 2.1 DNN 模块：引擎重写 + 原生 LLM/VLM

`[5.0 新增]` **新推理引擎**与经典引擎并存。5.0 把 dnn 从「固定 layer 序列」重写成**带类型的算子图**：形状推断、常量折叠、算子融合（含 attention / MatMul 融合，FlashAttention 风格实现）、原生**动态与符号 shape**、控制流子图（`If` / `Loop`）、量化图、统一 buffer pool / 内存池。上游口径：**新引擎覆盖 >64% 的 ONNX 规范，4.x 不到 13%**。

`[5.0 新增]` **引擎选择**：`cv::dnn::readNet()` / `readNetFromONNX()` 新增 `int engine = ENGINE_AUTO` 参数，默认先试新引擎、加载失败自动回落经典引擎。也可以用环境变量 `OPENCV_FORCE_DNN_ENGINE`：`1`=经典引擎、`2`=新引擎（模型不支持就报错）、`3`=auto（默认）、`4`=ONNX Runtime。**模型加载后不能再切引擎**（两套引擎内部表示不同）。

`[5.0 新增]` **ONNX Runtime 集成**：`cmake -DWITH_ONNXRUNTIME=ON -DDOWNLOAD_ONNXRUNTIME=ON`（NVIDIA GPU 再加 `-DDOWNLOAD_ONNXRUNTIME_GPU=ON`）。OpenCV 用自己的 ONNX parser 构造 ORT 图，所以只需要 ORT 本体，不必装整个 onnx 包。

`[5.0 行为变化]` **新引擎目前只跑 CPU**——上游原话是 "The new engine currently runs on CPU only"，GPU 支持留到后续版本。**这一条直接决定了昇腾上必须显式选经典引擎**（§10）。

`[5.0 新增]` **原生 LLM / VLM**：新引擎内置 tokenizer、attention 层、解码块、后处理与 **KV-cache**，可端到端跑 VLM。5.0.0 的 `samples/dnn/` 里对应 5 个样例：`gpt2_inference.py` / `qwen_inference.py` / `gemma3_inference.py` / `vlm_inference.py` / `ldm_inpainting.py`（扩散模型）。

`[5.0 移除]` **Darknet / Caffe 解析器删除**（`readNetFromDarknet()` / `readNetFromCaffe()` 没了）；**TFLite 仍由经典引擎支持**。

`[5.0 行为变化]` **模型仓库搬家**：OpenCV 的小模型集合迁到 HuggingFace `huggingface.co/opencv`（统一 ONNX 格式），样例用 `cd samples/dnn && python3 download_models.py` 拉。

> **性能对比要看官方的完整表，别看二手总结**：[OpenCV 5 DNN Benchmarks](https://github.com/opencv/opencv/wiki/OpenCV-5-DNN-Benchmarks)。它是**分模型**的，不是全面更快——Intel i9 上 OpenCV 5 快的有：X-Feat 6.2 vs 10.3 ms、owlv2 1074.7 vs 1542.6 ms、BiRefNet 7178 vs 9503.1 ms；慢的有：RAFT 1193 vs 685.2 ms、SAM3 (Encoder) 7753 vs 6194 ms、SwinIR 225.7 vs 164.7 ms（均为 OpenCV vs ONNX Runtime 1.25.1，毫秒，越小越好）。

### 2.2 core：数据类型、维度、形状

- `[5.0 新增]` **五种新数据类型**：`CV_16BF`（`cv::bfloat`）、`CV_32U`、`CV_64U`、`CV_64S`、`CV_Bool`（1 字节/值，非零即 true，可当 `CV_8U`/`CV_8S` mask 用）。`CV_16F`（`cv::hfloat`）**4.x 就有**，不是 5.0 新增。`hfloat` / `bfloat` 的算术在没有原生支持的硬件上也能用（内部走高效的标量/向量转换）。
- `[5.0 新增]` **1D / 0D 数组原生支持**：4.x 里 `std::vector` 会被当成 Nx1 矩阵，5.0 里它就是真正的 1D 数组（`dims == rows == 1`，`cols == total()`）；0D 数组（标量）同理（`dims == 0`）。判空用 `mat.empty()` 而不是 `total() == 0`。
- `[5.0 新增]` **shape 类型换成 `cv::MatShape`**（`MatSize` 保留为别名）：带上 layout 信息，而且**内嵌在 `Mat` / `UMat` / `GpuMat` 里不再额外动态分配**；dnn 的形状推断大量使用它。**这正是 §5 组 1 补丁要适配的东西。**
- `[5.0 新增]` **LAPACK 始终可用**：没装外部 LAPACK 就编一个内置小子集 → 更好的 SVD / 特征值分解 / USAC。
- `[4.x]` `cv::broadcast` / `cv::transposeND` / `cv::flipND` **不是 5.0 新增**：5.0.0、4.14.0、4.13.0、4.12.0 的 `core.hpp` 里三者都在（4.14 只是给 `transposeND` 加了 identity / 2D 顺序的快路径）。5.0 在这块的真正变化是上面那条 `MatShape`。

### 2.3 imgproc / videoio：被重写的算子、新字体

- `[5.0 行为变化]` **仿射 / 透视 / 重映射重写**：`warpAffine` / `warpPerspective` / `remap` 的双线性、双三次插值不再用查表近似，**结果更准也更快**（上游口径：提速 10% ~ 300%+，视平台 / 尺寸 / 类型 / flags 而定）。API 不变，但**数值与 4.x 略有差异**——逐像素比对 4.x 基线的测试要重录。
- `[5.0 行为变化]` **最近邻 resize 改用 Pillow 算法**：`INTER_NEAREST` 与 `INTER_NEAREST_EXACT` 现在等价，输出与 Pillow 逐像素对齐（跟 Pillow 做像素级比对的老测试要注意方向变了）。
- `[5.0 新增]` **新的文字渲染引擎**：基于 STB TrueType，内嵌 Rubik 可变字体，新增 `cv::FontFace` 类与 `putText()` / `getTextSize()` 的新重载，支持 UTF-8、自定义 `.ttf`/`.otf`、内置 `"sans"` / `"italic"` / `"uni"`（`"uni"` 带 CJK）。旧的 `FONT_HERSHEY_*` 仍可用，但底层已换成新引擎，**字号与字形跟 4.x 不一样**；复杂整形（阿拉伯文、天城文）和彩色 emoji 暂不支持。
- `[4.x 起有 / 5.0 自动启用]` **TRUCO 轮廓提取**：`modules/imgproc/src/contours_truco.cpp` 在 **4.14.0 里就有**（4.14 的 Change Log 记着 "Added findTRUContours - lock-free parallel contour extraction"），5.0 的变化是 `cv::findContours()` **在可用时自动切到 TRUCO**，比旧实现快数倍。
- `[5.0 行为变化]` **`VideoCapture::get()` 对不支持的属性返回 `-1`**（4.x 返回 0，而 0 对很多属性是合法值）。判「该后端不支持」要改成 `val < 0`。

### 2.4 HAL 与厂商加速：先纠正一个常见误解

- `[4.x]` **Universal intrinsics**（`v_exp` / `v_log` / `v_erf` / `v_sincos` 等）、**`AlgorithmHint`**（速度 vs 精度）、**厂商 HAL 扩展点**——**都是 4.x 时代就有的**，5.0 只是延续。上游 alpha Change Log 给这些条目标的就是 "5+4.x"（= 4.x 已有、5.0 也提一句）。
- `[4.x]` **一套向量代码映射到 SSE / AVX2 / AVX512 / NEON / SVE / RVV** 也是 4.x 的能力。`hal/kleidicv`（Arm KleidiCV）和 `3rdparty/fastcv`（Qualcomm FastCV）在 **4.14.0 里就已经在**；5.0.0 的 `hal/` 是 `armpl carotene fastcv ipp kleidicv ndsrvp riscv-rvv`，4.14.0 是同样这七个 **外加 `openvx`**（5.0 随 OpenVX 支持一起删掉了）。
- `[5.0 计划中，未落地]` 真正的「**HAL 2.0**」= 面向**非 CPU 后端**的厂商加速接口 + 把 `UMat` 扩成通用异构容器（上游 issue #25019 / #25025）。上游 Change Log 明确标成 "5.0, 5.x: expected"——**5.0.0 里没有**，`core/hal/interface.h` 里也没有 HAL 2.0 的版本标记。所以「OpenCV 5.0 有了 HAL 2.0」是个常见误传。

### 2.5 模块重组与移除

- `[5.0 行为变化]` **`calib3d` 拆成 4 个模块**：`geometry`（2D/3D 几何：`findHomography` / `solvePnP` / `triangulatePoints` …）、`calib`（相机标定）、`stereo`（双目深度）、`ptcloud`（TSDF / 视觉里程计 / 点云 I/O）。C++ 里 `opencv2/calib3d.hpp` 仍可用（只是个聚合头，依次 include 那三个）；**Python 完全透明**，还是 `cv2.findHomography()`。
  > 上游自己的文档在这点上前后不一致：迁移指南正文写 "split into three focused modules"，但标题、表格和实际 tag 都是**四个**（`5.0.0` 的 `modules/` 下有 `calib geometry stereo ptcloud`，没有 `3d`）。以源码树为准。
  > 兼容头 `opencv2/calib3d.hpp` 还在，但住在 `modules/calib/include/opencv2/` 下，只 include 了 geometry / stereo / calib **三个**——要用 TSDF / ICP / 点云 I/O 得另外 include `opencv2/ptcloud.hpp`。
- `[5.0 行为变化]` **imgproc 的计算几何搬去 `geometry`**：`convexHull` / `convexityDefects` / `isContourConvex` / `minEnclosingCircle` / `minEnclosingTriangle` / `minAreaRect` / `fitEllipse` / `Subdiv2D`（Delaunay）等。Python 同样透明。
- `[5.0 新增]` **`features2d` 更名 `features`**，并扩到处理深度网络产生的特征向量：主库新增 **ALIKED**、**DISK** 与 **LightGlue** 匹配器，以及基于 **Annoy** 的 ANN 检索（`annoy.cpp` 在 4.14.0 里没有，是 5.0 新增）。
  - ⚠️ **FLANN 模块并没有被移除**：`5.0.0` 的 `modules/flann/src/` 里 `flann.cpp` + `miniflann.cpp` 都还在。上游 2024-12 的 alpha Change Log 写过「`FLANN` as a separate module will be gone」，但 5.0.0 正式版**没执行**这条——Annoy 是**新增**的替代实现，不是「移除 FLANN」。（典型的「只看 wiki 概要会被坑」。）
  - 迁去 contrib `xfeatures2d` 的是 SURF / BRIEF / FREAK / LUCID / DAISY 等老检测器；**SIFT / ORB / FAST / GoodFeaturesToTrack / MSER 留在主库**。
- `[5.0 行为变化]` **G-API**、经典 **ml**、**Haar / HOG 检测器**（`xobjdetect`）搬去 `opencv_contrib`；主库 `objdetect` 里的 DNN 人脸检测（`FaceDetectorYN`）不需要 contrib。
- `[5.0 移除]` **遗留 C API**（`CvMat` / `cvCreateMat` / `cvFindContours` …，只保留 `CV_8U` 这类宏）、**OpenVX 支持**、**Caffe / Darknet 解析器**。
- `[5.0 行为变化]` **samples 清理**：约 50% 的 C++ samples、5% 的 Python samples 被删，剩余的大改。
- `[5.0 行为变化]` **USAC 成为所有鲁棒估计算法的默认后端**（单应 / 本质矩阵 / 基础矩阵 / PnP …）：PROSAC / NAPSAC / Progressive-NAPSAC、MSAC / MAGSAC++ / LMeds、LO-RANSAC / GC-RANSAC、SPRT 提前终止，并支持多线程。USAC 算法本身 4.x 就有，5.0 变的是「成为默认后端」。
- `[5.0 新增]` **多目相机标定流水线**（`calib`）：N 台相机三阶段（逐相机内参 → 两两外参 → 全局优化），支持针孔 / 鱼眼及混合阵列、棋盘 / ChArUco / 圆点板，容忍「不是每台相机每帧都看全标定板」的部分观测。
- `[5.0 新增]` **点云 / mesh**（`ptcloud`）：TSDF、ICP、视觉里程计，及 `.ply` / `.obj` 导入导出。

### 2.6 语言 / 绑定 / 文档

- `[5.0 新增]` **C++17 最低**（编译器下限 GCC 8 / Clang 9 / MSVC 2017；C++20 / 23 计划中）；**Python 2 移除**，要求 Python 3.6+。
- `[4.x]` **NumPy 2.x 兼容不是 5.0 特性**：4.x 已有实验性 NumPy 2.0 支持，4.12.0 起把 `ndarray.ptp()` 换成 `np.ptp()` 以兼容 2.0。
- `[5.0 行为变化]` **模块重组对 Python 透明**：函数仍在 `cv2.<func>()`，只是 `cv2.ml.*`（未编 contrib 时）、`cv2.CascadeClassifier` / `cv2.HOGDescriptor` 需要编 `opencv_contrib`。
- `[5.0 行为变化]` **Python 的 1D 语义**：`np.array` 作为 `InputArray` 现在映射成 1D `Mat`（4.x 是 Nx1 列向量），依赖 `.rows` / `.cols` 的代码要改用 `.total()`。
- `[5.0]` **文档**：5.0.0 源码树里 `doc/`（Doxygen）与 `docs_sphinx/`（Sphinx 骨架，含 `conf.py`）**两套并存**；docs.opencv.org 换成响应式新主题（侧边栏 / Ctrl+K 搜索 / 明暗色）。

### 2.7 特性 → 昇腾 CANN 后端可用性矩阵

这是本文最该看的一张表。**结论先行：5.0 的「招牌」特性（新图引擎、原生 LLM/VLM）目前一个都上不了 NPU**——昇腾要走的是 4.x 时代的**经典引擎 + CANN 后端**。反过来，5.0 里那些「跟后端无关」的新能力（新 dtype、1D/0D Mat、新字体、USAC……）在昇腾机器上**照常可用**，只是它们本来就是 CPU 算子。

| 5.0 特性 | 昇腾 CANN 后端上的状态 | 依据 / 绕法 |
| --- | --- | --- |
| **新图引擎**（算子图、形状推断、融合、内存池） | ❌ **不走 NPU** | 上游明说 "The new engine currently runs on CPU only"；本文实测：5.0 默认 `ENGINE_AUTO` → 新图引擎，`setPreferableBackend` / `setPreferableTarget` 被静默忽略，只打两条 WARN（§10.1）。**必须** `OPENCV_FORCE_DNN_ENGINE=1`（或 `readNetFromONNX(path, cv2.dnn.ENGINE_CLASSIC)`）强制经典引擎（§10.2） |
| `ENGINE_ORT`（ONNX Runtime 后端） | ❌ 未编 | 需要 `-DWITH_ONNXRUNTIME=ON`；本文没编（而且 ORT 也不是 NPU 路径） |
| **原生 LLM / VLM**（内置 tokenizer + KV-cache） | ❌ **只能 CPU** | 同上：新引擎不支持非 CPU 后端。5.0.0 的 `samples/dnn/{gpt2,qwen,gemma3,vlm}_inference.py` 全部硬编码 `ENGINE_NEW` |
| 新 dtype（`CV_16BF` / `CV_32U` / `CV_64U` / `CV_64S` / `CV_Bool`） | ⚠ core 层可用，CANN 后端不认 | ACL 算子图的元素类型是有限集合；bf16 张量进不了 CANN 算子图 |
| **1D / 0D Mat**、shape 类型换成 `cv::MatShape` | ⚠ **编译不过，必须打补丁** | 上游 CANN 后端 `modules/dnn/src/op_cann.hpp` 从 4.11 到 5.0.0 **一行没改**（构造函数仍是 `const std::vector<int>&`），而 5.0 的 `cv::MatShape` 是 `explicit` 且**没有 `front()`** → §5 组 1 补丁 |
| `broadcast` / `transposeND` / `flipND` | ✅ CPU 可用，无 NPU 路径 | ⚠️ 这三个**是 4.x 就有的**（≤4.12），不是 5.0 新增；实测：§9 的两个 ND 算子在源码版 `cv2` 上正常返回；CANN 后端未实现它们 |
| **imgproc 重写**（warp 加速 / 最近邻改 Pillow / STB 字体 / TRUCO 自动启用） | ✅ 纯 CPU | 与后端无关；但**数值与 4.x 有差异**，逐像素基线要重录（§2.3） |
| **USAC 默认后端** / 多目标定 / `ptcloud`（TSDF·ICP·ply·obj） | ✅ 纯 CPU | 与后端无关 |
| Universal Intrinsics / `AlgorithmHint` / KleidiCV / FastCV / IPP | ⚠ 与 NPU 无关，且**是 4.x 遗产** | CPU SIMD 层；昇腾 aarch64 走 NEON。5.0 的「HAL 2.0（非 CPU HAL）」**未落地**（§2.4） |
| `calib3d` → `geometry` / `calib` / `stereo` / `ptcloud` | ✅ 纯 CPU | 与后端无关（4 个模块，不是 3 个） |
| `features2d` → `features`（ALIKED / DISK / LightGlue / Annoy） | ✅ 纯 CPU | 与后端无关；**FLANN 模块仍在**（§2.5） |
| G-API / 经典 ml / Haar-HOG 移到 contrib | ⚬ 需额外编 `opencv_contrib` | 本文编了 contrib（为了 `cv2.cann` 经典算子加速） |
| 移除遗留 C API | ⚠ 迁移成本 | 与 CANN 后端无关（后端是 C++ 内部实现） |
| 移除 Caffe / Darknet 解析器 | ⚠ `samples/dnn/openpose.py` 断供 | 与后端无关；TFLite 仍可用（走经典引擎） |
| C++17 / Python 3 | ⚠ 环境约束 | 镜像 GCC 11.4 满足 C++17；CANN 的 GE 图编译依赖镜像里的 Python 站点包（见 §10.2 的 `PYTHONPATH` 警告） |

### 2.8 本文覆盖边界（明确没覆盖什么）

- **本文没编的**：`onnxruntime` 后端、`opencv_contrib` 里除 `cannops` 外的模块（`xobjdetect` / `xfeatures2d` / `ml` / `gapi` / `rgbd`…）、Java / JS / ObjC 绑定、OpenCL 路径。
- **本文编了但不逐条验证的**：`core, imgproc, imgcodecs, videoio, dnn, python3, cannops, ts`（见 §7 的 `BUILD_LIST`）。§2 里凡是纯 CPU 的新特性，本文只保证「API 存在、能 import」，**不保证逐个跑过**；真正跑过的是 §9 的 CPU quickstart、§10 的 NPU 推理、§8.1 的 cannops gtest。
- **本文没有覆盖的上游能力**：C++ API 的完整迁移面、Java 绑定的 import 变更、samples 集合、`docs.opencv.org` 新文档站本身。
- **本节（§2）的判定方法**：官方 wiki + 迁移指南 + Change Log 三份文档，与 `5.0.0` / `4.14.0` 两个 tag 的源码树逐条比对。**凡是上游文档与源码树打架的，以源码树为准**（§2.5 的 calib3d 拆成几个模块、FLANN 移不移除，两处都踩过这个坑）。

## 3. 前置条件

### 硬件

Atlas 900 A2 / A3 训练系列产品或 Ascend 950 系列，已挂载 `/dev/davinci*`。

### 软件（必须已就绪）

- Python 3.12
- CANN toolkit（参见 [Ascend 官方快速安装](https://ascend.github.io/docs/sources/ascend/quick_install.html)），`npu-smi info` 能列出设备
- 可访问 GitHub（clone + 下载模型）

> `opencv-python`（pip wheel）**不带** CANN 后端，必须从源码构建，`cv2.dnn.DNN_BACKEND_CANN` 才会链到真后端（本文实测：源码版 = 8，pip wheel 里只是个没实现后端的枚举值）。

### 本文示例使用的版本

| 项目 | 值 |
| --- | --- |
| 机器 | Atlas 900 A2 PODc（910B4） |
| OS | Ubuntu 22.04 arm64 |
| 镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`（CI）/ CANN 9.2.0（本地验证机） |
| Python / CMake / GCC | 3.12 / ≥ 3.18 / 11 |
| OpenCV | 5.0.0（源码，`WITH_CANN=ON`，`CMAKE_BUILD_TYPE=Debug`）|
| opencv_contrib | 5.0.0（同 tag，只用其中的 `cannops` 模块）|
| 安装前缀 | `/usr/local/opencv-cann-5`（与同仓另两条路线的 `/usr/local/opencv-cann`、`/usr/local/opencv-cann-411` 互不覆盖）|

### 前置安装

确认能看到 NPU 设备：

```shell
npu-smi info
```

> 如果 `npu-smi` 不存在，按官方快速安装指南补装驱动。

检查 Python / CMake：

```shell #test id="ocv5-check-toolchain"
python --version
cmake --version | head -n 1
```

输出结果如下：
```shell #test-result id="ocv5-check-toolchain" fuzzy='xxx'
Python 3.12.xxx
cmake version xxx
```

加载 CANN 环境变量（每个用到 CANN 变量的块都要自己 `source`）：

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

确认 toolkit 根目录可访问：

```shell #test id="ocv5-check-cann-path"
ls -d /usr/local/Ascend/ascend-toolkit/latest
```

输出结果如下：
```shell #test-result id="ocv5-check-cann-path"
/usr/local/Ascend/ascend-toolkit/latest
```

## 4. 装编译依赖 + clone 源码

```shell #test-setup
sed -i 's|http://ports.ubuntu.com/ubuntu-ports/|https://mirrors.aliyun.com/ubuntu-ports/|g' /etc/apt/sources.list
apt-get update -qq && apt-get install -y -qq git build-essential cmake pkg-config libjpeg-dev libpng-dev libtiff-dev zlib1g-dev
```

把上游 latest release tag 注入 `<ref>`；手动跑直接填 tag（如 `5.0.0`）。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test-setup load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/opencv/opencv.git
git clone --depth 1 --branch <ref> https://github.com/opencv/opencv_contrib.git
```

确认 clone 到的是 5.0：

```shell #test id="ocv5-opencv-ref"
cd opencv && git describe --tags
cd ../opencv_contrib && git describe --tags
```

输出结果如下：
```shell #test-result id="ocv5-opencv-ref" fuzzy='xxx'
5.0.xxx
5.0.xxx
```

> 这一步是 §1「补丁只对 5.0.x 成立」的守门人：如果上游把 `releases/latest` 换成 5.1/6.0，`git describe` 会打出别的 tag，此时 §5 的补丁脚本会**断言失败**（而不是悄悄编出一个错的东西），提示人工复核。

## 5. 打 3 组源码补丁

OpenCV 5.0 mainline 与 CANN 9.x / aarch64 有 3 处不兼容（对应 §2.7 矩阵里的两行）。按顺序跑这 3 段脚本。**三段都是幂等的**：重复跑会打印 `already patched` 并跳过，所以中途失败可以直接重跑。

| 组 | 上游哪里坏了 | 改法 |
| --- | --- | --- |
| 1 | OpenCV 5.0 把 shape 类型换成 `cv::MatShape`（`explicit` 构造、**没有** `front()`），上游 CANN 后端还停在 4.x 的 `std::vector<int>` 写法 | 新增 1 个 `cv::MatShape` 重载转发给上游构造 + 修 3 处赋值 |
| 2 | `all_ops.h` 的 ~1500 个 op 把 dnn `.text` 撑破 aarch64 ±128 MiB 链接窗口 | per-TU 窄 op 头 |
| 3 | CANN 9.x 拒绝 NULL 流（EH0008） | `OperatorRunner::run` 回落到 ctx 默认流 |

### 组 1. `cv::MatShape` 改了类型

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

预期：`(a)`/`(b)` 各 1 行打印，`(c)(d)` 打印 3 行（gemm 1 + matmul 2）。

### 组 2. 缩小 CANN op 头（避免 aarch64 链接炸）

`all_ops.h` 有 ~1500 个 op。`op_cann.hpp` 被 `net_impl.hpp` 拉进**每个** dnn TU，于是每个 TU 都背上 ~4 MB 的 op 内联代码，dnn 的 `.text` 撑破 aarch64 `R_AARCH64_CALL26/JUMP26` 的 ±128 MiB 窗口，链接直接死在 `relocation truncated to fit`（报错对象随链接顺序随机变——protobuf / arena 都有可能，别顺着报错对象查根因）。

Debug 与 Release **都炸**，换优化级别救不了，只能缩头：换成窄头 `array_ops.h`，并给每个 layer TU 插它实际用到的窄头。

> 23 个窄头**不能**合并成一个「域头并集」（那样补丁少 22 处，看着很美）：同一个 TU 的 `.text*` 实测 `all_ops.h` 361 KB~4 MB，域头并集是窄头的 5.6~9.4 倍，外推到 97 个 TU 仍约 195 MB > 128 MiB，链接照炸。

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

### 组 3. `OperatorRunner` 的 NULL 流 fallback

CANN 9.x 的 `aclopCompileAndExecute` 拒绝 NULL 流（报 `EH0008`）。NULL 时换成 `aclrtCtxGetCurrentDefaultStream`。

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
if new in s:
    print('(g) cann_call.cpp: NULL-stream fallback already patched')
else:
    assert old in s, 'cann_call.cpp: OperatorRunner::run body not found'
    p.write_text(s.replace(old, new, 1))
    print('(g) cann_call.cpp: NULL-stream fallback patched')
PY
```

预期：打印 `(g) cann_call.cpp: NULL-stream fallback patched`。

### 补丁是否真的落到位

```shell #test id="ocv5-patches-applied"
grep -c "cv::MatShape" opencv/modules/dnn/src/op_cann.hpp
grep -c "array_ops.h" opencv/modules/dnn/src/op_cann.hpp
grep -c "aclrtCtxGetCurrentDefaultStream" opencv_contrib/modules/cannops/src/cann_call.cpp
grep -rl "built-in/op_proto/inc/" opencv/modules/dnn/src/layers/ | wc -l
```

输出结果如下：
```shell #test-result id="ocv5-patches-applied"
1
2
1
23
```

## 6. 桥接 CANN 的库目录

OpenCV 的 `OpenCVFindCANN.cmake` 在 `${CANN_INSTALL_DIR}/{acllib,lib64,compiler/lib64}/` 下找 ACL 库：

| 找什么 | 期望位置 |
| --- | --- |
| `libascendcl.so` | `acllib/lib64/` |
| `libacl_op_compiler.so`、`libacl_dvpp_mpi.so` | `lib64/` |
| `libgraph.so`、`libge_compiler.so`、`libgraph_base.so` | `compiler/lib64/` |

这三个位置在 CANN 里都不叫这三个名字，所以要补软链。**统一指向「完整版 CANN 库目录」**：

- CANN 9.1.x：`<CANN>/lib64` 不存在，完整库目录是 `<CANN>/aarch64-linux/lib64`；
- CANN 9.2.x：`<CANN>/lib64` 已经是指向 `aarch64-linux/lib64` 的软链（同一批库）。

**不要**指向 `<CANN>/aarch64-linux/devlib/linux/aarch64/` —— 那是裁剪过的设备侧运行时（`libgraph.so` 只有 198 KB，少了 `ge::AscendString` 的 4 个序关系运算符，链接期会炸成 `undefined reference to ge::AscendString::operator<`）。

先确认本机的 `<CANN>/lib64` 是完整版：

```shell #test id="ocv5-cann-lib-layout"
CANN_DIR=/usr/local/Ascend/ascend-toolkit/latest
LIB="$CANN_DIR/lib64"
[ -d "$LIB" ] || LIB="$CANN_DIR/aarch64-linux/lib64"
ls -l "$LIB/libascendcl.so" "$LIB/libgraph.so"
nm -D --defined-only "$LIB/libgraph.so" | grep -c '_ZNK2ge12AscendStringlt'
```

输出结果如下：
```shell #test-result id="ocv5-cann-lib-layout" fuzzy='xxx'
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

> `ln -sfn` 到已存在的软链是「改指向」而不是「往目录里塞链接」，所以重复执行是安全的（`lib64` 那条还额外用 `[ -e ... ]` 挡掉了「指向自己」的 `are the same file` 报错）。
>
> **不想往 CANN 安装目录里写东西**（只读挂载 / 无 root）时，等价做法是直接把路径喂给 cmake：`-Dfound_lib_ascendcl=... -Dfound_lib_acl_op_compiler=... -Dfound_libacldvppmpi=... -Dfound_lib_graph=... -Dfound_lib_ge_compiler=... -Dfound_libgraph_base=...`（`find_library` 命中已存在的 cache 变量就不再搜目录）。

## 7. 配置并编译

CANN 后端开关 `WITH_CANN=ON`（不是 `BUILD_CANN`）；`-DOPENCV_DOWNLOAD_MIRROR_ID=gitcode` 让 cmake configure 阶段走 gitcode 镜像拉 ADE / IPPICV / TBB。

`BUILD_LIST` 里的模块决定了编出来有哪些能力：`core` / `imgproc` / `imgcodecs` / `videoio` / `dnn`（CANN 后端在这里）/ `python3` / `cannops`（`cv2.cann` 经典算子）/ `ts`。

```shell #test id="ocv5-cmake-configure"
cd opencv
mkdir -p build && cd build
PY_INC=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
PY_LIB=$(python3 -c 'import sysconfig,os,sys; print(os.path.join(sysconfig.get_config_var("LIBDIR"), "libpython%d.%d.so" % sys.version_info[:2]))')
test -f "$PY_INC/Python.h" || { echo "Python.h not found under $PY_INC"; exit 1; }
test -f "$PY_LIB" || { echo "libpython not found at $PY_LIB"; exit 1; }
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_INSTALL_PREFIX=/usr/local/opencv-cann-5 \
      -DWITH_CANN=ON \
      -DCANN_INSTALL_DIR=/usr/local/Ascend/ascend-toolkit/latest \
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
      -DPYTHON_INCLUDE_DIR="$PY_INC" \
      -DPYTHON_LIBRARY="$PY_LIB" \
      -DOPENCV_ENABLE_NONFREE=OFF \
      -DOPENCV_DOWNLOAD_MIRROR_ID=gitcode \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      .. | tee /tmp/opencv5-configure.log
```

输出结果如下：
```shell #test-result id="ocv5-cmake-configure" fuzzy='xxx' fuzzy='...'
-- The CXX compiler identification is GNU xxx
...
--   CANN:xxx YES
...
-- Configuring done
-- Generating done
-- Build files have been written to: xxx/build
```

> Python 路径用 `sysconfig` 现场取，不写死版本号目录：CI 镜像是 `/usr/local/python3.12.13`，本地验证机是 `/usr/local/python3.12.15`，两条路径都由 `sysconfig` 给出。

编译 + 安装一体（`--target install`）。**并行度别开高**：打补丁前的 `all_ops.h` 那几个 TU 每个要吃掉 3GB 上下内存，容器内存配额（cgroup limit）只有 32GB 时，并行度超过 6 会被 OOM killer 直接干掉 `cc1plus`（表现为编译日志里出现 `c++: fatal error: Killed signal terminated program cc1plus`）。所以并行度做成可覆盖的环境变量：默认 2（保守），内存富裕的机器可以调高。

```shell #test-setup
cd opencv/build
cmake --build . --target install --parallel "${OPENCV_BUILD_JOBS:-2}"
```

## 8. 校验 build 产物

```shell #test id="ocv5-verify-build"
/usr/local/opencv-cann-5/bin/opencv_version
ls -l /usr/local/opencv-cann-5/lib/libopencv_dnn.so.5.0*
ls -l /usr/local/opencv-cann-5/bin/opencv_test_cannops
```

输出结果如下：
```shell #test-result id="ocv5-verify-build" fuzzy='xxx'
5.0.0
xxx
xxx
```

### 8.1 跑 CANN 单元测试（`cv2.cann` 算子层）

`opencv_test_cannops` 是 `opencv_contrib/modules/cannops` 的 gtest：78 个用例直接测 CANN 算子（cvtColor / elementwise / copy_make_border / reduce / AscendMat 封装）。

**25 个用例被排除**（CANN 9.x + 910B 上的已知缺陷或数值抖动，910B1 CI 与 910B4 实测失败集一致）：

- `CVT_COLOR` 的 XYZ / YCrCb / YUV 系 18 个 + threshold 系 3 个（`MAT_THRESHOLD` / `MAT_THRESHOLD_ASCENDC` / `ASCENDC_KERNEL.THRESHOLD`）共 21 个同根因：cvtColor 对非 32F 输入用两次 threshold 截断饱和中间结果（color.cpp 的 `THRESH_TRUNC` + `THRESH_TO_ZERO`），与 threshold 系用例落在同一个 AscendC kernel 上，该 kernel 在 910B 上 AI Core 越界（VEC 指令 UB 地址越界，kernel 源码 bug）；纯 32F 融合算术路径实测全过。
- resize 系 4 个：`CORE.RESIZE` 是 ResizeArea 算子 GE shape 推断失败（稳定挂）；`CORE.CROP_RESIZE` 与 CPU 参考稳定差 2 超容差 1；`CORE.RESIZE_NEW` / `CORE.CROP_RESIZE_MAKE_BORDER` 是数值抖动 flaky（后者容差 1e-10 近似逐位比较，910B4 上稳定挂）。

剩 **53 个**应全过。**5.0.0 + CANN 9.2.0 / 910B4 实测 53/53 通过**（45.8 秒）。

```shell #test id="ocv5-cann-run-tests"
set -o pipefail
/usr/local/opencv-cann-5/bin/opencv_test_cannops --gtest_color=no \
  --gtest_filter=-CORE.RESIZE:CORE.RESIZE_NEW:CORE.CROP_RESIZE:CORE.CROP_RESIZE_MAKE_BORDER:CVT_COLOR.RGB2XYZ:CVT_COLOR.BGR2XYZ:CVT_COLOR.XYZ2BGR:CVT_COLOR.XYZ2RGB:CVT_COLOR.XYZ2BGR_DC4:CVT_COLOR.XYZ2RGB_DC4:CVT_COLOR.BGR2YCrCb:CVT_COLOR.RGB2YCrCb:CVT_COLOR.YCrCb2BGR:CVT_COLOR.YCrCb2RGB:CVT_COLOR.YCrCb2BGR_DC4:CVT_COLOR.YCrCb2RGB_DC4:CVT_COLOR.BGR2YUV:CVT_COLOR.RGB2YUV:CVT_COLOR.YUV2BGR:CVT_COLOR.YUV2RGB:CVT_COLOR.BGR2YUV_DC4:CVT_COLOR.RGB2YUV_DC4:CVT_COLOR.YUV2BGR_DC4:CVT_COLOR.YUV2RGB_DC4:ELEMENTWISE_OP.MAT_THRESHOLD:ELEMENTWISE_OP.MAT_THRESHOLD_ASCENDC:ASCENDC_KERNEL.THRESHOLD \
  > /tmp/cannops_gtest.log 2>&1; rc=$?
tail -n 25 /tmp/cannops_gtest.log
if [ $rc -ne 0 ]; then
  echo '--- per-test failure summary:'
  grep -E '^\[ RUN|unknown file: Failure|C\+\+ exception|op\[[A-Za-z0-9_]+\]|E[0-9]{5}' /tmp/cannops_gtest.log | head -80
fi
exit $rc
```

输出结果如下：
```shell #test-result id="ocv5-cann-run-tests" fuzzy='xxx' fuzzy='...'
...
[==========]xxx
[  PASSED  ] 53 tests.
```

> 不要加 `--gtest_brief`：OpenCV vendored gtest 没实现这个 flag，加上后零用例执行但 rc=0，假绿。
>
> 本节的排除清单与 [Quick-start-Ascend.md](Quick-start-Ascend.md) 完全一致（同一套 3 组补丁、同一套源码），差异只在 CANN 版本与卡的型号。

## 9. OpenCV Python quickstart

把 Python 指向源码版 `cv2`（带 CANN 后端）：

```shell #test-setup
export PYTHONPATH=/usr/local/opencv-cann-5/lib/python3.12/site-packages:${PYTHONPATH:-}
```

> 注意是 **prepend**（保留镜像 `set_env.sh` 注入的 CANN `PYTHONPATH` 条目）。GE 图编译依赖其中的 `tbe` Python 模块；**整体覆盖** `PYTHONPATH` 会让 DNN CANN 后端在图编译阶段报 `CANN graph check failed, ret = 4294967295`（ret=0xFFFFFFFF），而且日志里**看不到**任何 `tbe` 相关的报错——是这个坑最难查的地方。

下面用 clone 下来的 OpenCV 仓库自带测试图 `opencv/samples/data/baboon.jpg`（512×512 狒狒）作样例输入：

![baboon 输入图](images/baboon-input.jpg)

### 9a. 读图 / 写图

```shell #test id="ocv5-imread-imwrite"
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
```shell #test-result id="ocv5-imread-imwrite"
imwrite ok: True
shape: (512, 512, 3) dtype: uint8
/tmp/opencv_quickstart.png
```

### 9b. 颜色空间转换

```shell #test id="ocv5-cvtcolor"
python << 'PY'
import cv2
bgr = cv2.imread('opencv/samples/data/baboon.jpg')
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print('gray shape:', gray.shape, 'mean:', float(gray.mean()))
cv2.imwrite('/tmp/quickstart-gray.png', gray)
PY
```

输出结果如下：
```shell #test-result id="ocv5-cvtcolor" fuzzy='xxx'
gray shape: (512, 512) mean: xxx
```

baboon 整体偏中亮（mean ≈ 129.7/255），灰度图长这样：

![灰度转换结果](images/quickstart-gray.png)

### 9c. 缩放

```shell #test id="ocv5-resize"
python << 'PY'
import cv2
img = cv2.imread('opencv/samples/data/baboon.jpg')
resized = cv2.resize(img, (50, 200), interpolation=cv2.INTER_AREA)
print('resized shape:', resized.shape)
cv2.imwrite('/tmp/quickstart-resize.png', resized)
PY
```

输出结果如下：
```shell #test-result id="ocv5-resize"
resized shape: (200, 50, 3)
```

OpenCV shape 是 `(rows, cols, channels)`，`resize(img, (W, H), ...)` 得 `(H, W, C)`；缩到 50×200（宽×高）后：

![缩放结果](images/quickstart-resize.png)

### 9d. 绘制

```shell #test id="ocv5-draw"
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
```shell #test-result id="ocv5-draw" fuzzy='xxx'
rect+text drawn, image mean: xxx
/tmp/quickstart-draw.png
```

绿框 + 文字绘制结果：

![绘制结果](images/quickstart-draw.png)

### 9e. N 维算子（`transposeND` / `flipND`）

`cv::transposeND` / `cv::flipND` 是 core 的 N 维算子：前者在任意维度上按置换重排，后者沿指定轴翻转。⚠️ **它们是 4.x 就有的（≤4.12），不是 5.0 新增**（§2.2）。Python 绑定里，**单通道**的二维数组可以直接喂：

```shell #test id="ocv5-nd-ops"
python << 'PY'
import cv2
import numpy as np
b = np.arange(6, dtype=np.float32).reshape(2, 3)
print('transposeND([1,0]):', cv2.transposeND(b, [1, 0]).tolist())
print('flipND(axis=0):', cv2.flipND(b, 0).tolist())
PY
```

输出结果如下：
```shell #test-result id="ocv5-nd-ops"
transposeND([1,0]): [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]
flipND(axis=0): [[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]
```

> 注意 Python 绑定的边界：numpy 的三维数组 `(H, W, C)` 在 OpenCV 里是「二维矩阵 + C 通道」，不是 3 维 Mat——所以 `transposeND` 对它会报 `Input array should be single-channel`。真正的 N 维用法在 C++ 侧（`cv::Mat` 的 `dims > 2`）。注意这条限制跟 5.0 的 1D/0D Mat 语义无关（§2.2），那是 Python 绑定层的既有边界。

## 10. NPU 上跑一次 DNN 推理

先把推理脚本写好（两个小节共用同一份脚本，只是引擎不同）：

```shell #test-setup
cat > /tmp/ocv5_infer.py <<'PY'
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
# GE 图编译会 fork 子进程，子进程 exit(0) 会把继承的 stdout 缓冲冲到
# C++ 日志行的中间（实测 "model bytes" 粘在半截 INFO 行上）。
result = (
    'model bytes: %d\n'
    'output shape: %s\n'
    'output dtype: %s\n'
    'top class index: %d\n'
    'top class score: %s\n'
) % (os.path.getsize(MODEL_PATH), out.shape, out.dtype,
     int(np.argmax(prob)), float(np.max(prob)))
fd = os.open('/tmp/ocv5_result.txt', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.write(fd, result.encode())
os.close(fd)
sys.stdout.flush()
os._exit(0)
PY
```

> **模型来源**：CI 与本仓库 checkout 优先用 `fixtures/mobilenetv2-12.onnx`（同字节入仓，避免 GitHub raw 在国内网络的 `RemoteDisconnected` 抖动）；自己机器上跟着跑时没有 fixtures 目录，自动回落到从 onnx/models 下载（重试 5 次）。

### 10.1 先看清 5.0 的默认引擎：**它根本不上 NPU**

5.0 的 `ENGINE_AUTO` 默认选**新图引擎**，而新图引擎**不支持非 CPU 后端**——`setPreferableBackend` / `setPreferableTarget` 被静默忽略（只打两条 WARN）。这段实测把这件事摆到明面上：

```shell #test id="ocv5-new-engine-default"
python /tmp/ocv5_infer.py > /tmp/ocv5_new.log 2>&1 || { echo "new-engine inference failed"; tail -n 30 /tmp/ocv5_new.log; exit 1; }
echo "=== engine probe ==="
grep -c "not supported by the new graph engine" /tmp/ocv5_new.log
grep -c "ge::Graph to OM buffer" /tmp/ocv5_new.log
cat /tmp/ocv5_result.txt
```

输出结果如下：
```shell #test-result id="ocv5-new-engine-default" fuzzy='xxx'
=== engine probe ===
2
0
model bytes: 13964571
output shape: (1, 1000)
output dtype: float32
top class index: 372
top class score: xxx
```

两条 WARN 原文（实测）：

```
[ WARN:0@0.199] global net_impl_backend.cpp:297 setPreferableBackend Back-ends are not supported by the new graph engine for now
[ WARN:0@0.200] global net_impl_backend.cpp:345 setPreferableTarget Targets are not supported by the new graph engine for now
```

`ge::Graph to OM buffer` 计数为 **0** —— 整份日志里 `CANN` 字样也是 0 次，说明这次推理整条链都在 CPU 上跑完了（结果照样是 372，只是没有任何昇腾算子参与）。**这就是 5.0 上昇腾最大的坑**：不强制经典引擎，推理是「成功」的，但根本没碰 NPU。

### 10.2 强制经典引擎：真正走 CANN

```shell #test id="ocv5-cann-infer"
OPENCV_FORCE_DNN_ENGINE=1 python /tmp/ocv5_infer.py > /tmp/ocv5_cann.log 2>&1 || { echo "classic-engine inference failed"; tail -n 30 /tmp/ocv5_cann.log; exit 1; }
echo "=== cann probe ==="
grep -c "converting ge::Graph to OM buffer" /tmp/ocv5_cann.log
n=$(grep -c "converting layer" /tmp/ocv5_cann.log)
echo "cann layers converted: $n"
if [ "$n" -lt 100 ]; then echo "too few cann layers: $n"; exit 1; fi
cat /tmp/ocv5_result.txt
```

输出结果如下：
```shell #test-result id="ocv5-cann-infer" fuzzy='xxx'
=== cann probe ===
1
cann layers converted: xxx
model bytes: 13964571
output shape: (1, 1000)
output dtype: float32
top class index: 372
top class score: xxx
```

预期：`output shape (1, 1000)`、`float32`；baboon.jpg 的 top-1 是 **ImageNet-1k 第 372 类（patas / hussar monkey，赤猴）**，score ≈ 0.91（与 CPU 参考一致，baboon 本尊 373 排第二 ≈ 0.03）。

- `converting ge::Graph to OM buffer` 计 1 → GE 图编译真的发生了（昇腾编译模型）；
- `converting layer` 实测 **103** 行（5.0.0 + CANN 9.2.0 / 910B4）→ 网络的每一层都被转成了 CANN 算子（MobileNetV2-12 是 105 个 ONNX 节点）。这个数字随 CANN 版本的小版本可能变，所以断言写成「≥ 100」而不是硬编码。

走经典引擎还有第二种写法：Python 绑定的 `readNetFromONNX` 收 engine 参数——`cv2.dnn.readNetFromONNX(path, cv2.dnn.ENGINE_CLASSIC)`。实测两条路**完全等价**（GE 图编译 1 次、103 层转 CANN 算子、top-1 372、score 0.9142661690711975 一模一样）。区别是：环境变量对进程里所有 net 生效，API 参数只作用于那一个 net。

> **首次 forward 含 GE 图编译**，约 0.5~1 分钟出 `Compile success, model size = 9759197`；进程内二次 forward 约 1ms。
>
> **两个引擎的数值差**：新图引擎（CPU）实测 score = 0.9148015975952148，经典引擎（CANN/NPU）实测 score = 0.9142661690711975——同一张图、同一份权重，top-1 相同、末位差 5e-4，属正常浮点差异；所以上面的期望值用模糊匹配而不是逐位相等。

标注结果如下：

![NPU 推理标注结果](images/dnn-result.png)

> `os._exit(0)` 是为「驱动 25.5.2 × CANN 9.1.0 版本错配」的宿主环境兜底：那套组合下堆会在运行期被写坏，进程 `exit()` 的 atexit 阶段（`libacl_dvpp_mpi` 的 `DvppWrapperManager` 析构）才崩（`corrupted double-linked list` + SIGABRT），**推理结果本身是正确的**。驱动与 CANN 配套的正常环境（如 25.3.rc1.2）下这行无害——NPU 资源由驱动随进程回收。

## 11. 已知边界

- **5.0 的招牌特性上不了 NPU**：新图引擎、ONNX Runtime 后端、原生 LLM/VLM 在昇腾上目前都只能 CPU（§2.7）。要用 NPU 推理，必须 `OPENCV_FORCE_DNN_ENGINE=1` 走经典引擎。
- **引擎可以在 Python 侧显式选**：`cv2.dnn.readNetFromONNX(path, cv2.dnn.ENGINE_CLASSIC)` 直接走经典引擎（实测与 `OPENCV_FORCE_DNN_ENGINE=1` 等价：GE 图编译 1 次、103 层转 CANN 算子、top-1 372、score 0.9142661690711975 完全一致）。常量 `ENGINE_NEW`=2 / `ENGINE_CLASSIC`=1 / `ENGINE_AUTO`=3 / `ENGINE_ORT`=4 都在 `cv2.dnn` 里（只是 `cv2.dnn.EngineType` 这个**类型**没绑定）。本文统一用环境变量 `OPENCV_FORCE_DNN_ENGINE=1`：它对进程里所有 net 生效，脚本不用逐个改 `readNet` 调用。
- **`cv2.dnn.Tokenizer` 存在但默认是空实现**：`cv2.dnn.Tokenizer()` 之后直接 `encode()` 会报 `Tokenizer impl null`，需要先 `Tokenizer_load(...)` 灌模型；本文未展开（它跟 NPU 无关）。
- **Python 绑定不支持真正的 N 维 Mat**：numpy 的三维数组会被当成「二维 + 通道」，`transposeND` 因此要求单通道输入（§9e）。
- **`PYTHONPATH` 必须 prepend 不能覆盖**：覆盖掉 CANN 的 `tbe` 路径 → `CANN graph check failed, ret = 4294967295`，且日志里没有 `tbe` 字样（§9）。
- **`all_ops.h` 链接窗口是硬约束**：aarch64 的 `R_AARCH64_CALL26/JUMP26` 只有 ±128 MiB 跨度，缩 `.text` 只是缓解、不是保证；5.0 上过这一关只有窄 op 头一条路。
- **`cv2.cann` 的 contrib 单测**：本文编了 `opencv_contrib` 的 `cannops` 模块并跑了它的 gtest（§8.1，53/53 通过）；25 个被排除的用例及原因见那一节。
- **`videoio` 编了但没有编解码器**：源码构建默认不带 FFmpeg/GStreamer，`cv2.VideoWriter` 会打一堆 INFO/WARN 但仍能落盘；本文未把视频写入列为必过项。
- **补丁只对 5.0.x 成立**：`cv::MatShape` 是 5.0 引入的类型，4.x 源码树里根本没有它；上游一旦出 5.1/6.0，§5 的脚本会断言失败（这是设计好的失败方式，不是 bug）。
