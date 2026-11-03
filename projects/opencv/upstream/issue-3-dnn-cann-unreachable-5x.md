# 上游 issue 草稿 3：5.x 上 CANN 后端不可达（经典引擎已被移除）

> 状态：**待提交**。目标仓库 `opencv/opencv`，base 分支 `5.x`。
> 这是**第三个**独立问题（组 1 = 编译错，组 2 = 链接错，本条 = 后端不可达）。
> 组 1 的 issue 草稿见 `issue-1-dnn-cann-matshape.md`。
> **本条不需要配套 PR**：它是现状陈述 + 提问，最终怎么改由上游决定。

---

## Title

```
dnn: the CANN backend is unreachable on 5.x after the classic engine was removed
```

## Body

```markdown
### Problem

On 5.x the CANN backend cannot be reached by any model. `Net::setPreferableBackend(DNN_BACKEND_CANN)`
is accepted but has no effect -- the model runs entirely on the CPU.

Measured on 5.x HEAD with `WITH_CANN=ON` (Atlas 900 A2 PODc / 910B4, CANN 9.2.0,
gcc 11.4.0), MobileNetV2 in ONNX:

```
DNN/ONNX: loading ONNX v7 model produced by 'pytorch':1.8. Number of nodes = 105, initializers = 177, inputs = 1, outputs = 1
Back-ends are not supported by the new graph engine for now
Targets are not supported by the new graph engine for now
DNN/NewEngine: finalize() backend=8 target=0 useCUDA=0 over 1 graph(s)
DNN/NewEngine: finalize op #0 'Conv_0' (Conv2) -> CPU
DNN/NewEngine: finalize op #1 'Conv_2' (Conv2) -> CPU
...
DNN/NewEngine: finalize op #60 'Gemm_104' (Gemm) -> CPU
```

`backend=8` is `DNN_BACKEND_CANN` and `target=0` is `DNN_TARGET_CPU`: the request
is recorded and then ignored. Not a single layer is converted to a CANN
operator, and the forward pass on the "NPU" takes exactly as long as the one on
the CPU (measured ratio 1.00).

### Why

`modules/dnn/src/net_impl_backend.cpp` refuses every non-CPU back-end as soon as
the network lives on the new graph engine:

```cpp
    if (mainGraph)
    {
        if (backendId == DNN_BACKEND_OPENCV
#ifdef HAVE_CUDA
            || backendId == DNN_BACKEND_CUDA
#endif
            )
        { ... return; }
        CV_LOG_WARNING(NULL, "Back-ends are not supported by the new graph engine for now");
        preferableBackend = backendId;
        return;
    }

    clear();
    ...
    else if (backendId == DNN_BACKEND_CANN)
    {
#ifdef HAVE_CANN
        switchToCannBackend(net);
#else
        CV_Error(Error::StsNotImplemented, "CANN backend is not availlable in the current OpenCV build");
#endif
    }
```

`switchToCannBackend()` (`modules/dnn/src/net_cann.cpp`) is the only entry point
to the CANN backend, and it is reached only when `mainGraph` is null -- that is,
when the network was built by the classic engine.

Since #29341 ("disable_engine_classic", merged 2026-06-18, commit 49b8f7c03f)
there is no classic engine left to build a network with:

* `ENGINE_CLASSIC` was removed from `cv::dnn::EngineType`; the enum is now
  `ENGINE_AUTO=0, ENGINE_OPENCV=1, ENGINE_ORT=2`;
* `readNetFromONNX()` no longer dispatches to the classic importer
  (`detail::readNetDiagnostic<ONNXImporter>`) -- it always calls
  `readNetFromONNX2()`, which builds the new graph engine
  (`modules/dnn/src/onnx/onnx_importer2.cpp`: `netimpl->mainGraph = mainGraph;`);
* the Caffe, TensorFlow and Darknet readers are gone
  (`readNetFromCaffe()` / `readNetFromTensorFlow()` / `readNetFromDarknet()` are
  not declared any more), and `readNetFromTFLite()` builds the new engine too
  (`modules/dnn/src/tflite/tflite_importer.cpp`).

So every model that can be loaded in 5.x sets `mainGraph`, and the CANN branch
above is dead code.

### It is not CANN-specific: 5.x has no accelerator left but CUDA

The same two functions gate every other accelerator as well. The new graph
engine only lets through two back-ends and two targets:

| allowed | rejected (a warning, then the request is treated as CPU) |
|---|---|
| `DNN_BACKEND_OPENCV`, `DNN_BACKEND_CUDA` | `DNN_BACKEND_CANN`, `DNN_BACKEND_INFERENCE_ENGINE(_NGRAPH)`, `DNN_BACKEND_TIMVX`, `DNN_BACKEND_WEBNN`, `DNN_BACKEND_VKCOM` |
| `DNN_TARGET_CPU`, `DNN_TARGET_CUDA` / `CUDA_FP16` | `DNN_TARGET_NPU`, `DNN_TARGET_MYRIAD`, `DNN_TARGET_HDDL`, `DNN_TARGET_OPENCL` / `OPENCL_FP16`, `DNN_TARGET_VULKAN`, `DNN_TARGET_FPGA`, `DNN_TARGET_CPU_FP16` |

Measured on the same build (5.x + the CANN build fix, MobileNetV2 ONNX); the
"effective" column is the `backend` / `target` the engine finalized with, and
`cpu ops` is the number of operators that were finalized on the CPU:

```
combo                | effective      | cpu ops | warning
---------------------+----------------+---------+--------------------------
CANN + NPU           | b=8 t=0        | 120     | backend rejected
CANN + CPU           | b=8 t=0        | 120     | backend rejected
OPENCV + OPENCL      | b=3 t=0        | 120     | target rejected
OPENCV + VULKAN      | b=3 t=0        | 120     | target rejected
OPENCV + CPU_FP16    | b=3 t=0        | 120     | target rejected
OPENVINO + MYRIAD    | b=1000000 t=0  | 120     | backend rejected
CUDA + CUDA          | b=5 t=0        | 120     | backend rejected (no CUDA in this build)
OPENCV + CPU         | b=3 t=0        | 120     | -
```

Every accelerator combination ends up executing on the CPU. (`t=0` is
`DNN_TARGET_CPU`.)

### Why 4.x still has a working CANN backend

The CANN backend was hooked onto the *classic* engine, and 4.x was never
rewritten to the new engine:

* 4.x has only `modules/dnn/src/net_impl.cpp` -- there is no `net_impl2.cpp`,
  no `mainGraph`, and `net_impl_backend.cpp` has no `if (mainGraph)` branch at
  all, so `switchToCannBackend()` is reached directly;
* the Caffe, Darknet and TensorFlow importers are still present in 4.x;
* `modules/dnn/src/op_cann.cpp` in 4.x is a real implementation (the per-layer
  `initCannOp()` functions really build CANN GE operators).

The 4.x build break was a much smaller thing: issue #26175 and #26631, fixed
by #26632 ("dnn/gelu_cann"), whose whole diff is 9 lines -- a
`GeluFunctor::initCannOp()` that throws `CV_Error(Error::StsNotImplemented, "")`
because the CANN toolkit had no GELU operator. The CANN backend itself was not
disabled.

5.0.0 was the last release where both engines coexisted: `EngineType` still
had `ENGINE_CLASSIC` / `ENGINE_NEW` / `ENGINE_AUTO`, and `readNetFromONNX()`
still had the classic path. The CANN backend was therefore still reachable
there (see below).

5.x is where the classic engine disappeared:

| change | date | effect |
|---|---|---|
| #26056 "New dnn engine" | 2024-10-16 | the new graph engine and the `MatShape` structure were introduced |
| #26334 "dnn_engine_change" | 2024-11-08 | the `if (mainGraph)` branch that rejects non-CPU back-ends was added |
| "Darknet cleanup" | 2026-04-30 | the Darknet importer was removed |
| #28678 "caffe-importer-cleanup" | 2026-06-02 | the Caffe importer was removed |
| **5.0.0 tag** | **2026-06-05** | classic engine still present, CANN still reachable |
| **#29341 "disable_engine_classic"** (`49b8f7c03f`) | **2026-06-18** | `ENGINE_CLASSIC` and the classic ONNX importer were removed -- the classic engine is gone, and with it the only way to reach the CANN backend |
| #29474 "extra_parsers" | 2026-10-22 | the TensorFlow importer was removed; only ONNX and TFLite remain, and both build the new engine |

### It used to work

In 5.0.0 the classic engine was still selectable, and the ONNX reader still had
the classic path:

```cpp
    switch(engine)
    {
        case ENGINE_CLASSIC:
            return detail::readNetDiagnostic<ONNXImporter>(onnxFile.c_str());
        case ENGINE_NEW:
            return readNetFromONNX2(onnxFile);
```

With `OPENCV_FORCE_DNN_ENGINE=1` (which is `ENGINE_CLASSIC` in 5.0.0) on 5.0.0,
the CANN backend converts 124 layers of MobileNetV2 to CANN operators, the NPU
result matches the CPU reference (top-1 372 for both, score 0.91427 on the NPU
against 0.91480 on the CPU), and the NPU forward takes 1 ms against 34 ms on the
CPU.

`modules/dnn/src/net_cann.cpp`, `op_cann.cpp` and `op_cann.hpp` are
byte-identical between 5.0.0 and 5.x -- the backend itself did not change, only
the way a network is built.

### Why the backend cannot simply be re-wired

The CANN backend is a `Net::Impl` subclass (`NetImplCann`) whose entire
integration consists of two hooks:

* `NetImplCann::initBackend()` (`modules/dnn/src/net_cann.cpp:101`) walks the
  layers, builds the CANN GE graph, compiles it into one OM buffer
  (`compileCannGraph()`, `net_cann.cpp:292`) and attaches a `CannBackendNode`
  to every layer;
* `NetImplCann::forwardLayer()` (`net_cann.cpp:254`) does not execute any layer
  arithmetic -- it calls `cannNode->net->forward()` once, launching the whole
  compiled OM graph. The per-layer hook is only used as a trigger.

In 5.x both hooks are called only from the classic engine:

```
modules/dnn/src/net_impl.cpp:267:        initBackend(blobsToKeep_);
modules/dnn/src/net_impl.cpp:921:                    forwardLayer(ld);
modules/dnn/src/net_impl.cpp:987:        forwardLayer(ld);
modules/dnn/src/net_impl.cpp:991:    forwardLayer(ld);
```

The new engine's `Net::Impl::forwardMainGraph()` (`modules/dnn/src/net_impl2.cpp:917`)
never calls them: it calls `finalize()` (which selects a per-op executor) and
then `forwardGraph(mainGraph, inputs, outputs, true)`.

`NetImplOpenVINO` (`modules/dnn/src/net_openvino.cpp`) has exactly the same
shape -- `initBackend()` + `forwardLayer()` -- which is why the OpenVINO
back-end is dead on 5.x as well (measured below: `OPENVINO + MYRIAD` is
rejected and all 120 operators run on the CPU).

### The per-operator layer is intact -- it is the bridge that is missing

The Ascend-side operator knowledge does **not** have to be written again.
5.x renamed the hook from 4.x's `initCannOp` to `initCann`
(`modules/dnn/include/opencv2/dnn/dnn.hpp:474`) and 28 layer classes implement
it (convolution, deconvolution, gemm, matmul, fully connected, pooling,
softmax, concat, batch norm, elementwise, reduce, resize, slice, ...).
`net_cann.cpp:205` still calls it:

```cpp
auto backendNode = layer->initCann(ld.inputBlobsWrappers, ld.outputBlobsWrappers, layerInputNodes);
```

`op_cann.cpp` (329 lines, the CANN GE operator builders) and `net_cann.cpp`'s
graph builder / OM compiler / `CannNet` launcher are all present as well.

What is missing is the bridge to the new engine's data model. The hook speaks
`BackendWrapper` -- the classic engine's device-blob abstraction
(`getHostMat()`, `copyToHost()`, `setHostDirty()`) -- while the new engine
passes `Arg` / `Tensor` / `UMat` around. And `net_cann.cpp`'s `initBackend()`
walks the classic engine's `Net::Impl::layers` and its `LayerData`
(`ld.inputBlobsWrappers`, `ld.backendNodes`, `ld.skip`), none of which exists
in the new engine.

There is no generic slot to plug into either. CUDA got a new-engine-specific
overload -- `Layer::forwardCUDA()` and the `UMat`-based `initCUDA(context,
inputs, outputs)` -- with an explicit note in `modules/dnn/src/layer.cpp`:

```cpp
    // new graph engine only need to override initCUDA(context, inputs, outputs) with UMat.
```

Every other back-end (TimVX, VkCom, WebNN, OpenVINO) is still on the
`BackendWrapper` API and is therefore dead on 5.x as well.

### Two possible routes, and they differ by an order of magnitude

* **Per-operator port, like CUDA.** Give CANN a `UMat`-based `initCann`
  overload and a CANN branch in `finalizeGraph()` / `forwardGraph()`, then
  re-express the 28 operators in the new engine's `Arg`/`Tensor` semantics.
  This is what upstream did for CUDA: `modules/dnn/src/cuda4dnn/` is **103
  files, 46 of them primitives** -- i.e. every operator rewritten.
* **Whole-model offload, like ONNX Runtime.** Hand the whole model to CANN,
  compile it once and run it as a single program. The new engine already has
  this shape (see below), and the CANN back-end already has whole-model
  semantics (compile once into an OM buffer, launch once).

### A whole-model offload already exists on the new engine

The good news is that the new engine already has the shape a CANN port would
need. `forwardMainGraph()` starts with an ONNX Runtime whole-model bypass:

```cpp
void Net::Impl::forwardMainGraph(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
{
#ifdef HAVE_ONNXRUNTIME
    if (useOrtEngine && mainGraph && modelFormat == DNN_MODEL_ONNX && !modelFileName.empty())
        finalizeOrt();
    if (this->ort_session)
    {
        ...
        std::vector<Mat> ortOuts = runOrtSession(netInputLayer->blobs, std::vector<int>());
        ...
        return;
    }
#endif
    ...
    finalize();  // select per-op executors for the chosen backend/target (idempotent)
    ...
    forwardGraph(mainGraph, inputs, outputs, true);
```

The open PR #29726 ("dnn: add AMD MIGraphX (ROCm) backend for ONNX models")
states that it mirrors this bypass:

> The hook lives in the new graph engine's forwardMainGraph(), mirroring the
> ONNX Runtime whole-model bypass; inputs are read from the main graph tensors
> and bound to the program's own parameter names. On any parse/compile failure
> it falls back to the default engine, so results stay correct.

The CANN backend's semantics already are a whole-model offload (compile the
graph once into an OM buffer, launch it once), so the same shape would fit it:
build the OM once, run it as a single program, fall back to the default engine
on failure.

### Empirical validation: the classic-engine bridge alone is enough

Before asking upstream to pick a route, we checked whether what is missing is
the Ascend-side operator knowledge or only the bridge to the new engine's data
model. It is only the bridge -- and the bridge is one file.

The gate is a single condition. `Net::Impl::setPreferableBackend()`
(`modules/dnn/src/net_impl_backend.cpp:282`) refuses every non-OpenCV/CUDA
back-end once the new engine is in use, and reaches the CANN switch only
otherwise:

```cpp
    if (mainGraph)
    {
        if (backendId == DNN_BACKEND_OPENCV
    #ifdef HAVE_CUDA
            || backendId == DNN_BACKEND_CUDA
    #endif
            )
        { ...; return; }
        CV_LOG_WARNING(NULL, "Back-ends are not supported by the new graph engine for now");
        preferableBackend = backendId;
        return;
    }

    clear();
    ...
    else if (backendId == DNN_BACKEND_CANN)
    {
    #ifdef HAVE_CANN
        switchToCannBackend(net);
```

`mainGraph` is non-null exactly when the model was built by the new graph
engine's importer. The classic engine's importer -- still in the tree, its
`Net::Impl` API (`addLayer()`, `connect()`, `registerOutput()`,
`setInputShape()`) unchanged -- leaves it null. So a PoC that restores only the
classic importer and the dispatch that selects it, in
**one file** (`modules/dnn/src/onnx/onnx_importer.cpp`), puts `mainGraph` back
to null and makes the CANN back-end reachable again. Nothing in `net_cann.cpp`,
`op_cann.cpp` or any of the 28 `initCann()` implementations has to change.

Measured on an Atlas 900 A2 PODc (Ascend 910B4, CANN 9.2.0, gcc 11.4.0), 5.x
+ the classic-importer PoC, MobileNetV2-12 ONNX (13,964,571 B), both runs
through the *same* importer and the *same* input:

```
DNN: switching to CANN backend... (networkID=1)
DNN/CANN: converting layer Conv_0@Conv2@2 to CANN operator        (x118)
DNN/CANN: building ge::Graph
DNN/CANN: converting ge::Graph to OM buffer
Compile success, model size = 9762434
DNN/CANN: OM buffer size = 9762434
Load model to NPU memory
Create model description
createInputDataset length = 602112
```

| | first forward | second forward (pure inference) | top-1 | top-1 probability |
|---|---|---|---|---|
| OpenCV CPU | 57.60 ms | 30.97 ms | 549 | 0.051117 |
| CANN / NPU | 19242 ms (one-off GE -> OM compile + load) | **0.82 ms** | 549 | 0.050227 |

`npu-smi info` during the run:

```
| NPU | Chip | Process id | Process name | Process memory(MB) |
| 0   | 0    | 24183      | cann_e2e     | 108                |
```

Each back-end's second forward is bit-identical to its own first one
(`rerun_diff = 0`). The residual difference between the two back-ends (max abs
0.045 on the logits, top-1 probability 0.00089 apart) is the expected
fp16-vs-fp32 difference of the NPU.

So on 5.x the CANN back-end is not broken, and the Ascend operator layer does
not have to be rewritten: the whole model still converts into a single OM and
runs on the NPU in ~0.8 ms -- as long as the CANN hooks stay wired to the
classic engine's data model, which is what they were written for.

### Related work

* #25025 "Introducing non-CPU HAL for OpenCV 5+" -- the umbrella design
  discussion about non-CPU acceleration in 5.x (still open).
* #29726 "dnn: add AMD MIGraphX (ROCm) backend for ONNX models" -- a new
  back-end being added *to the new engine*, by mirroring the ORT bypass.
* #30439 "dnn(cuda): transformer layer coverage and CUDA Graph replay on the
  new engine" -- CUDA being ported to the new engine.

### Questions

1. Should the CANN backend be ported to the new graph engine, or dropped from
   5.x?
2. If it is kept: building 5.x with `WITH_CANN=ON` currently fails to compile
   (a separate issue, the CANN sources still use the pre-5.0 shape type), so
   today a CANN build of 5.x is not possible at all.
3. If it is ported: would a whole-model offload (the compiled OM launched as a
   single program from `forwardMainGraph()`, following the ONNX Runtime /
   MIGraphX pattern) be acceptable, or is a per-operator implementation in the
   new engine's executor model expected? The former can reuse `op_cann.cpp` and
   the 28 `initCann()` implementations; the latter is a rewrite of every
   operator, of the size of `modules/dnn/src/cuda4dnn/` (103 files).
4. Would re-enabling the classic engine path be acceptable as an interim
   measure? The classic engine is still in the tree and only the ONNX importer
   and its dispatch were removed; putting the classic importer back is a
   one-file change and it makes the CANN back-end work again (measured above:
   118 layers -> one OM -> 0.82 ms on the NPU). The alternative is to write
   the whole-model CANN bypass directly into the new engine.
```

---

## 提交前需要补的实证数据

- [x] 5.x 上 ONNX + CANN 请求被无视的**逐字**日志 —— 已取到并写进正文
      （`Back-ends are not supported by the new graph engine for now` /
      `DNN/NewEngine: finalize() backend=8 target=0` / 61 个 op 全部 `-> CPU`）
- [x] NPU 与 CPU 耗时比 = 1.00（说明确实没有上卡）
- [x] 5.x 里 CANN 不可达的两条独立证据：
      - `modules/dnn/src/net_impl_backend.cpp` 的 `if (mainGraph)` 分支
      - `EngineType` 里 `ENGINE_CLASSIC` 被删（commit `49b8f7c03f`
        "engine classic removed"，经 PR #29341 合并，2026-06-18，晚于 5.0.0 tag
        的 2026-06-05）
- [x] 5.x 里已无 Caffe / TensorFlow / Darknet 读取入口
      （`readNetFromCaffe` / `readNetFromTensorFlow` / `readNetFromDarknet`
      在 `modules/dnn/include` 里已不存在）
- [x] 5.0.0 上 CANN 后端可达且真的跑过 NPU（124 层 / top-1 372 /
      NPU 0.91427 vs CPU 0.91480 / 1 ms vs 34 ms）
- [x] `net_cann.cpp` / `op_cann.cpp` / `op_cann.hpp` 在 5.0.0 与 5.x 之间逐字节相同
- [x] **不是 CANN 专属**：探针实测 8 组 (backend, target) 组合，除纯 CPU 外
      全部塌回 CPU（120 个 op 全在 CPU）；5.x 只剩 OPENCV / CUDA 两个后端
      和 CPU / CUDA 两个 target 可用
- [x] 4.x 为什么还有：CANN 挂在经典引擎上，而 4.x 从没被新引擎改造过
      （4.x 没有 `net_impl2.cpp`、`net_impl_backend.cpp` 里没有 `mainGraph`
      分支、Caffe/Darknet/TF importer 都还在、`op_cann.cpp` 是真实现）；
      4.x 那次断裂只是 GELU 少了个 `initCannOp`，#26632 用 9 行桩修掉
- [x] 5.0.0 → 5.x 的时间线（#26056 → #26334 → darknet/caffe 清理 →
      5.0.0 tag → **#29341 删掉经典引擎** → #29474 删 TF importer）
- [x] 查清**逐算子那一层还在不在** —— 在：5.x 把钩子从 4.x 的
      `initCannOp` 改名成 `initCann`（`dnn.hpp:474`），**28 个 layer 类
      都实现了**（convolution / deconvolution / gemm / matmul / fc / pooling /
      softmax / concat / batch_norm / elementwise / reduce / resize / slice …），
      `net_cann.cpp:205` 仍在调用；`op_cann.cpp`（329 行 GE 算子构造）与
      `net_cann.cpp` 的建图 / 编 OM / 启动全在
      ⇒ **昇腾侧的算子知识不用重写**，缺的是到新引擎数据模型的桥
- [x] 查清**桥缺在哪**：钩子用的是经典引擎的设备 blob 抽象
      `BackendWrapper`（`getHostMat()` / `copyToHost()` / `setHostDirty()`），
      新引擎传的是 `Arg`/`Tensor`/`UMat`；`net_cann.cpp` 遍历的是经典引擎的
      `Net::Impl::layers` + `LayerData`，新引擎里不存在。
      没有通用插槽：CUDA 是靠新增 UMat 版 `initCUDA` 重载 +
      `Layer::forwardCUDA()` 接上的（`layer.cpp:67-73` 有明确注释），
      TimVX / VkCom / WebNN / OpenVINO 都还停在 `BackendWrapper` 形态
      ⇒ 在 5.x 上一起失效
- [x] 量化两条移植路线的差距：
      **A 逐算子**（跟 CUDA 一样）—— 上游给 CUDA 做了
      `modules/dnn/src/cuda4dnn/`：**103 个文件、其中 46 个 primitive**，
      等于每个算子重写一遍；CANN 28 个算子同量级
      **B 整模型旁路**（跟 ORT / PR #29726 一样）—— CANN 本来就是整模型语义，
      可保留 `op_cann.cpp` 与 28 个 `initCann`，只换「谁遍历图、谁喂 blob、
      谁启动」⇒ 大概是 `net_cann.cpp`（373 行）量级的改造
- [x] 查清**为什么不能只是把线重新接上**：`NetImplCann` 只有
      `initBackend()` + `forwardLayer()` 两个钩子，而这两个在 5.x 里
      **只被经典引擎调用**（`net_impl.cpp:267 / 921 / 987 / 991`）；
      新引擎的 `forwardMainGraph()`（`net_impl2.cpp:917`）走的是
      `finalize()` + `forwardGraph()`，完全不碰这两个钩子。
      `NetImplOpenVINO` 结构相同 ⇒ OpenVINO 后端在 5.x 上同样失效
      （实测 `OPENVINO + MYRIAD` 被拒、120 个 op 全在 CPU）。
- [x] 找到**移植的现成模板**：新引擎的 `forwardMainGraph()` 开头就有
      ONNX Runtime 的整模型旁路（`runOrtSession()` + `return`），
      进行中的 PR #29726（MIGraphX 后端）明说是「mirroring the ONNX Runtime
      whole-model bypass」；而 CANN 后端本身已经是整模型语义
      （`initBackend()` 编译出一个 OM，`forwardLayer()` 只调一次
      `cannNode->net->forward()`），形态吻合。
- [x] 查过上游**没有重复 issue**：`api.github.com/search/issues` 搜
      `repo:opencv/opencv` 的 title 含 CANN 的 25 条，最新一条是 2025-06 的
      #27402，没有任何一条讲 5.x 的 CANN 编译失败或后端不可达；
      搜 `disable_engine_classic` / `DNN_BACKEND_CANN` / `5.x accelerator`
      也都没有命中。
- [x] **实测验证方案 B 的桥足够**：在 npu-4（910B4 / CANN 9.2.0）上做了
      「只恢复经典 importer + 调度」的 PoC（**一个文件**
      `modules/dnn/src/onnx/onnx_importer.cpp`），`net_cann.cpp` /
      `op_cann.cpp` / 28 个 `initCann` 一行未改 ⇒ CANN 后端重新可达且
      **真跑在 NPU 上**：
      - 118 个层转成 CANN 算子 → GE 图 → OM（9,762,434 B）→ 装载上卡 →
        `createInputDataset` → `aclmdlExecute`
      - 纯推理 **0.82 ms** vs CPU 30.97 ms（**37.6×**）；首次 forward 的
        19.2 s 是一次性 GE→OM 编译，不是推理
      - `npu-smi info` 抓到进程：`NPU 0 | pid 24183 | cann_e2e | 108 MB`
      - CPU 与 NPU top-1 同为 549，概率差 0.00089（fp16 vs fp32 的正常差），
        各自二次 forward 逐位相同（`rerun_diff = 0`）
      ⇒ **缺的确实只是桥（`mainGraph` 非空与否），不是昇腾侧的算子知识**
- [x] 定位到闸门只有一处：`Net::Impl::setPreferableBackend()`
      （`net_impl_backend.cpp:316`）的 `if (mainGraph)` —— `mainGraph` 为空
      （经典 importer 导入）才会走到 `switchToCannBackend()`
- [ ] 上游是否接受「CANN 后端在 5.x 上不可达」这个结论（属于设计决策，
      本 issue 只陈述事实 + 提问，不带补丁）
