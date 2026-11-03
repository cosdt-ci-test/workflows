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

### Questions

1. Should the CANN backend be ported to the new graph engine, or dropped from
   5.x?
2. If it is kept: building 5.x with `WITH_CANN=ON` currently fails to compile
   (a separate issue, the CANN sources still use the pre-5.0 shape type), so
   today a CANN build of 5.x is not possible at all.
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
- [ ] 上游是否接受「CANN 后端在 5.x 上不可达」这个结论（属于设计决策，
      本 issue 只陈述事实 + 提问，不带补丁）
