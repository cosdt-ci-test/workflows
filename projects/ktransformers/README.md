# ktransformers

本目录是 [ktransformers](https://github.com/kvcache-ai/ktransformers) 的看护配套数据，不是 ktransformers 源码。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

上游默认分支是 `main`。上游没有昇腾 CI。本仓先走阶段 A。

用户向的昇腾入口在 [docs/Quick-start-Ascend.md](docs/Quick-start-Ascend.md)，只指向上游官方教程，**没有** `#test` 标签。本仓**不**接 Quick Start 流水线：不要增加 `*-quick-start.yml`，也不要在 `projects.yaml` 里写 `quick_start`。接上共享引擎会空跑通过（无标签块全部跳过）。

上游教程写明支持的 NPU 是 **Atlas 300I A2**。本仓 runner 是 910B，不能按那两篇的字面步骤做看护。

## 当前没有 example 流水线

清单 `supported` 为空（原因见「清单」一节），因此本仓没有 `ktransformers-examples.yml`，也没有 `projects/ktransformers/scripts/`。`supported` 为空时流水线唯一剩下的功能是「上游发 release 后核对清单是否覆盖新增文件」，抵不上维护成本，2026-08-28 起删除；清单保留为静态清单，README 记录每条 `unsupported` 的证据。

复审触发条件满足（见「清单」一节末尾）需要重建看护时，按 example 看护落地流程重新补 `supported` 条目、`setup_example.sh` / `run_example.sh` 和 workflow，每条 `supported` 在 coder 上逐条验证。

## 清单

`examples_manifest.yaml` 的扫描根是上游 `kt-kernel/examples`，只扫 `.py`。`kt-kernel/examples/test_rope.cpp` 不是扫描单位。重新生成会整文件覆盖 `--output`。生成器不会合并已经写好的分组注释。

```bash
python3 scripts/bootstrap_manifest.py \
  --target-root /path/to/ktransformers \
  --output projects/ktransformers/examples_manifest.yaml \
  --scan-root kt-kernel/examples \
  --include-extension .py
```

`supported` 为空，原因如下。`unsupported` 只表示本看护体系当前不跑，不是社区支不支持。

当前产品入口是 `kt-kernel`。它的 CMake 只有 CUDA / ROCm / MUSA / MACA / SYCL / CPU（AMX、AVX、KML、llamafile），**没有 CANN / NPU 后端**。这些 `.py` 是 CPU 或 NVIDIA 侧的 kernel 自检：

- 助手模块：`configuration_deepseek_v3.py`、`modeling_deepseek_v3.py`、`torch_attention.py`
- 调试 / 复现：`repro_llamafile_re.py`、`test-debug.py`
- 依赖本地 dump 文件：`test_softmax.py`、`test_rope.py`、`test_mla_quant.py`
- 未写完的代码片段（只定义函数、不执行任何逻辑）：`test_apply_rope.py`
- Intel AMX：`bench_moe_amx_int8.py`、`test_*amx*`、`test_bf16_moe.py`、`test_fp4_moe_v4.py`、`test_fp8_*.py`、`test_write_buffer.py`、`test_mxfp8_moe_m3.py` 等。aarch64 910B 没有 AMX
- AVX2：`test_fp4_moe_avx2.py`、`test_mxfp8_moe_avx2.py`
- 上游已坏的 KML 脚本：`test_moe_kml.py`、`test_deepseekv3.py`、`test_deepseekv3_prefill.py`、`test_deepseekv3_prefill_speed.py` 调用的 `KMLInt8_MOE` / `KMLInt4_MOE` 已不在当前 `ext_bindings.cpp` 的导出里；deepseekv3 三个还硬编码读 `/home/bd/models/DeepSeek-R1*` 整模权重
- 上游已坏的 CPUInfer 脚本：`test_linear.py`、`test_mlp.py` 用的 `linear.Linear` / `mlp.MLP` 在绑定源码里被注释掉了；`test_mla_torch.py` 用的 `mla.MLA` 也已不导出
- 硬编码本地整模 GGUF（`/home/bd/models/DeepSeek-R1-BF16`，两个文件都用 `if use_real_weights := True:` 恒真强制读取）：`test_mla.py`、`test_mla_qlen.py`
- 硬 NVIDIA 依赖：`test_moe.py`（三处 `device="cuda"` 无参数可绕）、`test_attention.py`（`flash_attn`）
- 纯 torch 参考实现，不 import 产品代码、无断言：`test_mla_simple.py`

2026-08-28 复审结论（对上游 `main` @ `0635007` 逐文件核对）：上游在 2025-11-03（f854d03b）和 2025-12-11（53f6a6d6）**删除了 KML 后端源码**（`operators/kml/`、`moe_kernel/mat_kernel/kml_kernel/`），CMake 里的 `KTRANSFORMERS_CPU_USE_KML` 等开关悬空——aarch64 上打开会对不存在的目录 `add_subdirectory`，配置期即失败。`test_gate.py`、`test_moe_kernel.py` 本是仅有的两条纯 CPU、带真实断言、逻辑上可在 aarch64 跑的脚本，但它们需要的 `gate.MoEGate` / `moe.Int8_KERNEL_MOE` 在 aarch64 上依赖已删除的 KML 源码。其中 `Int8_KERNEL_MOE` 的绑定条件已放宽为 `USE_MOE_KERNEL`，但 `operators/moe_kernel/la/` 只是分发层，四个 `*_cblas_gemm_s8s8s32` 符号在全仓唯一的实现是 AOCL（x86）；且 `api/common.h` 在 `__aarch64__` 上强制 `#define CPU_USE_KML`，导致 `MOE_KERNEL=ON` 的构建去 include 已删除的 `operators/kml/` 头文件，编译期即失败。PyPI 的 kt-kernel wheel 全部只有 x86_64。上游社区侧，ARM 支持（#1611）官方答复是 llamafile 已修、int8/int4 内核延后；A+K 平台进 kt-kernel（#1882）仍 open 无 roadmap。因此当前 `supported` 仍必须为空。

复审触发条件（满足任一即重新评估本清单）：上游为 `moe_kernel` 落地非 KML 的 aarch64 mat-kernel 后端；恢复 KML 后端；发布 aarch64 wheel；`test_moe.py` 的 `device="cuda"` 变为可参数化。首批候选是 `test_moe_kernel.py`（int8 路径）和 `test_gate.py`。

不扫 `archive/`。原始框架（含 `operators/ascend/` 和 `optimize_rules/npu/*300IA2*`）已经归档。仓内 `archive/ktransformers/tests/UT/test_kdeepseek_*_npu.py` 会 `pytest.importorskip("torch_npu")`，然后把 `npu_rms_norm` / fused attention 换成假对象，跑过也不证明上了卡。

上游昇腾教程走的是归档树里的 `ktransformers/server/main.py` + 300I A2 优化配置，还要满血 DeepSeek-R1 / Qwen3-235B 量级的内存和定制编译的 `torch_npu`。那是 Quick Start / 服务进程，不是 `kt-kernel/examples` 里的非交互 example。本仓 runner 也不是 300I A2。

没有 `fixtures/`。
