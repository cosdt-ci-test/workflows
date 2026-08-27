# ktransformers

本目录是 [ktransformers](https://github.com/kvcache-ai/ktransformers) 的看护配套数据，不是 ktransformers 源码。example 流水线在 [.github/workflows/ktransformers-examples.yml](../../.github/workflows/ktransformers-examples.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

上游默认分支是 `main`。上游没有昇腾 CI。本仓先走阶段 A。

本仓**不**写 Quick Start 线。上游已经发布昇腾教程：

- [doc/zh/DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/zh/DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md)
- [doc/zh/Qwen3-MoE_tutorial_zh_for_Ascend_NPU.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/zh/Qwen3-MoE_tutorial_zh_for_Ascend_NPU.md)

这两篇写明支持的 NPU 是 **Atlas 300I A2**，镜像和 CANN 也按 300I A2 / CANN 8.3 来。本仓 runner 是 910B。不在本仓再写一份对着 910B 的 Quick Start，避免和上游教程打架。

## 绿灯含义

当前清单 `supported` 是空的。`run-example` 不会占 NPU。流水线绿只表示：监控门过了、清单和上游 `kt-kernel/examples/` 对得上。**这不是昇腾推理绿。**

`setup_example.sh` 没有 `setup_*` 函数。未知 `profile` 在读 `TARGET_ROOT`、装包、编译之前非 0 退出。`run_example.sh` 拒绝任何路径。这两条是给以后补 `supported` 用的契约，现在被调用就是看护自己写错了。

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

当前产品入口是 `kt-kernel`。它的 CMake 只有 CUDA / ROCm / SYCL / CPU（AMX、AVX、KML、llamafile），**没有 CANN / NPU 后端**。这些 `.py` 是 CPU 或 NVIDIA 侧的 kernel 自检：

- 助手模块：`configuration_deepseek_v3.py`、`modeling_deepseek_v3.py`、`torch_attention.py`
- 调试 / 复现：`repro_llamafile_re.py`、`test-debug.py`
- 依赖本地 dump 文件：`test_softmax.py`、`test_apply_rope.py`、`test_rope.py`、`test_mla_quant.py`
- Intel AMX：`bench_moe_amx_int8.py`、`test_*amx*`、`test_bf16_moe.py`、`test_fp4_moe_v4.py`、`test_fp8_*.py`、`test_write_buffer.py`、`test_mxfp8_moe_m3.py` 等。aarch64 910B 没有 AMX
- AVX2：`test_fp4_moe_avx2.py`、`test_mxfp8_moe_avx2.py`
- Kunpeng KML（CPU 数学库，不是 NPU）：`test_moe_kml.py`、`test_deepseekv3.py`、`test_deepseekv3_prefill.py`、`test_deepseekv3_prefill_speed.py`
- CPUInfer / llamafile，且多数先在 CUDA 上 `randn` 再搬回 CPU，或 `import flash_attn`：`test_linear.py`、`test_mlp.py`、`test_moe.py`、`test_moe_kernel.py`、`test_gate.py`、`test_attention.py`、`test_mla*.py`

不扫 `archive/`。原始框架（含 `operators/ascend/` 和 `optimize_rules/npu/*300IA2*`）已经归档。仓内 `archive/ktransformers/tests/UT/test_kdeepseek_*_npu.py` 会 `pytest.importorskip("torch_npu")`，然后把 `npu_rms_norm` / fused attention 换成假对象，跑过也不证明上了卡。

上游昇腾教程走的是归档树里的 `ktransformers/server/main.py` + 300I A2 优化配置，还要满血 DeepSeek-R1 / Qwen3-235B 量级的内存和定制编译的 `torch_npu`。那是 Quick Start / 服务进程，不是 `kt-kernel/examples` 里的非交互 example。本仓 runner 也不是 300I A2。

没有 `fixtures/`。

## 触发

`ktransformers-examples.yml` 有两种入口。`monitor` job 跑在 `ubuntu-latest`，不占 NPU。

- `schedule`：cron 写在文件里但是注释掉的。接入阶段保持注释，不要打开。
- `workflow_dispatch`：手动触发。默认 `force=false`，和定时走同一套监控、同一份 cache。只有 `force=true` 才跳过监控门、必跑，并且不读不写 monitor cache。`target_repo` / `target_ref` 只在 `force=true` 时有意义。

两个监控信号都跑，是「或」，互不跳过、没有优先级、失败不重试：

1. 清单 `supported` 各 `path` 在上游 `main` 上的文件内容哈希（Contents API 的 blob SHA；目录会递归到文件。404 记成 `MISSING`）。`supported` 为空时，这条信号的哈希是空载荷的稳定值，不会因为 `kt-kernel/examples/` 新增文件而变化。新文件出现在扫描根、却不在清单里时，要等到 release 信号或手动 `force` 跑 `manifest-check` 才会写进 `new_paths`。
2. `/releases/latest` 的 release **id**（数字，不是 tag 字符串）。ktransformers 有 GitHub Releases（例如 `v0.7.0`），不要跳过这个信号。

谁亮了就测谁的树。都没亮则 `targets` 为空，后面的 job 跳过，不写 `result.json`，不产包。`force=true` 时 `targets` 只有一项，`reason=manual`。`has_supported` 为 false 时 `run-example` 和 `publish-result` 都跳过。

NPU job 不上传 artifact。以后若补了 `supported`，`result.json` 仍由托管 runner 上的 `publish-result` 按 job 名回看 conclusion 后上传。job 显示名必须保持 `run-example (${{ matrix.example.path }} @ ${{ matrix.example.target_ref }})`，与 `EXPECTED_JOB_NAME` 全等匹配。
