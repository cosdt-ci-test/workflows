# Liger-Kernel examples 看护

本项目复用公共 `examples-template.yml`；薄触发器为
`.github/workflows/liger-kernel-examples.yml`，被测仓库是 `linkedin/Liger-Kernel`。
手动触发的 `target_ref` 可指定分支、tag 或 SHA；留空时由公共引擎解析上游最新
release。setup 从同一个 `$TARGET_ROOT` checkout 安装 Liger 源码，因此被测内核
版本与 examples 始终来自同一个 release。Bring-up 期间 schedule 保持注释关闭。

## 为什么可行

v0.8.3 已原生带 Ascend NPU 后端：内核实现位于
`src/liger_kernel/ops/backends/_ascend/`，按设备自动选中；上层有
`infer_device()`（NPU 上返回 `"npu"`）与 `infer_comm_backend()`（返回 `"hccl"`）
两个抽象。上游 README 的 Ascend 安装段 pin 了 `torch==2.9.0` /
`torch_npu==2.9.0` / `triton-ascend==3.2.2` / `CANN==9.1.0`，并挂了华为官方
Ascend-CI 的 `liger_kernel.yml` 徽章——本项目 setup 即复刻该配方。

镜像用 `swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.1.0-910b-ubuntu22.04-py3.12`
（Ascend-CI 同款，内含 torch 2.9.0 + torch_npu 2.9.0）。`triton-ascend` 只发布在
`https://triton-ascend.osinfra.cn/pypi/simple`，不在默认 PyPI，setup 显式加
`--extra-index-url` 安装；CUDA 版 triton 必须先卸载，否则 `_ascend` 后端会被遮蔽。

## supported 清单

两条都在单卡 `linux-aarch64-a2-1` 上跑，模型统一走 ModelScope 的
`Qwen/Qwen2.5-0.5B-Instruct`，数据用仓内 8 行 fixture：

| example | 作用 | 压规模方式 |
|---|---|---|
| `examples/huggingface/training.py` | `AutoLigerKernelForCausalLM` 一行 monkey-patch 应用 Liger 内核，trl `SFTTrainer` 做 SFT | `--max_steps 3`、batch 2、seq 256，不开 FSDP |
| `examples/medusa/train.py` | 冻结 backbone、注入 medusa 多头，用 `fused_linear_cross_entropy` 训练多 token 预测头 | `--max_steps 1`、`--medusa_num_heads 2`、`--medusa_return True` |

两条 example 都不改源码：训练脚本本身无 CUDA/FSDP 依赖，设备由 liger 的
`infer_device()` 与 accelerate 自动落到 NPU。两条都带 upstream 自带的
`EfficiencyCallback`，其 `on_init_end` 强制要求 `--include_num_input_tokens_seen`
与 `--logging_steps 1`，overlay 必须带这两个。hf 例的 `--dataset` 走
`datasets.load_dataset(path)`，需要一个**目录**（内含 `train.jsonl`），setup 会
在 `$TARGET_ROOT/fixtures/` 下生成该目录；文件名不影响 split 名（都是 `train`）。

## 依赖版本窗口（关键约束）

`transformers==4.57.1` + `trl==0.12.1`。这个窗口被 example 源码钉死：

- `from_pretrained(..., dtype=...)` 需要 transformers >= 4.56.0（4.55.x 只有
  `torch_dtype=`，传 `dtype=` 抛 `Object of type dtype is not JSON serializable`）。
- `Trainer(tokenizer=...)`（medusa 用）在 transformers 5.0 被彻底移除。
- `SFTTrainer(max_seq_length=...)`（hf 例用）从 trl 0.13.0 起移入 `SFTConfig`。
- `DataCollatorForCompletionOnlyLM`（hf 例用）在 trl 0.20.0 被删除。

窗口内唯一同时满足四项的组合即 4.57.1 + 0.12.1，已用逐字节相同的 `training.py`
在 CPU 上跑通 2 步验证；上卡实测由手动 workflow 完成。

## unsupported

`examples/alignment/run_orpo.py`、`examples/lightning/training.py`、
`examples/megatron/*`、`examples/huggingface/training_multimodal.py` 及 `run_*.sh`
启动器家族全部入 unsupported，逐条在 manifest 里用一行中文写明原因（无 CLI 无法
压缩、Lightning 无 npu accelerator、Megatron 硬编码 CUDA/NCCL、多模态 trl 签名
漂移、启动器非独立入口等）。

## 已知边界

medusa 例的 `max_steps` 必须保持 1：其 callback 的 MFU 分支 `_get_gpu_peak_tflops`
只识别 A100/H100/V100，Ascend 上返回 `None`，第 2 步的 TFLOPS 除法会崩。warmup
步数为 2，故 1 步安全；这是上游代码问题，不是 NPU 不兼容，改 upstream 前不要放大步数。

artifact 由公共引擎命名为 `liger-kernel-examples-<run_id>-<job_index>`。手动验收时
确认 manifest-check 展开 2 个唯一 job、两条日志都出现 `NPU example passed` 且无 CPU
回落、publish-result 产出合规 result.json，全绿后再考虑启用 schedule。

