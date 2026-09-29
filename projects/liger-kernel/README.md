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

镜像用 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`，
与本仓 tensordict、trl 等项目同款。CANN 基础镜像可能完全不带 PyTorch（tensordict
的 run #1 就是这么失败的），所以 setup 先探测可复用的 torch/torch_npu 对，缺失时
从集群 pip 缓存 + Ascend 源装 `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，再断言
版本与 NPU 可用性。`triton-ascend` 只发布在
`https://triton-ascend.osinfra.cn/pypi/simple`：它的 wheel 名是 `triton_ascend-*`，
但装出来的 import 名仍是 `triton`（Ascend 版替换 PyPI 同名包），所以安装前先卸载
PyPI triton，幂等判断用 `pip show triton-ascend`。

## supported 清单

三条都在单卡 `linux-aarch64-a2-1` 上跑。前两条已通过手动 workflow 验收（run #3，
commit d4961fed）：v0.8.3 上跑完真实训练步，日志含逐步 loss 与 NPU 显存分配
（hf 例峰值 4594 MB 已分配、4824 MB reserved），以 `NPU example passed` 收尾，
result.json 均为 success。第三条按同一套配方新增，待下一轮手动 workflow 验证。

| example | 作用 | 压规模方式 |
|---|---|---|
| `examples/huggingface/training.py` | `AutoLigerKernelForCausalLM` 一行 monkey-patch 应用 Liger 内核，trl `SFTTrainer` 做 SFT | `--max_steps 3`、batch 2、seq 256，不开 FSDP |
| `examples/medusa/train.py` | 冻结 backbone、注入 medusa 多头，用 `fused_linear_cross_entropy` 训练多 token 预测头 | `--max_steps 1`、`--medusa_num_heads 2`、`--medusa_return True` |
| `examples/huggingface/training_multimodal.py` | Qwen2-VL 图文 SFT，monkey-patch 多模态 RoPE / RMSNorm / SwiGLU / FLCE | `--max_steps 1`、batch 1、seq 256，数据换 4 行本地 fixture |

三条都不改源码。前两条模型走 ModelScope 的 `Qwen/Qwen2.5-0.5B-Instruct`；多模态那条
走 ModelScope 的 `Qwen/Qwen2-VL-2B-Instruct`。设备由 liger 的 `infer_device()` 与
accelerate 自动落到 NPU。前两条带 upstream 自带的 `EfficiencyCallback`，其
`on_init_end` 强制要求 `--include_num_input_tokens_seen` 与 `--logging_steps 1`，
overlay 必须带这两个。

fixture 全部由 setup 现生成，不往仓库塞二进制：

- hf 例的 `--dataset` 走 `datasets.load_dataset(path)`，需要一个**目录**（内含 `train.jsonl`），
  文件名不影响 split 名（都是 `train`）；
- 多模态例的 `--dataset` 同理需要目录，且必须自带 dataset card 声明配置名，否则
  `load_dataset(dir, "ai2d")` 抛 `BuilderConfig not found`。图片列用 `Sequence(Image())`，
  文本列声明成**单元素 list** 的 dict schema（写成 `Sequence({...})` 会在编码时抛
  `'list' object has no attribute 'get'`）。setup 在
  `$TARGET_ROOT/fixtures/cauldron_ai2d/` 下生成 4 行 112x112 合成图文。

多模态那条额外装 `torchvision==0.24.0`：`AutoProcessor` 走 torchvision 图像后端，
纯文本栈里没有它，缺了会抛 `AutoVideoProcessor requires the Torchvision library`。

## 依赖版本窗口（关键约束）

`transformers==4.57.1` + `trl==0.12.1`。这个窗口被 example 源码钉死：

- `from_pretrained(..., dtype=...)` 需要 transformers >= 4.56.0（4.55.x 只有
  `torch_dtype=`，传 `dtype=` 抛 `Object of type dtype is not JSON serializable`）。
- `Trainer(tokenizer=...)`（medusa 用）在 transformers 5.0 被彻底移除。
- `SFTTrainer(max_seq_length=...)`（hf 与多模态例用）从 trl 0.13.0 起移入 `SFTConfig`。
- `DataCollatorForCompletionOnlyLM`（hf 例用）在 trl 0.20.0 被删除。

窗口内唯一同时满足的组合即 4.57.1 + 0.12.1，已用逐字节相同的 `training.py` 与
`training_multimodal.py` 在 CPU 上跑通训练步验证；上卡实测由手动 workflow 完成。

## unsupported

`examples/alignment/run_orpo.py`、`examples/lightning/training.py`、
`examples/megatron/*` 及 `run_*.sh` 启动器家族入 unsupported，逐条在 manifest 里
写明原因。两个值得记下的硬阻塞：

- ORPO：`LigerORPOTrainer` 的损失调用无条件走 `_FSDPForwardRedirection()`，该 helper
  在 `src/liger_kernel/transformers/fsdp.py:40` 直接 assert 模型是 `FullyShardedDataParallel`；
  单卡未包装模型第一步就 AssertionError，而走 accelerate + FSDP 需要多卡 HCCL。
  叠加脚本无 argparse（`max_steps=100`/batch 32/1B 模型全硬编码）与 gated 模型。
- Lightning：`pl.Trainer(accelerator=infer_device())` 传 `"npu"`，而 Lightning 只注册
  cpu/cuda/mps/xla，且 `torch.optim.AdamW(fused=True)` 在 NPU 无 `_fused_adamw_` 实现。

## 已知边界

medusa 例的 `max_steps` 必须保持 1：其 callback 的 MFU 分支 `_get_gpu_peak_tflops`
只识别 A100/H100/V100，Ascend 上返回 `None`，第 2 步的 TFLOPS 除法会崩。warmup
步数为 2，故 1 步安全；这是上游代码问题。

artifact 由公共引擎命名为 `liger-kernel-examples-<run_id>-<job_index>`。

