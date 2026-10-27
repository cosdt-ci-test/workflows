# Liger-Kernel examples 看护

本项目复用公共 `examples-template.yml`；薄触发器为
`.github/workflows/liger-kernel-examples.yml`，被测仓库是 `linkedin/Liger-Kernel`。
手动触发的 `target_ref` 可指定分支、tag 或 SHA；留空时由公共引擎解析上游最新
release。setup 从同一个 `$TARGET_ROOT` checkout 安装 Liger 源码，因此被测内核
版本与 examples 始终来自同一个 release。Bring-up 期间 schedule 保持注释关闭。

## 为什么可行

v0.8.3 起上游原生带 Ascend NPU 后端：内核实现位于
`src/liger_kernel/ops/backends/_ascend/`，按设备自动选中；上层有
`infer_device()`（NPU 上返回 `"npu"`）与 `infer_comm_backend()`（返回 `"hccl"`）
两个抽象。上游 README 的 Ascend 安装段 pin 了 `torch==2.9.0` /
`torch_npu==2.9.0` / `triton-ascend==3.2.2` / `CANN==9.1.0`，并挂了华为官方
Ascend-CI 的 `liger_kernel.yml` 徽章——本项目 setup 即复刻该配方。

v0.8.5 起 `src/liger_kernel/transformers/monkey_patch.py` 在 import 时硬校验
`transformers >= 5.0.0`（低于该版本直接 `ImportError`），而 examples 仍是 trl 0.12
的写法。项目因此把依赖窗口整体上移到 **transformers==5.3.0 + trl==0.12.1**：
5.3.0 是华为 Ascend-CI 在 torch 2.9.0 上验证过的 5.x 配对；0.12.1 是最后一个仍
接受 `SFTTrainer(max_seq_length=)`、`DataCollatorForCompletionOnlyLM` 和
`tokenizer=` 别名的 trl 版本（它的 `@deprecate_kwarg("tokenizer",
new_name="processing_class")` 让 transformers 5 的 Trainer/ORPOTrainer 也接受
`tokenizer=`）。examples 源码一行不改。

镜像用 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`，
与本仓 tensordict、trl 等项目同款。CANN 基础镜像可能完全不带 PyTorch（tensordict
的 run #1 就是这么失败的），所以 setup 先探测可复用的 torch/torch_npu 对，缺失时
从集群 pip 缓存 + Ascend 源装 `torch==2.9.0` 与 `torch_npu==2.9.0.post2`，再断言
版本与 NPU 可用性。`triton-ascend` 只发布在
`https://triton-ascend.osinfra.cn/pypi/simple`：它的 wheel 名是 `triton_ascend-*`，
但装出来的 import 名仍是 `triton`（Ascend 版替换 PyPI 同名包），所以安装前先卸载
PyPI triton，幂等判断用 `pip show triton-ascend`。

## supported 清单（8 条）

前 3 条是在新栈上已验证 API 兼容的核心入口；后 5 条是首轮取证用的实验条目，
按上游原样规模跑（不压规模），跑红即按 manifest 注释里的原因退回 unsupported。

| example | 说明 |
|---|---|
| `examples/huggingface/training.py` | 单卡 SFT：`AutoLigerKernelForCausalLM` 一行 monkey-patch 应用 Liger 内核，trl `SFTTrainer` 训练 |
| `examples/huggingface/run_qwen.sh` | 上述 SFT 的两卡 FSDP 配方，验证 torchrun、HCCL 与 full shard |
| `examples/huggingface/training_multimodal.py` | Qwen2-VL 图文 SFT：monkey-patch 多模态 RoPE / RMSNorm / SwiGLU / FLCE |
| `examples/alignment/run_orpo.py`（实验） | ORPO 偏好对齐：`LigerORPOTrainer` 的 chunked ORPO loss |
| `examples/huggingface/run_gemma.sh`（实验） | 上游四卡 Gemma-7B FSDP 启动包装 |
| `examples/huggingface/run_llama.sh`（实验） | 上游四卡 Llama-3-8B FSDP 启动包装 |
| `examples/huggingface/run_qwen2_vl.sh`（实验） | 上游四卡 Qwen2-VL-7B FSDP 启动包装 |
| `examples/huggingface/run_benchmarks.sh`（实验） | 30 次复跑（use_liger×batch×5 次）的基准脚本 |

三条核心条目的规模压缩方式：单卡 SFT 用 `--max_steps 3`、batch 2、seq 256；两卡
FSDP 复刻上游 launcher 但改 2 卡、ModelScope 0.5B、2 步、每卡 batch 1、本地 8 行
数据；多模态例 `--max_steps 1`、batch 1、数据换 4 行本地 fixture。文本与多模态
模型走 ModelScope 的 `Qwen/Qwen2.5-0.5B-Instruct` 与 `Qwen/Qwen2-VL-2B-Instruct`，
设备由 `infer_device()` 与 accelerate 自动落到 NPU。前两条带 upstream 的
`EfficiencyCallback`，其 `on_init_end` 强制要求 `--include_num_input_tokens_seen`
与 `--logging_steps 1`，overlay 必须带这两个。

`run_qwen.sh` 原脚本固定 `torchrun` 四进程、Qwen2-7B、每卡 batch 48，而且不透传额外
参数。项目 runner 按该脚本启动 `training.py`，用 manifest 覆盖模型、数据和规模，并保持
`--fsdp "full_shard auto_wrap"` 与上游 `config/fsdp_config.json`。这是同一 Python
example 的另一种分布式启动配方，不算新增的独立训练程序。两卡任务要求
`linux-aarch64-a2-2`，任务启动前检查 NPU 数量，`torchrun --standalone` 分配独立端口。

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

实验条目的额外准备：

- ORPO 的 example 写死 gated 模型 `meta-llama/Llama-3.2-1B-Instruct`、数据集
  `trl-lib/tldr-preference`、batch 32 / 100 步，且没有 CLI 开关。setup 用 ModelScope
  的 `LLM-Research/Llama-3.2-1B-Instruct` 把权重镜像进 HF hub cache
  （`models--meta-llama--…/refs/main` + `snapshots/` 布局），并预缓存偏好数据集，
  run 步以 `HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE` 离线解析。
  该例历史记录（Run #9，v0.8.3 栈）是首步在 `fused_linear_preference.py` 的
  `torch.func.grad_and_value` 内调 `F.logsigmoid` 报
  `NotImplementedError: Cannot access storage of TensorWrapper`；首轮实跑就是要判定
  它在 transformers 5.3.0 + trl 0.12.1 栈上是否仍然存在。
- 三个上游启动包装与基准脚本自带 torchrun、模型和数据集，没有 CLI 可压规模：
  gemma/llama/mistral 都是 gated 模型，`run_qwen2_vl.sh` 未覆盖数据集参数、会去拉
  gated 的 `the_cauldron`（168 GB），仓库侧没有 HF token，预计首轮在下载处失败。
  按结果决定保留或退回 unsupported。

## unsupported（9 条）

- `examples/medusa/train.py`：`Trainer(tokenizer=...)` 在 transformers 5.0 被移除，
  当前栈（liger 硬性要求 >=5.0.0）下必然 `TypeError`，需上游迁移到
  `processing_class`。
- `examples/lightning/training.py`：示例传 `pl.Trainer(accelerator="npu")`，但
  lightning 主线（2.6.6 实测）的 accelerator 注册表只有 cpu/cuda/mps/tpu，直接
  `ValueError: You selected an invalid accelerator name`。能注册 `"npu"` 的插件包
  在 PyPI（42 万+ 包索引里无 lightning+npu 组合）、昇腾 pypi 镜像和 Gitee 上都不
  存在，环境侧无解。
- 两条 Megatron：源码固定 `torch.cuda.set_device`、`.cuda()` 和 NCCL，增加卡数
  无法改变设备后端。
- `launch_on_modal.py`、`medusa/scripts/llama3_8b_medusa.sh` 及 3 个配套模块
  （`hf/callback.py`、`medusa/callback.py`、`medusa_util.py`）：Modal 云 GPU / nvidia-smi
  数卡，或非独立入口。

## 依赖版本窗口（关键约束）

- 数据栈仍由 `constraints-npu.txt` 统一锁定 NumPy 1.26.4 / PyArrow 20.0.0 /
  pandas 2.2.3 / datasets 3.6.0（Triton-Ascend 3.2.2 要求 NumPy 1.26），与
  transformers 5.3.0 + trl 0.12.1 不冲突。
- transformers 5.3.0 会把 huggingface-hub 拉到 1.x（`<2.0,>=1.3`）；datasets 3.6
  与 modelscope 1.37 在该组合下已本地验证可 import，模型下载仍走 ModelScope。
- 已用逐字节相同的 `training.py`、`training_multimodal.py` 在 CPU 上以
  5.3.0 + 0.12.1 跑通训练步（覆盖 `DataCollatorForCompletionOnlyLM`、
  `SFTTrainer(max_seq_length=)`、`tokenizer=` 别名与 EfficiencyCallback）；
  上卡由手动 workflow 完成。

## 已知边界

- transformers 5 的 FSDP/HCCL 路径与 Qwen2-VL（5.2 起新增 `Qwen2VLRMSNorm`）在
  NPU 上的行为需要在实机确认——这两条以前只在 4.57.1 上验证过。
- medusa 的 `max_steps` 必须为 1 的旧边界随条目一起失效（现已 unsupported）。
- artifact 由公共引擎命名为 `liger-kernel-examples-<run_id>-<job_index>`。

## 文件结构

入口集中在 `scripts/setup_example.sh`（`install_hf_stack` 统一装 transformers/trl
栈，各 profile 负责模型与数据准备）与 `scripts/run_example.sh`（按入口区分单卡/
两卡/四卡，并校验各自的训练后产物），依赖约束在 `constraints-npu.txt`，本地数据在
`fixtures/`。本项目不再使用 `scan.exclude`：所有被扫描单元都显式登记为 supported
或 unsupported，让 `examples-check` 的新增/失效差异更直观。
项目不保留 `tests/` 目录；独立 quick-start workflow 使用
`scripts/test_quick_start_ascend.py` 做文档验证，不影响 examples 启动契约。
