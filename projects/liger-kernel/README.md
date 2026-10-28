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

## supported 清单（2 条）

2026-10-27 的 NPU 实跑（run `40506606117`）确认这两条在新栈上通过：

| example | 说明 | 压规模方式 |
|---|---|---|
| `examples/huggingface/training.py` | `AutoLigerKernelForCausalLM` 一行 monkey-patch 应用 Liger 内核，trl `SFTTrainer` 做 SFT | `--max_steps 3`、batch 2、seq 256，不开 FSDP |
| `examples/huggingface/run_qwen.sh` | 上述 SFT 的两卡 FSDP 配方，验证 torchrun、HCCL 和 full shard | 复刻上游 launcher；改用 ModelScope 0.5B、2 卡、2 步、每卡 batch 1、本地 8 行数据 |

两条都不改源码。模型走 ModelScope 的 `Qwen/Qwen2.5-0.5B-Instruct`，设备由
`infer_device()` 与 accelerate 自动落到 NPU。两条都带 upstream 自带的
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
- 多模态例的同名参数同理（该条当前 unsupported，配方保留在 setup 里以便上游修复后复测）。

## 首轮实跑结论（run 40506606117，8 条探针）

| 条目 | 结果 | 原因 |
|---|---|---|
| `training.py` | 通过 | 3 步训练，loss 0.52，峰值 4596 MB |
| `run_qwen.sh` | 通过 | 两卡 FSDP 2 步，loss 0.85 |
| `training_multimodal.py` | 失败 | `Qwen2VLForConditionalGeneration.__init__() got an unexpected keyword argument 'use_cache'`：transformers 5 把 `use_cache` 移入 GenerationConfig，不再作为模型 config 属性被 `from_pretrained` 吸收 |
| `run_orpo.py` | 失败 | `liger_kernel/transformers/fsdp.py` 的 `_FSDPForwardRedirection` 断言模型必须被 FSDP 包装；该 example 是单进程 Trainer，必 `AssertionError`（上游代码问题） |
| `run_gemma.sh` | 失败 | `GatedRepoError: 403`，`google/gemma-7b-it` 是 gated 模型 |
| `run_llama.sh` | 失败 | `GatedRepoError: 403`，`meta-llama/Meta-Llama-3-8B` 是 gated 模型 |
| `run_qwen2_vl.sh` | 超时 | 脚本未覆盖数据集参数，会去拉 `the_cauldron` 整库（168 GB），90 分钟超时 |
| `run_benchmarks.sh` | 超时 | 30 次 × 20 步复跑（4 卡 7B，模型/数据本身公开），2 小时超时 |

失败条目已按上表原因逐条登记为 `unsupported`。ORPO 那轮验证了 ModelScope→HF cache
镜像可用：setup 把 `LLM-Research/Llama-3.2-1B-Instruct` 拷进
`models--meta-llama--…/snapshots/`，预缓存 `trl-lib/tldr-preference`（92858 行），
run 步离线解析成功，失败点确实在训练代码而非资产。

## unsupported（15 条）

- **transformers 5 删掉的示例 API**：`medusa/train.py` 用 `Trainer(tokenizer=...)`
  （5.0 起改为 `processing_class`）；`training_multimodal.py` 传
  `from_pretrained(use_cache=False)`（5.x 的 `use_cache` 只在 GenerationConfig 里）。
- **上游代码缺陷**：`alignment/run_orpo.py` 的 `LigerORPOTrainer` 假定 FSDP 包装，
  单进程示例必触发断言。
- **依赖装不上**：`lightning/training.py` 传 `accelerator="npu"`，而 lightning 主线
  2.6.6 的 accelerator 注册表只有 cpu/cuda/mps/tpu；能注册 `"npu"` 的插件包在 PyPI、
  昇腾 pypi 镜像和 Gitee 上都不存在。
- **上游原样规模**：`run_gemma.sh` / `run_llama.sh` 拉 gated 模型（无 HF token 即 403）；
  `run_qwen2_vl.sh` 会去拉 168 GB 的 `the_cauldron` 整库；`run_benchmarks.sh` 30 次复跑
  超出 runner 预算。三者都没有 CLI 可压规模。
- **CUDA/NCCL 硬编码**：两条 Megatron 固定 `torch.cuda.set_device`、`.cuda()` 和
  NCCL，增加卡数无法改变设备后端。
- **非昇腾路径/非独立入口**：`launch_on_modal.py`（Modal 云 GPU）、
  `medusa/scripts/llama3_8b_medusa.sh`（nvidia-smi 数卡），以及 3 个配套模块。

## 依赖版本窗口（关键约束）

- 数据栈由 `constraints-npu.txt` 统一锁定 NumPy 1.26.4 / PyArrow 20.0.0 /
  pandas 2.2.3 / datasets 3.6.0（Triton-Ascend 3.2.2 要求 NumPy 1.26），与
  transformers 5.3.0 + trl 0.12.1 不冲突。
- transformers 5.3.0 会把 huggingface-hub 拉到 1.x（`<2.0,>=1.3`）；datasets 3.6
  与 modelscope 1.37 在该组合下已实测可 import，模型下载仍走 ModelScope。
- `transformers==4.57.1 + trl==0.12.1` 是 v0.8.3 时代的窗口，v0.8.5 的
  `transformers>=5.0.0` 硬校验后不再可用。

## 已知边界

- transformers 5 的 FSDP/HCCL 路径已经在两卡 Qwen 配方上实测通过；单卡 SFT 也通过。
- 多模态、ORPO 与四个上游启动包装的 setup/run 分支仍保留在脚本里（不调度），
  上游修复对应问题后可直接把条目移回 supported 复测。
- artifact 由公共引擎命名为 `liger-kernel-examples-<run_id>-<job_index>`。

## 文件结构

入口集中在 `scripts/setup_example.sh`（`install_hf_stack` 统一装 transformers/trl
栈，各 profile 负责模型与数据准备）与 `scripts/run_example.sh`（按入口区分单卡/
两卡/四卡，并校验各自的训练后产物），依赖约束在 `constraints-npu.txt`，本地数据在
`fixtures/`。本项目不使用 `scan.exclude`：所有被扫描单元都显式登记为 supported 或
unsupported，让 `examples-check` 的新增/失效差异更直观。
项目不保留 `tests/` 目录；独立 quick-start workflow 使用
`scripts/test_quick_start_ascend.py` 做文档验证，不影响 examples 启动契约。
