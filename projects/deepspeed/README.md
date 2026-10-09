# deepspeed

本目录是 [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) 的看护配套数据，不是 DeepSpeed 源码。example 流水线在 [.github/workflows/deepspeed-examples.yml](../../.github/workflows/deepspeed-examples.yml)，quick-start 流水线在 [.github/workflows/deepspeed-quick-start.yml](../../.github/workflows/deepspeed-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：基础支持；阶段 A）。

上游默认分支是 `master`。上游有 Ascend NPU 加速器支持（`accelerator/npu_accelerator.py`），由华为贡献。外部 Ascend CI（`Ascend/Ascend-CI` 的 `deepspeed.yaml`）已停摆（自 2026-06-11 起连续失败，基础设施故障）。本仓先走阶段 A：在本仓流水线把 example 跑通，再考虑往上游推。

## 仓库关系（分离模式）

`deepspeed-examples.yml` 是调用公共 [.github/workflows/examples-template.yml](../../.github/workflows/examples-template.yml) 的薄触发器，并使用 `examples_repo` 分离模式：

- 监控仓 / 源码安装：`deepspeedai/DeepSpeed`（主仓，发布 release，驱动 schedule；`run-example` 容器内 checkout 到 `target/`，DeepSpeed 从这里 `pip install -e` 安装）。
- Example 来源：`deepspeedai/DeepSpeedExamples`（不发布 release，永远跟随其默认分支 `master`；checkout 到 `examples/`，本目录 manifest 的 `path` 相对该仓根）。
- 这是 `examples_repo` 分离模式（见 [docs/examples-guard-engine.md](../../docs/examples-guard-engine.md)）的首个使用方。

## 清单

[examples_manifest.yaml](examples_manifest.yaml) 的 `scan.root` 为 DeepSpeedExamples 仓根，`include_extensions` 为 `.sh` / `.py`。`supported` 与 `unsupported` 只登记 example 入口：被 import 的库、模型定义、安装脚本、单元测试、数据准备和结果处理工具不登记，也不为它们增加 exclude 字段。公共引擎只校验已声明的 supported 条目，不要求上游每个 `.py`/`.sh` 都出现在清单中。

当前 supported 共 19 条，按 example 的有效拓扑使用 1/2/4/8 卡 runner，统一用 CANN 9.1.0 镜像。Run #28 已验证原有 15 条全部成功；本次第一阶段新增 4 条（生成评测、奖励评分、固定长度 HF 训练、SuperOffload 目录的 ZeRO-Offload 模式），仍待 Actions 验收。模型走 ModelScope（`ms_download_models` 下载后经 `GITHUB_ENV` 传本地路径），数据集优先使用仓内 fixture，并用 `overlay_args` 压到 CI 规模：

| path（相对 examples 仓） | profile | 看护点 | 模型 / 数据 | 压规模 |
|---|---|---|---|---|
| `training/HelloDeepSpeed/run_ds.sh` | deepspeed / 1 卡 | Roberta MLM（ZeRO-1 + CPU offload + BF16） | wikitext 运行时重定向到 Salesforce/wikitext + roberta-base tokenizer | 2 层 10 步 |
| `applications/DeepSpeed-Chat/training/step1_supervised_finetuning`（exec: `main.py`） | ds_chat_sft | SFT（NPU-aware） | opt-125m（ModelScope）+ 8 行 local/jsonfile fixture | 1 epoch |
| `applications/DeepSpeed-Chat/training/step2_reward_model_finetuning`（exec: `main.py`） | ds_chat_rw | Reward Model | 同上 | 1 epoch |
| `applications/DeepSpeed-Chat/training/step2_dpo_finetuning`（exec: `main.py`） | ds_chat_dpo | DPO（ref model 内存翻倍） | 同上 | 1 epoch |
| `applications/DeepSpeed-Chat/training/step3_rlhf_finetuning`（exec: `main.py`） | ds_chat_rlhf | RLHF（官方 `--enable_test_mode`） | opt-125m ×2（ModelScope）+ fixture | test mode 5 步 |
| `inference/huggingface/text-generation/inference-test.py` | ds_infer | 推理（`--hf_baseline` 跳过 DS kernel） | opt-125m（ModelScope） | 8 token |
| `training/cifar`（exec: `run_ds.sh`） | ds_cifar | CIFAR10 分类（ZeRO-0 + BF16） | ModelScope 预置并校验的 CIFAR-10 | 1 epoch |
| `training/offload_states/offload_states.py` | deepspeed / 1 卡 | ZeRO offload_states | 随机合成数据 | 小规模 |
| `training/cifar/run_ds_moe.sh` | ds_cifar / 2 卡 | CIFAR10 MoE expert parallel（EP=2） | ModelScope 预置并校验的 CIFAR-10 | 1 epoch |
| `training/autotp_equivalence`（exec: `train.py`） | ds_autotp_equivalence / 4 卡 | AutoTP=1/3/4 loss 等价性 | Qwen3-0.6B（ModelScope）+ 随机 token | 每组 5 步 |
| `training/tensor_parallel/hf_integration/train.py` | ds_hf_autotp / 8 卡 | HF Trainer AutoTP=8，含最终 TP 权重保存 | opt-125m（ModelScope）+ 16 行 Alpaca fixture | 总共 3 步（含 2 步预热），序列 128，batch 1 |
| `training/cifar/run_ds_prmoe.sh` | ds_cifar / 2 卡 | 残差 PR-MoE，EP=2、experts=2/4 | 与普通 CIFAR/MoE 相同 | 1 epoch |
| `training/data_efficiency/variable_batch_size_and_lr/variable_batch_size_and_lr_example.py` | ds_variable_batch / 1 卡 | 动态序列打包、batch 与 LR 缩放 | 内置小模型和 1000 条合成序列 | 上游原生 2 epoch，pipeline=0 |
| `training/DeepSpeed-ZenFlow/benchmark/zf_benchmark.py` | ds_zenflow / 2 卡 | ZenFlow CPU optimizer offload 单配置 smoke | 256 维、2 层模型和合成数据 | iteration=3 × update_interval=2，共 6 次循环 |
| `compression/reasoning_aware_compression/prune.py` | ds_rac_prune / 1 卡 | Wanda 校准剪枝，显式使用 NPU | opt-125m（ModelScope）+ 8 行 prompt JSONL | 4×32 token，首三分之一 block 的线性层 |
| `applications/DeepSpeed-Chat/training/step1_supervised_finetuning/prompt_eval.py` | ds_chat_prompt_eval / 1 卡 | 真正先 SFT、再对比基础/微调模型生成 | opt-125m + 本 job 训练的 SFT 检查点 | 8 行数据训练 1 epoch；内置 6 个 prompt 各生成 8 token |
| `applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/rw_eval.py` | ds_chat_reward_eval / 1 卡 | 初始化 reward head 的 NPU forward/评分 smoke | opt-125m + 上游内置偏好对 | 2 对样本，原生 padding 到 512；不验证评分质量 |
| `training/tensor_parallel/hf_integration/train_bench_length.py` | ds_hf_bench_length / 8 卡 | 固定长度 padding/label masking + TP=8 + 最终权重保存 | opt-125m + 16 行 Alpaca fixture | 3 步（含 2 步预热），序列 128，batch 1 |
| `training/DeepSpeed-SuperOffload/finetune_zero3.py` | ds_superoffload / 2 卡 | 官方 ZeRO-Offload 模式，ZeRO-3 参数/优化器 CPU 卸载 | opt-125m + fixture 生成的本地 Parquet 目录 | 3 步，序列 128；每卡 batch 1、全局 batch 2 |

**启动方式的选择**：`cifar10_deepspeed.py` 的 main() 无条件读 launcher 注入的 `LOCAL_RANK` 并调 `init_distributed()`，因此普通 CIFAR 经支持 `$@` 的上游 `run_ds.sh` 启动。CIFAR MoE 的上游脚本不透传 `$@`，项目 runner 按原配方复刻两卡 launcher、EP=2 和 MoE 参数，再追加 CI overlay。DS-Chat 官方 training_scripts 硬编码 1.3B～66B 模型且不透传任意参数，项目 runner 等价执行 `deepspeed --num_gpus 1 main.py <overlay_args>`。AutoTP equivalence 同样复刻上游 `run_gpu.sh` 的 1/3/4 卡三次启动与 loss 比较，但显式传入 ModelScope 本地模型。offload_states 与 `--hf_baseline` inference 不依赖 launcher，直接运行 `.py`。

多卡条目启动前会检查 `ASCEND_RT_VISIBLE_DEVICES`：MoE、PR-MoE、ZenFlow 和 SuperOffload 入口至少需要 2 卡，AutoTP equivalence 至少需要 4 卡，两条 HF AutoTP 入口需要 8 卡；runner 未注入时按所需卡数选择从 0 开始的设备列表。这个变量只过滤子进程可见的物理 NPU 并把它们重新映射为进程内的逻辑设备，不负责创建 worker；实际 rank 数仍由 `deepspeed --num_gpus` 决定。各配方使用不同 master port，AutoTP 三次子运行也各自分配端口。

bf16_master_weight 与 pipeline_parallelism 曾进 supported，CI 实测其源码硬绑 CUDA（`torch.cuda.set_device` / `autocast(device_type="cuda")` / `--backend nccl`），装包无法解决，已移回 unsupported（需 patch，次轮候选）。

DeepSpeed-Chat 的 `--data_path local/jsonfile` 从 `applications/DeepSpeed-Chat/data/{train,eval}.json` 读取（JSON Lines，字段 `prompt`/`chosen`/`rejected`）；`scripts/setup_example.sh` 在对应 profile 下把 [fixtures/](fixtures/) 里的 8 行 fixture 拷到该目录。上游 `setup.py` 的 `find_packages(include=['dschat'])` 无法安装缺少根 `__init__.py` 的 namespace package，因此 setup 不依赖其空 editable wheel，而是把 `applications/DeepSpeed-Chat` 源码根目录写入 `PYTHONPATH` 并立即执行 `import dschat` 验证。上游 [issue #813](https://github.com/deepspeedai/DeepSpeedExamples/issues/813) 也记录了 DeepSpeed-Chat 在切换执行/缓存上下文后发生模块解析错误，但不是本次完全相同的报错。模型经 `ms_download_models` 从 ModelScope 下载到本地，路径写入 `GITHUB_ENV`（`OPT_125M_PATH` / `QWEN3_06B_PATH`），overlay_args 引用本地目录。setup 按上游 requirements/setup.py 显式安装依赖，但不会从 PyPI 覆盖镜像的 `torch + torch_npu` 或 `$TARGET_ROOT` 中的 DeepSpeed 源码；安装后会校验 `deepspeed.__file__` 位于目标源码树。

CIFAR 单卡、两卡 MoE 和 PR-MoE 共用 `ds_cifar` profile。setup 从固定 revision 的 ModelScope 镜像下载 CIFAR-10 zip，先校验 SHA-256，再解压到上游脚本使用的 `training/cifar/data`，最后调用 torchvision 自带的官方逐文件 MD5 清单复核。这样保留 example 的 `download=True` 原始逻辑，但完整数据已存在时不会访问 CI 中超时的 Toronto 源；其余 `deepspeed` profile 不承担这次约 170 MB 的下载。

Run #22 中 10 条已有 9 条通过；两卡 MoE 已完成两个 rank 的 HCCL 初始化并创建 EP=2 group，首次 forward 才在 DeepSpeed `sharded_moe._capacity()` 触发 `torch.compile`，随后因镜像没有 Triton 后端而报 `ModuleNotFoundError: triton`。这是 DeepSpeed 0.19.7 将 MoE helper 从 TorchScript 改为 `torch.compile` 后产生的可选编译路径（[issue #7835](https://github.com/deepspeedai/DeepSpeed/issues/7835)、[PR #7840](https://github.com/deepspeedai/DeepSpeed/pull/7840)）；后续 [PR #7875](https://github.com/deepspeedai/DeepSpeed/pull/7875) 的 fallback 无法捕获 `torch.compile` 在首次调用时才发生的懒编译失败。项目 runner 因此仅对 CIFAR MoE 命令设置 `TORCH_COMPILE_DISABLE=1`，让该 helper 走 eager；两卡 launcher、HCCL、MoE 和 EP=2 训练语义均保留，也不要求为一个未由 example 声明的可选优化安装版本敏感的 Triton-Ascend。

当前 107 条尚未接入的 example 列入 unsupported：同一逻辑 example 的启动包装尽量合并到主入口；每条上方保留一行中文注释，说明用途和当前未接入原因。原有 124 条非 example 配套项（包括 `applications/DeepSpeed-Chat/chat.py` 启动包装和库/测试目录）已移出清单，不代表从上游删文件。真正执行训练、推理、评测或基准的入口仍保留，即使暂时需要已有检查点或外部服务。独立扫描可能把未登记的配套文件列为清单差异，但不影响公共引擎调度；supported 路径不存在时 manifest-check 仍会立即失败。

新增配方通过 CLI、launcher、依赖版本和 fixture 接入，不修改上游 Python/启动脚本，也不为新增条目安装 API monkey patch。HF、动态 batch 和 ZenFlow 的运行缓存/报告写入本次 CI 输出目录；HF config 根据上游模板在输出目录生成。RAC 只通过正常 `import torch_npu` 注册设备后运行原入口，使用校准 forward 的 Wanda 路径，避免仅在 CPU 上完成 magnitude 剪枝却误判 NPU 通过。OPT 的前馈层名为 fc1/fc2，不匹配上游 `scope=mlp` 的命名筛选，因此这里使用 `scope=all`。PR-MoE 与普通 MoE 均局部禁用可选的 `torch.compile`。

HF setup 用上游慢速 tokenizer 和各入口自身的 SupervisedDataset 单进程生成共享数据缓存，并检查 fixture 在序列长度 128 下仍保留训练 label，八个 rank 只读该缓存；RAC setup 检查本地 prompt 可以组成完整的 4×32 token 校准窗口。新增依赖安装锁定镜像 torch/torch_npu 和被测源码 DeepSpeed，并检查 accelerator 为可用 NPU；ZenFlow 与新增 CPU offload profile 在 setup 编译/加载 CPU Adam 扩展，提前暴露工具链问题。动态 batch 没有 step/epoch 裁剪参数，保留完整的原生小规模运行，不用超时终止冒充成功。原有 15 条已在 #28 全绿，但扩展阶段仍不启用 schedule。

Run #27 使用 DeepSpeed `v0.19.7`、transformers `4.57.6` 和 accelerate `1.15.0`。HF AutoTP 的上游 config 使用 `WarmupDecayLR`，`warmup_num_steps: auto` 由 HF 的 `--warmup_steps` 填充；原 CI overlay 设为 0，被 DeepSpeed 的正整数校验拒绝。现在设为 2，与调度器内部最小有效预热长度一致；总训练仍为 3 步，TP=8、模型、数据和上游 config 模板不变。后续 `ERR99999` 是此次 Python 异常后的伴随日志，不据此判定为 NPU 算子不兼容。

Run #28 的 15 条原有配方及其 publish-result 已全部成功，HF 的三步训练和最终保存也已通过。本次远程验收改为 19 条训练/推理 job 与对应 publish-result 全部成功；结果发布成功本身不等于训练通过。新增条目失败时依据真实日志定位，不通过改写上游源码兜底。

### 第一阶段新增配方的语义和验收

- **生成评测**：在同一 job 内用原 SFT `main.py`、8 行 fixture、1 epoch 训练并保存小检查点，然后执行原 `prompt_eval.py`。基础模型与微调模型是不同的目录，不将基础模型复制两份冒充微调。验收要求 SFT 成功、检查点存在，并完成内置 6 个 prompt 的两组 greedy generation。
- **奖励评分**：原 `rw_eval.py` 调用 `create_critic_model(..., rlhf_training=False)`，从基础 OPT 构造模型并新建评分头，不恢复训练后的 reward head。本条只看护两组偏好样本的 NPU forward 与有限分数输出，不能宣传为训练检查点恢复，也不要求随机头满足 good > bad。
- **固定长度 HF AutoTP**：沿用已通过的 OPT/TP=8/3 步/2 步预热，但调用 `train_bench_length.py` 自己的固定 padding 和 label masking。setup 预生成独立工作目录中的 `dataset_dict128.pkl`，不会复用 `train.py` 的 `dataset_dict.pkl`。其 sibling `utils.py` 顶层导入旧 `openai_object`，因此只在此 profile 安装 `openai==0.28.1`；训练不调用 OpenAI 服务，不需要 API key。验收包括三步训练和上游强制执行的最终 TP 权重保存。
- **SuperOffload 目录入口**：采用上游允许的 `zerooffload` 模式，生成两卡 ZeRO-3 配置，CPU 参数/优化器 offload、BF16、每卡 micro batch 1、全局 batch 2、GA=1、pin_memory=false。**不启用 `super_offload=true`，不声称验证 superchip 原生 SuperOffload 或 pinned-memory 优化。** 模型走 ModelScope，数据由已有 16 行 Alpaca fixture 生成本地 `train.parquet` 目录，原 `load_dataset(directory)` 直接读取，无 shim。`--attn_implementation eager` 避免默认 FlashAttention CUDA 依赖；`--bench_steps 3` 真正停止训练，`--warmup_steps 1` 是计时预热而非 LR 预热。上游 CPUAdam 将实际 LR 固定为 0.001，`--lr` 只影响标记，所以 CI 同样填写 0.001，不伪装成低 LR 已生效。不传 `--save_checkpoint`，避免当前上游 rank0-only 分支调用分布式保存的挂起风险；本条不覆盖检查点保存。验收要求 world size 2、ZeRO-3/CPU offload 生效，并完成三个有限 loss 的训练 step。

第一阶段 19/19 全绿后，再单独预验证 finetune demo 和两条 pin-memory/offload 候选；本次不提升它们，不因缺少 GPU、默认大模型或单个受保护的 CUDA 调用就永久否定入口。

## 触发

`deepspeed-examples.yml` 是薄触发器：

- `workflow_dispatch`：手动运行只接受可选 `target_ref`（`deepspeedai/DeepSpeed` 的分支/tag/SHA），不经过 monitor 门。留空时优先选择最新 release；release 查询失败或无 release 时，使用薄触发器显式传入的 `default_branch: master`。要测试最新源码可填 `master`，不能填该仓不存在的 `main`。`upstream_repo` / `examples_repo` 固定在 workflow 内，不再支持旧版 `target_repo` 覆盖。
- `schedule`：当前以注释保留、暂不启用。启用后由公共引擎对主仓 `deepspeedai/DeepSpeed` 做 release-only 监控（latest release tag 变化时触发，上次失败时下一周期以 `release-retry` 重试）；examples 仓无 release，始终跟随 `master`。

Run #26 的 latest-release 查询返回 HTTP 403，旧触发器未声明 `default_branch`，引擎回退到 `main`，导致 `manifest-check` 的主仓 checkout 失败；15 个 example 均未启动。这次修复只补齐项目默认分支，不改公共引擎或 example 配方，也不把查询失败当成 example/NPU 兼容问题。

## 模型缓存边界

薄触发器不向公共引擎传宿主缓存卷，模型和 CIFAR-10 在各 matrix job 的容器内准备；同一 job 的 setup 与 run 步骤可复用该容器内文件，但不保证跨 job 或跨 workflow run 复用。本看护所需模型（opt-125m 等）和 CIFAR-10 均有 ModelScope 来源，正常路径不使用 [cache-seed](../../cache-seed/README.md)；仅当 ModelScope 资产变得不可用时才按 cache-seed 流程兜底投递。

## Quick Start

`deepspeed-quick-start.yml` 看护本仓 [docs/Quick-start-Ascend.md](docs/Quick-start-Ascend.md)。该文档描述了在昇腾 NPU 上安装 DeepSpeed、验证加速器、单卡跑通 CIFAR10 示例并双卡体验分布式训练的完整流程。

- 监控信号：doc 哈希、上游 latest release、master HEAD SHA。按 doc > release > commit 优先级，任一变化触发测试。
- 测试内容：pip 安装 DeepSpeed → 安装配套 torchvision → `get_accelerator()._name == 'npu'` → CIFAR10 内联示例（ZeRO-1 + BF16，1 个 epoch）→ 双卡分布式训练（`--num_gpus 2`，HCCL 通信）。
- **当前为节约 NPU 资源，`schedule` 已注释，只保留手动 `workflow_dispatch`。**

## 已知边界

- 版本错位：DeepSpeed 安装自主仓被监控的 release 版本，examples 仓跟随 `master`，二者存在小幅错位的可能；若某 release 与 examples `master` 不兼容导致失败，按结果定位后可将 manifest 的 `target_ref` 固定或向上游反馈。
- compression 的 `bert/gpt2/cifar` 三个 `*_no_trainer` 入口硬编码 `torch.device("cuda")`，需 patch 后方可接入；gan 需去掉 `--cuda` 硬编码；data_efficiency / deepspeed_finetune_demo 依赖远程 HF 数据集，换本地 fixture 后可升级。均列为下一轮候选，当前在 manifest 中以 unsupported + 注释记录。
