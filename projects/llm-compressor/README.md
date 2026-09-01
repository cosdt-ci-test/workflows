# llm-compressor

本目录是 [llm-compressor](https://github.com/vllm-project/llm-compressor) 的看护配套数据，不是上游源码。example 流水线在 [.github/workflows/llm-compressor-examples.yml](../../.github/workflows/llm-compressor-examples.yml)。Quick Start 流水线在 [.github/workflows/llm-compressor-quick-start.yml](../../.github/workflows/llm-compressor-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：基础支持；阶段 A）。

上游默认分支是 `main`，最新正式 release 是 `0.13.0`。上游 GitHub Actions / Buildkite 覆盖 CPU、NVIDIA GPU 和 Intel XPU，没有 Ascend / CANN / torch_npu CI，因此按阶段 A 在本仓落地。

两条线的 `schedule` 都保持注释。

## 绿灯含义

### example 线

四个 `profile` 的绿灯不是同一句话。未知 profile 会在安装任何包之前非 0 退出。

- **`npu_inference`**：上游 `examples/compressed_inference/fp8_compressed_inference.py` 能加载公开的 FP8 压缩 TinyLlama，并且 `compressed_model` 与 `inputs` 都在 `npu:0` 上完成 `generate`。进程 exit 0 但张量在 CPU 上，会被判红。这不是 `oneshot()` 量化绿。当前这条是诚实红：模型能加载到 `npu:0`，但 `generate` 在 FP8 解压时触发 `aclnnInplaceCopy`（错误码 561103）。同一条命令把 `device_map` 设成 `cpu` 时可以解压，说明失败在昇腾拷贝算子。
- **`cpu`**：上游 `examples/quantization_w8a8_fp8/llama3_example.py` 在 CPU 上做完数据无关 FP8 PTQ（`oneshot` + 保存）。这条**不是**昇腾推理绿。setup 只装 CPU `torch`，并卸掉 `torch_npu`，避免 `dispatch_model` 把后续 `generate` 派到 NPU、再撞上上面那条解压红灯。权重从 ModelScope `LLM-Research/Meta-Llama-3-8B-Instruct` 预取，种进 Hugging Face 缓存里的 `meta-llama/Meta-Llama-3-8B-Instruct`，不改上游脚本里的模型 ID。种完之后 setup 会把 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` 写进下一 step，避免 `from_pretrained` 再去 Hub 要真实 commit（门禁模型没 token 会 401，种进去的 snapshot 名字也对不上 Hub SHA）。
- **`npu_int8`**：上游 `examples/quantization_w8a8_int8/gemma2_example.py`。库的 `get_main_device()` 在装了 `torch_npu` 时会把 Sequential GPTQ 放到 `npu:0`。这条当前是诚实红：Gemma-2-2B 的 `mlp.down_proj` 输入维是 9216，`torch.linalg.cholesky` 走 `aclnnLinalgCholesky` 时要求 last dim ≤ 8192（错误码 161002 / EZ1001）。Gemma 权重从 ModelScope `LLM-Research/gemma-2-2b-it` 预取到 `google/gemma-2-2b-it`。上游调用是 `oneshot(dataset="perfectblend", num_calibration_samples=512)` 且不传 `splits`：库会把约 140 万条全部 tokenize 进内存，32GiB cgroup 会在 GPTQ 开始前 OOM（exit 137）。`run_example.sh` 补上 `splits=train[:512]`，只去掉这份看护噪音，不改 GPTQ 配方。
- **`npu_autoround`**：上游 `examples/autoround/quantization_wNa16/qwen3_example_custom_dataset.py`。Sequential AutoRound 在 `npu:0` 上跑完 36 层 W4A16（约 45 分钟），随后 `dispatch_for_generation` + `generate` 的参数仍在 `npu:0`。Qwen3-8B 从 ModelScope `Qwen/Qwen3-8B` 预取；校准数据是脚本自己切的 `ultrachat_200k` `train_sft` 前 128 条。AutoRound 初始化会打印 `Using default calibration dataset NeelNanda/pile-10k`，实际量化用的是已经 capture 的那 128 条，不会再去拉 pile-10k。

`run_example.sh` 用 `runpy.run_path()` 执行原始脚本。NPU profile 在 exit 0 之后检查设备；CPU profile 检查量化配置或保存目录，并拒绝参数落到 `npu:*`。Hugging Face 预取会设 `HF_HUB_DISABLE_XET=1`，否则新版 `huggingface_hub` 走 Xet 数据面（`cas-server.xethub.hf.co`），不受 `HF_ENDPOINT` 控制，国内会 401。oneshot 之后的 `generate` 会设 `TORCHDYNAMO_DISABLE=1`：transformers 5 默认会 `torch.compile`，8B 在 CPU 上第一次编译会把 90 分钟 timeout 吃完。字符串数据集别名若没带 `splits`，会在调用 `oneshot` 时补上 `train[:num_calibration_samples]`，避免把整份数据集 tokenize 进 32GiB cgroup。

清单仍要求每条 `supported` 写 `npu_devices`，所以 `cpu` job 也会占一台 NPU runner，只是不用卡。不要把 `cpu` 绿读成昇腾推理绿。

### Quick Start 线

文档绿灯表示：按 `docs/Quick-start-Ascend.md` 装上 `torch==2.10.0` / `torch_npu==2.10.0.post4` 和当前 release 的 `llmcompressor`，对 `nm-testing/tinysmokeqwen3` 做单层 W4A16 GPTQ，保存后重载，前向张量在 `npu:0`。探测块的 `torch.npu.is_available()` 不能单独当成量化成功。

## 看护范围

打了标签、会被看护的步骤：

- example：压缩推理；CPU 上的数据无关 FP8 `oneshot`；INT8 GPTQ 之后的 NPU 生成；AutoRound W4A16 之后的 NPU 生成
- Quick Start：检查 Python、安装 torch / torch_npu、安装 llm-compressor、`oneshot()` GPTQ、保存、重载、NPU 前向

无标签、**不看护**的步骤：

- `source set_env.sh` 和 `export PATH`（Quick Start 测试在 `prepare_environment` 里做等价的 CANN 注入）
- `npu-smi info`（设备表每次不同，正文只要求退出码 0）
- 清单 `unsupported` 里的其余上游 example，原因写在 [examples_manifest.yaml](examples_manifest.yaml)：CUDA 写死、NVIDIA NVFP4/MXFP 打包、多卡 DDP、门禁或超大模型、多模态、MoE / 剪枝 / QuIP 等非基础路径，以及 `.claude/skills` 模板

## 触发

example 线是脚本型 B 族：`examples` 树、`/releases/latest` tag、默认分支 HEAD 三信号，失败重试写成 `<signal>-retry`。手动入口是 `target_repo` / `target_ref`，没有 `force`。NPU job 不上传 artifact；`result.json` 由 `ubuntu-latest` 上的 `publish-result` 按 job 名 `run-example (${{ matrix.example.path }})` 回看 conclusion 后上传。

Quick Start 走共享引擎 `quick-start-template.yml`：互斥优先级 `release` > `doc` > `retry`。`doc_url` 走 GitHub Contents API（`ref=${{ github.sha }}`）。cache 前缀是引擎拥有的 `monitor-state-llm-compressor-`，与 example 线的 `llm-compressor-examples-monitor-state-` 以及 `llm-d` 的 `monitor-state-llm-d-` 都不互为前缀。
