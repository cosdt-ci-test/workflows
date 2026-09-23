# transformers

本目录是 [transformers](https://github.com/huggingface/transformers) 的看护配套数据，不是 transformers 源码。流水线位于 [`transformers-examples.yml`](../../.github/workflows/transformers-examples.yml) 和 [`transformers-quick-start.yml`](../../.github/workflows/transformers-quick-start.yml)。注册信息见根目录 [`projects.yaml`](../../projects.yaml)（分类：训练加速；支持程度：基础支持；阶段 A）。

在昇腾 NPU 上运行清单中 `supported` 的 example。example 退出码非 0 即判红，不比较 loss 或数值精度。

## 清单、fixture 和脚本

- `examples_manifest.yaml` 扫描目标仓 `examples/` 下的 Python、Shell 和 YAML 文件。`supported` 是实际调度的低成本 smoke 用例；其他路径先由 manifest-check 报为新增或失效，不自动占用 NPU。
- 当前 supported 用例覆盖：generation、GLUE/分类（Trainer + no-trainer）、语言建模（CLM/MLM/FIM，Trainer + no-trainer）、抽取式 QA（含 beam-search 与 seq2seq 变体）、多选（SWAG 双入口）、NER（双入口）、摘要和翻译（Trainer + no-trainer）。全部使用目标仓内小型 fixture、tiny 模型缓存、单卡和单步/少量样本训练。
- 不纳入 supported 的可执行 example 及原因（见 manifest 内注释）：run_plm（ModelScope 无 tiny xlnet）、run_xnli/run_greedy（硬编码 Hub 数据集，无本地文件入口）；视觉/语音家族待后续批次补 profile 与 fixture。
- `scripts/setup_example.sh` 按 profile 安装 editable transformers 和 example 依赖；`generation` / `glue` 保持原有依赖，`small-training` 用于 QA、SWAG、NER（额外装 `seqeval`），`lm` 用于 CLM/MLM/FIM，`seq2seq` 用于摘要/翻译/seq2seq-QA（额外装 `sacrebleu`、`rouge-score`、`nltk`）；并把扩展名缺失的 `wiki_text/wiki_00` fixture 拷贝为输出目录下的 `train.txt` 供 LM 用例引用。
- `scripts/run_example.sh` 对 `run_*_no_trainer.py` 使用 Accelerate 启动，其他普通 Python example 直接执行；脚本只修改目标 checkout 的临时副本来追加参数，不向上游仓库写入、提交或推送。
- 训练和模型缓存优先使用 runner 上的共享缓存；运行输出写入 `CI_OUTPUT_DIR`，不污染目标 checkout。

重新生成清单时，先确认 supported 段，再手工补回 profile、资源和 `overlay_args`：

```bash
python3 scripts/bootstrap_manifest.py \
  --target-root /path/to/transformers \
  --output projects/transformers/examples_manifest.yaml \
  --scan-root examples \
  --include-extension .py \
  --supported examples/pytorch/text-generation/run_generation.py \
  --supported examples/pytorch/text-classification/run_glue_no_trainer.py \
  --runner linux-aarch64-a2-2 \
  --npu-devices 0 \
  --image swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12 \
  --timeout-minutes 60
```

两条任务使用 `linux-aarch64-a2-2` runner，但只暴露设备 `0`；容器设备挂载由 manifest-check 根据 `npu_devices` 自动派生。镜像和超时属于清单条目，不写死在 workflow 中。

## Quick Start

[`docs/Quick-start-Ascend.md`](docs/Quick-start-Ascend.md) 是本仓专用的 Ascend Quick Start smoke，使用公开的 `Qwen/Qwen2.5-1.5B-Instruct`，覆盖 `AutoModelForCausalLM`、`pipeline` 两种推理方式与全流程对话示例，不需要 Hugging Face token。文档遵守 [`docs/markdown_doc_test_label.md`](../../docs/markdown_doc_test_label.md)，包含前置条件、环境检查、依赖安装和最终 smoke；`tests/test_quick_start_ascend.py` 从文档提取 `pycon` 示例并在 NPU runner 中执行。

## 触发和结果

- `transformers-examples.yml` 每 6 小时轮询上游 `examples/` 最新 commit、latest release 和 main HEAD。任一信号变化才 checkout 上游并运行 supported matrix；失败会在下个周期重试。`workflow_dispatch` 可指定 `target_repo` 和 `target_ref`，手动运行不经过 monitor 门。
- `transformers-quick-start.yml` 轮询 Quick Start 文档 hash、latest release 和 main HEAD。无变化时不占用 NPU；手动运行始终执行测试。
- 每条 example 和 Quick Start 都上传包含 `result.json` 的 artifact。外部机器可通过 Job API 读取结论，或按 `docs/artifacts.md` 下载 artifact。

当前范围是阶段 A 的下游轮询看护；CLM、分布式训练、需要 gated 模型或不会自行退出的服务暂不纳入 supported。
