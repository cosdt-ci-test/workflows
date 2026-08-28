# specforge

本目录是 [SpecForge](https://github.com/sgl-project/SpecForge) 的看护配套数据，不是 SpecForge 源码。流水线在 `[.github/workflows/specforge-quick-start.yml](../../.github/workflows/specforge-quick-start.yml)`。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：基础支持；阶段 A）。

在昇腾 NPU 上把 Quick Start 文档里所有 `#test` / `#test-setup` 代码块跑通，对照 `#test-result` 做输出断言。脚本退出码非 0 即判红。

## 文档

`docs/Quick-start-Ascend.md` 走的是 `docs/markdown_doc_test_label.md` 的契约：每个 `shell` 代码块都带 `#test` / `#test-result` / `#test-setup` 标签加 `id=` / `store=` / `load='x>>y'` / `fuzzy='xxx'` 参数，runner（`tests/test_quick_start_ascend.py` 的 `MarkdownDocTestBase.run_template`）会按文档顺序抓取、解析、执行并比对预期。

文档覆盖：

- CANN / torch 2.11.0 / torch_npu 2.11.0 / sglang 0.5.14 / modelscope 前置检查与安装（按 specforge `pyproject.toml` 的 pin 对齐，不绕 `--no-deps`）；
- specforge 源码安装（`git clone <ref>` + `pip install .`），PyPI 二进制 wheel 作为可选路径；
- `specforge --help` / `specforge train --help` CLI 自检；
- **端到端 smoke**：在 4 卡 NPU 上起 `mooncake_master` + SGLang capture server（卡 0）+ `specforge train` 1 步训练（卡 1），跑完 trap 兜底清理。命令直接写在 [Quick-start-Ascend.md 端到端 smoke 章节](docs/Quick-start-Ascend.md#端到端-smoke1-步训练) 的 `<details>` 折叠块里，CI 整段执行；
- **完整训练链路**（不在 CI 范围内，文档末尾展开）：多卡 managed-local 拓扑、SGLang capture patch 应用顺序、数据集准备、生产参数（`nproc_per_node=10`、`num_anchors=512` 等）、成功判据（`prompts_failed=0` / `step N: {...loss...}` / teardown 无 `could not drain`）、清理脚本、排错表（`ACL_ERROR_RT_CONTEXT_NULL` / USP 在 NPU 上不可用 / `--spec-capture-aux-layer-ids` 不一致等）。

## 触发

`specforge-quick-start.yml` 接受：

- `schedule`：每 6 小时轮询上游一次。`monitor` job（免费的 ubuntu-latest）对比三个信号与上次记录（状态存 actions/cache）：文档 hash、latest release tag、main HEAD SHA 任一变化才 checkout 上游、在 NPU runner 上跑测试，被测 ref 按 release > doc > commit 优先级取自变化的信号；全部无变化则 monitor 后直接结束。上次看护失败时，下个周期即使无变化也会自动重试（`record-outcome` job 回写成败）。
- `workflow_dispatch`：手动触发；绕过 monitor 门，直接拿 latest release tag 跑测试。