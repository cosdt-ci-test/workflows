# specforge

本目录是 [SpecForge](https://github.com/sgl-project/SpecForge) 的看护配套数据，不是 SpecForge 源码。流水线在 `[.github/workflows/specforge-quick-start.yml](../../.github/workflows/specforge-quick-start.yml)`。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：基础支持；阶段 A）。

在昇腾 NPU 上把 Quick Start 文档里所有 `#test` / `#test-setup` 代码块跑通，对照 `#test-result` 做输出断言。脚本退出码非 0 即判红。

## 文档

`docs/Quick-start-Ascend.md` 走的是 `docs/markdown_doc_test_label.md` 的契约：每个 `shell` 代码块都带 `#test` / `#test-result` / `#test-setup` 标签加 `id=` / `store=` / `load='x>>y'` / `fuzzy='xxx'` 参数，runner（`tests/test_quick_start_ascend.py` 的 `MarkdownDocTestBase.run_template`）会按文档顺序抓取、解析、执行并比对预期。

文档覆盖：

- 镜像预装 torch 2.10.0 + torch_npu 2.10.0 + sglang 0.5.18 + CANN 9.0.0 + Python 3.11.15（`check-torch` 步校验；specforge 源码 `pip install --no-deps .` 跳过上游 `pyproject.toml` 的 sglang==0.5.14 pin）；再装 modelscope 1.37.0 + mooncake-transfer-engine 0.3.13；
- specforge 源码安装（`git clone <ref>` + `pip install .`），PyPI 二进制 wheel 作为可选路径；
- `specforge --help` / `specforge train --help` CLI 自检；
- **端到端 smoke**：在 4 卡 NPU 上起 `mooncake_master` + SGLang capture server（卡 0）+ `specforge train` 1 步训练（卡 1）。Smoke 拆成 5 个独立 `#test` 块（`smoke-download-model` / `smoke-apply-patches` / `smoke-start-mooncake` / `smoke-start-sglang` / `smoke-train`），每段独立失败定位；mooncake + sglang 是 `nohup` 后台进程跨段复用，store `model_path` 把下载路径传到 sglang / trainer 段。命令直接写在 [Quick-start-Ascend.md 端到端 smoke 章节](docs/Quick-start-Ascend.md#端到端-smoke1-步训练)；CI smoke 已经覆盖端到端主路径，本地不再补完整训练示例（参考 xtuner 删「完整 5 epoch 本地手动块」的同一取舍）。

## 触发

`specforge-quick-start.yml` 接受：

- `schedule`：每 6 小时轮询上游一次。`monitor` job（免费的 ubuntu-latest）对比三个信号与上次记录（状态存 actions/cache）：文档 hash、latest release tag、main HEAD SHA 任一变化才 checkout 上游、在 NPU runner 上跑测试，被测 ref 按 release > doc > commit 优先级取自变化的信号；全部无变化则 monitor 后直接结束。上次看护失败时，下个周期即使无变化也会自动重试（`record-outcome` job 回写成败）。
- `workflow_dispatch`：手动触发；绕过 monitor 门，直接拿 latest release tag 跑测试。