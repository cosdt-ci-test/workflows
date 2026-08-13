# cosdt-ci-test/workflows

本仓库是昇腾 example CI 看护流水线的**唯一部署点**。红绿只出现在本仓的 Actions。

## 本阶段范围

- 被 checkout、被跑 example 的代码只来自测试靶 [`cosdt-ci-test/ms-swift`](https://github.com/cosdt-ci-test/ms-swift)。
- `target_repo` / `target_ref` 是 workflow 入参，避免把 fork URL 写死进脚本逻辑。本阶段验证只传入 `cosdt-ci-test/ms-swift`。
- **不** checkout、clone、请求或互动 [`modelscope/ms-swift`](https://github.com/modelscope/ms-swift)（不提 PR、不开 issue、不评论、不要 secret、不 push）。

## 清单分类

`examples_manifest.yaml` 由 `scripts/bootstrap_manifest.py` 扫描目标仓 `examples/` 下的 `.sh` / `.py` / `.yaml` 生成。

除 `examples/ascend/train/qwen3/qwen3_lora_megatron.sh` 外，其余全部标 `unsupported`。这是本任务规定的分类，不是社区结论。新增文件只打印路径，不使 job 失败。

## 触发

- `workflow_dispatch`：手动指定 `target_repo`（默认 `cosdt-ci-test/ms-swift`）和 `target_ref`。
- `repository_dispatch`（`ms-swift-ci-completed`）：由测试 fork 上的 notifier 在 `citest-npu` 成功后发送。payload 必须带 `repo` 与 `sha`。

无 `schedule`。`repository_dispatch` 不改 golden。标定 golden 只通过 `workflow_dispatch` 且 `update_golden=true`。

对 example 的 `"$@"` 只在 CI 临时工作区 sed，不改任何 ms-swift 仓库。
