# cosdt-ci-test/workflows

Example CI 的唯一部署点，红绿只出现在本仓的 Actions。每个被看护的项目在根目录有一个同名目录存放配套清单、脚本与数据；流水线本身放在 `.github/workflows/<项目>.yml`。

```
.github/workflows/ms-swift.yml   ms-swift 流水线
ms-swift/                        ms-swift 的清单、overlay、fixture、脚本
```

## ms-swift

在昇腾 NPU 上把清单里 `supported` 的 example 跑通。example 退出码非 0 即判红，不比对 loss 等数值。

- 目标仓由入参 `target_repo` / `target_ref` 决定，默认 `cosdt-ci-test/ms-swift`。**不** checkout、clone、请求或互动 [`modelscope/ms-swift`](https://github.com/modelscope/ms-swift)（不提 PR、不开 issue、不评论、不要 secret、不 push）。
- `examples_manifest.yaml` 由 `scripts/bootstrap_manifest.py` 扫描目标仓 `examples/` 下的 `.sh` / `.py` / `.yaml` 生成。除 `examples/ascend/train/qwen3/qwen3_lora_megatron.sh` 外全部标 `unsupported`，这是任务规定的分类，不是社区结论。清单与磁盘的差异只打印路径，不使 job 失败。
- `overlays/*.args` 把 example 压到 CI 规模（仓内 8 条 fixture、短序列、输出到 CI 目录）。给 example 补 `"$@"` 只发生在 CI 临时工作区，不改任何 ms-swift 仓库。

### 触发

- `workflow_dispatch`：手动指定 `target_repo` 与 `target_ref`。
- `repository_dispatch`（`ms-swift-ci-completed`）：由测试 fork 上的 notifier 在 `citest-npu` 成功后发送，payload 必须带 `repo` 与 `sha`。

无 `schedule`。
