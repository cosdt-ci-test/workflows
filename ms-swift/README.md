# ms-swift

本目录是 [ms-swift](https://github.com/modelscope/ms-swift) 的看护配套数据，不是 ms-swift 源码。流水线在 [`.github/workflows/ms-swift-examples.yml`](../.github/workflows/ms-swift-examples.yml) 和 [`.github/workflows/ms-swift-quick-start.yml`](../.github/workflows/ms-swift-quick-start.yml)。注册信息见根目录 [projects.yaml](../projects.yaml)（分类：训练加速；支持程度：基础支持；阶段 A）。

在昇腾 NPU 上把清单里 `supported` 的 example 跑通。example 退出码非 0 即判红，不比对 loss 等数值。

## 目标仓与红线

目标仓由入参 `target_repo` / `target_ref` 决定，默认 `cosdt-ci-test/ms-swift`。**不** checkout、clone、请求或互动 [`modelscope/ms-swift`](https://github.com/modelscope/ms-swift)（不提 PR、不开 issue、不评论、不要 secret、不 push）。

`ms-swift-quick-start.yml` 是这条红线的**显式例外**：它只读监控 `modelscope/ms-swift` 的 latest release 与 main HEAD，仍然不写、不提 PR、不评论、不要 secret、不 push。

## 清单、overlay、fixture

- `examples_manifest.yaml` 由仓库根目录 `scripts/bootstrap_manifest.py` 扫描目标仓 `examples/` 下的 `.sh` / `.py` / `.yaml` 生成。除 `examples/ascend/train/qwen3/qwen3_lora_megatron.sh` 外全部标 `unsupported`，这是任务规定的分类，不是社区结论。清单与磁盘的差异只打印路径，不使 job 失败。
- `overlays/*.args` 把 example 压到 CI 规模（仓内 8 条 fixture、短序列、输出到 CI 目录）。
- `scripts/run_example.sh` 是 ms-swift 专用：在 CI 临时工作区给 example 补 `"$@"` 并展开 overlay。不改任何 ms-swift 仓库。

重新生成清单（会覆盖本目录的 yaml，先确认 supported 段）：

```bash
python3 scripts/bootstrap_manifest.py \
  --target-root /path/to/ms-swift \
  --output ms-swift/examples_manifest.yaml \
  --supported examples/ascend/train/qwen3/qwen3_lora_megatron.sh \
  --runner linux-aarch64-a2-2 \
  --npu-devices 0,1 \
  --overlay overlays/qwen3_lora_megatron.args \
  --timeout-minutes 180
```

## 触发

`ms-swift-examples.yml` 接受：

- `workflow_dispatch`：手动指定 `target_repo` 与 `target_ref`。
- `repository_dispatch`，三种 `event_type`，均由测试 fork `cosdt-ci-test/ms-swift` 上的 notifier 发送：
  - `ms-swift-ci-completed`：payload 必须带 `repo` 与 `sha`。
  - `ms-swift-examples-changed`：`examples/**` 被 push 到 fork 的 main。payload 必须带 `repo` 与 `sha`。
  - `ms-swift-release`：fork 上推了 `v**` 标签。payload 必须带 `repo` 与 `ref`（tag 名）。

这条流水线没有 `schedule`。阶段 B 兜底（定时指向上游）见 [docs/guarding-examples.md](../docs/guarding-examples.md)。

`ms-swift-quick-start.yml` 每 6 小时跑一次监控；文档、上游 release 或 main HEAD 有变化才在 NPU 上跑 `tests.docs.test_quick_start_ascend`。也可以 `workflow_dispatch` 并勾选 `force_install`。
