# ms-swift

本目录是 [ms-swift](https://github.com/modelscope/ms-swift) 的看护配套数据，不是 ms-swift 源码。流水线在 `[.github/workflows/ms-swift-examples.yml](../../.github/workflows/ms-swift-examples.yml)` 和 `[.github/workflows/ms-swift-quick-start.yml](../../.github/workflows/ms-swift-quick-start.yml)`。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：基础支持；阶段 A）。

在昇腾 NPU 上把清单里 `supported` 的 example 跑通。example 退出码非 0 即判红，不比对 loss 等数值。

## 清单、fixture

- `examples_manifest.yaml` 由仓库根目录 `scripts/bootstrap_manifest.py` 扫描目标仓 `examples/` 下的 `.sh` / `.py` / `.yaml` 生成。`supported` 是本仓实际调度的条目：runner 按 example 脚本写明的卡数选 `linux-aarch64-a2-1` / `-2` / `-4` / `-8`，不把小任务挂到更大的机器上。多机 2×8 卡、16 卡 / A3 SuperPoD、8×96GiB A5，以及 `swift deploy` 这种不会退出的常驻服务，仍标 `unsupported`。这是任务规定的分类，不是社区结论。清单与磁盘的差异只打印路径，不使 job 失败；例外：`supported` 条目的 path 已不在磁盘上时 manifest-check 立即判红。看护目标是诚实地暴露 example 退出码，不修上游 example。
- 该 supported 条目的 `overlay_args` 把 example 压到 CI 规模（仓内 8 条 fixture、短序列、输出到 CI 目录）。bootstrap 只写占位注释，参数要手写进清单。
- `scripts/run_example.sh` 是 ms-swift 专用：在 CI 临时工作区给 example 补 `"$@"` 并展开 `OVERLAY_ARGS`。不改任何 ms-swift 仓库。

重新生成清单（会覆盖本目录的 yaml，先确认 supported 段，并重新写上 `overlay_args`）：

```bash
python3 scripts/bootstrap_manifest.py \
  --target-root /path/to/ms-swift \
  --output projects/ms-swift/examples_manifest.yaml \
  --supported examples/ascend/train/qwen3/qwen3_lora_megatron.sh \
  --runner linux-aarch64-a2-2 \
  --npu-devices 0,1 \
  --image swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.9.0.post2-910b-ubuntu22.04-py3.11 \
  --timeout-minutes 180
```



## 触发

`ms-swift-examples.yml` 接受：

- `schedule`：每 6 小时轮询上游一次。`monitor` job（免费的 ubuntu-latest）对比三个信号与上次记录（状态存 actions/cache）：`examples/` 最新 commit、latest release tag、main HEAD SHA。任一变化才 checkout 上游、在 NPU runner 上跑 example，被测 ref 按 examples > release > commit 优先级取自变化的信号；全部无变化则 monitor 后直接结束。上次看护失败时，下个周期即使无变化也会自动重试（`record-outcome` job 回写成败）。机制详见 [docs/guarding-examples.md](../../docs/guarding-examples.md)「要求 2」。**当前为节约 NPU 资源，`ms-swift-examples.yml` 里的 `schedule` 已注释，只保留手动 `workflow_dispatch`。**
- `workflow_dispatch`：手动指定 `target_repo` 与 `target_ref`（默认 `modelscope/ms-swift` 的 `main`），不经过 monitor 门。

`ms-swift-quick-start.yml` 每 6 小时跑一次监控；文档、上游 release 或 main HEAD 有变化才在 NPU 上跑 `tests.test_quick_start_ascend`。也可以 `workflow_dispatch` 手动触发（手动一律直接跑测试）。
