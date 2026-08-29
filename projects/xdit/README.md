# xDiT

本目录是 [xDiT](https://github.com/xdit-project/xDiT)（PyPI 包名 [xfuser](https://pypi.org/project/xfuser/)，命令行入口 `xdit`）的看护配套数据，不是 xDiT 源码。流水线在 `[.github/workflows/xdit-quick-start.yml](../../.github/workflows/xdit-quick-start.yml)`。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：基础支持；阶段 A）。

在单卡昇腾 NPU 上跑通 xfuser：从源码安装 + import + 验证 `xfuser/envs.py` 里的 NPU 分发路径（`_is_npu()` / `get_torch_distributed_backend() == "hccl"`）+ `xdit --help` 解析通过。文档副作用：单步 inference smoke 用 `--num_inference_steps 1` + `256×256` 的 SD3-medium 把"启动 xfuser runtime + `torch.distributed` init + `xdit` runtime"从头到尾跑一遍（CI 时间预算允许时不跑，模型下载会撑爆 cold-cache 预算）。

## 触发

`xdit-quick-start.yml` 接受：

- `schedule`：每 6 小时轮询上游一次，文档 / 上游 release / main HEAD 有变化才在 NPU 上跑 `tests.test_quick_start_ascend`。
- `workflow_dispatch`：手动 trigger。

cwd 是 `workflows/projects/xdit`，由 GitHub Actions 自动 checkout。环境契约：`MONITORED_DOC_URL` / `UPSTREAM_REF` / `NPU_READY` 由 engine 注入，测试逻辑都在这个 `tests/` 目录下。

详细 trigger 模式与 cache I/O 流程参见父引擎 [quick-start-template.yml](../../.github/workflows/quick-start-template.yml) 与项目文档说明 [docs/guarding-examples.md](../../docs/guarding-examples.md)。
