# xDiT

本目录是 [xDiT](https://github.com/xdit-project/xDiT)（PyPI 包名 [xfuser](https://pypi.org/project/xfuser/)）的看护配套数据，不是 xDiT 源码。流水线在 [.github/workflows/xdit-quick-start.yml](../../.github/workflows/xdit-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

在单卡昇腾 NPU 上跑通 xfuser：文档自装 torch 栈（torch 2.9.0 + torch_npu 2.9.0.post2 + triton 3.5.*）→ 源码安装 xfuser（`git clone` 默认分支 + `uv pip install -e ./xDiT`）→ 验证 NPU 分发（`xfuser/envs.py` 输出 `npu hccl True`）→ `modelscope download --local_dir` 拉 SD3 medium（全仓 ~30 GB）→ 最小单卡脚本经 `torchrun --nproc_per_node=1` 生成一张 256×256 单步图片并做 PNG 结构校验。多卡并行（USP / DP / TP）为文档内指引链接，不在看护范围。

## 触发

`xdit-quick-start.yml` 接受：

- `schedule`：每 6 小时轮询上游一次，文档 / 上游 release / main HEAD 有变化才在 NPU 上跑 `tests.test_quick_start_ascend`（当前暂时注释，0.6.0 适配跑通后恢复）。
- `workflow_dispatch`：手动 trigger。

cwd 是 `workflows/projects/xdit`（测试类 chdir 到 `/root/xdit-test` 钉文档执行目录，`models/` 子目录由宿主卷 `/data/ci-cache/xdit-models` 持久化）。环境契约：`MONITORED_DOC_URL` / `UPSTREAM_REF` / `NPU_READY` 由 engine 注入，测试逻辑都在这个 `tests/` 目录下。

详细 trigger 模式与 cache I/O 流程参见父引擎 [quick-start-template.yml](../../.github/workflows/quick-start-template.yml) 与项目文档说明 [docs/guarding-examples.md](../../docs/guarding-examples.md)。
