# DGL-Ascend

本目录是 [DGL-Ascend](https://github.com/BUPT-GAMMA/dgl-ascend)（[dmlc/dgl](https://github.com/dmlc/dgl) 的华为×北邮昇腾适配 fork）的看护配套数据，不是 DGL 源码。流水线在 [.github/workflows/dgl-quick-start.yml](../../.github/workflows/dgl-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：新兴适配；阶段 A）。

在单卡昇腾 NPU 上跑通 DGL-Ascend：文档自装 torch 栈（torch 2.9.0 + torch_npu 2.9.0.post2，复用 CI 现有 torch 线而非 fork 文档锁定的 2.8.0/py3.10）→ 源码编译 DGL-Ascend（`git clone` fork + submodule + `build_dgl_ascend.sh` + `pip install -e .`）→ 验证 `import dgl` + NPU 可用 → 最小单卡 LightGCN 示例经 `python main.py --device npu` 训练一轮并做 `Average BPR Loss` 日志校验。上游 fork 无 release/tag，看护靠引擎 `/commits/HEAD` 回退锚定 master HEAD sha 作为变更键。多模型（GraphSAGE / HERec）为文档内指引，不在看护范围。

## 触发

`dgl-quick-start.yml` 接受：

- `schedule`：每 6 小时轮询上游 fork 一次，文档 / 上游 HEAD 有变化才在 NPU 上跑 `tests.test_quick_start_ascend`（当前暂时注释，适配跑通后恢复）。
- `workflow_dispatch`：手动 trigger。

cwd 是 `workflows/projects/dgl`（测试类 chdir 到 `/root/dgl-test` 钉文档执行目录，clone + 编译产物落这里；gowalla 示例数据每次从 hf-mirror 的 RecZoo ``Gowalla_m1`` 数据拉取 train/test.txt，不跨 run 持久化）。环境契约：`MONITORED_DOC_URL` / `UPSTREAM_REF` / `NPU_READY` 由 engine 注入，测试逻辑都在这个 `tests/` 目录下。

详细 trigger 模式与 cache I/O 流程参见父引擎 [quick-start-template.yml](../../.github/workflows/quick-start-template.yml) 与项目文档说明 [docs/guarding-examples.md](../../docs/guarding-examples.md)。