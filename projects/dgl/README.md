# DGL-Ascend

本目录是 [DGL-Ascend](https://github.com/BUPT-GAMMA/dgl-ascend)（[dmlc/dgl](https://github.com/dmlc/dgl) 的华为×北邮昇腾适配 fork）的看护配套数据，不是 DGL 源码。流水线在 [.github/workflows/dgl-quick-start.yml](../../.github/workflows/dgl-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：训练加速；支持程度：新兴适配；阶段 A）。

文档按 Liger-Kernel 快速入门的结构组织，使用 Python 3.10、CANN 9.1.0、torch 2.9.0 和 torch_npu 2.9.0.post2 源码编译 DGL-Ascend，并使用 DGL 官方 Gowalla 数据运行单卡 LightGCN 一轮训练。数据中不存在编号范围内的孤立用户或物品，不修改上游模型源码。

上游 GitHub fork 没有 release/tag，引擎通过 `/commits/HEAD` 解析完整 SHA。测试夹具在文档 clone 后、子模块更新前检出该 SHA，并核对已安装原生库与本次编译产物的 SHA256，保证 artifact 的 `target_ref` 与实际构建源码一致。摘要采用分块读取计算，兼容 Python 3.10。验证不在源码包目录中运行。

文档先安装 `Cython==3.0.12`，兼容上游 Cython 代码中的 Python 2 `long` 名称；新版编译器会报 `undeclared name not builtin: long`。随后进入 `dgl-ascend/python`，按上游安装文档依次执行 `python setup.py install` 和 `python setup.py build_ext --inplace`，最后用 `cd ../..` 返回原工作目录。每条命令独占一行，安装后的独立导入检查拦截安装失败。夹具检查已安装的 Cython 扩展和原生库，并执行 NPU 图消息传递及反向传播检查；训练输出必须包含完整一轮结束日志和有限的 BPR 损失。GraphSAGE、HERec 和 GitCode 发行包不在本次范围。

编译直接使用上游脚本，不额外添加 `torch.libs` 搜索路径；训练使用默认 NPU 执行模式，不注入强制同步配置。夹具在训练前使用与训练命令相同的环境检查完整 Gowalla 图：从原始边独立计算 CPU 参考度数，交错执行 PyTorch 运算和 DGL 查询，重复三次逐项核对 NPU 入度、出度。保留 `GraphConv` 的零入度检查，不使用 `allow_zero_in_degree=True` 绕过错误。若新环境运行失败，再根据具体日志补充所需适配。

## 触发

`dgl-quick-start.yml` 接受：

- `schedule`：预留每 6 小时轮询，当前仍注释，待本版文档上卡验收后再启用。
- `workflow_dispatch`：手动 trigger。

夹具创建独立临时目录并切换 cwd，源码、编译产物和数据均落在其中；正常结束或失败后恢复 cwd 和环境变量并删除该目录。环境契约为 `MONITORED_DOC_URL` / `UPSTREAM_REF` / `NPU_READY`。文档通过带触发提交 SHA 的 Contents API 读取。普通块超时为 1800 秒，源码编译块为 7200 秒，workflow 总预算为 240 分钟。

详细 trigger 模式与 cache I/O 流程参见父引擎 [quick-start-template.yml](../../.github/workflows/quick-start-template.yml) 与项目文档说明 [docs/guarding-examples.md](../../docs/guarding-examples.md)。
