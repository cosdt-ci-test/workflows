# DGL 昇腾接入评估：暂不支持

| 项目 | 分类 | 支持程度 | 评估日期 | 结论 |
| --- | --- | --- | --- | --- |
| [dgl](https://github.com/dmlc/dgl)（dmlc/dgl） | 训练加速 | 待支持 | 2026-08-29 | 暂不接入昇腾 CI |

## 总之

DGL 官方不支持昇腾 NPU，且上游已事实上冻结（最新 release v2.4.0 发布于 2024-09-03，此后 24 个月无新版本）。昇腾侧唯一的适配渠道是华为与北京邮电大学合作的实验性 fork [BUPT-GAMMA/dgl-ascend](https://github.com/BUPT-GAMMA/dgl-ascend)：没有 release、没有 tag、没有预编译包。而本仓 quick-start 看护由上游 release 信号驱动，需要能锚定一个含昇腾适配的版本——对 dgl-ascend 这三级信号全部落空，对 dmlc/dgl 锚定到的版本又不含昇腾代码。因此**暂不接入**。本文档记录完整原因与证据，并给出重评触发条件。

## 原因一：官方层——无昇腾支持，且上游冻结

**dmlc/dgl 的代码、构建系统、发布渠道全链路均无昇腾 NPU 支持：**

- 设备后端只有两种：`src/array` 下仅有 `cpu` 与 `cuda` 两个设备实现目录（[src/array](https://github.com/dmlc/dgl/tree/master/src/array)），运行时 `DeviceAPI` 抽象只有 CPU/CUDA 两套。
- 官方安装文档只提供 CPU 与 CUDA（11.8 / 12.1 / 12.4）组合（[安装文档](https://docs.dgl.ai/install/index.html)、[Get Started](https://www.dgl.ai/pages/start.html)），README 与文档全文无 Ascend / NPU / torch_npu / CANN 字样。
- 构建系统（[CMakeLists.txt](https://github.com/dmlc/dgl/blob/master/CMakeLists.txt)）中唯一的加速器开关是 `USE_CUDA`，无第三种设备后端的构建路径。
- 官方 wheel 源（[data.dgl.ai/wheels](https://data.dgl.ai/wheels/repo.html)）中 linux-aarch64 仅有 CPU 版 wheel，且止步于 2.1.0；2.2–2.4 的专项索引只有 x86_64。社区实践（在昇腾 aarch64 服务器上 `pip install dgl==2.1.0` + torch_npu 可完成 import）走的是纯 CPU 计算路径，不构成 NPU 支持。

**上游维护事实上冻结：**

- 最新 release [v2.4.0](https://github.com/dmlc/dgl/releases/tag/v2.4.0) 发布于 2024-09-03，至评估日 24 个月无新版本；[master 分支](https://github.com/dmlc/dgl/commits/master)最后提交为 2025-08-01，此后无提交。
- v2.3.0 起官方宣布停发 Windows/macOS 预编译包——预编译范围正在收缩。
- 社区提交的 NPU 支持 RFC（[#7920](https://github.com/dmlc/dgl/issues/7920)）被机器人标记 stale，无维护者回应。

## 原因二：生态层——适配仅存在于实验性 fork，未达可看护成熟度

**唯一适配渠道是 [BUPT-GAMMA/dgl-ascend](https://github.com/BUPT-GAMMA/dgl-ascend)**（华为联合北邮石川团队的 DGL-NPU 专项攻坚项目，2026-07-23 经[华为计算官方渠道](https://cj.sina.cn/article/norm_detail?froms=ttmp&url=https%3A%2F%2Ffinance.sina.com.cn%2Froll%2F2026-07-23%2Fdoc-iniivayy4560038.shtml%3Ffinpagefr=ttzz)官宣）：

- **无 release、无 tag、无预编译 wheel**：仓库 [Releases 页面](https://github.com/BUPT-GAMMA/dgl-ascend/releases)为空；`git ls-remote --tags https://github.com/BUPT-GAMMA/dgl-ascend.git` 实测返回为空（连 tag 都没有）；也没有任何 PyPI/wheel 分发。官方新闻同时承认此前"DGL 原生框架暂未提供昇腾 NPU 适配能力"，项目目标是"为后续合入上游奠定基础"——即截至官宣时仍未进入 dmlc/dgl。
- **未合入上游**：上游 PR [#7912（Integrate NPU SpMM Demo into DGL）](https://github.com/dmlc/dgl/pull/7912) 处于 Open 状态、0 review；配套 RFC [#7920](https://github.com/dmlc/dgl/issues/7920) 标记 stale。
- **实验性支持范围**：仅覆盖 SpMM / BSpMM / SegmentReduce（sum/mean/max/min）/ RandomWalk 算子与 LightGCN、GraphSAGE、HEREC 三个模型，另有基于 HCCL 的分布式；无稳定性与版本化承诺。

**昇腾官方组织渠道排查结果（全部为空）：**

| 渠道 | 结果 |
| --- | --- |
| Gitee：[gitee.com/ascend/dgl](https://gitee.com/ascend/dgl) | 404 |
| GitCode：[gitcode.com/Ascend/dgl](https://gitcode.com/Ascend/dgl) | 404 |
| GitHub：[github.com/Ascend/dgl](https://github.com/Ascend/dgl) | 404 |
| [ascend.github.io](https://ascend.github.io/) 社区文档站 | 无任何 DGL / 图神经网络内容 |
| CANN / torch_npu 官方文档 | 无 DGL 兼容性声明 |

**成熟度对比（判定标尺）**：本仓已接入的 tilelang 同样依赖昇腾生态仓，但 `tile-ai/tilelang-ascend` 有正式 release（`v0.1.1.010-release`）、文档化的安装路线（`build_wheel_ascend.sh`）与版本化依赖要求（CANN ≥8.3.RC1 等），满足"可看护"条件。dgl-ascend 三者皆无——差异的核心是**可看护性**，而不是"有没有适配"。

## 原因三：机制层——release 驱动的看护无法锚定版本

本仓 quick-start 看护由 monitor 轮询上游 release 触发，`UPSTREAM_REF` 按三级回退链取版本：`/releases/latest` → `/releases?per_page=20`（排除 pre-release）→ `/tags?per_page=1`。

- 对 **dgl-ascend**：无 releases、无 tags，三级回退全部落空，无法锚定任何 ref，看护流程根本无法建立。
- 对 **dmlc/dgl**：可以锚定 v2.4.0，但该版本内容不含任何昇腾适配代码，checkout 后跑 NPU quick-start 必然失败——锚定无意义。

即：无论把 upstream 指向哪一侧，都不存在"有 release 信号且含昇腾能力"的可看护对象。

## 实验性路线（参考，非看护对象）

如需立即在昇腾上试用 DGL，可参考 dgl-ascend 的 [AscendInstallation.md](https://github.com/BUPT-GAMMA/dgl-ascend/blob/master/AscendInstallation.md)（**实验性，未合入上游，无版本化承诺**）：

1. 安装 CANN 并创建 conda 环境（python 3.10）。
2. 安装 CPU 版 PyTorch 与 torch_npu：`pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu`，再 `pip install torch_npu==2.8.0`。
3. 克隆仓库并更新子模块：`git clone https://github.com/BUPT-GAMMA/dgl-ascend.git && cd dgl-ascend && git submodule update --init --recursive`。
4. 源码编译：`bash ./script/build_dgl_ascend.sh`，然后 `cd python && python setup.py install && python setup.py build_ext --inplace`。
5. 快速开始示例：[LightGCN](https://github.com/BUPT-GAMMA/dgl-ascend/blob/master/examples/pytorch/lightgcn/README.md)。

## 重评触发条件

满足以下任一条件时重新评估，并按本仓 guard-\<project\> 模式接入（参照 tilelang 先例：upstream 指向适配仓 + 源码构建 wheel），同时将本文档迁出 `docs/unsupported/`：

1. `BUPT-GAMMA/dgl-ascend` 发布正式 release 或 tag（使版本可锚定）。
2. 适配合入 dmlc/dgl 上游，且上游发布包含昇腾能力的 release。
