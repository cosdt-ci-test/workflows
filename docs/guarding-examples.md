# 看护 Examples

## 总之

一般每个项目都存在 examples。我们的工作是让这些 Examples 被 github workflow 看护。一个项目的 Examples 至少一条能在昇腾上跑通。workflow 理想情况进入上游社区，随 CI 触发，而最不理想情况是在下游定时监听上游社区的改动而触发。核心思路是，在下游仓库准备好 workflow 和 examples 并跑通，然后尽可能地往上游社区推送。

## 阶段划分

1. 阶段 A：fork 上游仓库当靶场，用 fork 仓上的 notifier 模拟「上游 CI 完成」等事件。在 <https://github.com/cosdt-ci-test/workflows/> 仓内，开发看护 example 用的 workflow。
2. 阶段 B：按上游的接受度递进，一步一步来，每一步都可以停在原地继续用阶段 A 的能力：
   1. 先推 example 本身（如果上游还缺昇腾 example）。
   2. 再争取上游接受 notifier（上游 CI 完成后通知我们）——对上游来说只是一个几十行、不影响他们的小 workflow。
   3. 最理想是看护 workflow 直接合入上游、随上游 CI 触发，workflow 的成功、失败在上游社区可见。**这一步通常需要我们向上游提供 self-hosted runner 和容器镜像**。为上游社区接入 self-hosted runner 参考：<https://ascend-gha-runners.github.io/docs/user-manual-gha-zh/#github-app-runner_1>。注意这些 runner 全部位于中国大陆、无 docker hub 代理，这点需要和社区沟通。
   4. 兜底：上游什么都不接受。此时在 workflows 仓给该项目的流水线加 schedule 定时触发，把 target_repo 直接指向上游、target_ref 指向上游最新 main 或最新 release，实现「下游监听上游」。

## 零、前提

上游社区至少存在昇腾支持的一条 example。如果不存在，需要新增一条。example 需要展示软件的基本功能。

动手前先判断你负责的项目属于哪种情况：

- 上游已有健康的昇腾 CI（如 LlamaFactory、veRL、VeOmni）：先确认 example 是否已被该 CI 覆盖。若已覆盖，直接进入阶段 B（在上游补 example 看护），fork 靶场只用于开发调试。
- 上游有昇腾 CI 但停摆或不健康（如 DeepSpeed、ONNXRuntime、llama.cpp、SGLang）：进入阶段 A，先在靶场把 example 跑通。
- 上游无昇腾 CI（如 transformers、PEFT、diffusers）：按本文完整走。example 若不存在，先在 fork 里新增并调通，作为将来推向上游的内容。未来需要和上游社区沟通接入 self-hosted runner 的事宜。

## 一、Fork 上游仓用于 workflow 靶场

推荐将上游仓 fork 到 <https://github.com/organizations/cosdt-ci-test/> 组织下，该组织已接入昇腾 runner。在 workflows 仓 Settings → Actions → Runners 可以看到当前可用的 runner 及其标签。

## 二、workflow 要求

推荐在 <https://github.com/cosdt-ci-test/workflows> 下新增目标软件的 github workflow，workflow 的核心要求是：

- 对用昇腾能跑通的 examples，必须在合适的 runner 机器上跑通。
- 触发条件：CI 完成、example 变动、Release。
- 能感知新增的 example。
- Job 的结果能被外部机器通过 job api 或者读 Artifact 文件而获取。

以下是为了满足各个要求的详细步骤。

### 要求 1

1. 选择合适的 runner。在 workflows 仓 Settings → Actions → Runners 查看可用 runner。标签形如 linux-aarch64-a2-N，后缀 N 是该机器可用的昇腾卡数。按 example 需要的卡数选：例如 ms-swift 那条 example 用 tensor_model_parallel_size 2（2 卡张量并行），所以选 linux-aarch64-a2-2、npu_devices 用 "0,1"。选好后写进清单 supported 条目的 runner / npu_devices / timeout_minutes 字段，example 流水线按条目调度，不在 workflow 里硬编码。
2. 选择合适的镜像。注意，**cosdt-ci-test 下的 runners 全部位于中国大陆**，代理没有配置 docker hub，所以无法从 docker hub 拉取镜像。推荐从 Ascend 官方镜像仓拉取，在 <https://www.hiascend.com/developer/ascendhub> 中，选择合适的镜像版本，点下载时，即可看到 SWR 地址。
3. 配置安装依赖命令。注意，由于 runners 位于国内，安装时可能存在网络问题，所以需要配置镜像。昇腾包用镜像：<https://repo.huaweicloud.com/ascend/repos/pypi>。
4. 运行 example。有些 example 很大，例如训练加速的有些项目的 example 会真实训练模型，需要通过参数覆盖的方式控制 steps，减少 CI 资源占用。

### 要求 2

GitHub 的 workflow_run（一个 workflow 完成后触发另一个）不能跨仓库，所以「fork/上游发生了事通知 workflows 仓跑看护」的模式，只能用 repository_dispatch，也就是一方调用 GitHub API 向 workflows 仓发消息，消息带 event_type（事件名）和 client_payload（自定义数据）。

### 要求 3

机制是「清单 + 差集」：

1. 初始化清单。参考 scripts/，扫描目标仓 examples/ 下的 .sh / .py / .yaml（扫描根目录和扩展名记录在清单的 scan 段，检查脚本按它执行，可按项目调整），生成 examples_manifest.yaml。你确认能在 CI 跑通的 example 列入 supported——supported 条目必须写全 path、runner、npu_devices、overlay、timeout_minutes；其余全部自动写入 unsupported。
2. 每次 CI 比对。创建一个跑在免费的 ubuntu-latest 上的 job，这个 job 把目标仓磁盘上实际存在的 example 与清单求差集：新增的路径（磁盘有、清单无）和失效的路径（清单有、磁盘无）打印到日志和 job summary，然后正常退出，不让 job 失败。新 example 出现是「提示有待办」。

### 要求 4

外部获取有两条路径：github Job API 和 Artifact 文件。两条都要满足，各自对 workflow 写法有前提约定。

每个 example job 无论成败（if: always()）写一个 result.json 上传为 Artifact，字段如下：

```json
{
  "trigger": "workflow_dispatch | <项目>-ci-completed | <项目>-examples-changed | <项目>-release",
  "target_repo": "目标仓库（例如 cosdt-ci-test/ms-swift）",
  "target_ref": "被测的分支 / tag / sha",
  "path": "被测example路径，例如 examples/ascend/train/qwen3/qwen3_lora_megatron.sh",
  "image": "所用容器镜像",
  "job_status": "success | failure | cancelled"
}
```

## 契约

项目名称从 <https://docs.google.com/spreadsheets/d/1GtLB4Zvi_rzGqsH6dWShebk3DvRg6b9NNBeNeeDuNUM/edit?gid=753359062#gid=753359062> 表格的「项目名称」获取，全小写。

workflows 仓目录结构

```
workflows/
├── projects.yaml                     # 项目注册表
├── schemas/
│   └── result.schema.json            # result.json 的 JSON Schema
├── scripts/                          # 全项目共用脚本
├── templates/                        # 可复制的 workflow 骨架
├── .github/workflows/
│   ├── <project>-examples.yml        # example 看护流水线
│   └── <project>-quick-start.yml     # quick start 看护流水线
└── projects/
    └── <project>/                    # 项目专属数据
```
