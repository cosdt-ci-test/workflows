# Artifact 输出规范

每个看护 job 无论成败都要留下机器可读的产物，让外部机器不必打开 GitHub UI 就能判断这次跑了什么、结果如何。设计依据见 [guarding-examples.md](guarding-examples.md) 的「要求 4」。

## 命名

所有 artifact 名称必须以 `<project>-` 开头。`<project>` 是 [projects.yaml](../projects.yaml) 里的 `name`，全小写。

| 种类 | 名称 | 上传时机 |
| --- | --- | --- |
| example 结果 | `<project>-examples-<run_id>-<job_index>` | `publish-results` 矩阵下的每个 `publish-result` job，`if: always()` |
| example 汇总 | `<project>-examples-result-<timestamp>` | 托管 runner 上的 `summarize-results`，包括定时命中缓存的运行 |
| quick-start 结果 | `<project>-quick-start-<run_id>`，一轮测多棵树时用 `<project>-quick-start-<run_id>-<job_index>` | 托管 runner 上的 `publish-result`（或仍在同一 job 里写结果的旧流水线）。NPU 上的测试 job 不上传。无变化时整个 run 在 monitor 后结束，不产生此 artifact |

`run_id` 是 GitHub Actions 的 `github.run_id`。`job_index` 是 matrix 的 `strategy.job-index`（从 0 起）。失败也必须上传，不能只在绿的时候留文件。

## 内容与 schema

### `<project>-examples-<run_id>-<job_index>`

单个文件 `result.json`，符合 [schemas/result.schema.json](../schemas/result.schema.json)。

GitHub 托管 runner 上的 `publish-result` job 通过 Jobs API 读取对应 `run-example` 的 conclusion，生成并上传 `result.json`。完整训练输出在 Actions 页面的 `Run example` 步骤日志中。清单中没有 `supported` 条目，或定时看护复用缓存时，只发布汇总 artifact。

`result.json` 必填字段：`trigger`、`target_repo`、`target_ref`、`path`、`image`、`job_status`。`job_status` 只能是 `success`、`failure` 或 `cancelled`。允许附加字段。

### `<project>-examples-result-<timestamp>`

`<timestamp>` 是生成汇总时的 UTC−8 时间，格式为 `YYYYMMDDHHMM`，例如 `tensordict-examples-result-202611250146`。

单个文件 `result.json`，包含 `project`、`trigger`、`target_ref`、`reason`、`machines` 和 `workflow_status`。看板按 `machines.a2`、`machines.a3` 读取各自的 `total`、`success`、`failed`。每个计划内的条目只有拿到有效的成功结果才计入 `success`，其余计入 `failed`。没有配置 A3 时 A3 三项均为零。无法确定执行计划时 `machines` 为 `null`。

定时看护复用上次成功的统计时，`reason` 为 `cached`，本轮只发布汇总。确认为同一版本和文件哈希但没有有效统计时，`reason` 为 `cache-miss`，本轮实跑并发布明细。`workflow_status` 根据汇总阶段可见的任务状态和统计判定，失败和取消统一记为 `failure`。GitHub 最终运行状态仍可从 Actions API 获取。

### `<project>-quick-start-<run_id>` 或 `<project>-quick-start-<run_id>-<job_index>`

至少包含 `result.json`（同一份 [result.schema.json](../schemas/result.schema.json)）。还可以附带 unittest 日志、release API 响应等调试文件。一轮只测一棵树时用前者。一轮测多棵树时用后者，避免两个 job 重名。

llama.cpp 的 quick-start 结果由跑在 GitHub 托管 runner 上的 `publish-result` 生成，不由 NPU runner 上传。它通过 Jobs API 读取对应 `run-quick-start` 的 conclusion 再写文件。`schedule` 触发但 monitor 未检测到变化时，后面的 job 跳过，不产生 artifact。

example 流水线里，每个 `publish-result` job 在上传前用 `check-jsonschema --schemafile schemas/result.schema.json` 校验自己生成的 `result.json`；quick-start 流水线同样在上传前校验。缺文件或不合规即红。

## 外部机器如何读取

两条路径都要能用。把 `{run_id}` 换成一次 Actions run 的数字 id。

### Jobs API

看每个 job 的 `conclusion`（success / failure / cancelled）：

```bash
gh api repos/cosdt-ci-test/workflows/actions/runs/{run_id}/jobs
```

### Artifacts API

列出这次 run 的 artifact，再下载 zip，从中读 `result.json`：

```bash
gh api repos/cosdt-ci-test/workflows/actions/runs/{run_id}/artifacts
gh run download {run_id} --repo cosdt-ci-test/workflows --name <project>-examples-{run_id}-0
```

下载后打开 `result.json`，用 `job_status` 判断该 example 是否跑通，用 `path` / `target_ref` 判断测的是哪条、哪个提交。
