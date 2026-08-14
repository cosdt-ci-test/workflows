# Artifact 输出规范

每个看护 job 无论成败都要留下机器可读的产物，让外部机器不必打开 GitHub UI 就能判断这次跑了什么、结果如何。设计依据见 [guarding-examples.md](guarding-examples.md) 的「要求 4」。

## 命名

所有 artifact 名称必须以 `<project>-` 开头。`<project>` 是 [projects.yaml](../projects.yaml) 里的 `name`，全小写。

| 种类 | 名称 | 上传时机 |
| --- | --- | --- |
| example 结果 | `<project>-examples-<run_id>-<job_index>` | 每个 `run-example` matrix job，`if: always()` |
| manifest 检查 | `<project>-manifest-check` | `manifest-check` job，`if: always()` |
| quick-start 结果 | `<project>-quick-start-<run_id>` | `monitor-and-test` job，`if: always()` |

`run_id` 是 GitHub Actions 的 `github.run_id`。`job_index` 是 matrix 的 `strategy.job-index`（从 0 起）。失败也必须上传，不能只在绿的时候留文件。

## 内容与 schema

### `<project>-examples-<run_id>-<job_index>`

单个文件 `result.json`，符合 [schemas/result.schema.json](../schemas/result.schema.json)。

artifact 刻意只装 result.json，不装运行日志和训练产物：上传越小，越不容易被 runner 到 GitHub 之间的网络波动打断（曾出现过 example 跑通、仅因上传 stall 而全线判红的 run）。完整训练输出在 Actions 页面 `Run example` 步骤的控制台日志里。上传失败自动重试一次，两次都失败才判红。清单里还没有 supported 条目时，`run-example` 和 `validate-results` 整体跳过，不产生此 artifact、也不判红。

`result.json` 必填字段：`trigger`、`target_repo`、`target_ref`、`path`、`image`、`job_status`。`job_status` 只能是 `success`、`failure` 或 `cancelled`。允许附加字段。

### `<project>-manifest-check`

单个文件 `manifest_check_result.json`，符合 [schemas/manifest_check_result.schema.json](../schemas/manifest_check_result.schema.json)。字段：`trigger`、`target_repo`、`target_ref`、`new_paths`、`stale_paths`、`supported`。`new_paths` / `stale_paths` 只记录、不使 job 失败。`supported` 条目的 `path`、`runner`、`npu_devices`、`overlay`、`timeout_minutes` 均为必填——example 流水线按这些字段调度 runner、卡和超时，缺字段会在 manifest-check 被 schema 校验拦下。

### `<project>-quick-start-<run_id>`

至少包含 `result.json`（同一份 [result.schema.json](../schemas/result.schema.json)）。quick-start 额外写 `tests_ran`（布尔）：监控无变化、未跑测试时为 `false`。还可以附带 unittest 日志、release API 响应等调试文件。

流水线里的 `validate-results` job 用 `check-jsonschema --schemafile schemas/result.schema.json` 校验下载到的每一份 `result.json`。缺文件或不合规即红。

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
