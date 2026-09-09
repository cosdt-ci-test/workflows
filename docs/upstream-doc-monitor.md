# upstream-doc-monitor 使用文档

监控所有在维护项目的**上游侧文档**变化：上游仓库内的文档（README/quickstart 等）与外部官方网页（如 ascend.github.io 文档页）。检测到内容变化或检测异常时，**在仓库内生成/更新工单 Issue 并 @ 处理人**——工单即待办，处理人核对上游变更、更新本项目看护文档后关闭工单。

- 独立 workflow：[.github/workflows/upstream-doc-monitor.yml](../.github/workflows/upstream-doc-monitor.yml)
- 检测引擎：[scripts/upstream_doc_monitor.py](../scripts/upstream_doc_monitor.py)
- 监控清单：[.github/upstream-doc-monitor.yaml](../.github/upstream-doc-monitor.yaml)（唯一需要人工维护的文件）

## 1. 运行方式

| 触发 | 说明 |
| --- | --- |
| schedule | cron `0 */6 * * *`（UTC 0/6/12/18 点 = 香港 8/14/20/2 点），**仅对默认分支（main）上的 workflow 定义生效** |
| workflow_dispatch | 手动即时触发，Actions 页 → upstream-doc-monitor → Run workflow |

单轮全量检测约 1 分钟（每项 1 次轻量 API 调用 + 网页条件请求）。

**运行结果在哪看**：Issues（交付面，标题形如 `\[项目\] 事件类型 — 文档` 的工单）→ 处理人；Step Summary（Actions 对应 run → Summary 页）→ 运维；job 日志→ 逐项检测明细；result artifact（前端消费面）→ 机器读取，见 §5。

## 2. 监控清单配置

`.github/upstream-doc-monitor.yaml`，**完全自包含**（不依赖 projects.yaml 解析）。

```yaml
owners:                              # 项目负责人名单（owner 从这里选）
  - zhangsan
  - lisi
maintainer: wangwu                   # 本 workflow 管理人（可不在 owners 中）
projects:
  - project: transformers            # 展示标签（工单标题与报告分组用）
    owner: zhangsan                  # 处理人 GitHub 用户名
    url: https://github.com/huggingface/transformers/blob/main/README.md

  - project: lm-eval-ascend-doc      # 外部官方网页（标签可自拟）
    owner: lisi
    url: https://ascend.github.io/docs/sources/lm_evaluation/quick_start.html
```

**规则**：每个条目一个 `url`，**源类型自动识别**——`https://github.com/<owner>/<repo>/blob/<branch>/<path>` 链接→仓库文档（branch 填该仓库默认分支名）；其他 http(s) 地址→外部网页。GitHub 链接必须指到单个文件 blob 链接，仓库首页等链接会校验失败。`owner` 取值必须命中 `owners ∪ {maintainer}` 校验集合，否则记 config_error（检测照常，项目工单内 @ maintainer 提示修复）。`owners` 和 `maintainer` 均为必填。

**接入新监控项**：在 `projects:` 下加条目，提交合入 main 即生效。新增条目首轮记 `first_seen`（只登记基线、不建工单），第二轮起正常检测变化。同一项目可加多个条目（`project` 标签可重复、可配不同 `owner`）。

**更换处理人**：改条目的 `owner` 字段即可。已 open 工单不受影响，新事件的新评论 @ 新 owner。

**字段填错如何定位**：语法错误（url 非法 / owner 格式错 / 缺 `owners` / `maintainer`）→ job 直接失败，日志含条目索引与字段名；url 拼错→该条目记检测异常，工单可见；owner 不在校验集合→检测照常，工单追加「## 配置错误」段落。

## 3. 工单生命周期（处理人须知）

**每个监控文档一张工单**（标题固定 `\[<project>\] <事件类型> — <文档>`（事件类型取建票时首事件，后续不更新）），记录该文档的变化、异常、恢复时间线。新建工单时本轮全部事件段落直接并入正文（正文顶部 @ 处理人一次），工单 open 后的新事件以评论追加（每条评论 @ 处理人）：

| 事件 | 段落 | 频控 |
| --- | --- | --- |
| 文档变化 | `## 文档变化`（前后哈希、文档与提交历史链接、建议动作） | 每次变化都记录 |
| 检测异常 | `## 检测异常 (类型)`（doc_not_found / repo_error / fetch_error） | 同型异常持续**不重复评论**；错误类型变化才追加 |
| 异常恢复 | `## 异常恢复`（此前异常在本轮观测中恢复） | 恢复时记录一次，**不自动关闭** |
| 配置错误 | `## 配置错误`（owner 不在 `owners ∪ {maintainer}`，@ maintainer） | 未修复持续不重复评论；修复后走异常恢复 |

**处理闭环**：收到 @ → 点开正文/评论里的文档与提交历史链接核对上游变更 → 更新本项目看护文档（如受影响）→ **关闭工单**。关闭后再有事件会新建新工单。open 工单即未处理事项。

不产生工单的情形：新增条目的首轮（first_seen）、无变化且无未关闭异常、本仓库侧运行故障（限流/配置错误——此时 job 标红，看 Actions 失败通知）。

## 4. 检测机制

**纯哈希校验**：仓库文档用 GitHub Contents API 返回的 git blob SHA（每项 1 次轻量调用）；网页用 HTTP 条件请求（304 = 无变化，零正文）+ 响应体 SHA-256 兜底。**基线**：各监控项的哈希、工单链接与频控状态存于 actions/cache（键前缀 `upstream-doc-monitor-state-`），跨轮持久。**错误分级**：上游侧问题（文档 404 / 仓库异常 / 网页不可达）不中断其余监控项、job 保持绿色、走工单通知；本仓库侧问题（配置错误 / 基线损坏 / API 限流 / 整轮无法观测）中断执行、job 标红。

## 5. 前端对接（result artifact）

三个交付面并存：工单 Issue 面向处理人、Step Summary 面向运维、**result artifact 面向前端（机器消费）**。本节是前端/后端服务的对接契约。

**产出物与时机**：artifact 名 `upstream-doc-monitor-result-<run_id>`（`<run_id>` = `github.run_id`），内容为单个 `result.json`，符合 [schemas/upstream_doc_monitor_result.schema.json](../schemas/upstream_doc_monitor_result.schema.json)。**每轮必产出**（`if: always()`），检测失败或限流中断的轮次同样上传（此时条目多为 `error`），`if-no-files-found: error`。上传前用 `check-jsonschema` 校验，不合规即 job 红。

**数据形态**：顶层是**裸数组**（不是对象、无 summary 包裹），一份**全量快照**——每轮包含监控清单的**全部**条目，本轮无变化的也在内。每项恰好 5 个字段：

| 字段 | 含义 |
| --- | --- |
| `project` | 项目名（监控清单的 project 标签） |
| `doc` | 文档完整 URL，与配置 url 一致：仓库文档为 GitHub blob 链接，外部网页为网页 URL |
| `version` | 上游版本号，三级降级解析：`/releases/latest` → `/releases?per_page=20` 取最新非 draft → `/tags?per_page=1`；均无则 `null`。web_page 条目恒为 `null` |
| `status` | 综合状态，三值枚举，判定优先级 **error > pending > unchanged** |
| `ticket` | 该监控项当前未关闭工单的 html url，无则 `null`。`status=pending` 时必非 `null`；`unchanged` 时必为 `null` |

`status` 三值：`error` = 检测异常（本轮检测失败：文档 404 / 仓库不可达 / 网页抓取失败 / 限流未检测）；`pending` = 已变化待确认（本轮无异常，但该监控项存在未关闭工单）；`unchanged` = 未变化（本轮无异常且无未关闭工单，含新增条目首轮登记基线）。

**消费语义**：**整表替换**——拉最新一份 artifact 覆盖本地视图即可，无需增量合并，漏拉几轮可自愈。`pending` 是持续状态——工单未被人工关闭则跨轮保持 `pending`（即使本轮无新变化），工单关闭后下一轮自动回到 `unchanged`。待确认数由前端统计 `status == "pending"` 的项数得出，产物不提供 summary 计数。**新鲜度与 run 标识从 artifact 元数据取**——产物内没有 `generated_at` / `run_id` / `schema_version` 等运行元信息，读 Artifacts API 返回的 `created_at` 与 artifact 名称中的 `run_id`。

**后端拉取（持 GitHub token）**：

```bash
# 1. 列最新一次成功的 run，取 workflow_runs[0].id
gh api "repos/cosdt-ci-test/workflows/actions/runs?branch=main&status=success&per_page=1"

# 2. 取该 run 的 artifacts，找 name 为 upstream-doc-monitor-result-{run_id} 的项
gh api repos/cosdt-ci-test/workflows/actions/runs/{run_id}/artifacts

# 3. 下载并解包，得到单个 result.json
gh run download {run_id} --repo cosdt-ci-test/workflows --name upstream-doc-monitor-result-{run_id}
```

**契约演进**：新增字段向后兼容，前端应忽略未知字段；破坏性变更会同步更新 schema 文件与本节。

**游离工单**（监控条目已从配置删除、但工单仍未关闭）**不出现在产物中**，仅在 job 日志提示，由运维排查清理。

**与内部报告 report.json 的关系**：同一轮另产出 `report.json`（`schema_version: 1`，含 changes / errors 双数组与 owner、前后哈希、工单动作、错误消息等细节），驱动 Step Summary 渲染与日志审计，**不上传 artifact**。两者并存：`result.json` 是对外前端契约（极简、稳定），`report.json` 是内部审计视图（可随实现调整）。

更多用法见 [检测引擎](../scripts/upstream_doc_monitor.py) 与 [result schema](../schemas/upstream_doc_monitor_result.schema.json)。

