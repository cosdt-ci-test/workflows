# upstream-doc-monitor 使用文档

监控所有在维护项目的**上游侧文档**变化：上游仓库内的文档（README/quickstart 等）与外部官方网页（如 ascend.github.io 文档页）。检测到内容变化或检测异常时，**在仓库内生成/更新工单 Issue 并 @ 处理人**——工单即待办，处理人核对上游变更、更新本项目看护文档后关闭工单。

- 独立 workflow：[.github/workflows/upstream-doc-monitor.yml](../.github/workflows/upstream-doc-monitor.yml)（与 quick-start 引擎完全解耦，不触发任何测试）
- 检测引擎：[scripts/upstream_doc_monitor.py](../scripts/upstream_doc_monitor.py)
- 监控清单：[.github/upstream-doc-monitor.yaml](../.github/upstream-doc-monitor.yaml)（唯一需要人工维护的文件）

## 1. 运行方式

| 触发 | 说明 |
| --- | --- |
| schedule | cron `0 */6 * * *`（UTC 0/6/12/18 点 = 香港 8/14/20/2 点），**仅对默认分支（main）上的 workflow 定义生效** |
| workflow_dispatch | 手动即时触发，Actions 页 → upstream-doc-monitor → Run workflow |

单轮全量检测约 1 分钟（43 个监控项 × 每项 1 次轻量 API 调用 + 网页条件请求）。

### 运行结果在哪看

1. **Issues**（交付面）：标题前缀 `[upstream-doc-monitor]` 的工单，处理人被 @ 提及
2. **Step Summary**：Actions → 对应 run → Summary 页，有人读的变化表与异常表
3. **job 日志**：逐监控项的检测明细（`[i/N] 键: 状态`）
4. **result artifact**（前端消费面）：artifact 名 `upstream-doc-monitor-result-<run_id>`，内容为 `result.json`（顶层裸数组、每项 5 字段），供前端/后端服务机器读取，详见 [§5 前端对接](#5-前端对接result-artifact)

## 2. 监控清单配置

`.github/upstream-doc-monitor.yaml`，**完全自包含**（不依赖 projects.yaml 解析，与项目注册表各自独立演进）。

```yaml
owners:                              # 项目负责人名单（条目 owner 从这里选）
  - zhangsan
  - lisi
maintainer: wangwu                   # 本 workflow 管理人（可不在 owners 中）
projects:
  - project: transformers            # 展示标签（工单标题与报告分组用）
    owner: zhangsan                  # 处理人 GitHub 用户名（建议必填）
    url: https://github.com/huggingface/transformers/blob/main/README.md

  - project: lm-eval-ascend-doc      # 外部官方网页（标签可自拟）
    owner: lisi
    url: https://ascend.github.io/docs/sources/lm_evaluation/quick_start.html
```

规则：

- 每个条目一个 `url`，**源类型自动识别**：`https://github.com/<owner>/<repo>/blob/<branch>/<path>` 链接 → 仓库文档（branch 填该仓库默认分支名）；其他 http(s) 地址 → 外部网页。GitHub 链接必须指到单个文件（blob 链接），仓库首页等链接会校验失败
- `owner`：该条目的工单在**建票正文顶部（一次）与事件评论**中 @owner；**取值必须命中 `owners ∪ {maintainer}` 校验集合**（项目负责人名单 + 管理人），否则记 config_error（检测照常，项目工单内 @ maintainer 提示修复）
- `owners`：项目负责人名单（必填）
- `maintainer`：管理人（必填，owner 配置问题的通知对象）
- 校验失败（URL 非法、github.com 链接未指到单文件、条目键 `<project>::<文档路径或 URL>` 重复）→ job 直接失败并在日志指出错误条目

### 接入新监控项

1. 在 `projects:` 下加一个条目（注意 `owner` 填实际处理人）
2. 提交合入 main 即生效；**新增条目首轮记 `first_seen`（只登记基线、不建工单）**，第二轮起正常检测变化
3. 同一项目要监控第二个文档：再加一个条目（`project` 标签可重复、可配不同 `owner`）

### 指派/更换处理人

改条目的 `owner` 字段即可。已 open 的工单不受影响（处理人看后续评论的 @ 提及）；新事件的新评论 @ 新 owner。

### 字段填错如何定位

| 错误 | 表现 | 处理 |
| --- | --- | --- |
| 语法错误（url 非法 / owner 格式错 / 缺 `owners` / `maintainer`） | job 直接失败，日志含条目索引与字段名（如 `projects[3] (deepspeed): 'url' is required`） | 按提示改配置 |
| url 值拼错（仓库 / 分支 / 路径任一段） | 该条目记检测异常，message 含完整 url 与核对指引；工单「检测异常」段落可见 | 核对 url，上游确实变更则更新配置 |
| owner 不在 `owners ∪ {maintainer}` | 该条目检测照常，项目工单追加「## 配置错误」段落并 @ maintainer（每型仅首轮评论，修复后自动恢复） | 修正 owner 或加入 `owners` 名单 |

## 3. 工单生命周期（处理人须知）

**每个监控文档一张工单**（标题固定 `[upstream-doc-monitor] <project> / <path-or-url>`），记录该文档的变化、异常、恢复时间线。**新建工单时，本轮的全部事件段落直接并入工单正文**（正文顶部 @ 处理人一次），不再对首事件单独发评论——建票即该轮唯一通知；工单 open 后的新事件才以评论追加（每条评论 @ 处理人）：

| 事件 | 段落 | 频控 |
| --- | --- | --- |
| 文档变化 | `## 文档变化`（前后哈希、文档与提交历史链接、建议动作） | 每次变化都记录 |
| 检测异常 | `## 检测异常 (类型)`（doc_not_found / repo_error / fetch_error） | 同型异常持续**不重复评论**；错误类型变化才追加 |
| 异常恢复 | `## 异常恢复`（此前异常在本轮观测中恢复） | 恢复时记录一次，**不自动关闭** |
| 配置错误 | `## 配置错误`（owner 不在 `owners ∪ {maintainer}`，@ maintainer） | 未修复持续不重复评论；修复后走异常恢复 |

**处理闭环**：收到 @ → 点开正文/评论里的文档与提交历史链接核对上游变更 → 更新本项目看护文档（如受影响）→ **关闭工单**。关闭后再有事件会新建新工单（事件同样并入正文）——open 工单即未处理事项。

不产生工单的情形：新增条目的首轮（first_seen）、无变化且无未关闭异常、本仓库侧运行故障（限流/配置错误——此时 job 标红，看 Actions 失败通知）。

## 4. 检测机制

- **纯哈希校验**：仓库文档用 GitHub Contents API 现成返回的 git blob SHA（每项 1 次轻量调用）；网页用 HTTP 条件请求（304 = 无变化，零正文）+ 响应体 SHA-256 兜底
- **基线**：各监控项的哈希、工单链接与频控状态存于 actions/cache（键前缀 `upstream-doc-monitor-state-`），跨轮持久
- **错误分级**：上游侧问题（文档 404 / 仓库异常 / 网页不可达）不中断其余监控项、job 保持绿色、走工单通知；本仓库侧问题（配置错误 / 基线损坏 / API 限流 / 整轮无法观测）中断执行、job 标红

## 5. 前端对接（result artifact）

三个交付面并存：工单 Issue 面向处理人（通知）、Step Summary 面向运维（人读）、**result artifact 面向前端（机器消费）**。本节是前端/后端服务的对接契约。

### 产出物与时机

- artifact 名：`upstream-doc-monitor-result-<run_id>`（`<run_id>` = `github.run_id`）
- 内容：单个 `result.json`，符合 [schemas/upstream_doc_monitor_result.schema.json](../schemas/upstream_doc_monitor_result.schema.json)
- **每轮必产出**：`Upload result artifact` step 带 `if: always()`——检测失败、限流中断的轮次同样上传（此时条目多为 `error`），`if-no-files-found: warn`
- 上传前用 `check-jsonschema` 校验，不合规即 job 红
- 命名例外：本 workflow 是**全局 workflow**（不对应 projects.yaml 里的单一 project），故以 workflow 名 `upstream-doc-monitor-` 作前缀，见 [docs/artifacts.md](artifacts.md)

### 后端拉取（持 GitHub token）

```bash
# 1. 列最新一次成功的 run，取 workflow_runs[0].id
#    也可按 workflow 名过滤：&workflow=upstream-doc-monitor.yml
gh api "repos/cosdt-ci-test/workflows/actions/runs?branch=main&status=success&per_page=1"

# 2. 取该 run 的 artifacts，找 name 为 upstream-doc-monitor-result-{run_id} 的项
#    同一响应里的 created_at 即数据新鲜度
gh api repos/cosdt-ci-test/workflows/actions/runs/{run_id}/artifacts

# 3. 下载并解包，得到单个 result.json
gh api repos/cosdt-ci-test/workflows/actions/runs/{run_id}/artifacts/{artifact_id}/zip > result.zip
#    或用 gh CLI 一步下载解包：
gh run download {run_id} --repo cosdt-ci-test/workflows --name upstream-doc-monitor-result-{run_id}
```

### 数据形态

顶层是**裸数组**（不是对象、无 summary 包裹），一份**全量快照**：每轮包含监控清单的**全部**条目，本轮无变化的也在内。每项恰好 5 个字段：

| 字段 | 含义 |
| --- | --- |
| `project` | 项目名（监控清单的 project 标签） |
| `doc` | 文档标识：仓库文档（repo_file）为仓库内路径（如 `README.md`），外部网页（web_page）为完整 URL。用于消歧——同一 project 标签可配多个文档条目 |
| `version` | 上游版本号，三级降级解析：`/releases/latest` → `/releases?per_page=20` 取最新非 draft → `/tags?per_page=1`；均无则 `null`。web_page 条目无关联仓库，恒为 `null`；版本查询失败降级为 `null`，不影响监控 |
| `status` | 综合状态，三值枚举，判定优先级 **error > pending > unchanged**（见下表） |
| `ticket` | 该监控项当前未关闭工单的 html url，无则 `null`。`status=pending` 时必非 `null`；`unchanged` 时必为 `null`；`error` 时可有可无 |

`status` 三值：

| 值 | 判定条件 | 中文显示 | 处置 |
| --- | --- | --- | --- |
| `error` | 本轮检测失败：文档 404 / 仓库不可达 / 网页抓取失败 / 限流未检测 | 检测异常 | 排查上游或配置 |
| `pending` | 本轮无异常，但该监控项存在未关闭工单（含往轮变化遗留与配置错误工单） | 已变化待确认 | 核对变化 → 更新看护文档 → 关闭工单 |
| `unchanged` | 本轮无异常且无未关闭工单（含新增条目首轮登记基线） | 未变化 | 无需处理 |

### 消费语义

- **整表替换**：拉最新一份 artifact 覆盖本地视图即可——无需增量合并、无需维护 diff 状态；漏拉几轮可自愈（下一份仍是完整快照）
- **`pending` 是持续状态**：工单未被人工关闭则跨轮保持 `pending`（即使本轮无新变化）；工单关闭后下一轮自动回到 `unchanged`
- **待确认数**由前端统计 `status == "pending"` 的项数得出，产物不提供 summary 计数
- **新鲜度与 run 标识从 artifact 元数据取**：产物内没有 `generated_at` / `run_id` / `schema_version` 等运行元信息，读 Artifacts API 返回的 `created_at` 与 artifact 名称中的 `run_id`

### 契约演进

新增字段向后兼容，前端应忽略未知字段；破坏性变更会同步更新 schema 文件与本节。

> 游离工单（监控条目已从配置删除、但工单仍未关闭）**不出现在产物中**，仅在 job 日志提示，由运维排查清理。

### 与内部报告 report.json 的关系

同一轮另产出内部详细报告 `report.json`（`schema_version: 1`，含 changes / errors 双数组与 owner、前后哈希、工单动作、错误消息等细节），驱动 Step Summary 渲染与日志审计，**不上传 artifact**。两者并存：`result.json` 是对外前端契约（极简、稳定），`report.json` 是内部审计视图（可随实现调整）。

## 6. 已知限制（v1）

- **SPA 页面**：哈希基于服务器响应体，JS 客户端渲染的内容不可见。官方文档站（SSG 静态输出，如 ascend.github.io）不受影响
- **哈希抖动**：页面若嵌时间戳/随机 token，内容没变哈希也会变（误报变化）。官方静态文档站实践上稳定；个别页面出现抖动时反馈维护者按条目归一化
- **反爬/需登录页面**：取不到内容会归为检测异常（工单可见，job 绿）

## 7. 常见操作

```bash
# 本地全链路 dry-run（不建工单：缺省 --repo 时工单同步跳过）
GH_TOKEN=$(gh auth token) python scripts/upstream_doc_monitor.py \
  --config .github/upstream-doc-monitor.yaml \
  --state /tmp/udm-state.json \
  --output-dir /tmp/udm-report

# 指定工单目标仓库（会真实创建/评论 Issue，谨慎使用）
... --repo cosdt-ci-test/workflows

# 校验 workflow 语法
actionlint .github/workflows/upstream-doc-monitor.yml
```

手动重置某条目的基线（让它重新走 first_seen）：编辑缓存中的状态文件不可行（actions/cache 只读），最简单的方式是**改一下条目的 project 标签**（键变化 → 旧键作废 → 新键首轮 first_seen）。
