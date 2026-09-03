# upstream-doc-monitor 使用文档

监控所有在维护项目的**上游侧文档**变化：上游仓库内的文档（README/quickstart 等）与外部官方网页（如 ascend.github.io 文档页）。检测到内容变化或检测异常时，**在仓库内生成/更新工单 Issue 并 @ 处理人**——工单即待办，处理人核对上游变更、更新本项目看护文档后关闭工单。

- 独立 workflow：[.github/workflows/upstream-doc-monitor.yml](workflows/upstream-doc-monitor.yml)（与 quick-start 引擎完全解耦，不触发任何测试）
- 检测引擎：[scripts/upstream_doc_monitor.py](../scripts/upstream_doc_monitor.py)
- 监控清单：[.github/upstream-doc-monitor.yaml](upstream-doc-monitor.yaml)（唯一需要人工维护的文件）

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

## 2. 监控清单配置

`.github/upstream-doc-monitor.yaml`，**完全自包含**（不依赖 projects.yaml 解析，与项目注册表各自独立演进）。

```yaml
projects:
  - project: transformers            # 展示标签（工单标题与报告分组用）
    owner: zhangsan                  # 处理人 GitHub 用户名（建议必填）
    source:
      type: repo_file                # 采集源一：GitHub 仓库文档
      repo: huggingface/transformers # owner/repo 形式
      branch: main                   # 可选，默认上游默认分支
      path: README.md                # 仓库内文档路径（相对仓库根）

  - project: lm-eval-ascend-doc      # 采集源二：外部官方网页（标签可自拟）
    owner: lisi
    source:
      type: web_page
      url: https://ascend.github.io/docs/sources/lm_evaluation/quick_start.html
```

规则：

- 每个条目**有且仅有一个采集源**，`type` 二选一：`repo_file`（`repo` + `path` 必填，`branch` 可选）或 `web_page`（`url` 必填，合法 http(s)）
- `owner`：该条目产生的工单在**事件评论**中 **@owner**（对任何 GitHub 用户生效）；建票正文不含 @（不触发建票通知）。**缺省则评论也不 @人**——强烈建议每条目填写（不再自动设置 assignee）
- 校验失败（格式错误、未知 type、条目键 `<project>::<source>` 重复）→ job 直接失败并在日志指出错误条目

### 接入新监控项

1. 在 `projects:` 下加一个条目（注意 `owner` 填实际处理人）
2. 提交合入 main 即生效；**新增条目首轮记 `first_seen`（只登记基线、不建工单）**，第二轮起正常检测变化
3. 同一项目要监控第二个文档：再加一个条目（`project` 标签可重复、可配不同 `owner`）

### 指派/更换处理人

改条目的 `owner` 字段即可。已 open 的工单不受影响（处理人看后续评论的 @ 提及）；新事件的新评论 @ 新 owner。

## 3. 工单生命周期（处理人须知）

**每个监控文档一张工单**（标题固定 `[upstream-doc-monitor] <project> / <path-or-url>`），该文档的变化、异常、恢复都以评论追加到同一张工单的时间线：

| 事件 | 评论 | 频控 |
| --- | --- | --- |
| 文档变化 | `## 文档变化`（前后哈希、文档与提交历史链接、建议动作） | 每次变化都追加 |
| 检测异常 | `## 检测异常 (类型)`（doc_not_found / repo_error / fetch_error） | 同型异常持续**不重复评论**；错误类型变化才追加 |
| 异常恢复 | `## 异常恢复`（此前异常在本轮观测中恢复） | 恢复时评论一次，**不自动关闭** |

**处理闭环**：收到 @ → 点开评论里的文档/提交历史链接核对上游变更 → 更新本项目看护文档（如受影响）→ **关闭工单**。关闭后再有事件会新建新工单——open 工单即未处理事项。

不产生工单的情形：新增条目的首轮（first_seen）、无变化且无未关闭异常、本仓库侧运行故障（限流/配置错误——此时 job 标红，看 Actions 失败通知）。

## 4. 检测机制

- **纯哈希校验**：仓库文档用 GitHub Contents API 现成返回的 git blob SHA（每项 1 次轻量调用）；网页用 HTTP 条件请求（304 = 无变化，零正文）+ 响应体 SHA-256 兜底
- **基线**：各监控项的哈希、工单链接与频控状态存于 actions/cache（键前缀 `upstream-doc-monitor-state-`），跨轮持久
- **错误分级**：上游侧问题（文档 404 / 仓库异常 / 网页不可达）不中断其余监控项、job 保持绿色、走工单通知；本仓库侧问题（配置错误 / 基线损坏 / API 限流 / 整轮无法观测）中断执行、job 标红

## 5. 已知限制（v1）

- **SPA 页面**：哈希基于服务器响应体，JS 客户端渲染的内容不可见。官方文档站（SSG 静态输出，如 ascend.github.io）不受影响
- **哈希抖动**：页面若嵌时间戳/随机 token，内容没变哈希也会变（误报变化）。官方静态文档站实践上稳定；个别页面出现抖动时反馈维护者按条目归一化
- **反爬/需登录页面**：取不到内容会归为检测异常（工单可见，job 绿）

## 6. 常见操作

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
