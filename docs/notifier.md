# 在 fork 仓部署 notifier

本文是 [guarding-examples.md](guarding-examples.md)「要求 2」的落地步骤：让 fork 仓在「CI 完成 / examples 变动 / 打 release tag」时通知本仓（workflows 仓）跑 example 看护流水线。接入新项目时照本文在该项目的 fork 仓部署，不需要读过其他任何上下文。

## 为什么需要 notifier

GitHub 的 `workflow_run` 触发器（一个 workflow 完成后触发另一个）**不能跨仓库**。所以「fork 仓发生了事、workflows 仓跑看护」只能反过来做：fork 仓上放两个很小的 workflow（下称 notifier），在事件发生时调 GitHub API 向 workflows 仓发 `repository_dispatch`（带自定义事件名和数据的跨仓消息）。workflows 仓的 `<project>-examples.yml` 监听这些事件名。

## 事件与 payload 契约

事件名（`event_type`）必须与 `.github/workflows/<project>-examples.yml` 里 `repository_dispatch.types` 列的三个名字完全一致。`client_payload` 的必填字段：

| event_type | 触发时机 | client_payload 必填字段 |
| --- | --- | --- |
| `<project>-ci-completed` | fork 上游 CI（NPU CI）成功完成 | `repo`（fork 全名）、`sha`（被测 commit） |
| `<project>-examples-changed` | `examples/**` 被 push 到 fork 的 main | `repo`、`sha` |
| `<project>-release` | fork 上推了 `v**` 标签 | `repo`、`ref`（tag 名） |

接收端（examples workflow 的 `resolve` job）的解析规则：`ref` 优先于 `sha` 作为被测 ref。所以 release 事件可以同时带 `ref` 和 `sha`（会用 tag），而 ci-completed / examples-changed 只带 `sha` 即可。两者都缺时 `resolve` 直接判红。

## 准备 PAT（EXAMPLE_GUARD_PAT）

向 workflows 仓发 `repository_dispatch` 的 API（`POST /repos/<owner>/workflows/dispatches`）要求调用方对 workflows 仓有写权限，而 fork 仓 workflow 自带的 `GITHUB_TOKEN` 只对 fork 仓自身有效，所以需要一个 PAT（Personal Access Token，个人访问令牌）：

1. 用对 workflows 仓有写权限的账号，在 GitHub Settings → Developer settings → Personal access tokens 创建 **fine-grained PAT**：Resource owner 选 workflows 仓所在组织（如 `cosdt-ci-test`），Repository access 只勾 workflows 仓，Permissions 里 Contents 给 Read and write。（classic PAT 的 `repo` scope 也可用，但授权面更大，不推荐。）
2. 在 **fork 仓**的 Settings → Secrets and variables → Actions → New repository secret，名字填 `EXAMPLE_GUARD_PAT`，值填上一步的 token。
3. token 有效期到期后 notifier 会开始报 401/403，需要换新并更新 secret。

## 可复制的 notifier

以下两个文件放到 fork 仓的 `.github/workflows/` 下。替换所有 `<project>`（项目名，全小写）、`<fork_repo>`（fork 全名，如 `cosdt-ci-test/ms-swift`）、`<guard_repo>`（workflows 仓全名，如 `cosdt-ci-test/workflows`）、`<upstream-ci-workflow>`（fork 上被监听的 CI workflow 的 `name`，如 ms-swift 的 `citest-npu`）。已部署的真实例子见 `cosdt-ci-test/ms-swift` 仓的 `notify-example-guard.yml` 和 `notify-example-guard-release.yml`。

`notify-example-guard.yml`（CI 完成 + examples 变动）：

```yaml
name: notify-example-guard

on:
  workflow_run:
    workflows: ["<upstream-ci-workflow>"]
    types: [completed]
  push:
    branches:
      - main
    paths:
      - 'examples/**'

jobs:
  notify:
    if: ${{ github.event_name == 'push' || github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    steps:
      - name: Dispatch to example guard
        env:
          TOKEN: ${{ secrets.EXAMPLE_GUARD_PAT }}
          EVENT_NAME: ${{ github.event_name }}
          PUSH_SHA: ${{ github.sha }}
          WR_SHA: ${{ github.event.workflow_run.head_sha }}
        run: |
          set -euo pipefail
          if [ -z "${TOKEN:-}" ]; then
            echo "EXAMPLE_GUARD_PAT is not configured"
            exit 1
          fi
          if [ "$EVENT_NAME" = "push" ]; then
            event_type="<project>-examples-changed"
            sha="$PUSH_SHA"
          else
            event_type="<project>-ci-completed"
            sha="$WR_SHA"
          fi
          payload="$(jq -n --arg event_type "$event_type" --arg sha "$sha" \
            '{event_type: $event_type,
              client_payload: {repo: "<fork_repo>", sha: $sha}}')"
          curl -sS --fail-with-body -X POST \
            -H "Authorization: Bearer ${TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            https://api.github.com/repos/<guard_repo>/dispatches \
            -d "$payload"
          echo "dispatched $event_type for $sha"
```

`notify-example-guard-release.yml`（release tag）：

```yaml
name: notify-example-guard-release

on:
  push:
    tags:
      - 'v**'

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Dispatch <project>-release
        env:
          TOKEN: ${{ secrets.EXAMPLE_GUARD_PAT }}
          TAG: ${{ github.ref_name }}
          SHA: ${{ github.sha }}
        run: |
          set -euo pipefail
          if [ -z "${TOKEN:-}" ]; then
            echo "EXAMPLE_GUARD_PAT is not configured"
            exit 1
          fi
          payload="$(jq -n --arg tag "$TAG" --arg sha "$SHA" \
            '{event_type: "<project>-release",
              client_payload: {repo: "<fork_repo>", ref: $tag, sha: $sha}}')"
          curl -sS --fail-with-body -X POST \
            -H "Authorization: Bearer ${TOKEN}" \
            -H "Accept: application/vnd.github+json" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            https://api.github.com/repos/<guard_repo>/dispatches \
            -d "$payload"
          echo "dispatched <project>-release for $TAG ($SHA)"
```

## 手动验证

不必等真实事件，本地用 PAT 直接发一条测试消息（把占位符换成真实值）：

```bash
curl -sS --fail-with-body -X POST \
  -H "Authorization: Bearer <PAT>" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/<guard_repo>/dispatches \
  -d '{"event_type":"<project>-examples-changed","client_payload":{"repo":"<fork_repo>","sha":"<40位commit sha>"}}'
```

API 成功返回 204（无响应体）。然后到 workflows 仓的 Actions 页面确认 `<project>-examples` 被触发，且 run 里 `resolve` job 打印的 `trigger` / `target_repo` / `target_ref` 与你发的一致。
