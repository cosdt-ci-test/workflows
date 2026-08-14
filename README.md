# cosdt-ci-test/workflows

workflows的开发、测试仓。被看护的项目各自有一条 example 流水线和 一条 quick-start 流水线；项目专属的清单、overlay、fixture 放在 `projects/<项目名>/`。

当前只有 [ms-swift](projects/ms-swift/) 一个已接入项目，作为后续项目的样板。设计说明见 [docs/guarding-examples.md](docs/guarding-examples.md)。

## 目录

```
projects.yaml                        # 项目注册表
schemas/                             # result.json 等产物的 JSON Schema
scripts/                             # 全项目共用：初始化清单、CI 差集检查
templates/                           # 新项目可复制的 workflow 骨架
docs/guarding-examples.md            # 看护 Examples 设计文档
docs/artifacts.md                    # artifact 命名与读取约定
.github/workflows/<project>-examples.yml
.github/workflows/<project>-quick-start.yml
projects/<project>/                  # 该项目的清单、overlay、fixture、专用脚本
```

`projects.yaml` 里的 `name`、`category`（分类）、`support_level`（支持程度）取自[项目表](https://docs.google.com/spreadsheets/d/1GtLB4Zvi_rzGqsH6dWShebk3DvRg6b9NNBeNeeDuNUM/edit?gid=753359062#gid=753359062)的「项目名称」「分类」「支持程度」列。`name` 全小写；`category` 和 `support_level` 保留表里的中文值。

## 接入新项目

1. 读 [docs/guarding-examples.md](docs/guarding-examples.md)，判断该项目处于阶段 A 还是阶段 B。
2. 参考 [templates/project-examples.yml](templates/project-examples.yml)，完成 `.github/workflows/<project>-examples.yml`、`.github/workflows/<project>-quick-start.yml`。
3. 需要看护 Quick Start 文档时，请参考 [templates/project-quick-start.yml](templates/project-quick-start.yml)。
4. 在 `projects.yaml` 注册该项目。
5. 建 `projects/<project>/` 目录。用 `scripts/bootstrap_manifest.py --target-root <checkout> --output projects/<project>/examples_manifest.yaml --supported <path>` 生成清单，再手工补全 supported 条目的 runner / overlay / timeout。
6. 按 [docs/artifacts.md](docs/artifacts.md) 确认 artifact 名称和 `result.json` 字段。

`scripts/check_examples_manifest.py` 由 example 流水线的 `manifest-check` job 调用：对比目标仓磁盘与清单，差集只打印不失败，并把 supported 列表写成 matrix。`scripts/bootstrap_manifest.py` 只在接入时离线使用，CI 不调用。

## Artifact

每个 job 无论成败都上传产物。外部机器用 Jobs API 看 conclusion，或下载 artifact 读 `result.json`。字段与命名见 [docs/artifacts.md](docs/artifacts.md)，schema 在 [schemas/](schemas/)。
