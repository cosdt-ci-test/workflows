# torchtitan

[`pytorch/torchtitan`](https://github.com/pytorch/torchtitan) — PyTorch native
platform for large-scale generative-AI pretraining (Llama 3.x, Llama 4,
DeepSeek-V3, Flux, GPT-OSS, …). torchtitan 在 NPU 上的最小验证场景见
[`docs/Quick-start-Ascend.md`](docs/Quick-start-Ascend.md)；端到端 CI 跑在
[`.github/workflows/torchtitan-quick-start.yml`](../../.github/workflows/torchtitan-quick-start.yml)。

## 项目元信息

| 字段 | 值 |
| --- | --- |
| `category` | 训练加速 |
| `support_level` | 新兴适配 |
| `upstream_repo` | `pytorch/torchtitan` |
| `phase` | A |
| `runner` | `linux-aarch64-a2-1` |
| `workflow` | `.github/workflows/torchtitan-quick-start.yml` |

## 接入特殊性

torchtitan 截至 `v0.2.2` 所有 release 都标了 `prerelease: true`，GitHub 的
`/releases/latest` 端点会直接 404。本项目的 workflow 在
`.github/workflows/quick-start-template.yml` 的 `include_prerelease` 输入
上置 `true`，让引擎走 `/releases?per_page=20` + `sort_by(.published_at)
| last` 取最新非 draft tag。如果未来出了 stable release，记得在
`.github/workflows/torchtitan-quick-start.yml` 里把 `include_prerelease`
改回 `false` / 删掉，恢复其他项目一致的 `/releases/latest` 路径。

## 目录

```
projects/torchtitan/
├── README.md                  # 本文件
├── docs/
│   └── Quick-start-Ascend.md  # NPU 端到端最小验证 spec
└── tests/
    ├── __init__.py            # sys.path bootstrap
    └── test_quick_start_ascend.py
```