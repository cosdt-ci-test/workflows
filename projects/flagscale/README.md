# flagscale

本目录是 [FlagScale](https://github.com/flagos-ai/FlagScale) 的看护配套数据，不是上游源码。Quick Start 流水线在 [.github/workflows/flagscale-quick-start.yml](../../.github/workflows/flagscale-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类与支持程度仍是 `TODO`，待与项目表核对；阶段 A）。

当前只落地 Quick Start 线，没有 example 清单和 `*-examples.yml`。

上游 **已经有健康的昇腾 CI**（`.github/configs/ascend.yml`）。按仓库总则这类本应直接走阶段 B，把流水线推到上游。本仓仍先落一条 Quick Start 看护（阶段 A），方便在 AscendHub / 910B4 上单独验证入门文档。

`upstream_repo` 指向 `flagos-ai/FlagScale`。monitor 跟上游 Release 走（撰写时最新是 `v2.0.0`）。文档用隐藏 `#test-setup` 把 `$UPSTREAM_REF` 填进 `git clone -b <UPSTREAM_REF>`。共享引擎仍会把 `result.json` 的 `target_ref` 写成解析到的最新 Release tag。

配套镜像只用 AscendHub 的 CANN 9.1 / 910B tag，不用上游文档里的 `harbor.baai.ac.cn` 镜像。那份镜像里的 vllm-ascend 自定义算子是按 A3（`ascend910_9391`）编的，本仓 runner 全是 A2 / 910B4。文档在本机用 `SOC_VERSION=Ascend910B4` 重编算子。工作负载是离线 `flagscale inference`（TP=2），不是 `vllm serve`。版本与命令以 `docs/Quick-start-Ascend.md` 为准。

## Quick Start 看护范围

文档在 `docs/Quick-start-Ascend.md`。方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。

打了 `#test`、会被看护的步骤：

- 检查 Python 版本
- 生成 CUDA 屏蔽约束并安装 `torch==2.10.0` / `torch-npu==2.10.0.post4` / `triton-ascend==3.2.2`，断言 `npu_available True`
- 源码安装 vLLM 0.20.2（克隆到 `vllm-src`）以及 `import vllm.LLM` 需要的 Python 依赖，断言 `vllm 0.20.2`
- 源码安装 `vllm-ascend` `v0.20.2rc1`（克隆到 `vllm-ascend-src`，`SOC_VERSION=Ascend910B4`），断言版本
- 安装 FlagGems `v5.3.0`（`--no-deps --no-build-isolation`）与 `vllm-plugin-FL` commit `53adefb26`
- 按 `$UPSTREAM_REF` 克隆 FlagScale 并 `pip install --no-build-isolation --no-deps -e ./FlagScale`，再装 Hydra / typer
- 平台探测：`VLLM_PLUGINS=fl` 时 `device_type npu`、`dist_backend hccl`、`PlatformFL`
- 离线 `flagscale inference` 对 Qwen2.5-0.5B 做 TP=2 短生成。工作负载块的预期输出含 `Platform plugin fl is activated`、`NPU compatibility enabled`、`backend=hccl` 与 `output.outputs[0].text=`

无标签、**不看护**的步骤：

- `source set_env.sh` 和 `export PATH`（测试进程在 `prepare_environment` 里做等价的 CANN / ATB 注入）
- `npu-smi info`（设备表每次不同，正文只要求退出码 0）
- `vllm serve`、上游功能测试原文里的 Qwen3-4B 与 16 卡 `ASCEND_VISIBLE_DEVICES`
- `flagtree` 厂商索引、`harbor.baai.ac.cn` 镜像、`vllm-plugin-FL@main`

`schedule` 保持注释。`doc_url` 走 GitHub Contents API（`ref=${{ github.sha }}`），不走 `raw.githubusercontent.com`。
