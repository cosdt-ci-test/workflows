# lmcache

本目录是 [LMCache-Ascend](https://github.com/LMCache/LMCache-Ascend) 的看护配套数据，不是上游源码。Quick Start 流水线在 [.github/workflows/lmcache-quick-start.yml](../../.github/workflows/lmcache-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类与支持程度仍是 `TODO`，待与项目表核对；阶段 A）。

当前只落地 Quick Start 线，没有 example 清单和 `*-examples.yml`。

`upstream_repo` 指向 `LMCache/LMCache-Ascend`，不是主仓 `LMCache/LMCache`。主仓没有昇腾实现。monitor 跟昇腾仓的 Release 走。文档用隐藏 `#test-setup` 把 `$UPSTREAM_REF` 填进 `git clone -b <UPSTREAM_REF>`，并把去掉 `v` 前缀的同一 tag 填进 `pip install lmcache==<LMCACHE_VER>`（pip 不接受 `lmcache==v0.4.4`）。共享引擎仍会把 `result.json` 的 `target_ref` 写成解析到的最新 Release tag。

安装栈按仓内已验证的 vLLM-Ascend 0.23 配方：`torch==2.10.0` / `torch-npu==2.10.0.post4`、源码 vLLM `v0.23.0`（`VLLM_TARGET_DEVICE=empty`）、`vllm-ascend==0.23.0`。上游 CI 用过 vllm-ascend 0.18.0，但那份轮子是 cp311，本镜像是 Python 3.12，不能退回 0.18。

## Quick Start 看护范围

文档在 `docs/Quick-start-Ascend.md`。方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。

打了 `#test`、会被看护的步骤：

- 检查 Python 版本
- 生成 CUDA 屏蔽约束并安装 `torch==2.10.0` / `torch-npu==2.10.0.post4`，断言 `npu_available True`
- 源码安装 vLLM 0.23.0 与 `vllm-ascend==0.23.0`（克隆到 `vllm-src`，避免 `./vllm` 挡住 import），断言 `vllm_device npu`
- 安装 `lmcache==<LMCACHE_VER> --no-deps --no-build-isolation` 以及导入期 Python 依赖，按 `$UPSTREAM_REF` 用 HTTP/1.1 克隆 LMCache-Ascend 并拉子模块，补上 HIXL 的 `pkg_inc` include 和 KV connector 的第三个参数，再 `SOC_VERSION=Ascend910B4` 编译，断言 `soc Ascend910B4` 与 `c_ops_ok True`
- 把离线脚本存成文件，设 `VLLM_WORKER_MULTIPROC_METHOD=spawn`，用 `vllm.LLM` + `LMCacheAscendConnectorV1Dynamic` 对 Qwen2.5-0.5B 做一次短生成。工作负载块的预期输出含 `Platform plugin ascend is activated`、`Using NPU for LMCache engine.` 与 `current_device npu:0`

无标签、**不看护**的步骤：

- `source set_env.sh` 和 `export PATH`（测试进程在 `prepare_environment` 里做等价的 CANN / ATB 注入）
- `npu-smi info`（设备表每次不同，正文只要求退出码 0）
- `vllm serve`、磁盘后端、Llama-3.1-8B 原文示例

`schedule` 保持注释。`doc_url` 走 GitHub Contents API（`ref=${{ github.sha }}`），不走 `raw.githubusercontent.com`。
