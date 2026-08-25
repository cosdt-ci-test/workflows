# llm-d

本目录是 [llm-d](https://github.com/llm-d/llm-d) 的看护配套数据，不是 llm-d 源码。Quick Start 流水线在 [.github/workflows/llm-d-quick-start.yml](../../.github/workflows/llm-d-quick-start.yml)。没有 example 线。上游没有 `examples/` 目录，其余指南是 Kubernetes 形态，ARC runner 的 pod 没有集群。

注册信息见根目录 [projects.yaml](../../projects.yaml)。分类 推理加速，支持程度 新兴适配，phase A，`upstream_repo: llm-d/llm-d`。

## 看护范围

文档在 `docs/Quick-start-Ascend.md`。方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。形态是无 Kubernetes 的三进程栈：vLLM 经 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 跑模型，EPP 从 [llm-d-router](https://github.com/llm-d/llm-d-router) 构建，Envoy 用官方 ARM64 二进制。

打了 `#test`、会被看护的步骤：

- `npu-smi info`
- 安装 vLLM、vllm-ascend、modelscope、triton-ascend 并打印版本
- 克隆 llm-d 配置并把模型改成 `Qwen/Qwen3-0.6B`
- 经 Envoy `:8081` 打一次 `/v1/completions`
- 在 vLLM 日志里匹配 `backend=hccl`（设备锚点）

`#test-setup` 会执行但不比对输出。它注入上游 Release tag、构建 EPP、获取 Envoy，并按 PID 文件启动和清理三个进程。

无标签、不看护的步骤：

- 文档开头的 `source` CANN / NNAL 与 `export PATH`。测试进程在 `prepare_environment` 里做等价注入，并且覆盖写回 `os.environ`。缺脚本直接失败。
- 上游 Kubernetes quickstart 及其余 guides。ARC pod 没有集群。

清理只读 `/root/llm-d/envoy.pid`、`/root/llm-d/epp.pid`、`/root/llm-d/vllm.pid`，并按这个顺序停进程组。不要 `pkill`。

`schedule` 保持注释。`doc_url` 走 GitHub Contents API（`ref=${{ github.sha }}`），不走 `raw.githubusercontent.com`。
