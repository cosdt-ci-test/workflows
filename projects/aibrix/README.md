# aibrix

本目录是 [aibrix](https://github.com/vllm-project/aibrix) 的看护配套数据，不是 aibrix 源码。example 流水线在 [.github/workflows/aibrix-examples.yml](../../.github/workflows/aibrix-examples.yml)。Quick Start 流水线是另一条线，文件在 `.github/workflows/aibrix-quick-start.yml`。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

上游默认分支是 `main`。当前看护钉在 local mode（v0.7.0 起提供），后端用 vLLM-Ascend。上游没有健康的昇腾 CI。本仓先走阶段 A。

## 清单

`examples_manifest.yaml` 的扫描根是 `deployment`，只扫 `.sh`。

```bash
python3 scripts/bootstrap_manifest.py \
  --target-root /path/to/aibrix \
  --output projects/aibrix/examples_manifest.yaml \
  --scan-root deployment \
  --include-extension .sh \
  --max-depth 3
```

`supported` 有一条：`deployment/local/run-local.sh`，profile `local_gateway`，挂 `linux-aarch64-a2-1`、`npu_devices: '0'`、镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

`unsupported` 只表示本看护体系当前不跑，不是社区支不支持。

`profile` 由 `setup_example.sh` 的 `setup_*` 解释。未知 profile 在读 `TARGET_ROOT`、装包、编译之前非 0 退出，并打印 `local_gateway`。

## 绿灯含义

- `local_gateway`（路径 `deployment/local/run-local.sh`）。先用文档同一套 vLLM-Ascend 栈在 `127.0.0.1:8000` 拉起 `Qwen/Qwen2.5-0.5B-Instruct`，日志必须出现 `backend=hccl`，再原样执行上游 `run-local.sh`，经 Envoy `:10080` 拿到 HTTP 200 且 `choices[0].message.content` 非空。**绿灯 = 昇腾后端上跑过一次经 AIBrix local mode 转发的 completion，不是“Go 编过了”或“网关进程还活着”。**

## Quick Start 看护范围

文档在 `docs/Quick-start-Ascend.md`。标签方言见仓里 `docs/markdown_doc_test_label.md`。

看护会执行的块：

- `install-system-prereqs`、`install-go`、`install-envoy`
- `clone-aibrix`、`build-gateway`
- `install-vllm`
- `start-backend`（`#test-setup`，轮询 `/health`）
- `backend-on-npu`（日志必须含 `backend=hccl`）
- `configure-endpoints`、`start-gateway`、`infer`
- 停进程收尾块（`#test-setup`，只看退出码，不比对输出）

不看护的无标签块（用户仍应按文档做）：

- `npu-smi info`
- `source` CANN / ATB 的说明块（测试的 `prepare_environment` 会加载同一对 `set_env.sh`，和文档语义等价）

CI 容器里如果 `127.0.0.1:6060` 已被占用（例如本机调试时的 coder agent），文档里有一段隐藏的 `#test-setup` 只给 `gateway-plugins` 套一层 `localhost -> 127.0.0.2` 的 hosts remap。vLLM 仍在主机网络里跑。example 线用 `scripts/with_free_pprof.sh` 只包 `run-local.sh`。用户文档不写这一步。用户机器上 6060 空闲时，原样 `run-local.sh` 即可。

## 缓存与下载源

example 线工具目录默认 `/root/.cache/cosdt-ci-test/aibrix/tools`（可用 `AIBRIX_TOOLS_DIR` 覆盖）。Go 工具链、Envoy、`GOPATH` / `GOCACHE`、vllm 源码树都在这里跨 run 复用。Envoy 和 vllm clone 的主源是组织代理 `https://gh-proxy.test.osinfra.cn/<原 GitHub URL>`，失败再直连 GitHub。

Quick Start 文档的可见命令仍写官方 GitHub URL。跨 run 复用只在隐藏 `#test-setup` 里，路径是 `/root/.cache/cosdt-ci-test/aibrix/`（Envoy `envoy/1.39.0/envoy`，源码 `src/aibrix-<ref>` 和 `src/vllm-v0.23.0`）。workflow 把该目录按同路径挂进容器。Envoy 的隐藏 restore 块在缓存未命中时会经组织代理下载（CI 直连 GitHub Release 只有十几 KB/s，可见块必然超 1200 秒块超时），sha256 校验后原子写入缓存再预置工作目录；代理也失败则该块直接非 0，秒级失败，不再烧一个块超时。

## 触发

`aibrix-examples.yml` 有两种入口。`monitor` 跑在 `ubuntu-latest`，不占 NPU。

- `schedule`。cron 写在文件里但是注释掉的。接入阶段保持注释，不要打开。
- `workflow_dispatch`。手动触发。默认 `force=false`，和定时走同一套监控。只有 `force=true` 才跳过监控门。`target_repo` 和 `target_ref` 只在 `force=true` 时有意义。

两个监控信号是「或」，没有优先级，失败不重试。

1. 清单 `supported` 各 `path` 在上游 `main` 上的 Contents API 哈希。
2. `/releases/latest` 的 release 数字 id。

Quick Start 线没有 `force` / `target_repo` / `target_ref`。`workflow_dispatch` 本身就会跑，并自动测上游最新 release tag。
