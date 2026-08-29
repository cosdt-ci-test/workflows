# mooncake

本目录是 [Mooncake](https://github.com/kvcache-ai/Mooncake) 的看护配套数据，不是 Mooncake 源码。example 流水线在 [.github/workflows/mooncake-examples.yml](../../.github/workflows/mooncake-examples.yml)。Quick Start 流水线在 [.github/workflows/mooncake-quick-start.yml](../../.github/workflows/mooncake-quick-start.yml)。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：新兴适配；阶段 A）。

上游默认分支是 `main`。上游有 `ci_ascend.yml`（nightly / 带 `run-e2e-ci` 标签的 E2E）：用私有镜像编 `-DUSE_ASCEND_DIRECT=ON`，`BUILD_EXAMPLES=OFF`，跑的是 HIXL 仓里的 Mooncake Store Python 样例，不跑 Transfer Engine 的 C++ example。本仓阶段 A 看护那条 C++ Ascend Direct 例程，以及一份会真正上 NPU 的 Quick Start。

## 清单

- `examples_manifest.yaml` 由仓库根目录 `scripts/bootstrap_manifest.py` 扫描目标仓 `mooncake-transfer-engine/example/` 下的 `.cpp` / `.py` / `.cu` 文件。`http-metadata-server/` 是 Go，不在这个扩展名集合里。
- `supported` 只有 `transfer_engine_ascend_direct_perf.cpp`。这是上游文档里的 Ascend Direct 测试程序，双进程（target + initiator），两张 910B。`unsupported` 只表示本看护体系当前不跑它们，不是社区支不支持。
  - `transfer_engine_ascend_one_sided.cpp`、`transfer_engine_ascend_perf.cpp` 走已弃用的 `USE_ASCEND` / HCCL 传输。
  - `transfer_engine_heterogeneous_ascend_perf_initiator.cpp` 要 GPU 对端。
  - `transfer_engine_bench.cpp`、`transfer_engine_bench_with_notify.cpp`、`transfer_engine_bench_with_retry.cpp`、`transfer_engine_validator.cpp`、`memory_pool.cpp`、`show_link.cpp` 是 RDMA / TCP 通用程序，绿灯不等于昇腾传输绿。
  - `device_transport_example.cu`、`nccl_device_transport_example.cu`、`nccl_host_transport_example.cpp` 是 CUDA / NCCL。
  - `efa_first_submit_probe.cpp`、`efa_per_transfer_latency_bench.py`、`batch_register_bench.py`、`kvcache_prefix_bench.py` 是 AWS EFA。
  - `http-metadata-server-python/bootstrap_server.py` 是主机侧 metadata 服务。
- `supported` 条目挂 `linux-aarch64-a2-2`、`npu_devices: '0,1'`、镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。
- `profile` 由 `setup_example.sh` 解释。未知 profile 在 apt / cmake 之前非 0 退出，并打印 `ascend-direct`。
  - `ascend-direct`：`cmake -DUSE_ASCEND_DIRECT=ON`，只编 `basename($EXEC)`。`run_example.sh` 先起 target（NPU 0），从日志解析 `listening on <IP>:<port>`，再起 initiator（NPU 1）。initiator 退出后再杀掉 target。HIXL 需要 `/etc/hccn.conf`（setup 缺文件即失败；workflow 从宿主机只读挂入）。日志必须有 `Success to initialize adxl engine`、`Test completed:`，以及 `npu:<logicid>` 或 `mem type:device`（device buffer 登记）。`Failed to install Ascend transport`、`getTransferStatus FAILED` 或 `Sync data transfer timeout` 判红。上游 initiator 在 FAILED/TIMEOUT 时仍会打印 `Test completed:` 并 `return 0`，所以必须扫这些失败串。**绿灯 = 这次 Ascend Direct 写传输在两张 NPU 之间跑完，不是二进制编过了。** `--protocol` 在该 `.cpp` 里声明了但运行时不用，传输后端是编译期 `USE_ASCEND_DIRECT`。不要把 `--mode` / `--segment_id` 写进 `overlay_args`。
  - overlay 只压规模：`--block_iteration=1 --batch_size=2 --block_size=16384`。默认 `block_iteration=10` 会按 2 的幂放大块，CI 里不合适。
  - 二进制默认 `local_server_name` 是实验室 IP。脚本强制 `127.0.0.1`。
- 编译依赖用 `apt-get` 装 glog / gflags / ibverbs 等；镜像已有 cmake / g++，不再为编译器 `apt-get`。examples 管线（`setup_example.sh`）不改 `/etc/apt/sources.list`；quick-start doc 管线（`docs/Quick-start-Ascend.md`）在每次重建的测试容器里临时 sed 成阿里云 `ubuntu-ports`（原生 `ports.ubuntu.com` 在国内 runner 上拉索引极慢）。`extern/pybind11` 在 setup 里 `submodule update`，失败则按目标仓 gitlink SHA 从 `ghfast.top` 拉同一份 commit，对不上就失败。

重新生成清单会**整文件覆盖** `--output` 指向的 yaml。生成器不会合并已填好的 `profile` / `exec` / `overlay_args`。

```bash
python3 scripts/bootstrap_manifest.py \
  --target-root /path/to/Mooncake \
  --output projects/mooncake/examples_manifest.yaml \
  --scan-root mooncake-transfer-engine/example \
  --include-extension .cpp \
  --include-extension .py \
  --include-extension .cu \
  --supported mooncake-transfer-engine/example/transfer_engine_ascend_direct_perf.cpp
```

## 触发

`mooncake-examples.yml` 有两种入口。`monitor` job 跑在 `ubuntu-latest`，不占 NPU。

- `schedule`：cron 写在文件里但是注释掉的。接入阶段保持注释，不要打开。
- `workflow_dispatch`：手动触发。默认 `force=false`，和定时走同一套监控、同一份 cache。只有 `force=true` 才跳过监控门、必跑，并且不读不写 monitor cache。`target_repo` / `target_ref` 只在 `force=true` 时有意义。

两个监控信号都跑，是「或」，互不跳过、没有优先级：

1. 清单 `supported` 各 `path` 在上游 `main` 上的文件内容哈希（Contents API 的 blob SHA；目录会递归到文件。404 记成 `MISSING`）。本信号亮了，测的是这一轮解析到的 `main` commit SHA。
2. `/releases/latest` 的 release **id**（数字，不是 tag 字符串）。本信号亮了，测的是该 release tag 当前指到的 commit SHA。

`force=true` 时 `targets` 只有一项，`reason=manual`。

不做失败重试。NPU job 不上传 artifact。`result.json` 由托管 runner 上的 `publish-result` 按 job 名回看 conclusion 后上传。

## Quick Start 看护

文档在 `docs/Quick-start-Ascend.md`。流水线是 `.github/workflows/mooncake-quick-start.yml`。文档方言见 [docs/markdown_doc_test_label.md](../../docs/markdown_doc_test_label.md)。无标签的 `shell` 块给用户复制，看护跳过。

### 看护范围

- **看护**：第 3 节安装编译依赖；第 4 节克隆并按 `-DUSE_ASCEND_DIRECT=ON` 编译 `transfer_engine_ascend_direct_perf`；第 5 节双进程写传输（target NPU 0 + initiator NPU 1，`P2PHANDSHAKE`）。
- **不看护**（无标签块）：第 1 节加载 CANN 环境（由测试 `prepare_environment` 合并进 `os.environ`，与文档同一份 `set_env.sh`）；第 2 节 `npu-smi info`（设备表数值每次不同）；`/etc/hccn.conf` 由驱动/宿主机提供，文档只说明需要它。
- **本文也不覆盖**：`pip install mooncake-transfer-engine-npu`、vLLM / SGLang 接入、Mooncake Store、已弃用的 HCCL Ascend Transport、异构 GPU+NPU、UBSHMEM。

`schedule` 保持注释。薄触发器 `doc_url` 走 Contents API + `ref=${{ github.sha }}`。
