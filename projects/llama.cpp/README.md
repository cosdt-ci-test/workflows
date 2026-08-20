# llama.cpp

本目录是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的看护配套数据，不是 llama.cpp 源码。example 流水线在 `[.github/workflows/llama.cpp-examples.yml](../../.github/workflows/llama.cpp-examples.yml)`。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：基础支持；阶段 A）。

上游默认分支是 `master`（不是 `main`）。上游有 CANN 后端文档 [docs/backend/CANN.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)，以及已停用的 `build-cann.yml`（为节省 GitHub-hosted runner 而关掉）。本仓先走阶段 A：在本仓流水线把 example 跑通，再考虑往上游推。

## 清单

- `examples_manifest.yaml` 由仓库根目录 `scripts/bootstrap_manifest.py` 按目录单位扫描目标仓 `examples/`：下一层里带 `CMakeLists.txt` 的目录算一条 example（`scan.unit: directories`）。语法转换脚本、`model-conversion`、Android / Swift 应用不在这份清单里。
- `supported` 覆盖除 `idle` / `sycl` / `training` 以外的全部扫描条目。`unsupported` 只表示本看护体系当前不跑它们，不是社区支不支持。`examples/idle` 是 Metal 空闲微基准。`examples/sycl` 要 Intel oneAPI。`examples/training` 是 WIP 全量 FP32 finetune，CANN 没有对应反向算子。
- 每条 `supported` 都挂 `linux-aarch64-a2-1`、`npu_devices: '0'`、镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。workflow 用 `npu_devices` 派生 `--device=/dev/davinciN`，托管 `ubuntu-latest` 没有这块设备，所以 CPU 工具也不能分叉到托管 runner。
- 四个 `profile`，由 `setup_example.sh` 的 `setup_<profile>` 函数解释。未知 profile 在 cmake / 下载之前非 0 退出。
  - `cann`：`cmake -DGGML_CANN=on`，只编 `basename($EXEC)` 这一个 target（`examples/simple` 仍是 `llama-simple`）。从 ModelScope 下载 `qwen2.5-0.5b-instruct-q4_0.gguf`，把绝对路径写进 `LLAMA_CI_MODEL`。GGUF 先写到 `.part`，下载成功再改名。半截文件头也是 `GGUF`，不能当完整模型复用。
  - `cpu`：cmake **不开** `GGML_CANN`，只编对应 target，不下 Qwen。绿灯含义是「这个目录在 aarch64 CI 机上还能编、还能跑完」，**不是**「用了 NPU」。外部汇总不要把这些条目读成昇腾推理绿。
  - `cann-diffusion`：同样开 CANN，编 `llama-diffusion-cli`。Dream 架构 GGUF 另下，不塞进 `cann`，避免每条 simple/batched 都拉 8GB。现用钉死 URL：`https://hf-mirror.com/mradermacher/Dream-v0-Instruct-7B-GGUF/resolve/main/Dream-v0-Instruct-7B.Q8_0.gguf`（约 8.1GB）。落盘路径是 `/root/.cache/cosdt-ci-test/llama.cpp/`（自建 NPU runner 已把 SFS Turbo 挂在 `/root/.cache`，跨 pod 还在）。`fetch_gguf` 看到完整 `GGUF` 文件头就跳过下载。不要用 `actions/cache` 传这 8GB。ModelScope 上没有这份副本；该仓库也没有 Q4_0。下载失败或文件头不是 `GGUF` 是 setup 失败（看护噪音），不会改用 Qwen。
  - `cmake-pkg`：`cmake --install` 之后用 `find_package(Llama)` 再编 `examples/simple-cmake-pkg`。看护的是安装包还能链上 CANN 后端。不开 `GGML_BACKEND_DL`。消费方二进制的 RUNPATH 只覆盖直接依赖，`libggml` 再加载的 `libggml-cpu` / `libggml-cann` 仍会找不到。setup 一律把 `LD_LIBRARY_PATH=$TARGET_ROOT/inst/lib` 写进 `$GITHUB_ENV`。Run step 会再 `source` CANN，把 toolkit 路径接到前面。
- 量化选 Q4_0 / Q8_0 而不是 Q4_K_M：上游 `docs/backend/CANN.md` 的 910B DataType 表列出 FP16 / Q8_0 / Q4_0 / BF16。
- `overlay_args` 把步数、并行度、prompt 压到 CI 规模，模型路径用 `${LLAMA_CI_MODEL}`。flag 和值写在同一项，避免 YAML 把裸数字收成 int。
- `run_example.sh` 按清单相对路径分叉：`examples/gguf` 对 `$CI_OUTPUT_DIR/ci-demo.gguf` 先 `w` 再 `r`；`examples/simple-chat` / `examples/retrieval` 用 `printf` 喂一句非空 stdin。其余是一次 `exec` + overlay。`examples/lookahead` 源码已经把 `kv_unified` 写死，overlay 不再传 `-kvu`（该 flag 只挂在 batched/parallel 等 example 枚举上，lookahead 走 `LLAMA_EXAMPLE_COMMON` 会 parse 失败）。`examples/debug`、`examples/passkey`、`examples/diffusion` 默认日志太少，overlay 加 `-v` 才能看到 `CANN0`。
- `CANN[0-9]` 断言只在 `profile` 为 `cann` / `cann-diffusion` / `cmake-pkg` 时执行。`cpu` 跳过。ggml 在卡没挂上时会静默退回 CPU 且进程仍退出 0；NPU 条目用这条断言挡住假绿。
- 工作副本补丁只打在 `$TARGET_ROOT`，禁止 `git add` / `commit` / `push`。`examples/simple-chat` 去掉源码里只放行 ERROR 的 `llama_log_set`，否则加载阶段的 `CANN0` 被掐掉会假红。`examples/retrieval` 在 `getline` 之后对空 query / EOF `break`，否则 `while (true)` 永不退出，timeout 是看护噪音。
- `examples/convert-llama2c-to-ggml` 和 `examples/gguf-hash` 用本目录 `fixtures/`：`stories260K.bin` + `tok512.bin`（与上游 CPU CI 同款），以及 `llama-gguf w` 生成的极小 `ci-demo.gguf`。不要在 NPU job 里现拉 huggingface.co。
- 清单与磁盘的差异只打印路径，不使 job 失败；例外：`supported` 条目的 path 已不在磁盘上时 manifest-check 立即判红。
- 编译工具用镜像自带的 `cmake` / `g++` / `make` / `git` / `curl`，不 `apt-get`，不改 `/etc/apt/sources.list`。

重新生成清单（会覆盖本目录的 yaml，先确认 supported 段）：

```bash
python3 scripts/bootstrap_manifest.py \
  --target-root /path/to/llama.cpp \
  --output projects/llama.cpp/examples_manifest.yaml \
  --unit directories
```

## 触发

`llama.cpp-examples.yml` 有两种入口。`monitor` job 跑在 `ubuntu-latest`，不占 NPU。

- `schedule`：cron 是 `30 */6 * * *`。**接入阶段保持注释**，第一次 `force=true` 跑绿后再打开。打开后每 6 小时跑监控，任一信号亮了才上 NPU。
- `workflow_dispatch`：手动触发。默认 `force=false`，和定时走**同一套监控、同一份 cache**。只有 `force=true` 才跳过监控门、必跑，并且**不读不写** monitor cache。`target_repo` / `target_ref` 只在 `force=true` 时有意义。

两个监控信号都跑，是「或」，互不跳过、没有优先级：

1. 清单 `supported` 各 `path` 在上游 `master` 上的文件内容哈希（Contents API 的 blob SHA；目录会递归到文件。404 记成 `MISSING`，哈希会变）。本信号亮了，测的是这一轮解析到的 `master` commit SHA。
2. `/releases/latest` 的 release **id**（数字，不是 tag 字符串；不过滤 `bxxxxx` / `v*`）。本信号亮了，测的是该 release tag 当前指到的 commit SHA。

谁亮了就测谁的树：

- 只有 supported 亮：测 `MASTER_SHA`，`reason=supported`。
- 只有 release 亮：测 `RELEASE_SHA`，`reason=release`。
- 两个都亮且两个 SHA 不同：跑两份，`reason` 分别是 `supported` 和 `release`。
- 两个都亮且 SHA 相同：只跑一份（同一份代码不编两遍），`reason=supported,release`。
- 都没亮：`targets` 为空，后面的 job 跳过。

`force=true` 时 `targets` 只有一项，`reason=manual`，repo/ref 用输入解析出的 commit SHA。

不做失败重试。看见新哈希或新 release id 就写入 cache；后面 NPU 红了也不为同一对值再跑。要再跑：等信号再变，或 `force=true`。

`ggml/`、CANN 后端、本仓 `setup_example.sh` / `overlay_args` 不算进 supported 哈希。那些要靠新 release，或 `force=true`。

## Quick Start 看护

文档在 `docs/Quick-start-Ascend.md`。流水线是 `.github/workflows/llama.cpp-quick-start.yml`。它按字面执行文档里的 `shell` 块，跳过 `shell skip` 块，并把紧邻的 `text` 和实际输出做正则比对。

触发契约和上面的 example 看护相同，只是信号 A 换成这篇文档的 sha256。

- `schedule`：cron 是 `15 */6 * * *`。**接入阶段保持注释**，第一次 `force=true` 跑绿后再打开。打开后每 6 小时跑监控，任一信号亮了才上 NPU。
- `workflow_dispatch`：手动触发。默认 `force=false`，和定时走**同一套监控、同一份 cache**。只有 `force=true` 才跳过监控门、必跑，并且**不读不写** monitor cache。`target_repo` / `target_ref` 只在 `force=true` 时有意义。

两个监控信号都跑，是「或」，互不跳过、没有优先级：

1. 本仓这篇 Quick Start 文档的 sha256。`MONITORED_DOC_URL` 走 GitHub Contents API（`ref` 是这次 run 的 `github.sha`），不用 `raw.githubusercontent.com`：NPU runner 的出口到不了那个域名。monitor 和测试拉同一 URL。本信号亮了，测的是这一轮解析到的上游 `master` commit SHA。
2. `/releases/latest` 的 release **id**（数字，不是 tag 字符串。不过滤 `bxxxxx` / `v*`）。本信号亮了，测的是该 release tag 当前指到的 commit SHA。

谁亮了就测谁的树：

- 只有 doc 亮：测 `MASTER_SHA`，`reason=doc`。
- 只有 release 亮：测 `RELEASE_SHA`，`reason=release`。
- 两个都亮且两个 SHA 不同：跑两份，`reason` 分别是 `doc` 和 `release`。
- 两个都亮且 SHA 相同：只跑一份（同一份代码不编两遍），`reason=doc,release`。
- 都没亮：`targets` 为空，后面的 job 跳过，不写 `result.json`。

`force=true` 时 `targets` 只有一项，`reason=manual`，repo/ref 用输入解析出的 commit SHA。

不做失败重试。看见新哈希或新 release id 就写入 cache。后面 NPU 红了也不为同一对值再跑。要再跑：等信号再变，或 `force=true`。

NPU job 不上传 artifact。`result.json` 由托管 runner 上的 `publish-result` 按 job 名回看 conclusion 后上传。
