# llama.cpp

本目录是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的看护配套数据，不是 llama.cpp 源码。example 流水线在 `[.github/workflows/llama.cpp-examples.yml](../../.github/workflows/llama.cpp-examples.yml)`。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：基础支持；阶段 A）。

上游默认分支是 `master`（不是 `main`）。上游有 CANN 后端文档 [docs/backend/CANN.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)，以及已停用的 `build-cann.yml`（为节省 GitHub-hosted runner 而关掉）。本仓先走阶段 A：在本仓流水线把 example 跑通，再考虑往上游推。

## 清单

- `examples_manifest.yaml` 由仓库根目录 `scripts/bootstrap_manifest.py` 按目录单位扫描目标仓 `examples/`：下一层里带 `CMakeLists.txt` 的目录算一条 example（`scan.unit: directories`）。语法转换脚本、`model-conversion`、Android / Swift 应用不在这份清单里。
- `examples/simple` 已进 `supported`。`profile` 是 `cann`，runner 是 `linux-aarch64-a2-1`（1 张卡），`npu_devices` 是 `'0'`，镜像是 CANN 9.1.0 ubuntu22.04。真正启动的是编译产物，所以条目写了 `exec: build/bin/llama-simple`。其余目录仍在 `unsupported`：那只表示本看护体系当前不跑它们，不是社区支不支持。
- profile `cann` 做两件事：`cmake -DGGML_CANN=on` 只编 `llama-simple` 这一个 target；从 ModelScope 下载 `qwen2.5-0.5b-instruct-q4_0.gguf`，把绝对路径写进 `LLAMA_CI_MODEL`。编译工具用镜像自带的 `cmake` / `g++` / `make` / `git` / `curl`，不 `apt-get`，不改 `/etc/apt/sources.list`。
- 量化选 Q4_0 而不是 Q4_K_M：上游 `docs/backend/CANN.md` 的 910B DataType 表列出 FP16 / Q8_0 / Q4_0 / BF16，Q4_K_M 等不在表内。
- `overlay_args` 把生成长度压到 `-n 16`，并显式写 `-ngl 99`（全部层放到 NPU），模型路径用 `${LLAMA_CI_MODEL}`。
- `run_example.sh` 会把 `llama-simple` 的输出 tee 到日志，并断言日志里出现 `CANN` 加数字（例如 `CANN0`）。ggml 在卡没挂上时会静默退回 CPU 且进程仍退出 0；这条断言用来挡住那种假绿。
- 清单与磁盘的差异只打印路径，不使 job 失败；例外：`supported` 条目的 path 已不在磁盘上时 manifest-check 立即判红。

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
