# onnxruntime

本目录是 [onnxruntime](https://github.com/microsoft/onnxruntime) 的看护配套数据，不是 onnxruntime 源码。example 流水线在 [.github/workflows/onnxruntime-examples.yml](../../.github/workflows/onnxruntime-examples.yml)。Quick Start 流水线是另一条线，文件在 `.github/workflows/onnxruntime-quick-start.yml`。注册信息见根目录 [projects.yaml](../../projects.yaml)（分类：推理加速；支持程度：基础支持；阶段 A）。

上游默认分支是 `main`。输出键仍叫 `master_sha` 和 `MASTER_SHA`，只为了和 decide 步骤对齐。上游有 CANN Execution Provider，没有健康的昇腾 CI。本仓先走阶段 A。

## 清单

`examples_manifest.yaml` 的扫描根是 `samples`。`unit: mixed` 且 `marker` 为空。`samples` 下一层每个目录都算一条，不要求 `CMakeLists.txt`。深度 1 的 `.py` 也会进扫描。重新生成会整文件覆盖 `--output`。生成器不会合并已填的 `profile`、`exec`、`overlay_args`。

```bash
python3 scripts/bootstrap_manifest.py \
  --target-root /path/to/onnxruntime \
  --output projects/onnxruntime/examples_manifest.yaml \
  --scan-root samples \
  --unit mixed \
  --marker '' \
  --max-depth 1 \
  --include-extension .py
```

`supported` 有两条。都挂 `linux-aarch64-a2-1`、`npu_devices: '0'`、镜像 `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`。

`unsupported` 只表示本看护体系当前不跑，不是社区支不支持。`samples/nodejs` 需要 npm，也不走 CANN EP。

`profile` 由 `setup_example.sh` 的 `setup_*` 解释。未知 profile 在读 `TARGET_ROOT` 和 `GITHUB_ENV`、装包、编译之前非 0 退出，并打印 `cann-gtest` `cmake-consumer`。

## 绿灯含义

- `cann-gtest`（路径 `onnxruntime/test/providers/cann`）。用缓存里的 `onnxruntime_provider_test`，过滤 `CannExecutionProviderTest.*`。上游把 `test/providers/*` 从 `onnxruntime_test_all` 拆到这个二进制。对 `onnxruntime_test_all` 用同一条 filter 会 `0 tests` 且 exit 0。日志必须出现 `CannExecutionProviderTest.FunctionTest`，并且 `[  PASSED  ]` 的计数至少为 1。`0 tests from 0 test suites` 必须红。**绿灯 = CANN EP 上跑过并通过对这个 gtest，不是“ORT 编过了”。**
- `cmake-consumer`（路径 `samples/cxx`）。用同一份安装前缀编 `onnxruntime_sample_program`。日志必须有 `ONNX Runtime version:` 和 `Result: PASS`。**这条不是昇腾推理绿。** 上游 `samples/cxx/main.cc` 用默认 `SessionOptions`，不注册 CANN EP。绿灯只证明安装包能被下游 cmake 找到、链上、跑通 CPU Add。

## 不看护的相邻仓库

不看护 [microsoft/onnxruntime-training-examples](https://github.com/microsoft/onnxruntime-training-examples)。仓库已归档，0 个 release。`ORTModule` 走 NVIDIA 和 AMD。CANN 没有训练 EP。

不看护 [microsoft/onnxruntime-inference-examples](https://github.com/microsoft/onnxruntime-inference-examples)。0 个 release，release 信号是死的。plugin EP ABI 缺 `CreateEpFactories`。`onnxruntimesetup.cmake` 把 TensorRT 标成 `REQUIRED`。默认 CPU fallback 会假绿。CANN 的 `symbols.def` 只导出 `GetProvider`。

本线看的是 `microsoft/onnxruntime` 自己的 CANN gtest 和 `samples/cxx`。

## 扫描根和 cann 测试路径

`scan.root` 是 `samples`，因为那是仓内样本树。`onnxruntime/test/providers/cann` 不在这个根下，仍然可以写进 `supported`。`check_examples_manifest.py` 只在 supported 路径磁盘上不存在时失败。扫描多出来的 `new_paths` 只记录，不让 job 红。

## 构建缓存

ORT 从源码编 CANN 要数小时。产物按目标 commit SHA 放。

`/root/.cache/cosdt-ci-test/onnxruntime/<sha>/{install,bin}`

三层不要省。

- `/root/.cache` 是宿主机持久化根。pip 和 HuggingFace 也会写自己的子目录。不要把看护产物堆在这一层。
- `/root/.cache/cosdt-ci-test/` 是本看护体系前缀。容器 volume 挂这一层。
- `/root/.cache/cosdt-ci-test/onnxruntime/` 是本项目根。再往下按 SHA 分。`install` 是 cmake 安装前缀。`bin` 放 `onnxruntime_provider_test`。

写入先落到 `<sha>.part`，用 `flock` 串行，再 `mv` 成 `<sha>`。命中条件是 `bin/onnxruntime_provider_test`、`bin/` 下的 ORT `.so`（含 `libonnxruntime_providers_cann.so`）、`install/include`、`install/lib/libonnxruntime_providers_cann.so` 都在。缺一块就整份重编。同一 SHA 再跑应对齐到同一套文件。

workflow 的 `max-parallel` 是 1。两条 job 共用这份缓存，并行会抢同一把锁并打满编译机。

## cmake FetchContent 镜像

`setup_example.sh` 在编译前填充 `/root/.cache/cosdt-ci-test/onnxruntime/cmake-mirror/`。布局与 ORT 的 `--cmake_deps_mirror_dir` 约定一致：`cmake/deps.txt` 里每条 `https://` URL 对应 `<mirror>/<去掉 https:// 的 URL>`。github.com 和 codeload.github.com 先走 `https://gh-proxy.test.osinfra.cn/<原 URL>`，失败再直连。下完按 deps.txt 的 SHA1 校验，错了删除。单文件失败只告警，cmake 会回退在线下载。`www.nuget.org` 条目跳过（Windows 遥测，本配置用不到）。已命中 ORT 编译缓存的暖跑不会再填镜像。

## 触发

`onnxruntime-examples.yml` 有两种入口。`monitor` 跑在 `ubuntu-latest`，不占 NPU。

- `schedule`。cron 写在文件里但是注释掉的。接入阶段保持注释，不要打开。
- `workflow_dispatch`。手动触发。默认 `force=false`，和定时走同一套监控。只有 `force=true` 才跳过监控门。`target_repo` 和 `target_ref` 只在 `force=true` 时有意义。

两个监控信号是「或」，没有优先级，失败不重试。

1. 清单 `supported` 各 `path` 在上游 `main` 上的 Contents API 哈希。目录会递归到文件。404 记成 `MISSING`。
2. `/releases/latest` 的 release 数字 id。

monitor 另外盯 `onnxruntime/core/providers/cann`。EP 源码不是 example 路径。只改 provider、不改 `samples/` 或 gtest 目录时，也必须再跑。

`Decide targets` 成功之后才写 `.monitor-state`。任一步失败都不保存候选状态。

## Quick Start

文档在 `docs/Quick-start-Ascend.md`。用户路径是：检出 `microsoft/onnxruntime` 当前 GitHub 正式 Release，用 `./build.sh --use_cann --build_wheel` 编出 `onnxruntime-cann`，再当场造最小 Add 模型、关掉 CPU 回退做推理。monitor 盯的就是这份上游仓的 `/releases/latest`，所以文档跟最新正式 tag 走，不再钉 PyPI 上的 `onnxruntime-cann==1.24.4`。

第三方包索引用 `https://repo.huaweicloud.com/repository/pypi/simple`。这个 wheel 按 NumPy 1.x 编，文档钉 `numpy<2`。CANN 算子编译依赖是 `decorator`、`scipy>=1.11,<1.15`、`attrs`、`psutil`、`sympy`。缺其中任何一个时，会话能建，`sess.run()` 会在 `aclgrphBuildInitialize` / `aclopCompileAndExecute` 上红——根因是 `import tbe` 失败。

Ubuntu 22.04 默认是 gcc 11.4 和 cmake 3.22。文档会装 gcc-12、ninja，并用华为通用 PyPI 装 `cmake>=3.28,<4`、`packaging`、`wheel`、`setuptools`。不要覆盖 `/etc/apt/sources.list`。不要装 cmake 4。

创建会话时只请求 `CANNExecutionProvider`，并关掉 fallback。`get_providers()` 仍可能同时列出 CPU。那是注册表，不是「请求了 CPU」。

工作目录是 `/root/onnxruntime-qs`。跨 run 缓存在 `/root/.cache/cosdt-ci-test/onnxruntime/`：`wheels/<tag>/` 放编好的 wheel，`src/<tag>/` 放浅克隆，`cmake-mirror/` 与 example 线共用。薄触发器按同路径挂 volume。这些路径只出现在隐藏 `#test-setup`，不进用户看得见的正文。

## 看护范围

被看护的可见 `#test` 块：安装编译工具、克隆上游、编译 wheel、安装 wheel 与 TBE 依赖、打印 `get_available_providers()`、生成 `add_model.onnx`、关掉 CPU 回退后的 Add 推理。推理块的预期含 `CANNExecutionProvider` 和 `result [4.0, 6.0]`。

无标签、不执行：开头的 `source set_env.sh` / `PATH`，以及 `npu-smi info`。CI 面由 `prepare_environment` 覆盖式合并同一份 `set_env.sh`。

隐藏 `#test-setup`：捕获 `UPSTREAM_REF`；从宿主机缓存预置 wheel / 源码 / cmake-mirror 软链；编译成功后把 wheel 和浅克隆写回缓存。可见 clone / compile 仍是官方 GitHub 与 `./build.sh`；缓存命中时工作目录里已经有 `.git` 或 `dist/*.whl`，`if [ ! -d .../.git ]` / `compgen -G dist/*.whl` 自然跳过。

QS 绿只证明当前正式 Release 的 Python CANN EP 能跑通这篇文档里的 Add。不等于 example 线 `cann-gtest` 绿。`CannExecutionProviderTest.FunctionTest` 在 CANN 9.1.0 上仍是 `aclrtAllocatorGetByStream failed. Parameter stream is invalid`，保持诚实红。

## 已知的诚实红 / 噪音

coder 上（A2-910B，CANN 9.1.0，无 docker，不是 CI 镜像）目前测到：

- 当前正式 Release 源码 + `--use_cann --build_wheel` + CPython 3.12 + `numpy<2` + `decorator` / `scipy>=1.11,<1.15` / `attrs` / `psutil` / `sympy`：wheel 元数据名是 `onnxruntime-cann`，`get_available_providers()` 含 `CANNExecutionProvider`，关掉 fallback 的最小 Add 推理得到 `[4.0, 6.0]`。不钉 `numpy<2` 时 import 会炸。只装 wheel、不装 TBE 依赖时，`import tbe` 失败，`sess.run()` 报 `ge::aclgrphBuildInitialize` 或 `aclopCompileAndExecute("Add")`。缺 `sympy` 时同样是 `import tbe` 失败。
- `--use_cann` 在 CANN 9.1.0 + gcc-12 + cmake 3.31 上能编过，并链出 `libonnxruntime_providers_cann.so`。`--build_wheel` 打出 `onnxruntime_cann-*-linux_aarch64.whl`。PyPI 上 FFrog 的 `onnxruntime-cann==1.24.4` 不再是 Quick Start 的安装路径。
- `cann-gtest`：`CannExecutionProviderTest.FunctionTest` 能匹配到并在 NPU=0 上跑。官方 `run_example.sh` 路径上的失败是 `aclrtAllocatorGetByStream failed. Parameter stream is invalid`。环境已就绪（CANN 已 source、`--use_cann` 编过、filter 命中）。这是诚实红，不是「编错二进制」或假绿。空 `--gtest_filter` 被守卫判红（`0 tests from 0 test suites`）。不 source CANN 时加载 `libonnxruntime_providers_cann.so` 失败（缺 `libmsprofiler.so`），不会假绿。
- `cmake-consumer`：绿。日志有 `ONNX Runtime version: 1.30.0` 和 `Result: PASS`。默认 SessionOptions，没有 NPU 锚点。
- 默认是 gcc 11.4 和 cmake 3.22。`setup_example.sh` 会装 gcc-12，并用华为通用 PyPI 装 `cmake>=3.28,<4`。不要覆盖 `/etc/apt/sources.list`。不要从 pypi.org 拉 cmake。不要装 cmake 4。ORT 的 FetchContent 还依赖 cmake 3 的兼容行为。

某条在环境就绪后仍挂，就保持诚实红，不把它藏进 `unsupported`。
