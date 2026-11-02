# Quick Start（Ascend NPU）

在单卡 Atlas 900 A2（Ascend 910B）上从源码构建 RTP-LLM 的 Ascend 分支，并用一个 0.6B 稠密模型完成一次端到端推理：源码编译 → wheel 安装 → C++/Python 算子加载 → HTTP 服务 → 一次生成请求。

> RTP-LLM 是阿里云的大模型推理引擎（C++ engine + Python 前端，bazel 构建）。本文针对上游 `feature/ascend` 分支——Ascend 适配目前合入在该特性分支上（PR #1481），尚未进入 `main`，官方验证硬件为 Ascend 950。

## 前置条件

### 硬件

Atlas 900 A2 单卡（Ascend 910B），并按需完成物理机或容器内的设备挂载。

### 基础软件

在跑本文档**之前**，你的机器上需要已经装好并可用：

- Python 3.10 环境（本仓 lock 全部按 `cp310` 生成），且解释器位于 `/opt/conda310/bin/python3`（上游 `deps/pip.bzl` 与该仓的 bazel 配置都写死了这个路径）
- 可用的 CANN ≥ 9.1.0（参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)）
- 基本编译工具：`git`、`curl`、`gcc/g++` 11、`make`

### 本文档示例使用的版本

**配套机器**：

- **机器类型**：Atlas 900 A2 单卡（Ascend 910B）
- **操作系统**：Ubuntu 22.04

**配套镜像**：

swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12

**软件版本**：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10 |
| CANN | 9.1.0 |
| PyTorch / torch_npu | 2.9.0+cpu / 2.9.0.post2 |
| Bazel | 6.4.0 |
| RTP-LLM | \<ref>（源码构建） |
| 模型 | Qwen3-0.6B |

### 检查前置是否满足

检查 Python 版本（应指向 `/opt/conda310/bin/python3`）：

```shell #test id="check-py"
which python
python --version
```

输出结果如下：

```shell #test-result id="check-py" fuzzy='xxx'
/opt/conda310/bin/pythonxxx
Python 3.10.xxx
```

> `xxx` 是版本号与解释器后缀的占位（`python` 或 `python3` 都算通过）。

> 若 `which python` 不是 `/opt/conda310/...`，把该目录放到 `PATH` 最前面：`export PATH=/opt/conda310/bin:$PATH`。路径不能换成别处——bazel 构建阶段取的是写死的 `/opt/conda310/bin/python3`，两者必须一致。

确认能看到 NPU 设备：

```shell
npu-smi info
```

输出为设备列表（型号与数量因机器而异），只要能看到 `910B` 且 `Health` 为 `OK` 即可。

### 安装 bazelisk

构建由 `bazelisk` 拉起，本文固定用 Bazel 6.4.0。镜像里没有预装，先装到 `/usr/local/bin`：

```shell #test id="install-bazelisk"
case $(uname -m) in
  x86_64)  bazelisk_arch=amd64 ;;
  aarch64) bazelisk_arch=arm64 ;;
  *)       echo "unsupported arch: $(uname -m)"; exit 1 ;;
esac
if [ ! -x /usr/local/bin/bazelisk ]; then
  curl -fL --connect-timeout 20 --retry 8 --retry-all-errors --retry-delay 3 \
    -o /usr/local/bin/bazelisk \
    "https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-${bazelisk_arch}"
  chmod +x /usr/local/bin/bazelisk
fi
export USE_BAZEL_VERSION=6.4.0
export BAZELISK_BASE_URL=https://mirrors.huaweicloud.com/bazel/
bazelisk version | grep -m1 'Build label'
```

输出结果如下：

```shell #test-result id="install-bazelisk"
Build label: 6.4.0
```

> 两个变量必须和 `bazelisk` 命令在同一个 shell 里：不指定版本时 `bazelisk` 会去 `www.googleapis.com` 查 `latest`，没外网的机器上直接超时；仓里的 `.bazeliskrc` 只在 `rtp-llm/` 目录内生效。`BAZELISK_BASE_URL` 指向华为云镜像，Bazel 本体（约 60MB）从这里取。

## 获取代码

本文把所有产物放在 `/root/rtp-llm-test` 下（克隆的 `rtp-llm/` 与 `models/`）；换成你自己的目录时，把后文出现的这个前缀一并替换即可。

<!-- 工作流注入的 UPSTREAM_REF（被测分支名）通过这个隐藏的 #test-setup 捕获并注入到下方 clone 命令中 -->

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆 RTP-LLM 并切到被测分支（本仓看护锚定 `feature/ascend`）：

```shell #test id="clone-repo" load="upstream_ref>>ref"
git clone --depth 1 --branch <ref> https://github.com/alibaba/rtp-llm.git
cd rtp-llm
echo "branch $(git rev-parse --abbrev-ref HEAD)"
echo "HEAD $(git log -1 --format=%h)"
```

输出结果如下：

```shell #test-result id="clone-repo" fuzzy='xxx'
branch xxx
HEAD xxx
```

## 环境准备

### 适配 aarch64 宿主的构建输入

上游这份分支的构建输入是按 x86_64 宿主写的，在 aarch64 上直接构建会因为「x86 轮子 + 平台不符」在很靠前的位置失败。下面几处改动把轮子与版本钉到 aarch64 可用的组合；**x86 宿主不需要这一步**。

```shell #test-setup
cd rtp-llm

# 1) 运行时依赖清单：torch 直链换 aarch64，torch_npu 钉到与本镜像配套的 post2
cat > deps/requirements_ascend_x86.txt <<'REQ'
-r requirements_base.txt
https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
torch_npu==2.9.0.post2
torchvision==0.24.0
numpy==1.26.4
REQ

# 2) lock 里的两条直链同步改写（其余 4400 行是 == + 全平台 hash，aarch64 能命中）
mkdir -p /opt/rtp_wheels
curl -fsSL -o /opt/rtp_wheels/torch-2.9.0+cpu-cp310-cp310-manylinux_2_28_aarch64.whl \
  "https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl"
SHA_T=$(sha256sum /opt/rtp_wheels/torch-2.9.0+cpu-*.whl | awk '{print $1}')
test -n "$SHA_T"
sed -i "s|torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl|torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl|" deps/requirements_lock_ascend_x86.txt
sed -i "s|bd2a257e670ede9fc01c6d76dccdc473040913b8e9328169bf177dbdc38e2484|$SHA_T|" deps/requirements_lock_ascend_x86.txt
sed -i "s|https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-beta.1-pytorch2.9.0/torch_npu-2.9.0.post3-cp310-cp310-manylinux_2_28_x86_64.whl|https://mirrors.aliyun.com/pypi/packages/97/d8/685fc488350ba63e158620d3ea04367c908456f03a01f9a9cc158fec380b/torch_npu-2.9.0.post2-cp310-cp310-manylinux_2_28_aarch64.whl|" deps/requirements_lock_ascend_x86.txt
sed -i "s|3d436f74af70089b4320e6dae374641ddfb76712cdd593e67763ded33cf31a2a|727d97f5436161a94035ae00002d657369664e1dd397141db9d8fd54912f346b|" deps/requirements_lock_ascend_x86.txt

# 3) wheel 元数据里的 torch_npu 版本（漏了这处，装 wheel 时会把 post2 降级）
sed -i 's/"torch_npu==2\.9\.0"/"torch_npu==2.9.0.post2"/' arch_config/arch_select.bzl

# 4) 上游依赖死结：transformers==5.2.0 与 xgrammar==0.2.5（要求 transformers<5）无法同时满足
sed -i 's/^xgrammar==0\.2\.5$/xgrammar==0.2.8/' deps/requirements_base.txt

# 5) 校验：aarch64 直链 2 条、x86 直链 0 条、gitcode 0 条、旧 hash 0 条
test "$(grep -c 'manylinux_2_28_aarch64.whl' deps/requirements_lock_ascend_x86.txt)" = 2
test "$(grep -c 'manylinux_2_28_x86_64.whl' deps/requirements_lock_ascend_x86.txt)" = 0
test "$(grep -c 'gitcode.com' deps/requirements_lock_ascend_x86.txt)" = 0
test "$(grep -c '3d436f74' deps/requirements_lock_ascend_x86.txt)" = 0
echo "aarch64 build inputs patched"
```

> 第 5 步的四条 `test` 就是判据：`2 / 0 / 0 / 0`。若第一条不是 2，说明直链没替换成功；若 `$SHA_T` 为空（轮子没下载到），`sed` 会把 hash 替换成空串，第 4 条校验会拦住。

### 放宽 Ubuntu 默认开启的警告

Ubuntu 的 gcc 默认打开 `-Wformat-security`，叠加仓里的 `-Werror` 会把第三方头文件里的一处 `snprintf` 用法升成硬错误。把它降级为警告（保留提示，不中断构建）：

```shell #test-setup
cd rtp-llm
sed -i '/^build:ascend --copt="-D_GLIBCXX_USE_CXX11_ABI=1"$/a build:ascend --copt="-Wno-error=format-security"' .bazelrc
grep -n "format-security" .bazelrc
```

### 安装运行时依赖

```shell #test id="install-deps"
cd rtp-llm
python -m pip install -r deps/requirements_ascend_x86.txt \
  --index-url https://mirrors.aliyun.com/pypi/simple/ > /tmp/pip-deps.log 2>&1 \
  || { tail -n 60 /tmp/pip-deps.log; exit 1; }
python - <<'PY'
from importlib.metadata import version

for name in ('torch', 'torch-npu', 'torchvision', 'numpy'):
    print(name.replace('-', '_'), version(name))
PY
```

输出结果如下：

```shell #test-result id="install-deps"
torch 2.9.0+cpu
torch_npu 2.9.0.post2
torchvision 0.24.0
numpy 1.26.4
```

> 装完务必复核这四行：`torch 2.9.0+cpu`、`torch-npu 2.9.0.post2`、`torchvision 0.24.0`、`numpy 1.26.4`。任何一次 `pip install` 漏掉 `--no-deps` 都可能把 `torchvision` 解析到最新版（对应更新的 torch），随后 `import torchvision` 会报 `operator torchvision::nms does not exist`。

## 构建

```shell #test id="build"
cd rtp-llm
export USE_BAZEL_VERSION=6.4.0
export BAZELISK_BASE_URL=https://mirrors.huaweicloud.com/bazel/
# 钳制并行度：runner 核数高，--jobs=$(nproc) 会因单个 TU 峰值内存过大被 OOM killer 杀掉，
# 而且报出来的是 signal 9 不是编译错误。本机内存更大可上调。
bazelisk build //rtp_llm:rtp_llm_whl --config=ascend --jobs=32 > /tmp/build.log 2>&1 \
  && echo "build ok" || { tail -n 80 /tmp/build.log; exit 1; }
```

输出结果如下：

```shell #test-result id="build"
build ok
```

验证 wheel 产物：

```shell #test id="verify-whl"
cd rtp-llm
ls bazel-bin/rtp_llm/*.whl > /dev/null && echo "whl ok"
```

输出结果如下：

```shell #test-result id="verify-whl"
whl ok
```

> - 首次构建要拉取全部 bazel 外部依赖（grpc / abseil / googletest / havenask / protobuf 等）并编译一万多个动作，冷构建 40–90 分钟属正常。
> - 日志里成片的 `alibaba-aios-dev.oss-cn-hangzhou.aliyuncs.com ... 403 Forbidden` 是**噪声**：仓里的 `bazel/bazel_downloader.cfg` 把所有 github 下载重写到阿里内网 OSS，规则表的第二条会回退到真实地址，下载继续。
> - 如果 fetch 阶段长时间停在 `Fetching repository @grpc ... (shallow fetch failed, fetching full history)`，是 bazel 的 `--shallow-since` 对很老的 `shallow_since` 退化成全量克隆，见文末「常见问题」。

## 安装 wheel 并加载算子

```shell #test id="install-whl"
cd rtp-llm
python -m pip install --no-deps --force-reinstall bazel-bin/rtp_llm/*.whl > /tmp/whl-install.log 2>&1 \
  && echo "whl installed" || { tail -n 40 /tmp/whl-install.log; exit 1; }
```

输出结果如下：

```shell #test-result id="install-whl"
whl installed
```

> `--no-deps` 是必须的：依赖已在上一步按 lock 装好，让 pip 再解析一遍依赖树很可能把 torch 或 torch_npu 换成别的版本。

自检 C++ 算子是否能加载（注意 `libth_transformer` 必须由 `_load_engine_ops()` 按 `librtp_compute_ops` → `libth_transformer` 的顺序加载，直接 `import libth_transformer` 会在进程退出时崩）：

```shell #test id="ops-load"
cd /tmp
python - <<'PY' > /tmp/ops-load.log 2>&1
from rtp_llm.ops import ensure_engine_ops_loaded

ensure_engine_ops_loaded()
print('OPS_LOAD_OK')
PY
grep -q 'OPS_LOAD_OK' /tmp/ops-load.log && echo "engine ops ok" \
  || { tail -n 40 /tmp/ops-load.log; exit 1; }
```

输出结果如下（`rtp_llm.ops` 导入时会在 stdout 打一行耗时，所以这里用标记串断言，不比对全文）：

```shell #test-result id="ops-load"
engine ops ok
```

> `libth_transformer.so` 依赖一批带内容 hash 的库（`libarm_compute-*.so`、`libgomp-*.so` 等），它们来自 **torch 自己的 `torch.libs` 目录**，不是漏打包。`import torch` 后这些 SONAME 会进入进程的已加载集合，因此顺序必须是「先 torch、再 engines ops」——`ensure_engine_ops_loaded()` 内部就是这么做的。

## 下载模型

本文用一个 0.6B 稠密模型：它是该分支当前唯一声明支持并验证过的模型族（Qwen3）。

```shell #test-setup
cd /tmp
python -m pip install -q modelscope > /dev/null 2>&1
# modelscope 收尾时会 import torch，在 torch_npu 环境下析构可能报 Segmentation fault（rc=139），
# 而下载本身已经完成，所以这里不看 rc，改为校验产物。
python - <<'PY' > /tmp/model-dl.log 2>&1 || true
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='/root/rtp-llm-test/models/Qwen3-0.6B')
print('model ready')
PY
ls /root/rtp-llm-test/models/Qwen3-0.6B/config.json \
  /root/rtp-llm-test/models/Qwen3-0.6B/*.safetensors > /dev/null
python - <<'PY'
from importlib.metadata import version

assert version('torch-npu') == '2.9.0.post2'
assert version('torchvision') == '0.24.0'
PY
```

## 启动服务并调用

```shell #test id="serve"
cd /tmp
export ASCEND_RT_VISIBLE_DEVICES=0
python -m rtp_llm.start_server \
  --checkpoint_path=/root/rtp-llm-test/models/Qwen3-0.6B \
  --model_type=qwen_3 \
  --start_port=30000 > /tmp/server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for _ in $(seq 1 180); do
  curl -sf http://127.0.0.1:30000/health > /dev/null 2>&1 && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died"; tail -n 60 /tmp/server.log; exit 1; }
  sleep 2
done

echo "health: $(curl -s http://127.0.0.1:30000/health)"
curl -s -XPOST http://127.0.0.1:30000 -H 'Content-Type: application/json' \
  -d '{"prompt":"hello, what is your name","generate_config":{"max_new_tokens":16}}' > /tmp/gen.json
python - <<'PY'
import json

with open('/tmp/gen.json', encoding='utf-8') as fh:
    data = json.load(fh)
aux = data['aux_info']
print('has_response', bool(data.get('response')))
print('output_len_positive', aux['output_len'] > 0)
print('finished', data.get('finished'))
PY
```

输出结果如下：

```shell #test-result id="serve"
health: "ok"
has_response True
output_len_positive True
finished True
```

> - `/health` 返回的是带引号的 JSON 字符串 `"ok"`，不是 `{"status":"ok"}`。
> - `output_len` 的具体值取决于采样结果（可能在 `max_new_tokens` 之前遇到 EOS），所以这里只断言「大于 0」，不断言具体 token 数。
> - 服务在同一个块内启动并回收（`trap ... EXIT`），避免依赖跨块的常驻进程。

## 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `is not a supported wheel on this platform` | 构建输入还是 x86 轮子（未做「适配 aarch64 宿主」一节） | 按该节重写 requirements 与 lock 的两条直链 |
| fetch 阶段卡在 `@grpc ... full history` | bazel 的 `--shallow-since` 对老时间戳退化成全量克隆 | 等它跑完，或本地预建只含目标 commit 的镜像后按仓 `insteadOf` 指向本地 |
| `-Werror=format-security` | Ubuntu 默认开启该 warning × 仓里 `-Werror` | 已在「放宽 Ubuntu 默认开启的警告」一节处理；其他 `-Werror=xxx` 同理降级 |
| `operator torchvision::nms does not exist` | torchvision 版本与 torch 不匹配 | 复核「安装运行时依赖」末尾的四行版本 |
| `malloc_consolidate(): invalid chunk size` | 绕过顺序直接 `import libth_transformer` | 用 `ensure_engine_ops_loaded()` |
| `THESE PACKAGES DO NOT MATCH THE HASHES` | lock 里 hash 与下载到的轮子不符 | 重做「适配 aarch64 宿主」第 2 步，确认三个计数为 `2`/`0`/`0` |
