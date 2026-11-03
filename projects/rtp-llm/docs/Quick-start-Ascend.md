# RTP-LLM

RTP-LLM 是阿里云的大模型推理引擎，C++ 引擎加 Python 前端，用 bazel 构建。本文在单张昇腾 NPU 上从源码构建它的 Ascend 分支，安装产出的 wheel，启动 HTTP 服务并完成一次生成请求。

Ascend 适配目前在上游的 `feature/ascend` 分支上，`main` 里还没有。本文的命令都在容器内以 root 执行，工作目录统一用 `/root/rtp-llm-test`。

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列，芯片为 Ascend **910B**。本文示例为单卡。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit 与驱动固件已安装，并可 `source set_env.sh`，9.1.0 或以上 |
| gcc / g++ | 11 |
| Python | 3.10，须落在 `/opt/conda310`，见第 2.3 节 |
| PyTorch | `torch` 2.9.0 的 CPU 构建版 + `torch_npu` 2.9.0.post2，见第 4 节 |
| Bazel | 6.4.0，由 `bazelisk` 拉起，见第 2.2 节 |
| 模型 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |

### 本文验证环境

| 项目 | 内容 |
| --- | --- |
| 机器 | Atlas 900 A2，Ascend 910B，单卡 |
| 镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` |
| Python | 3.10.12，用发行版自带的 3.10 建 venv 到 `/opt/conda310` |
| torch / torch_npu | 2.9.0+cpu / 2.9.0.post2 |
| Bazel | 6.4.0 |

阅读本文前，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。

## 1. 加载 CANN 环境

加载 CANN，并把 `/usr/local/sbin` 和 `/usr/local/bin` 加入 `PATH`。

```shell
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

输出为设备列表，型号与数量因机器而异，至少一张卡且 `Health` 为 `OK` 即可。若提示找不到 `npu-smi`，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载。

### 2.2 安装 bazelisk

构建由 `bazelisk` 拉起，Bazel 固定用 6.4.0。发行版通常不带 `bazelisk`，装到 `/usr/local/bin`。

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

```{admonition} Note
:class: note
`USE_BAZEL_VERSION` 与 `BAZELISK_BASE_URL` 要和 `bazelisk` 在同一次调用里生效：不指定版本时 `bazelisk` 会去查 `latest`，国内直连常超时；镜像地址让它从华为云取 Bazel 本体。仓里的 `.bazeliskrc` 只在 `rtp-llm/` 目录内生效。
```

### 2.3 准备 Python 3.10

上游把解释器路径写死成 `/opt/conda310/bin/python3`（`deps/pip.bzl`、`.bazelrc`、`WORKSPACE` 等多处引用），所以 3.10 必须装在这个路径下。用发行版自带的 3.10 建 venv 即可。

```shell #test id="prepare-python310"
if [ ! -x /opt/conda310/bin/python3 ]; then
  apt-get update -qq
  apt-get install -y -qq python3.10 python3.10-venv python3.10-dev
  python3.10 -m venv /opt/conda310
  /opt/conda310/bin/python3 -m pip install -q --upgrade pip
  # //3rdparty/py 的 genrule 直接从这个前缀拷 libpython3.10.so，venv 自身不复制
  mkdir -p /opt/conda310/lib
  ln -sf /usr/lib/aarch64-linux-gnu/libpython3.10.so /opt/conda310/lib/libpython3.10.so
fi
export PATH=/opt/conda310/bin:$PATH
python --version
```

输出结果如下，其中 `xxx` 表示实际的补丁版本号：

```shell #test-result id="prepare-python310" fuzzy='xxx'
Python 3.10.xxx
```

```{admonition} Note
:class: note
不要用 conda 建这个环境。conda 自带的 `libstdc++` 最高只到 `GLIBCXX_3.4.29`，而编译产物要求 `3.4.30`，加载时会报 `version GLIBCXX_3.4.30 not found`；venv 直接用系统那份库，没有这个问题。
```

## 3. 获取代码

克隆上游的 `feature/ascend` 分支。

```shell #test id="clone-repo"
cd /root/rtp-llm-test
rm -rf rtp-llm
git clone --depth 1 --branch feature/ascend https://github.com/alibaba/rtp-llm.git
cd rtp-llm
echo "branch $(git rev-parse --abbrev-ref HEAD)"
echo "HEAD $(git log -1 --format=%h)"
```

输出结果如下，其中 `xxx` 表示实际 commit 的短哈希：

```shell #test-result id="clone-repo" fuzzy='xxx'
branch feature/ascend
HEAD xxx
```

## 4. 适配 aarch64 构建输入

上游这套构建输入是按 x86_64 宿主写的：运行时依赖清单、lock 里的直链、bazel 自己的两个 `http_archive` 全都指向 x86 轮子。不改的话 pip 要么直接拒装，要么一路跑到链接阶段才报 `cannot find -ltorch_npu`。**x86 宿主跳过本节。**

```shell #test id="patch-aarch64"
cd /root/rtp-llm-test/rtp-llm

# 1) 运行时依赖清单：torch 换 aarch64 直链，torch_npu 钉到与本镜像配套的 post2
cat > deps/requirements_ascend_x86.txt <<'REQ'
-r requirements_base.txt
https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
torch_npu==2.9.0.post2
torchvision==0.24.0
numpy==1.26.4
REQ

# 2) lock 里的两条直链同步改写（其余行是版本号 + 全平台 hash，aarch64 能命中）
mkdir -p /opt/rtp_wheels
curl -fsSL -o /opt/rtp_wheels/torch-2.9.0+cpu-cp310-cp310-manylinux_2_28_aarch64.whl \
  "https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl"
SHA_T=$(sha256sum /opt/rtp_wheels/torch-2.9.0+cpu-*.whl | awk '{print $1}')
test -n "$SHA_T"
sed -i "s|torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64.whl|torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl|" deps/requirements_lock_ascend_x86.txt
sed -i "s|bd2a257e670ede9fc01c6d76dccdc473040913b8e9328169bf177dbdc38e2484|$SHA_T|" deps/requirements_lock_ascend_x86.txt
sed -i "s|https://gitcode.com/Ascend/pytorch/releases/download/v26.1.0-beta.1-pytorch2.9.0/torch_npu-2.9.0.post3-cp310-cp310-manylinux_2_28_x86_64.whl|https://mirrors.aliyun.com/pypi/packages/97/d8/685fc488350ba63e158620d3ea04367c908456f03a01f9a9cc158fec380b/torch_npu-2.9.0.post2-cp310-cp310-manylinux_2_28_aarch64.whl|" deps/requirements_lock_ascend_x86.txt
sed -i "s|3d436f74af70089b4320e6dae374641ddfb76712cdd593e67763ded33cf31a2a|727d97f5436161a94035ae00002d657369664e1dd397141db9d8fd54912f346b|" deps/requirements_lock_ascend_x86.txt

# 3) bazel 自己的两个 http_archive（deps/http.bzl）同样是 x86 直链
curl -fsSL -o /opt/rtp_wheels/torch_npu-2.9.0.post2-cp310-cp310-manylinux_2_28_aarch64.whl \
  "https://mirrors.aliyun.com/pypi/packages/97/d8/685fc488350ba63e158620d3ea04367c908456f03a01f9a9cc158fec380b/torch_npu-2.9.0.post2-cp310-cp310-manylinux_2_28_aarch64.whl"
SHA_N=$(sha256sum /opt/rtp_wheels/torch_npu-2.9.0.post2-*.whl | awk '{print $1}')
test "$SHA_N" = 727d97f5436161a94035ae00002d657369664e1dd397141db9d8fd54912f346b
sed -i "s|torch-2\.9\.0%2Bcpu-cp310-cp310-manylinux_2_28_x86_64\.whl|torch-2.9.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl|g" deps/http.bzl
sed -i -E 's#https://[^"]*torch_npu-2\.9\.0[^"]*manylinux_2_28_x86_64\.whl#https://mirrors.aliyun.com/pypi/packages/97/d8/685fc488350ba63e158620d3ea04367c908456f03a01f9a9cc158fec380b/torch_npu-2.9.0.post2-cp310-cp310-manylinux_2_28_aarch64.whl#g' deps/http.bzl
#    两个 sha256 不写死：先读出文件里的当前值当锚点，再替换（上游改过这些 hash）
SHA_OLD_T=$(grep -A1 'name = "torch_cpu_ascend"' deps/http.bzl | sed -n 's/.*sha256 = "\([0-9a-f]\{64\}\)".*/\1/p')
SHA_OLD_N=$(grep -A1 'name = "torch_npu_ascend"' deps/http.bzl | sed -n 's/.*sha256 = "\([0-9a-f]\{64\}\)".*/\1/p')
test -n "$SHA_OLD_T" -a -n "$SHA_OLD_N"
sed -i "s|$SHA_OLD_T|$SHA_T|g" deps/http.bzl
sed -i "s|$SHA_OLD_N|$SHA_N|g" deps/http.bzl

# 4) wheel 元数据里的 torch_npu 版本（漏了这处，装 wheel 时会把 post2 降级成裸 2.9.0）
sed -i 's/"torch_npu==2\.9\.0"/"torch_npu==2.9.0.post2"/' arch_config/arch_select.bzl

# 5) 上游依赖冲突：transformers==5.2.0 与 xgrammar==0.2.5（要求 transformers<5）无法同时满足
sed -i 's/^xgrammar==0\.2\.5$/xgrammar==0.2.8/' deps/requirements_base.txt

# 6) 校验：lock 里 2 条 aarch64 直链、0 条 x86 直链、0 条 gitcode、0 条旧 hash；
#    http.bzl 里两个 ascend archive 各 1 条 aarch64 直链
test "$(grep -c 'manylinux_2_28_aarch64.whl' deps/requirements_lock_ascend_x86.txt)" = 2
test "$(grep -c 'manylinux_2_28_x86_64.whl' deps/requirements_lock_ascend_x86.txt)" = 0
test "$(grep -c 'gitcode.com' deps/requirements_lock_ascend_x86.txt)" = 0
test "$(grep -c '3d436f74' deps/requirements_lock_ascend_x86.txt)" = 0
test "$(grep -A6 'name = "torch_cpu_ascend"' deps/http.bzl | grep -c 'manylinux_2_28_aarch64.whl')" = 1
test "$(grep -A6 'name = "torch_npu_ascend"' deps/http.bzl | grep -c 'manylinux_2_28_aarch64.whl')" = 1
echo "aarch64 build inputs patched"
```

输出结果如下：

```shell #test-result id="patch-aarch64"
aarch64 build inputs patched
```

```{admonition} Note
:class: note
块内末尾六条 `test` 就是判据。前四条管 lock，依次为 `2 / 0 / 0 / 0`；后两条管 `deps/http.bzl`，各为 `1`。少了后两条时 pip 阶段全绿，直到链接 `libth_transformer.so` 才报 `skipping incompatible ... cannot find -ltorch_npu`。
```

## 5. 安装运行时依赖

按 lock 装依赖，torch 与 torch_npu 的 aarch64 轮子在这一步落地。

```shell #test id="install-deps"
cd /root/rtp-llm-test/rtp-llm
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

```{admonition} Note
:class: note
这四行要逐个对上。之后任何一次 `pip install` 漏掉 `--no-deps`，都可能把 `torchvision` 解析到最新版，`import torchvision` 会报 `operator torchvision::nms does not exist`。
```

## 6. 构建 wheel

### 6.1 放宽 Ubuntu 默认开启的警告

Ubuntu 的 gcc 默认打开 `-Wformat-security`，叠加仓里的 `-Werror` 会把第三方头文件里的一处 `snprintf` 用法升成硬错误。降级为警告即可，提示保留、构建不中断。

```shell #test id="allow-format-security"
cd /root/rtp-llm-test/rtp-llm
if ! grep -q 'Wno-error=format-security' .bazelrc; then
  sed -i '/^build:ascend --copt="-D_GLIBCXX_USE_CXX11_ABI=1"$/a build:ascend --copt="-Wno-error=format-security"' .bazelrc
fi
sed -n '/Wno-error=format-security/p' .bazelrc
```

输出结果如下：

```shell #test-result id="allow-format-security"
build:ascend --copt="-Wno-error=format-security"
```

### 6.2 编译

用 `bazelisk` 构建 wheel，目标 `//rtp_llm:rtp_llm_whl`，`--config=ascend` 选中 Ascend 后端。

```shell #test id="build"
cd /root/rtp-llm-test/rtp-llm
export USE_BAZEL_VERSION=6.4.0
export BAZELISK_BASE_URL=https://mirrors.huaweicloud.com/bazel/
bazelisk build //rtp_llm:rtp_llm_whl --config=ascend --jobs=32 > /tmp/build.log 2>&1 \
  && echo "build ok" || { tail -n 80 /tmp/build.log; exit 1; }
```

输出结果如下：

```shell #test-result id="build"
build ok
```

```{admonition} Note
:class: note
首次构建要拉取全部外部依赖并编译上万个动作，几十分钟属正常；日志里成片的 `alibaba-aios-dev.oss-cn-hangzhou.aliyuncs.com ... 403 Forbidden` 是噪声，规则表会回退到真实地址。`--jobs` 别开满，单个编译单元峰值内存不小，被 OOM killer 杀掉时报的是 signal 9 而不是编译错误。
```

确认产物：

```shell #test id="verify-whl"
cd /root/rtp-llm-test/rtp-llm
ls bazel-bin/rtp_llm/*.whl > /dev/null && echo "whl ok"
```

输出结果如下：

```shell #test-result id="verify-whl"
whl ok
```

## 7. 安装 wheel 并加载算子

### 7.1 安装 wheel

把产出的 wheel 装进当前解释器。

```shell #test id="install-whl"
cd /root/rtp-llm-test/rtp-llm
python -m pip install --no-deps --force-reinstall bazel-bin/rtp_llm/*.whl > /tmp/whl-install.log 2>&1 \
  && echo "whl installed" || { tail -n 40 /tmp/whl-install.log; exit 1; }
```

输出结果如下：

```shell #test-result id="install-whl"
whl installed
```

```{admonition} Note
:class: note
`--no-deps` 是必须的：依赖已按 lock 装好，让 pip 再解析一遍依赖树很可能把 `torch` 或 `torch_npu` 换成别的版本。
```

### 7.2 加载 C++ 算子

自检 wheel 里的 C++ 引擎能否加载：

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

```{admonition} Note
:class: note
`libth_transformer.so` 依赖 `torch.libs` 里一批带内容 hash 的库（`libarm_compute-*.so`、`libgomp-*.so` 等），不是漏打包。`import torch` 之后这些 SONAME 才进入进程的已加载集合，所以顺序必须是先 `torch`、再 engine ops，`ensure_engine_ops_loaded()` 内部就是这么做的。也不要在源码树里直接 `import libth_transformer`：cwd 里的源码 `rtp_llm` 会抢在 wheel 之前，进程退出时还可能崩。
```

## 8. 下载模型

用 0.6B 的稠密模型，Ascend 分支当前声明支持的模型族是 Qwen3。

```shell #test id="download-model"
cd /root/rtp-llm-test
python -m pip install -q modelscope
# modelscope 收尾时会 import torch，在 torch_npu 环境下析构可能报 Segmentation fault（rc=139），
# 而下载本身已经完成，所以这里不看 rc，改为校验产物。
python - <<'PY' > /tmp/model-dl.log 2>&1 || true
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='/root/rtp-llm-test/models/Qwen3-0.6B')
print('model ready')
PY
# modelscope 的依赖解析不能动 torch_npu / torchvision 的版本
python - <<'PY'
from importlib.metadata import version

assert version('torch-npu') == '2.9.0.post2', version('torch-npu')
assert version('torchvision') == '0.24.0', version('torchvision')
PY
ls models/Qwen3-0.6B/config.json models/Qwen3-0.6B/*.safetensors > /dev/null
echo "model ok"
```

输出结果如下：

```shell #test-result id="download-model"
model ok
```

## 9. 启动服务并发一次请求

在 0 号卡上启动 HTTP 服务，端口 `30000`，等 `/health` 就绪后发一条生成请求。

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

```{admonition} Note
:class: note
`/health` 返回的是带引号的 JSON 字符串 `"ok"`，不是 `{"status":"ok"}`。`output_len` 取决于采样结果，可能在 `max_new_tokens` 之前遇到 EOS，所以只断言大于 0。服务在这个块内启动并回收，不依赖跨块的常驻进程。
```

## 10. 更多用法

- 上游仓库：[alibaba/rtp-llm](https://github.com/alibaba/rtp-llm)
- Ascend 分支：[feature/ascend](https://github.com/alibaba/rtp-llm/tree/feature/ascend)
