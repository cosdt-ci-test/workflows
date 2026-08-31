# 昇腾 Quick Start：AIBrix local mode + vLLM-Ascend

本文介绍如何在一台 Atlas 900 A2 / Ascend 910B（aarch64，CANN 9.1）机器上完成快速开始。先用 vLLM-Ascend 拉起一个小模型的 OpenAI 兼容服务，再用 AIBrix 的 local mode（使用本机 Envoy + Go 网关，不需要 Kubernetes）转发一次 chat completion。

## 硬件与 CANN

确认驱动可用、设备可见（`npu-smi` 通常装在 `/usr/local/sbin`）：

```shell
export PATH="/usr/local/sbin:/usr/local/bin:$PATH"
npu-smi info
```

加载 CANN 与 ATB 环境，后面的安装和启动步骤都依赖它：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh
```

## 系统工具

`run-local.sh` 用 `ss` 探测网关端口，没有就装 `iproute2`：

```shell #test id="install-system-prereqs"
if ! command -v ss >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y iproute2
fi
ss --version
```

输出结果如下：

```shell #test-result id="install-system-prereqs"
...ss utility, iproute2-...
```



## 安装 Go 1.22.6

Go 1.22.6 解压到 `.aibrix-quick-start/toolchain/go`。

```shell #test id="install-go"
mkdir -p .aibrix-quick-start/toolchain
curl -fL --connect-timeout 20 --retry 5 --retry-delay 3 --max-time 180 \
  -o .aibrix-quick-start/go.tar.gz \
  https://mirrors.aliyun.com/golang/go1.22.6.linux-arm64.tar.gz
tar -C .aibrix-quick-start/toolchain -xzf .aibrix-quick-start/go.tar.gz
.aibrix-quick-start/toolchain/go/bin/go version
```

输出结果如下：

```shell #test-result id="install-go"
go version go1.22.6 linux/arm64
```



## 安装 Envoy 1.39.0

local mode 经 Envoy 监听 `:10080`。从 GitHub Release 下载官方 aarch64 包：

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/cosdt-ci-test/aibrix'
cached="$ci/envoy/1.39.0/envoy"
sum='ee53a4f5375566f15944dc9cb03afb1fc228df38f61737c677f139213215afcf'
url='https://gh-proxy.test.osinfra.cn/https://github.com/envoyproxy/envoy/releases/download/v1.39.0/envoy-1.39.0-linux-aarch_64'
mkdir -p .aibrix-quick-start/bin "$ci/envoy/1.39.0"
if [ -f "$cached" ] && ! echo "$sum  $cached" | sha256sum -c >/dev/null 2>&1; then
  rm -f "$cached"
fi
if [ ! -f "$cached" ]; then
  (
    flock 9
    if [ ! -f "$cached" ]; then
      curl -fL --connect-timeout 20 --retry 3 --retry-all-errors --retry-delay 3 --max-time 600 \
        -o "$cached.part" "$url"
      echo "$sum  $cached.part" | sha256sum -c
      mv "$cached.part" "$cached"
    fi
  ) 9> "$ci/envoy/1.39.0/.lock"
fi
cp -a "$cached" .aibrix-quick-start/bin/envoy
chmod 0755 .aibrix-quick-start/bin/envoy
```
-->

```shell #test id="install-envoy"
mkdir -p .aibrix-quick-start/bin
if [ ! -x .aibrix-quick-start/bin/envoy ]; then
  curl -fL --connect-timeout 20 --retry 8 --retry-all-errors --retry-delay 3 --max-time 300 \
    -C - -o .aibrix-quick-start/bin/envoy.part \
    https://github.com/envoyproxy/envoy/releases/download/v1.39.0/envoy-1.39.0-linux-aarch_64
  mv .aibrix-quick-start/bin/envoy.part .aibrix-quick-start/bin/envoy
  chmod +x .aibrix-quick-start/bin/envoy
fi
.aibrix-quick-start/bin/envoy --version
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/cosdt-ci-test/aibrix'
src='.aibrix-quick-start/bin/envoy'
sum='ee53a4f5375566f15944dc9cb03afb1fc228df38f61737c677f139213215afcf'
echo "$sum  $src" | sha256sum -c
mkdir -p "$ci/envoy/1.39.0"
if [ ! -f "$ci/envoy/1.39.0/envoy" ]; then
  cp -a "$src" "$ci/envoy/1.39.0/envoy.part"
  mv "$ci/envoy/1.39.0/envoy.part" "$ci/envoy/1.39.0/envoy"
fi
```
-->

输出结果如下：

```shell #test-result id="install-envoy"
...1.39.0...
```



## 克隆 AIBrix 并编译网关

<!-- 工作流注入的 UPSTREAM_REF（最新 release tag）通过这个隐藏的 #test-setup 捕获；markdown 渲染器会丢掉注释，读者看不到，runner 仍会执行 -->
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

克隆 release tag（`<ref>` 可改为例如 `v0.7.0`）。只用下面的 `go build`，勿用 `make build-gateway-plugins-nozmq`（会跑全仓 `manifests generate fmt vet`）。

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/cosdt-ci-test/aibrix'
ref="${UPSTREAM_REF}"
printf '%s\n' "$ref" | grep -Eq '^[A-Za-z0-9._/-]+$'
cached="$ci/src/aibrix-${ref}"
dest='.aibrix-quick-start/aibrix'
if [ -d "$cached/.git" ]; then
  got=$(git -C "$cached" describe --tags --exact-match 2>/dev/null || true)
  if [ "$got" = "$ref" ]; then
    mkdir -p .aibrix-quick-start
    rm -rf "$dest"
    cp -a "$cached" "$dest"
  else
    rm -rf "$cached"
  fi
fi
```
-->

```shell #test id="clone-aibrix" load="upstream_ref>>ref"
if [ ! -d .aibrix-quick-start/aibrix/.git ]; then
  rm -rf .aibrix-quick-start/aibrix
  for _ in 1 2 3; do
    GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1 \
      git clone --depth 1 --branch <ref> https://github.com/vllm-project/aibrix.git .aibrix-quick-start/aibrix && break
    rm -rf .aibrix-quick-start/aibrix
    sleep 5
  done
fi
git -C .aibrix-quick-start/aibrix describe --tags --exact-match
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/cosdt-ci-test/aibrix'
ref="${UPSTREAM_REF}"
printf '%s\n' "$ref" | grep -Eq '^[A-Za-z0-9._/-]+$'
src='.aibrix-quick-start/aibrix'
cached="$ci/src/aibrix-${ref}"
if [ -d "$src/.git" ] && [ ! -d "$cached/.git" ]; then
  mkdir -p "$ci/src"
  rm -rf "${cached}.part"
  cp -a "$src" "${cached}.part"
  mv "${cached}.part" "$cached"
fi
```
-->

输出结果如下：

```shell #test-result id="clone-aibrix" load="upstream_ref>>ref"
<ref>
```

`GOPROXY` 指向国内代理（直连 `proxy.golang.org` 在大陆网络经常拉不下依赖）；`GOPATH` / `GOCACHE` 落在工作目录内，便于事后整体清理。

```shell #test id="build-gateway"
export GOPROXY=https://goproxy.cn,direct
export GOPATH="$PWD/.aibrix-quick-start/gopath"
export GOCACHE="$PWD/.aibrix-quick-start/gocache"
mkdir -p "$GOPATH" "$GOCACHE"
cd .aibrix-quick-start/aibrix
CGO_ENABLED=0 "$PWD/../toolchain/go/bin/go" build -tags=nozmq -o bin/gateway-plugins cmd/plugins/main.go
"$PWD/../toolchain/go/bin/go" version -m bin/gateway-plugins
```

输出结果如下（首行是二进制内嵌的 Go 版本，其余为构建信息）：

```shell #test-result id="build-gateway"
bin/gateway-plugins: go1.22.6
...
```

<!-- CI/coder 在 127.0.0.1:6060 已被占用时，只给 gateway-plugins 套一层 hosts remap。vLLM 不能进这个 namespace。用户机器上 6060 空闲时这里是 no-op。 -->
<!--
```shell #test-setup
set -euo pipefail
plugin="$PWD/.aibrix-quick-start/aibrix/bin/gateway-plugins"
if python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 6060))' >/dev/null 2>&1; then
  echo pprof_port_free
  exit 0
fi
test -x "$plugin"
real="$plugin.real"
mv "$plugin" "$real"
cat > "$plugin" <<'WRAP'
#!/usr/bin/env bash
set -euo pipefail
hosts=$(mktemp)
{
  printf '%s\t%s\n' '127.0.0.2' 'localhost'
  printf '%s\t%s\n' '127.0.0.1' "$(hostname)"
} > "$hosts"
exec unshare --user --map-root-user --mount --fork -- \
  bash -c 'mount --bind "$1" /etc/hosts; shift; exec "$@"' \
  bash "$hosts" \
  REAL_PLACEHOLDER "$@"
WRAP
sed -i "s|REAL_PLACEHOLDER|${real}|g" "$plugin"
chmod +x "$plugin"
echo pprof_port_wrapped
```
-->

## 安装 vLLM-Ascend

分步安装，勿合并为一次 `pip install`：先钉 torch / torch-npu 2.10，再 `VLLM_TARGET_DEVICE=empty` 装 vLLM 0.23.0 源码（避免官方 wheel 把 torch 升到 2.11），最后 `vllm-ascend==0.23.0`。

本步骤依赖第一节加载的 CANN/ATB 环境（构建 `vllm-ascend` 时会加载 `torch_npu`，找不到 `libhccl.so` 会直接失败）。`VLLM_USE_MODELSCOPE=True` 留到启动服务再设。pip 默认走清华镜像；当前环境已设 `PIP_INDEX_URL` 时沿用已有镜像。

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/cosdt-ci-test/aibrix'
cached="$ci/src/vllm-v0.23.0"
dest='.aibrix-quick-start/src/vllm'
if [ -d "$cached/.git" ]; then
  mkdir -p .aibrix-quick-start/src
  rm -rf "$dest"
  cp -a "$cached" "$dest"
fi
```
-->

```shell #test id="install-vllm"
python3 -m venv .aibrix-quick-start/venv
.aibrix-quick-start/venv/bin/python -m pip install -U pip
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=5
.aibrix-quick-start/venv/bin/python -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/ \
  torch==2.10.0 torch-npu==2.10.0.post4 torchvision==0.25.0 torchaudio==2.10.0 triton-ascend==3.2.2
.aibrix-quick-start/venv/bin/python -m pip install 'cmake>=3.26' nanobind ninja setuptools-rust wheel 'setuptools-scm>=8' 'setuptools>=77,<81'
if [ ! -d .aibrix-quick-start/src/vllm/.git ]; then
  rm -rf .aibrix-quick-start/src/vllm
  for _ in 1 2 3; do
    GIT_TERMINAL_PROMPT=0 GIT_HTTP_VERSION=HTTP/1.1 \
      git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git .aibrix-quick-start/src/vllm && break
    rm -rf .aibrix-quick-start/src/vllm
    sleep 5
  done
fi
export VLLM_TARGET_DEVICE=empty
.aibrix-quick-start/venv/bin/python -m pip install --no-build-isolation -e .aibrix-quick-start/src/vllm
.aibrix-quick-start/venv/bin/python -m pip install grpcio-tools
export CMAKE_PREFIX_PATH="$(.aibrix-quick-start/venv/bin/python -m nanobind --cmake_dir)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export Python_EXECUTABLE="$PWD/.aibrix-quick-start/venv/bin/python"
export PYTHON_EXECUTABLE="$PWD/.aibrix-quick-start/venv/bin/python"
.aibrix-quick-start/venv/bin/python -m pip install --no-build-isolation \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  vllm-ascend==0.23.0
.aibrix-quick-start/venv/bin/python -m pip install modelscope==1.31.0
.aibrix-quick-start/venv/bin/python -c "import importlib.metadata as m
for n in ['torch', 'torch-npu', 'vllm', 'vllm-ascend', 'modelscope']:
    print(n, m.version(n))"
```

<!--
```shell #test-setup
set -euo pipefail
ci='/root/.cache/cosdt-ci-test/aibrix'
src='.aibrix-quick-start/src/vllm'
cached="$ci/src/vllm-v0.23.0"
if [ -d "$src/.git" ] && [ ! -d "$cached/.git" ]; then
  mkdir -p "$ci/src"
  rm -rf "${cached}.part"
  cp -a "$src" "${cached}.part"
  mv "${cached}.part" "$cached"
fi
```
-->

输出结果如下：

```shell #test-result id="install-vllm"
...
torch 2.10.0
torch-npu 2.10.0.post4
vllm 0.23.0+empty
vllm-ascend 0.23.0
modelscope 1.31.0
```



## 启动 vLLM-Ascend 后端

后台启动服务、日志落盘，并轮询 `/health` 直到就绪（模型首次运行会从 ModelScope 下载，需等待数分钟）：

```shell #test-setup
export PYTHONUNBUFFERED=1
export VLLM_USE_MODELSCOPE=True
mkdir -p .aibrix-quick-start
: > .aibrix-quick-start/vllm.log
setsid bash -c '
  echo $$ > .aibrix-quick-start/vllm.pid
  exec .aibrix-quick-start/venv/bin/vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --served-model-name Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 2048 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.2
' </dev/null >> .aibrix-quick-start/vllm.log 2>&1 &
for i in $(seq 1 180); do
  pid=$(cat .aibrix-quick-start/vllm.pid 2>/dev/null || true)
  if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
    echo 'vLLM exited before /health succeeded. Full log:'
    cat .aibrix-quick-start/vllm.log
    exit 1
  fi
  if [ -n "${pid}" ] \
      && curl -sf --connect-timeout 2 -- 'http://127.0.0.1:8000/health' >/dev/null \
      && ss -ltnp 'sport = :8000' | grep -q "pid=${pid},"; then
    echo 'vLLM /health OK'
    exit 0
  fi
  sleep 2
done
echo 'timed out waiting for http://127.0.0.1:8000/health. Full log:'
cat .aibrix-quick-start/vllm.log
exit 1
```

`--max-model-len` / `--max-num-seqs` / `--gpu-memory-utilization` 压到单卡演示规模。`--served-model-name` 须与 `endpoints.yaml` 一致，否则 Envoy 报 `no healthy upstream`。

`VLLM_USE_MODELSCOPE=True` 从 ModelScope 拉权重；`PYTHONUNBUFFERED=1` 让日志实时写入文件，便于观察启动进度。日志/PID 在 `.aibrix-quick-start/`。`/health` 200 不能排除 8000 被占；块内用 `ss` 核对 PID。日志里应有 `backend=hccl`。

```shell #test id="backend-on-npu"
grep -F 'backend=hccl' .aibrix-quick-start/vllm.log
```

输出结果如下：

```shell #test-result id="backend-on-npu"
...backend=hccl...
```



## 配置网关并启动 local mode

模型名与 `--served-model-name` 一致；endpoint 用 `127.0.0.1:8000`。

```shell #test id="configure-endpoints"
cat > .aibrix-quick-start/endpoints.yaml <<'YAML'
models:
  - name: "Qwen/Qwen2.5-0.5B-Instruct"
    engine: "vllm"
    endpoints:
      - "127.0.0.1:8000"
YAML
cat .aibrix-quick-start/endpoints.yaml
```

输出结果如下：

```shell #test-result id="configure-endpoints"
models:
  - name: "Qwen/Qwen2.5-0.5B-Instruct"
    engine: "vllm"
    endpoints:
      - "127.0.0.1:8000"
```

`run-local.sh` 从 PATH 找 `envoy`；`endpoints.yaml` 须绝对路径。

```shell #test id="start-gateway"
export PATH="$PWD/.aibrix-quick-start/bin:$PATH"
bash .aibrix-quick-start/aibrix/deployment/local/run-local.sh \
  -e "$PWD/.aibrix-quick-start/endpoints.yaml"
```

输出结果如下：

```shell #test-result id="start-gateway"
...
AIBrix gateway is running!
...
```



## 发一次推理

经 `:10080` 发 chat completion；生成内容非确定性，勿对原文。

```shell #test id="infer"
.aibrix-quick-start/venv/bin/python - <<'PY'
import json
import urllib.request

payload = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct',
    'messages': [{'role': 'user', 'content': 'Say hi in one sentence.'}],
    'max_tokens': 32,
    'temperature': 0,
}
request = urllib.request.Request(
    'http://127.0.0.1:10080/v1/chat/completions',
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(request, timeout=120) as response:
    body = json.load(response)
content = (body['choices'][0]['message']['content'] or '').strip()
print('model', body['model'])
print('content_nonempty', 'true' if content else 'false')
print('completion_tokens', body['usage']['completion_tokens'])
PY
```

输出结果如下：

```shell #test-result id="infer"
model Qwen/Qwen2.5-0.5B-Instruct
content_nonempty true
completion_tokens ...
```



## 停掉本机进程

先 `stop-local.sh`，再按 `.aibrix-quick-start/vllm.pid` 停 vLLM。

```shell #test-setup
if [ -x .aibrix-quick-start/aibrix/deployment/local/stop-local.sh ]; then
  bash .aibrix-quick-start/aibrix/deployment/local/stop-local.sh || true
fi
if [ -f .aibrix-quick-start/vllm.pid ]; then
  pid=$(cat .aibrix-quick-start/vllm.pid)
  if [ -n "${pid}" ]; then
    kill "${pid}" 2>/dev/null || true
  fi
fi
```

