# 昇腾 Quick Start：AIBrix local mode + vLLM-Ascend

这篇教你在一台 Atlas 900 A2 / Ascend 910B（aarch64，CANN 9.1）机器上做成一件事：先用 vLLM-Ascend 拉起一个小模型的 OpenAI 兼容服务，再用 AIBrix 的 local mode（本机 Envoy + Go 网关，不需要 Kubernetes）转发一次 chat completion。

下面命令默认你已经在一个空目录里，并且会把下载、编译、虚拟环境和日志都放进相对路径 `.aibrix-quick-start/`。每一条都可以单独复制到终端；不要指望前一条的 `cd`、`source` 或 `activate` 还在。

## 硬件与 CANN

先确认 NPU 健康，再加载 CANN 和 ATB（Ascend Transformer Boost）。后面真正启动 vLLM 的命令也会再加载一次，因为新开的 shell 不会继承这里的环境。

```shell
export PATH="/usr/local/sbin:/usr/local/bin:$PATH"
npu-smi info
```

```shell
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh
```

`npu-smi` 和上面两次 `source` 只用来让你对照本机环境。看护不会执行这些无标签块。

## 系统工具

`setsid` 来自 util-linux，一般已经有。`ss` 来自 `iproute2`。AIBrix 的 `run-local.sh` 用它们拉起并探测网关端口。缺 `ss` 时按提示安装。

```shell #test id="install-system-prereqs"
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:$PATH"
if ! command -v ss >/dev/null 2>&1; then
  echo 'ss is missing. Install it with: apt-get update && apt-get install -y iproute2'
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y iproute2
fi
command -v setsid >/dev/null
command -v ss >/dev/null
command -v curl >/dev/null
command -v git >/dev/null
echo setsid_ok
echo ss_ok
echo curl_ok
echo git_ok
```

输出结果如下：

```shell #test-result id="install-system-prereqs"
setsid_ok
ss_ok
curl_ok
git_ok
```

## 安装 Go 1.22.6

网关是 Go 写的。装到 `.aibrix-quick-start/toolchain/go`，后面编译用绝对路径，不依赖系统 `go`。国内用阿里云镜像；SHA256 必须对上。

```shell #test id="install-go"
set -euo pipefail
mkdir -p .aibrix-quick-start/toolchain
curl -fL --connect-timeout 20 --retry 5 --retry-delay 3 --max-time 180 \
  -o .aibrix-quick-start/go.tar.gz \
  https://mirrors.aliyun.com/golang/go1.22.6.linux-arm64.tar.gz
echo 'c15fa895341b8eaf7f219fada25c36a610eb042985dc1a912410c1c90098eaf2  .aibrix-quick-start/go.tar.gz' | sha256sum -c
tar -C .aibrix-quick-start/toolchain -xzf .aibrix-quick-start/go.tar.gz
.aibrix-quick-start/toolchain/go/bin/go version
```

输出结果如下：

```shell #test-result id="install-go"
.aibrix-quick-start/go.tar.gz: OK
go version go1.22.6 linux/arm64
```

## 安装 Envoy 1.39.0

local mode 用 Envoy 把 HTTP `:10080` 转到网关。下面这条走 GitHub 的国内加速前缀，并校验官方发布包的 SHA256。如果加速前缀不可用，把 URL 换成官方地址再下同一文件：

`https://github.com/envoyproxy/envoy/releases/download/v1.39.0/envoy-1.39.0-linux-aarch_64`

```shell #test id="install-envoy"
set -euo pipefail
mkdir -p .aibrix-quick-start/bin
curl -fL --connect-timeout 20 --retry 8 --retry-all-errors --retry-delay 3 --max-time 300 \
  -C - -o .aibrix-quick-start/bin/envoy.part \
  https://gh.ddlc.top/https://github.com/envoyproxy/envoy/releases/download/v1.39.0/envoy-1.39.0-linux-aarch_64
echo 'ee53a4f5375566f15944dc9cb03afb1fc228df38f61737c677f139213215afcf  .aibrix-quick-start/bin/envoy.part' | sha256sum -c
mv .aibrix-quick-start/bin/envoy.part .aibrix-quick-start/bin/envoy
chmod +x .aibrix-quick-start/bin/envoy
.aibrix-quick-start/bin/envoy --version
```

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

克隆工作流注入的 release tag（你本地可以改成 `v0.7.0`）。编译只用这一条 `go build`，不要跑 `make build-gateway-plugins-nozmq`，那个 target 会先跑全仓 `manifests generate fmt vet`。

```shell #test id="clone-aibrix" load="upstream_ref>>ref"
set -euo pipefail
rm -rf .aibrix-quick-start/aibrix
git clone --depth 1 --branch <ref> https://github.com/vllm-project/aibrix.git .aibrix-quick-start/aibrix
git -C .aibrix-quick-start/aibrix describe --tags --exact-match
```

输出结果如下：

```shell #test-result id="clone-aibrix" load="upstream_ref>>ref"
<ref>
```

```shell #test id="build-gateway"
set -euo pipefail
export GOPROXY=https://goproxy.cn,direct
export GOPATH="$PWD/.aibrix-quick-start/gopath"
export GOCACHE="$PWD/.aibrix-quick-start/gocache"
mkdir -p "$GOPATH" "$GOCACHE"
cd .aibrix-quick-start/aibrix
CGO_ENABLED=0 "$PWD/../toolchain/go/bin/go" build -tags=nozmq -o bin/gateway-plugins cmd/plugins/main.go
test -x bin/gateway-plugins
echo gateway-plugins_ok
```

输出结果如下：

```shell #test-result id="build-gateway"
gateway-plugins_ok
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

版本要按这个顺序装，不要合成一次 `pip install`。先钉 torch / torch-npu 2.10，再用 `VLLM_TARGET_DEVICE=empty` 装 vLLM 0.23.0 源码（避免官方 vLLM wheel 把 torch 升到 2.11，和 torch-npu 2.10 冲突），最后才装 `vllm-ascend==0.23.0`。

装 vLLM 源码时会 import 已经装好的 `torch`，它会自动加载 `torch_npu`，所以这一块必须先 `source` CANN 和 ATB。`VLLM_USE_MODELSCOPE=True` 是后面启动服务才设；这里先把包装对。

```shell #test id="install-vllm"
set -euo pipefail
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:$PATH"
python3 -m venv .aibrix-quick-start/venv
PY="$PWD/.aibrix-quick-start/venv/bin/python"
"$PY" -m pip install -U pip
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_DEFAULT_TIMEOUT=120
export PIP_RETRIES=5
"$PY" -m pip install \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --find-links https://repo.huaweicloud.com/ascend/repos/pypi/triton-ascend/ \
  torch==2.10.0 torch-npu==2.10.0.post4 torchvision==0.25.0 torchaudio==2.10.0 triton-ascend==3.2.2
"$PY" -m pip install 'cmake>=3.26' nanobind ninja setuptools-rust wheel 'setuptools-scm>=8' 'setuptools>=77,<81'
rm -rf .aibrix-quick-start/src/vllm
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git .aibrix-quick-start/src/vllm
export VLLM_TARGET_DEVICE=empty
"$PY" -m pip install --no-build-isolation -e .aibrix-quick-start/src/vllm
"$PY" -m pip install grpcio-tools
export CMAKE_PREFIX_PATH="$("$PY" -m nanobind --cmake_dir)${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export Python_EXECUTABLE="$PY"
export PYTHON_EXECUTABLE="$PY"
"$PY" -m pip install --no-build-isolation \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/variant \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  vllm-ascend==0.23.0
"$PY" -m pip install modelscope==1.31.0
"$PY" -c "import importlib.metadata as m
for n in ['torch', 'torch-npu', 'vllm', 'vllm-ascend', 'modelscope']:
    print(n, m.version(n))"
```

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

`--max-model-len` / `--max-num-seqs` / `--gpu-memory-utilization` 把这次演示压到单卡、小模型可接受的规模。`--served-model-name` 必须和后面网关 `endpoints.yaml` 里的模型名完全一致，否则 Envoy 会报 `no healthy upstream`。

`VLLM_USE_MODELSCOPE=True` 从 ModelScope 拉权重（不要写成 `=1`）。日志和 PID 都写在 `.aibrix-quick-start/`，方便你对照，也方便结束后按 PID 停掉。`/health` 返回 200 还不够：机器上如果已经有别人的服务占着 8000，curl 也会成功。下面用 `ss` 确认监听这个端口的就是刚才记下的 PID。

```shell #test-setup
set -euo pipefail
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:$PATH"
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
    echo backend_health_200
    exit 0
  fi
  sleep 2
done
echo 'timed out waiting for http://127.0.0.1:8000/health. Full log:'
cat .aibrix-quick-start/vllm.log
exit 1
```

服务起来之后，在日志里确认这次推理走了 HCCL。只有昇腾后端被选中时才会出现 `backend=hccl`。`Platform plugin ascend is activated` 在卡不可见时也可能出现，不能单独当成功证据。

```shell #test id="backend-on-npu"
set -euo pipefail
grep -F 'backend=hccl' .aibrix-quick-start/vllm.log
```

输出结果如下：

```shell #test-result id="backend-on-npu"
...backend=hccl...
```

## 配置网关并启动 local mode

模型名必须和 `--served-model-name` 一致。endpoint 用 `127.0.0.1:8000`，不要写 `localhost`，避免 hosts 解析把流量送到别的地址。

```shell #test id="configure-endpoints"
set -euo pipefail
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

`run-local.sh` 会在 PATH 里找名为 `envoy` 的二进制，所以这一块把 `.aibrix-quick-start/bin` 放到 PATH 最前面。脚本要求 `endpoints.yaml` 用绝对路径。

```shell #test id="start-gateway"
set -euo pipefail
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

请求打到网关的 `127.0.0.1:10080`，由 Envoy 转到 vLLM。用虚拟环境里的 Python 解析 JSON：模型名要对，回复不能为空。生成内容每次都会变，不要拿原文去对。

```shell #test id="infer"
set -euo pipefail
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

先停网关和 Envoy，再按 PID 文件停 vLLM。不要 `pkill vllm`，那会误杀别人的任务。

```shell #test id="cleanup"
set -euo pipefail
if [ -x .aibrix-quick-start/aibrix/deployment/local/stop-local.sh ]; then
  bash .aibrix-quick-start/aibrix/deployment/local/stop-local.sh || true
fi
if [ -f .aibrix-quick-start/vllm.pid ]; then
  pid=$(cat .aibrix-quick-start/vllm.pid)
  if [ -n "${pid}" ]; then
    kill "${pid}" 2>/dev/null || true
  fi
fi
echo stopped
```

输出结果如下：

```shell #test-result id="cleanup"
...
stopped
```
