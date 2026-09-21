# FastChat 快速入门（Ascend NPU）

在单张昇腾 NPU 上安装 FastChat，完成一次命令行对话，再通过 OpenAI 兼容接口调用同一个模型。

## 环境准备

请先安装 CANN，以及与 CANN 匹配的 `torch` 和 `torch_npu`。安装方法见[昇腾环境快速安装指南](https://ascend.github.io/docs/sources/ascend/quick_install.html)和 [Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)。

本文示例使用以下版本：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.9.0 |
| torch_npu | 2.9.0.post2 |
| fschat | 0.2.36 |
| transformers | 4.57.6 |
| modelscope | 1.37.0 |
| fastapi | 0.141.1 |
| uvicorn | 0.52.0 |
| 模型 | [Qwen/Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct)，约 1 GB |

**检查 NPU 运行环境。** 确认 Python 和 PyTorch 版本正确，并且至少有一张 NPU 可用。

```shell #test id="check-runtime"
python <<'PY'
import sys
import torch
import torch_npu

assert torch.npu.is_available()
assert torch.npu.device_count() > 0
print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
PY
```

```shell #test-result id="check-runtime" fuzzy='xxx'
Python 3.12.xxx
torch 2.9.0+cpu
torch_npu 2.9.0.post2
NPU available: True
NPU count: xxx
```

## 安装 FastChat

**安装运行模型与 API 服务所需的包。** PyPI 包名是 `fschat`，导入名是 `fastchat`。

```shell #test id="install-fastchat"
python -m pip install "fschat[model_worker]==0.2.36" "transformers==4.57.6" "modelscope==1.37.0" "fastapi==0.141.1" "uvicorn==0.52.0"
python -c "import fastapi, fastchat, modelscope, transformers, uvicorn; print('fastchat', fastchat.__version__); print('transformers', transformers.__version__); print('modelscope', modelscope.__version__); print('fastapi', fastapi.__version__); print('uvicorn', uvicorn.__version__)"
```

```shell #test-result id="install-fastchat" fuzzy='...'
...
fastchat 0.2.36
transformers 4.57.6
modelscope 1.37.0
fastapi 0.141.1
uvicorn 0.52.0
```

## 命令行对话

**运行一轮命令行对话。** 模型首次运行时会自动下载到 ModelScope 默认缓存，回答完成后以空行退出。

```shell #test id="cli-chat"
printf '你好\n\n' | FASTCHAT_USE_MODELSCOPE=True \
  python -m fastchat.serve.cli \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --revision master \
  --device npu \
  --max-new-tokens 64
```

```shell #test-result id="cli-chat" fuzzy='...' fuzzy='xxx'
...你好...
...xxx...
```

## OpenAI 兼容 API

FastChat 用 controller 管理 model worker，并通过 API server 提供 OpenAI 兼容接口。

**启动 controller。** controller 负责注册和调度 model worker，日志保存在当前目录的 `.fastchat` 中。

```shell #test-setup id="start-controller"
mkdir -p .fastchat
python -m fastchat.serve.controller >.fastchat/controller.log 2>&1 &
echo $! >.fastchat/controller.pid
```

**启动 model worker。** worker 在 NPU 上加载模型，并以 `Qwen2.5-0.5B-Instruct` 为服务名注册到 controller。

```shell #test-setup id="start-worker"
FASTCHAT_USE_MODELSCOPE=True python -m fastchat.serve.model_worker \
  --model-path Qwen/Qwen2.5-0.5B-Instruct \
  --model-names Qwen2.5-0.5B-Instruct \
  --revision master \
  --device npu \
  >.fastchat/worker.log 2>&1 &
echo $! >.fastchat/worker.pid
```

**启动 API server。** 服务在 `http://127.0.0.1:8000/v1` 提供 OpenAI 兼容接口。

```shell #test-setup id="start-api"
python -m fastchat.serve.openai_api_server \
  --host 127.0.0.1 \
  --port 8000 \
  >.fastchat/api.log 2>&1 &
echo $! >.fastchat/api.pid
```

**等待模型就绪。** `/v1/models` 返回服务名后即可发送请求。

```shell #test id="wait-model"
set -e
models=""
for _ in $(seq 1 180); do
  models=$(curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null || true)
  echo "$models" | grep -q 'Qwen2.5-0.5B-Instruct' && break
  sleep 5
done
echo "$models" | grep -q 'Qwen2.5-0.5B-Instruct' || { tail -50 .fastchat/{controller,worker,api}.log; exit 1; }
echo "model ready"
```

```shell #test-result id="wait-model"
model ready
```

**发送一次对话请求。** 调用 OpenAI 兼容的 Chat Completions 接口并打印模型回复。

```shell #test id="api-chat"
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"temperature":0}' \
  | python -c "import json, sys; data=json.load(sys.stdin); reply=data['choices'][0]['message']['content'].strip(); assert reply; print('model:', data['model']); print('reply:', reply)"
```

```shell #test-result id="api-chat" fuzzy='xxx'
model: Qwen2.5-0.5B-Instruct
reply: xxx
```

**停止服务。** 使用启动时保存的进程号停止 API server、model worker 和 controller。

```shell #test-setup id="stop-api"
kill $(cat .fastchat/api.pid .fastchat/worker.pid .fastchat/controller.pid) 2>/dev/null || true
rm -f .fastchat/*.pid
```

更多 Web UI、多 worker 和评测用法见 [FastChat 官方文档](https://github.com/lm-sys/FastChat)。
