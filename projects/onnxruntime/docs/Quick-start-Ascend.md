# 快速开始：在昇腾 NPU 上用 ONNX Runtime 做第一次推理

你将在单卡昇腾 NPU 上安装 `onnxruntime-cann`，当场生成一个最小加法模型，并确认推理走的是昇腾后端，而不是 CPU。

[ONNX Runtime](https://github.com/microsoft/onnxruntime) 用 **CANN Execution Provider** 把计算调度到昇腾。包名是 `onnxruntime-cann`，导入名仍是 `onnxruntime`。不要再装一份 CPU 包 `onnxruntime`，两个包会抢同一个导入名。

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 装好 CANN 与驱动。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装，并且可以 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| Python | 3.12 |
| 包管理 | `pip`（下面用 `python -m pip`） |

新开终端后 CANN 变量不会自动生效。`npu-smi` 在常见容器里位于 `/usr/local/sbin` 或 `/usr/local/bin`。后面每一个真正调用 `onnxruntime` 的命令块都会再 `source` 一次，单独复制也能跑。

```shell
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 确认 NPU 在线

```shell
npu-smi info
```

命令退出码应为 0，并打印设备表。功耗、温度、HBM 占用每次都不同，不必和任何截图逐字一致。

若提示找不到 `npu-smi`，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载（例如 `/dev/davinci0`）。

---

## 本文档验证过的版本

**配套机器**

- **机器类型**：Atlas 900 A2（Ascend 910B，单卡）
- **操作系统**：Ubuntu 22.04

**配套镜像**

`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`

**软件版本**

| 组件 | 版本 |
| --- | --- |
| Python | 3.12 |
| CANN | 9.1.0 |
| onnxruntime-cann | 1.24.4 |
| onnx | 1.22.0 |
| numpy | 1.26.x（必须 `<2`） |

官方 [CANN Execution Provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html) 兼容表目前只列了 ONNX Runtime 1.20.0 / 1.21.0 / 1.22.1 对应 CANN 8.2.0。本文按上表验证，没有跟着那张旧表降版本。

---

## 安装 onnxruntime-cann

用华为云通用 PyPI 镜像安装钉死版本。昇腾专用源 `https://repo.huaweicloud.com/ascend/repos/pypi` **没有** `onnxruntime-cann` 这个包。

`onnx` 用来在下一步当场生成模型，不从网上下载权重。这个 wheel 按 NumPy 1.x 编译。不钉 `numpy<2` 时，pip 会拉到 NumPy 2，`import onnxruntime` 会直接失败。

CANN 编译算子还要用 `decorator`、`scipy`、`attrs`、`psutil`。不装的话，会话能建起来，第一次 `sess.run()` 会报 `aclgrphBuildInitialize` 或 `aclopCompileAndExecute`。`scipy` 钉在 1.15 以下，避免把 NumPy 升到 2。

```shell #test id="install"
python -m pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple \
    onnxruntime-cann==1.24.4 \
    onnx==1.22.0 \
    'numpy<2' \
    decorator \
    'scipy>=1.11,<1.15' \
    attrs \
    psutil
python -c "from importlib.metadata import version; print('onnxruntime-cann', version('onnxruntime-cann')); print('onnx', version('onnx'))"
```

输出结果如下：

```shell #test-result id="install"
...onnxruntime-cann 1.24.4
onnx 1.22.0
```

---

## 确认昇腾后端

单独复制这一块时也必须先 `source`。不加载 CANN 的动态库，`CANNExecutionProvider` 不会出现在列表里。

```shell #test id="providers"
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

输出结果如下（列表里必须有 `CANNExecutionProvider`，前后还可能有 `CPUExecutionProvider` 等）：

```shell #test-result id="providers"
...CANNExecutionProvider...
```

若没有 `CANNExecutionProvider`，先看文末「常见问题」，不要继续推理。列表里有 CPU 并不等于这次推理会走 CPU。真正决定后端的是下一节创建 `InferenceSession` 时传入的 `providers`。

---

## 造一个最小 ONNX 模型

用已安装的 `onnx` 在当前目录写一个两向量相加的图，保存为相对路径 `add_model.onnx`。不下载任何文件。

```shell #test id="make-model"
python <<'PY'
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [2])
graph = helper.make_graph(
    [helper.make_node("Add", ["X", "Y"], ["Z"])],
    "add",
    [x, y],
    [z],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.save(model, "add_model.onnx")
print("wrote add_model.onnx")
PY
```

输出结果如下：

```shell #test-result id="make-model"
wrote add_model.onnx
```

---

## 用昇腾跑第一次推理

下面这段**关掉了 CPU 回退**。ONNX Runtime 默认 `enable_fallback=1`。CANN 会话创建失败时，它会静默改在 CPU 上重建会话，加法结果仍然正确，进程退出码也是 0。你会以为昇腾已经跑通。

官方 CANN 示例常把 `CPUExecutionProvider` 接在后面。第一次验证不要抄那种写法。这里只注册 `CANNExecutionProvider`，并同时关掉会话级和 Python 级回退。装错包、CANN 没 `source`、版本对不上时，进程必须失败。

输入是 `[1.0, 2.0]` 和 `[3.0, 4.0]`，昇腾上的加法结果应是 `[4.0, 6.0]`。模型文件是当前目录下的 `add_model.onnx`。

```shell #test id="infer"
export PATH=/usr/local/sbin:/usr/local/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python <<'PY'
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
sess = ort.InferenceSession(
    "add_model.onnx",
    sess_options=so,
    providers=["CANNExecutionProvider"],
    enable_fallback=False,
)
providers = sess.get_providers()
print(providers)
assert providers[0] == "CANNExecutionProvider", providers
x = np.array([1.0, 2.0], dtype=np.float32)
y = np.array([3.0, 4.0], dtype=np.float32)
out = sess.run(None, {"X": x, "Y": y})[0]
print("result", [float(v) for v in out])
PY
```

输出结果如下。第一项必须是 `CANNExecutionProvider`。列表里是否还出现 CPU，以你机器上的实际打印为准。

```shell #test-result id="infer"
...CANNExecutionProvider...
result [4.0, 6.0]
```

---

## 常见问题

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `get_available_providers()` 没有 `CANNExecutionProvider` | 没 `source set_env.sh`，或装的是 CPU 包 `onnxruntime`，或两个包叠在一起 | 在同一段命令里重新 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。若叠装过 CPU 包，先 `python -m pip uninstall -y onnxruntime onnxruntime-cann`，再只装 `onnxruntime-cann==1.24.4` |
| `import onnxruntime` 报找不到 CANN 动态库 | 当前 shell 没有 CANN 环境变量 | 先 `source`，不要只在另一个终端里 source 过 |
| 创建 `InferenceSession` 失败，日志提到 CANN / ACL 版本 | 本文验证的是 CANN 9.1.0 + `onnxruntime-cann==1.24.4`。官方兼容表只写到 ORT 1.20–1.22.1 ↔ CANN 8.2.0 | 对齐本文版本表，或按你本机 CANN 换已验证过的 `onnxruntime-cann` |
| 会话能建，`get_providers()` 第一项也是 `CANNExecutionProvider`，但 `sess.run()` 报 `aclgrphBuildInitialize` 或 `aclopCompileAndExecute("Add")` / `ACL_ERROR_FAILURE` | 当前 Python 环境缺 CANN 算子编译依赖，`import tbe` 失败 | 回到「安装 onnxruntime-cann」，确认 `decorator`、`scipy`、`attrs`、`psutil` 都装上了，再 `source` 一次后重跑推理 |
| 推理结果正确，但 `get_providers()` 里同时出现 CPU | 创建会话时把 CPU 写进了 `providers`，或没有关掉回退。有的版本建好 CANN 会话后仍会在列表里留下 CPU | 不要把 `CPUExecutionProvider` 写进创建会话时的 `providers`。第一项必须是 `CANNExecutionProvider`，并且推理没有落到 CPU 回退日志 |
| `npu-smi: command not found` | `npu-smi` 在 `/usr/local/sbin` 或 `/usr/local/bin`，不在默认 `PATH` | `export PATH=/usr/local/sbin:/usr/local/bin:$PATH` 后再执行 `npu-smi info` |
