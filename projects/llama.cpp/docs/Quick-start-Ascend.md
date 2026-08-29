# 快速开始：在昇腾 NPU 上用 llama.cpp 做推理

> **阅读本文前**，请先按 [安装指南](https://ascend.github.io/docs/sources/llama_cpp/install.html) 或 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**：从源码编译 llama.cpp（CANN 后端），下载一份 GGUF 模型，在单卡 NPU 上完成一次文本生成。

[llama.cpp](https://github.com/ggml-org/llama.cpp) 是面向 GGUF 格式大模型的轻量推理引擎。昇腾侧通过 **CANN 后端**（`-DGGML_CANN=on`）把计算图调度到 NPU；设备在日志里显示为 `CANN0`、`CANN1` 等。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。上游已验证的设备列表见 [CANN 后端 — Devices](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| 编译工具 | cmake ≥ 3.14、g++（C++17）、make、git |
| 下载工具 | curl |
| llama.cpp | 本文从 GitHub 源码编译，见下文 |

---

## 1. 加载 CANN 环境

新开终端后 CANN 变量不会自动生效。`cmake` 配置阶段还会调用 `npu-smi` 探测 SoC 型号；在常见容器布局里，`npu-smi` 位于 `/usr/local/sbin`，需把该目录加入 `PATH`。

```shell #test-setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

---

## 2. 检查环境是否就绪

### 2.1 确认 NPU 在线

```shell
npu-smi info
```

**预期**：命令退出码为 0，并打印设备列表。表格中的功耗、HBM 占用每次不同，**不必**与任何样例逐字一致。

### 2.2 确认 CANN 与编译工具

```shell #test
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
cmake --version
```

**预期**：`ASCEND_HOME_PATH` 非空；`npu-smi` 与 `cmake` 均能打印出版本信息。

若 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载（如 `/dev/davinci0`）。

---

## 3. 获取源码并编译

克隆上游仓库，检出你要用的 ref，开启 CANN 后端并编译。编译成功后应生成 `llama-completion`（一次性文本生成；**不要**用交互式 `llama-cli` 做首次验证）。

将 `<UPSTREAM_REF>` 换成目标**分支、tag 或 commit**（上游默认分支为 `master`）。


```shell #test-setup
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout <UPSTREAM_REF>
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
ls build/bin/llama-completion
```

**预期**：最后一条 `ls` 列出 `llama-completion` 可执行文件；整段脚本退出码为 0。首次全量编译可能耗时较久（数十分钟量级，视机器而定）。

---

## 4. 准备模型（GGUF）

llama.cpp 推理输入为 **GGUF** 文件。你可以：

- 使用社区已转换好的 GGUF（本文做法）；
- 或用上游 `convert_hf_to_gguf.py` / [GGUF-my-repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo) 自行从 Hugging Face 权重转换。

**910B 上 CANN 后端支持的量化类型**：FP16、BF16、Q8_0、Q4_0（详见 [CANN 后端 — DataType](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)）。请勿使用 Q4_K_M 等未列出的格式。

下面从 ModelScope 下载小体积示例模型（无需 token，国内可直连）。模型保存在工作区根目录，与 `llama.cpp/` 同级。

下载完成后用文件头 magic 校验——正确 GGUF 的前 4 字节为 `GGUF`；若得到 HTML，说明 URL 错误或下载到了错误页。

```shell #test
curl -L -o qwen2.5-0.5b-instruct-q4_0.gguf \
  https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_0.gguf
head -c 4 qwen2.5-0.5b-instruct-q4_0.gguf
```

```text
GGUF
```

---

## 5. 推理

llama.cpp 通过 **split mode**（`-sm`）与 **main GPU**（`-mg`）选择设备布局，与 [昇腾开源文档 — 推理](https://ascend.github.io/docs/sources/llama_cpp/quick_start.html#id5) 的约定一致：

| 场景 | 参数 | 说明 |
| --- | --- | --- |
| **单设备** | `-sm none -mg <设备号>` | 全部算子跑在指定的一张卡上 |
| **多设备** | `-sm layer`（默认） | 按层拆分到多张同后端设备 |

此外常用参数：

| 参数 | 含义 |
| --- | --- |
| `-m` | GGUF 模型路径 |
| `-p` | 提示词（prompt） |
| `-n` | 最大生成 token 数（本文用 `64` 做快速验证，可按需调大） |
| `-ngl` |  offload 到 NPU 的层数（`99` 表示尽量全部） |
| `-no-cnv` | 关闭自动对话模式；Instruct 类 GGUF 自带 chat template，不加此项会等待交互输入 |
| `-v` | 提高日志详细度，便于确认 NPU 是否真正参与计算 |
| `ASCEND_RT_VISIBLE_DEVICES` | 限制进程可见的 NPU 编号 |

### 5.1 单卡推理

下列命令自带 `cd llama.cpp`，可**单独复制**执行；模型路径 `../qwen2.5-0.5b-instruct-q4_0.gguf` 相对仓库根目录。

**怎样算成功**

1. 进程退出码为 0；
2. 标准输出末尾出现与 prompt 相关的续写文本；
3. 日志中出现 `load_tensors: ... CANN0 model buffer size = ...`——**这是模型权重已加载到 NPU 的关键证据**。若缺少 `CANN0`，进程仍可能返回 0，但往往在 CPU 上静默回退。

```shell #test
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-completion \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "Building a website can be done in 10 simple steps:" \
    -n 64 -no-cnv -ngl 99 -sm none -mg 0 -v
```

```text
load_tensors: ...CANN0 model buffer size = ...
```

### 5.2 多卡推理（可选）

有两张及以上 NPU 时，可用 `-sm layer` 把层拆到多卡。下面示例暴露 0、1 号卡：

```shell skip
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/bin/llama-completion \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "Building a website can be done in 10 simple steps:" \
    -n 64 -no-cnv -ngl 99 -sm layer -v
```

### 5.3 交互式对话（可选）

`llama-cli` 面向多轮对话，启动后会等待键盘输入，不适合脚本化流水线。想手动体验模型时可以使用：

```shell skip
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -ngl 99 -sm none -mg 0
```

---

## 6. 下一步

| 目标 | 参考 |
| --- | --- |
| CANN 环境变量、性能调优、完整模型支持表 | 上游 [docs/backend/CANN.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md) |
| HTTP/OpenAI 兼容 API 服务 | 编译产物 `llama-server` |
| Hugging Face → GGUF 转换与量化 | 上游 `convert_hf_to_gguf.py`、`llama-quantize` |
| 昇腾侧安装与更多示例 | [昇腾开源 — llama.cpp](https://ascend.github.io/docs/sources/llama_cpp/quick_start.html) |

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `cmake` 报 SoC 探测失败 | 未 `source set_env.sh`，或 `npu-smi` 不在 `PATH` | 重做第 1–2 节 |
| 编译很久无报错 | 可能在等 HuggingFace Web UI 资源 | 加 `-DLLAMA_USE_PREBUILT_UI=OFF` 后重新配置 |
| `head -c 4` 不是 `GGUF` | 下载到了 HTML 错误页 | 检查 URL、网络与磁盘空间 |
| 推理退出 0 但无 `CANN0` | NPU 未挂载或层未 offload 到 CANN | 检查 `ASCEND_RT_VISIBLE_DEVICES`、`-ngl`、`-v` 日志 |
| `llama-completion` 卡住不输出 | 未加 `-no-cnv`，进入对话等待 | 加上 `-no-cnv` 或使用 `-p` 一次性生成 |
