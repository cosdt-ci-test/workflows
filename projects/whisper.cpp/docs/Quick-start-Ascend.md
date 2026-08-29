# 快速开始：在昇腾 NPU 上用 whisper.cpp 做语音转写

> **阅读本文前**，请先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备好 CANN 与驱动。本文聚焦**第一次跑通**：从源码编译 whisper.cpp（CANN 后端），下载一个小模型，在单卡 NPU 上把一段英文语音转成文字。

[whisper.cpp](https://github.com/ggml-org/whisper.cpp) 是 OpenAI Whisper 语音识别模型的纯 C/C++ 推理实现。昇腾侧通过 **CANN 后端**（`-DGGML_CANN=on`）把计算调度到 NPU；设备在日志里显示为 `CANN0`、`CANN1` 等。

---

## 前置条件

### 硬件

Atlas **800T** / **900 A2** 训练系列（Ascend **910B**）。本文示例为**单卡**。

### 软件

| 类别 | 要求 |
| --- | --- |
| CANN | toolkit + 驱动固件已安装并可 `source set_env.sh` |
| 编译工具 | cmake、g++（C++17）、make、git |
| 下载工具 | curl |
| whisper.cpp | 本文从 GitHub 源码编译，见下文 |

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

```shell #test-setup
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
cmake --version
```

**预期**：`ASCEND_HOME_PATH` 非空；`npu-smi` 与 `cmake` 均能打印出版本信息。

若 `npu-smi` 找不到，回到 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 检查驱动与设备挂载（如 `/dev/davinci0`）。

---

## 3. 获取源码并编译

克隆上游仓库，检出你要用的 ref，开启 CANN 后端并编译。编译成功后应生成 `whisper-cli`（命令行转写工具）。

将 `<UPSTREAM_REF>` 换成目标**分支、tag 或 commit**（上游默认分支为 `master`）。
<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="compile" load="upstream_ref>>UPSTREAM_REF"
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
git checkout <UPSTREAM_REF>
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
ls build/bin/whisper-cli
```

输出结果如下：

```shell #test-result id="compile"
...
build/bin/whisper-cli
...
```

首次全量编译可能耗时较久（数十分钟量级，视机器而定）。

---

## 4. 准备模型（ggml 格式）

whisper.cpp 的推理输入是 **ggml 格式**的模型文件。上游提供 `models/download-ggml-model.sh` 从 Hugging Face 下载；国内直连 huggingface.co 可能超时，下面改用镜像站 hf-mirror.com 的直链下载 `tiny.en`（英文小模型，约 77 MB），保存到上游约定的 `models/` 目录。

下载完成后用文件头 magic 校验——正确 ggml 模型文件的前 4 字节为 `lmgg`；若得到 HTML，说明 URL 错误或下载到了错误页。

```shell #test id="download-model"
cd whisper.cpp
curl -fL -o models/ggml-tiny.en.bin \
  https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
head -c 4 models/ggml-tiny.en.bin
```

输出结果如下：

```shell #test-result id="download-model"
lmgg
```

---

## 5. 转写

常用参数：

| 参数 | 含义 |
| --- | --- |
| `-m` | ggml 模型路径 |
| `-f` | 输入音频（需 16-bit WAV；仓内自带样例 `samples/jfk.wav`） |
| `-t` | 推理线程数（本文用 `4`） |
| `--device` | 使用的 NPU 编号 |
| `ASCEND_RT_VISIBLE_DEVICES` | 限制进程可见的 NPU 编号 |

### 5.1 单卡转写

下列命令自带 `cd whisper.cpp`，可**单独复制**执行；模型与音频路径相对仓库根目录。

**怎样算成功**

1. 进程退出码为 0；
2. 输出末尾打印出 JFK 演讲的转写文本；
3. 日志中出现 `whisper_backend_init_gpu: using CANN0 backend`——**这是模型已在 NPU 上初始化的关键证据**。若缺少 `CANN0`，进程仍可能返回 0，但往往在 CPU 上静默回退。

```shell #test id="transcribe"
cd whisper.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/whisper-cli \
    -m models/ggml-tiny.en.bin \
    -f samples/jfk.wav \
    -t 4 --device 0 2>&1
```

输出结果如下：

```shell #test-result id="transcribe"
...
whisper_backend_init_gpu: using CANN0 backend
...ask not what your country...
```

### 5.2 HTTP 推理服务（可选）

`whisper-server` 把转写包成 HTTP 服务。它启动后不会自己退出（用 Ctrl+C 结束），想手动体验时可以使用：

<!--
```shell
cd whisper.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/whisper-server \
    -m models/ggml-tiny.en.bin \
    -t 4 --device 0 \
    --host 127.0.0.1 --port 8080
```
-->

服务就绪后用浏览器打开 `http://127.0.0.1:8080`，或从另一个终端向 `POST /inference` 上传音频。

---

## 6. 下一步

| 目标 | 参考 |
| --- | --- |
| 更多模型（多语言、更大体积）与转换方式 | 上游 [models/README.md](https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md) |
| 量化（减小模型体积） | 编译产物 `whisper-quantize` |
| VAD 语音活动检测 | 编译产物 `whisper-vad-speech-segments` |
| 上游昇腾说明与已验证设备表 | [README — Ascend NPU support](https://github.com/ggml-org/whisper.cpp#ascend-npu-support) |
| 昇腾侧安装与更多示例 | [昇腾开源 — whisper.cpp](https://ascend.github.io/docs/sources/whisper_cpp/quick_start.html) |

---

## 故障排查

| 现象 | 可能原因 | 建议 |
| --- | --- | --- |
| `cmake` 报 SoC 探测失败 | 未 `source set_env.sh`，或 `npu-smi` 不在 `PATH` | 重做第 1–2 节 |
| `head -c 4` 不是 `lmgg` | 下载到了 HTML 错误页或文件被截断 | 检查 URL、网络与磁盘空间后重下 |
| huggingface.co 下载超时 | 国内直连受限 | 用第 4 节的 hf-mirror.com 直链 |
| 转写退出 0 但无 `CANN0` | NPU 未挂载或 `--device` 指向不可用设备 | 检查 `ASCEND_RT_VISIBLE_DEVICES` 与 `npu-smi info` |
| 命令打印 usage 后退出 0 | 参数拼写错误（whisper-cli 对未知参数打印用法后以 0 退出） | 对照 `./build/bin/whisper-cli -h` 检查参数 |
