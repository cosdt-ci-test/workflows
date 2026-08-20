# 在昇腾 NPU 上编译 llama.cpp

你将在一张昇腾 NPU 上，用 CANN 后端从源码编译 llama.cpp，并完成一次文本生成。

下面的命令都假设你在一个空的工作目录里。这个目录稍后会包含 `llama.cpp/` 文件夹。点 GitHub 代码块右上角的复制，整块贴进终端即可。折行用 bash 的 `\`。

## 前置条件

机器是 Atlas 800T 或 Atlas 900 A2（Ascend 910B 系列）。本文只用一张卡。上游已验证的设备见 [CANN 后端文档的 Devices 表](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)。

先装好 CANN toolkit 和驱动固件，见 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。还需要 cmake 3.14 或更新、支持 C++17 的 g++、make、git、curl。

本文示例用过的版本如下。

| 项目 | 版本 |
| --- | --- |
| 机器 | Atlas 900 A2 PODc（910B4 64GB × 1） |
| 操作系统 | Ubuntu 22.04 aarch64 |
| 镜像 | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12` |
| CANN | 9.1.0 |
| cmake | 3.22 |
| g++ | 11.4 |
| llama.cpp | 从源码编译，见下面的 checkout |
| 模型 | Qwen2.5-0.5B-Instruct-GGUF，量化 Q4_0 |

## 加载 CANN 环境

`npu-smi` 在容器里通常在 `/usr/local/sbin`。cmake 配置阶段靠它探测 SoC。这一步不能省。

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH=/usr/local/sbin:$PATH
```

## 检查前置是否满足

先确认 NPU 在线。表格里的功耗和 HBM 每次都会变，这一步只看退出码。

```shell
npu-smi info
```

再确认 CANN 已加载、`npu-smi` 在 PATH 上、cmake 能跑。

```shell
test -n "$ASCEND_HOME_PATH"
command -v npu-smi
cmake --version
```

## 获取源码并编译

把 `<UPSTREAM_REF>` 换成你要用的分支、tag 或 commit。CI 会把它换成这次要测的 SHA。

国内网络下，这一步可能会向 HuggingFace 拉 server 的 Web UI，失败后自动跳过，不影响后面的推理，但可能多等几分钟。不想等的话，在 cmake 配置那一行加上 `-DLLAMA_USE_PREBUILT_UI=OFF`。

```shell
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout <UPSTREAM_REF>
cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
ls build/bin/llama-completion
```

## 准备模型

CANN 在 910B 上支持 FP16、BF16、Q8_0、Q4_0。本文用 Q4_0。用 `curl` 从 ModelScope 下载，不要装 modelscope 命令行。

`head -c 4` 应打出 `GGUF`。如果打出的是 HTML，说明下到了错误页。

```shell
curl -L -o qwen2.5-0.5b-instruct-q4_0.gguf \
  https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_0.gguf
head -c 4 qwen2.5-0.5b-instruct-q4_0.gguf
```

```text
GGUF
```

## 单卡推理

这一块自己 `cd llama.cpp`，单独复制也能跑。模型文件在上一级目录。

各参数的含义如下。

- `-m` 是 GGUF 路径。
- `-p` 是提示词。
- `-n 64` 是生成长度。这是 CI 规模，想更长就把数字调大。
- `-no-cnv` 关掉自动对话。Qwen2.5-Instruct 的 GGUF 自带 chat template，不加这个会停下来等输入。
- `-ngl 99` 把全部层放到 NPU。
- `-sm none` 表示单卡，不拆层。
- `-mg 0` 指定 0 号卡。
- `-v` 打开完整日志。默认 verbosity 不打印 `load_tensors`，看不到设备名。
- `ASCEND_RT_VISIBLE_DEVICES=0` 只暴露 0 号卡。

日志里必须出现 `CANN0`。没有这行时，进程仍可能退出 0，但模型是在 CPU 上跑的。

```shell
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-completion \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "Building a website can be done in 10 simple steps:" \
    -n 64 -no-cnv -ngl 99 -sm none -mg 0 -v
```

```text
load_tensors: ...CANN0 model buffer size = ...
```

## 多卡推理

两张卡可以把层拆开。CI 和本文的默认机器只有一张卡，所以自动化测试不执行下面这一块。框顶的 `skip` 不会进复制内容。

```shell skip
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0,1 ./build/bin/llama-completion \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -p "Building a website can be done in 10 simple steps:" \
    -n 64 -no-cnv -ngl 99 -sm layer -v
```

## 交互式对话

`llama-cli` 是对话客户端，适合你自己打字试模型。它会停下来等输入，不适合写进脚本。

```shell skip
cd llama.cpp && ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
    -m ../qwen2.5-0.5b-instruct-q4_0.gguf \
    -ngl 99 -sm none -mg 0
```

## 下一步

环境变量 `GGML_CANN_*` 和模型支持表见上游 [CANN 后端文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/CANN.md)。

要开 HTTP 服务，用编译产物里的 `llama-server`。

要把其他权重转成 GGUF 或改量化，用上游的 convert 和 quantize 工具。
