# 快速开始：在昇腾 NPU 上使用 KTransformers

[KTransformers](https://github.com/kvcache-ai/ktransformers) 是面向大模型异构推理与微调的框架。昇腾上的安装、权重准备和启动命令以上游官方教程为准。本文只做入口，不重复那些会随上游改动的步骤。

阅读官方教程前，先按 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 准备 CANN 与驱动。

项目主页：[KTransformers 文档站](https://kvcache-ai.github.io/ktransformers/)。

## 官方教程

上游在 2025-10-27 起声明支持 Ascend NPU。可复现步骤在下面两篇中文教程里（仓库 `main` 分支）：

| 场景 | 教程 |
| --- | --- |
| DeepSeek-R1 / V3 | [DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/zh/DeepseekR1_V3_tutorial_zh_for_Ascend_NPU.md) |
| Qwen3-MoE | [Qwen3-MoE_tutorial_zh_for_Ascend_NPU.md](https://github.com/kvcache-ai/ktransformers/blob/main/doc/zh/Qwen3-MoE_tutorial_zh_for_Ascend_NPU.md) |

入口声明见上游 [README](https://github.com/kvcache-ai/ktransformers/blob/main/README.md) 的 Updates 列表。

两篇教程都写明当前支持的 NPU 型号是 **Atlas 300I A2**，并给出对应的 CANN 版本、MindIE 镜像和主机内存要求（满血 DeepSeek-R1 约 400GB，Qwen3-MoE 约 200GB）。版本号、镜像 tag、合并权重的命令和 `balance_serve` 启动脚本都以教程正文的当前版本为准。

Qwen3-MoE 那篇只写与 DeepSeek 教程不同的部分（权重来源、合并脚本、启动参数）。物理机、系统、HDK、镜像和 CANN 仍按 DeepSeek 那篇做。

## 和上游默认分支布局的关系

默认分支上面向用户的入口是 `kt-kernel`（CPU / CUDA 等 kernel）。官方昇腾教程里的 `ktransformers/server/main.py` 和 `optimize/optimize_rules/npu/` 在上游 `archive/` 目录。按教程操作时以教程正文给出的路径为准，不要只在仓库根目录找同名文件。
