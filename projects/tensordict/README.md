# TensorDict examples 看护

本项目复用公共 `examples-template.yml`；薄触发器为 `.github/workflows/tensordict-examples.yml`，被测仓库是 `pytorch/tensordict`。手动触发的 `target_ref` 可指定分支、tag 或 SHA；留空时由公共引擎选择最新 release。Bring-up 期间 schedule 保持注释关闭。

上游没有独立的 `examples/` 目录；清单扫描 `tutorials/sphinx_tuto` 中的 14 个 Python 教程。首批 supported 为 `tensordict_keys.py`、`tensordict_shapes.py`、`tensordict_preallocation.py`，其余 11 个逐条在 manifest 中说明用途和暂缓原因。supported 目前表示静态评估后的待验收候选，**不代表已经在 CI NPU 上跑通**。

项目 setup 从受测 checkout 安装 TensorDict 源码，并保护镜像中的 `torch`/`torch_npu`。启动器先设置 PyTorch 默认设备为 `npu:0`，再用 `runpy` 执行未经修改的上游教程；最后检查教程留下的 TensorDict 张量叶子全部位于 NPU。这样既运行了原教程的完整代码与断言，也不会把 CPU 回落误报为 NPU 成功。三个教程只使用小型合成张量，不下载模型或数据，无需 ModelScope/cache-seed。

此方案与 DeepSpeed 的 CLI overlay 不同：这些教程没有缩规模参数。FashionMNIST 两个训练教程以及 ImageNet 教程在上游源码中硬选 CUDA/CPU，原样运行不会使用 NPU；需在后续阶段单独设计可验证的适配方式。`torch.export`、`vmap`、`torch.compile` 等教程也需在基础操作上卡通过后再逐项评估。

手动验收时检查 manifest-check 展开 3 个唯一 job，三个日志都出现 `NPU tutorial passed` 且无 CPU 回落，publish-result 产出符合 schema 的 `result.json`；完整 workflow 成功后再考虑启用 schedule。artifact 由公共引擎命名为 `tensordict-examples-<run_id>-<job_index>`。
