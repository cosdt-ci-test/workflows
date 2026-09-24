# TensorDict examples 看护

本项目复用公共 `examples-template.yml`；薄触发器为 `.github/workflows/tensordict-examples.yml`，被测仓库是 `pytorch/tensordict`。手动触发的 `target_ref` 可指定分支、tag 或 SHA；留空时由公共引擎选择最新 release。Bring-up 期间 schedule 保持注释关闭。

上游没有独立的 `examples/` 目录；清单扫描 `tutorials/sphinx_tuto` 中的 14 个 Python 教程。首批 `tensordict_keys.py`、`tensordict_shapes.py`、`tensordict_preallocation.py` 已在 `tensordict-examples #2` 跑通。第二批增加 `functional.py`（参数替换、vmap、函数式调用）和 `export.py`（小型 TensorDictModule 导出），两条都只用合成张量，仍待手动 NPU workflow 验收。其余 9 个逐条在 manifest 中说明用途和暂缓原因。

项目 setup 先探测 `torch`/`torch_npu`：若镜像里有兼容版本就复用，否则按 quick-start 的已验证配方安装 `torch==2.9.0` 与 `torch_npu==2.9.0.post2`；再从受测 checkout 安装 TensorDict 源码，且源码安装使用 `--no-deps` 防止覆盖 NPU 栈。启动器先设置 PyTorch 默认设备为 `npu:0`，再用 `runpy` 执行未经修改的上游教程；最后检查原教程生成的 TensorDict 叶子或函数式/导出结果确实位于 NPU。这样既运行了原教程的完整代码与断言，也不会把 CPU 回落误报为 NPU 成功。五个 supported 教程都只使用小型合成张量，不下载模型或数据，无需 ModelScope/cache-seed。

此方案与 DeepSpeed 的 CLI overlay 不同：这些教程没有缩规模参数。FashionMNIST 两个训练教程以及 ImageNet 教程在上游源码中硬选 CUDA/CPU，原样运行不会使用 NPU；需在后续阶段单独设计可验证的适配方式。`torch.export` 和 `vmap` 已进入第二批待验收；包含无条件 `torch.compile` 性能基准的 `tensordict_module.py` 继续暂缓。

手动验收时检查 manifest-check 展开 5 个唯一 job；原有 3 条保持成功，新加的 functional/export 日志都出现 `NPU tutorial passed` 且无 CPU 回落，5 个 publish-result 均产出符合 schema 的 `result.json`。完整 workflow 成功后再考虑启用 schedule。artifact 由公共引擎命名为 `tensordict-examples-<run_id>-<job_index>`。
