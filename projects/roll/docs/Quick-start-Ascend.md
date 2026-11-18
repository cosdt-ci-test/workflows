# ROLL

在昇腾 NPU 上安装 ROLL，并用 FrozenLake agentic 强化学习跑通一次完整的训练闭环。

## 前置条件

- **硬件**：Atlas 900 A2 或 A3 训练系列产品，单机 8 张 NPU 卡，并按需完成物理机或容器内的设备挂载。
- **软件**：已装好 CANN（toolkit 与驱动），并能 `source set_env.sh`，Python 版本不低于 3.10。参考[快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html)。

本文档示例使用 Python 3.12、CANN 9.1.0。

本文档配套镜像：

- Atlas 900 A2：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12`
- Atlas 900 A3：`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-a3-ubuntu22.04-py3.12`

## 加载 CANN 环境

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## 安装 ROLL

**使用源码安装 ROLL** ：以下命令均在打开终端时所在的工作目录执行，`ROLL` 将克隆到该目录下。

<!--
```shell #test-setup store="upstream_ref"
echo "${UPSTREAM_REF}"
```
-->

```shell #test id="install-roll" load="upstream_ref>>ref"
git clone --branch <ref> https://github.com/alibaba/ROLL.git
echo "ROLL $(git -C ROLL describe --tags --exact-match HEAD)"
pip install -e ROLL
```

```{admonition} Note
:class: note
`<ref>` 为最新正式 release 的 tag
```

输出结果如下：

```shell #test-result id="install-roll" fuzzy='xxx'
ROLL xxx
```

```{admonition} Note
:class: note
输出中的 `xxx` 为实际安装到的 release 版本号
```

## 运行示例：FrozenLake agentic 强化学习

FrozenLake 是 ROLL 官方快速入门的示例：Qwen2.5-0.5B-Instruct 作为策略模型，在 4×4 冰面网格中逐轮输出移动方向，绕开冰洞到达终点，环境按结果返回奖励。

**安装 vLLM 与 triton。** 安装 `vllm-ascend` 时，配套的 `torch`、`torch_npu`、`torchvision` 和 `torchaudio` 会作为依赖自动安装，无需单独安装 PyTorch 软件栈。

```shell #test-setup id="install-npu-runtime"
set -e
pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple vllm==0.23.0
pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi vllm-ascend==0.23.0
pip install --index-url https://repo.huaweicloud.com/repository/pypi/simple --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi triton-ascend==3.2.2
```

核对安装的 vLLM 与 triton 版本，下面代码用python执行：

```python #test id="verify-vllm-triton"
from importlib.metadata import version

expected = {
    "vllm": "0.23.0",
    "vllm-ascend": "0.23.0",
    "triton-ascend": "3.2.2",
}
for package, required in expected.items():
    actual = version(package)
    assert actual == required, f"{package}: expected {required}, got {actual}"
    print(f"{package} {actual}")
```

输出结果如下：

```shell #test-result id="verify-vllm-triton"
vllm 0.23.0
vllm-ascend 0.23.0
triton-ascend 3.2.2
```

**安装示例依赖。** 按 ROLL 的依赖清单安装 agentic 示例所需组件。约束 PyTorch 软件栈版本以匹配 `vllm-ascend==0.23.0`，并选用与 ROLL 所需 NumPy 1.26 兼容的数据处理依赖：

```shell #test-setup id="install-example-deps"
set -e
cat > ROLL/constraints-quick-start.txt <<'EOF'
torch==2.10.0
torch-npu==2.10.0.post4
torchvision==0.25.0
torchaudio==2.10.0
numpy==1.26.4
pyarrow==20.0.0
pandas==2.2.3
scipy==1.16.3
EOF

grep -v '^gem-llm' ROLL/requirements_common.txt | sed 's#^\./mcore_adapter$#ROLL/mcore_adapter#' > ROLL/requirements_npu.txt
sed -i 's/^decord /decord2 /' ROLL/requirements_vision.txt
pip install -c ROLL/constraints-quick-start.txt -r ROLL/requirements_npu.txt
pip install -c ROLL/constraints-quick-start.txt --ignore-requires-python gem-llm==0.0.4
pip install -c ROLL/constraints-quick-start.txt "transformers==4.57.6" "tensorboard==2.20.0" "antlr4-python3-runtime==4.9.3"
rm ROLL/requirements_npu.txt ROLL/constraints-quick-start.txt
```

核对示例依赖的版本，下面代码用python执行：

```python #test id="verify-example-deps"
from importlib.metadata import version

expected = {
    "gem-llm": "0.0.4",
    "numpy": "1.26.4",
    "pyarrow": "20.0.0",
    "transformers": "4.57.6",
    "tensorboard": "2.20.0",
    "antlr4-python3-runtime": "4.9.3",
}
for package, required in expected.items():
    actual = version(package)
    assert actual == required, f"{package}: expected {required}, got {actual}"
    print(f"{package} {actual}")
```

输出结果如下：

```shell #test-result id="verify-example-deps"
gem-llm 0.0.4
numpy 1.26.4
pyarrow 20.0.0
transformers 4.57.6
tensorboard 2.20.0
antlr4-python3-runtime 4.9.3
```

**准备示例配置。** 从检出的 ROLL 源码读取[上游昇腾指南](https://github.com/alibaba/ROLL/blob/v0.4.0/docs_roll/docs/User%20Guides/Hardware%20Support/ascend_npu_examples.md)中的八卡 FrozenLake 配置。示例沿用原配置的 FSDP2 策略和卡位分配（卡 0–3 训练，卡 4–7 推理与参考模型）。为便于快速验证，训练缩为一步，训练采样和验证批量分别缩为 32 和 8，验证只跑 FrozenLake。ROLL 0.4 的 NPU 环境关闭 TransferQueue，训练产物写入当前目录的 `output/`。

```python #test id="write-config"
from pathlib import Path

import yaml

example_dir = Path("ROLL/examples/agentic_frozen_lake_npu")
guide = Path("ROLL/docs_roll/docs/User Guides/Hardware Support/ascend_npu_examples.md")
_, section_heading, section = guide.read_text(encoding="utf-8").partition(
    "## Example 1: Single-Node Agentic Pipeline"
)
section = section.split("\n## ", 1)[0]
_, yaml_fence, yaml_text = section.partition("```yaml\n")
if not section_heading or not yaml_fence:
    raise ValueError(f"未找到上游指南中的八卡 FrozenLake YAML：{guide}")
config = yaml.safe_load(yaml_text.split("\n```", 1)[0])
if config["num_gpus_per_node"] != 8:
    raise ValueError("上游 FrozenLake 配置已变更，请重新核对 Runner 和卡位分配")

config.update(
    exp_name="roll-quick-start-npu",
    max_steps=1,
    rollout_batch_size=32,
    val_batch_size=8,
    sequence_length=2048,
    tracker_kwargs={"log_dir": "./output/tensorboard"},
    transfer_backend={"backend_name": None},
)
config["checkpoint_config"]["output_dir"] = "./output/checkpoints"
config["actor_train"]["training_args"]["gradient_accumulation_steps"] = 4
config["train_env_manager"].update(num_env_groups=4, num_groups_partition=[4])
config["val_env_manager"].update(
    tags=["FrozenLake"], num_env_groups=8, num_groups_partition=[8]
)

example_dir.mkdir(parents=True, exist_ok=True)
path = example_dir / "quick_start_npu.yaml"
path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print("config written", path.as_posix())
```

输出结果如下：

```shell #test-result id="write-config"
config written ROLL/examples/agentic_frozen_lake_npu/quick_start_npu.yaml
```

**启动训练。** 在当前工作目录运行 agentic pipeline 入口脚本，ROLL 自动拉起 Ray 集群。

```shell #test-setup id="run-agentic"
python ROLL/examples/start_agentic_pipeline.py --config_path agentic_frozen_lake_npu --config_name quick_start_npu
```

**查看训练产物。** 训练完成后，TensorBoard 指标保存在当前工作目录的 `output/tensorboard/roll-quick-start-npu` 下。用 Python 执行下面的代码，查看训练结果指标：

```python #test id="verify-output"
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

tensorboard_dir = Path("output/tensorboard/roll-quick-start-npu")
event_files = sorted(tensorboard_dir.glob("*/events.out.tfevents.*"))
assert event_files, f"no tensorboard event file under {tensorboard_dir}"
metrics = EventAccumulator(str(event_files[-1]))
metrics.Reload()

print("训练产物已生成")
print(f"训练指标路径：{tensorboard_dir}")
print(f"训练步数：{len(metrics.Scalars('time/step_total'))}")
print(f"训练样本数：{int(metrics.Scalars('system/samples')[-1].value)}")
print(f"FrozenLake 平均得分：{metrics.Scalars('critic/score/mean')[-1].value:.3f}")
```

输出结果如下：

```shell #test-result id="verify-output" fuzzy='xxx'
训练产物已生成
训练指标路径：output/tensorboard/roll-quick-start-npu
训练步数：1
训练样本数：32
FrozenLake 平均得分：xxx
```

```{admonition} Note
:class: note
输出中的 `xxx` 为本次训练实际得到的平均得分，随训练随机性变化
```

## 外部链接

- GitHub：[alibaba/ROLL](https://github.com/alibaba/ROLL)
- 文档中心：[ROLL Docs](https://alibaba.github.io/ROLL/zh-Hans/)
