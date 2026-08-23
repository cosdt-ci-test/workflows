# Ray Ascend NPU Quick Start

This guide turns the Huawei Ascend examples in Ray's upstream
[Accelerator Support](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html)
document into a real two-device smoke test. It verifies Ray's native `NPU`
resource, `ASCEND_RT_VISIBLE_DEVICES` isolation, and a real `torch_npu`
operation inside each Ray worker.

## Prerequisites

- Linux aarch64 with two visible Ascend NPUs;
- CANN toolkit and driver available in the container;
- a matching `torch` and `torch_npu` stack;
- Python 3.12, supported by Ray's Linux aarch64 wheels.

Install the current Ray release with dashboard support:

```shell #test-setup
python -m pip install -U "ray[default]"
```

Check the environment before starting Ray:

```shell #test id="check-environment"
python - <<'PY'
import ray
import torch
import torch_npu

print("ray=", ray.__version__)
print("torch=", torch.__version__)
print("torch_npu=", torch_npu.__version__)
print("npu_available=", torch.npu.is_available())
print("npu_count=", torch.npu.device_count())
PY
```

```shell #test-result id="check-environment" fuzzy="xxx"
ray= xxx
torch= xxx
torch_npu= xxx
npu_available= True
npu_count= 2
```

## Verify automatic NPU discovery

Ray detects Ascend devices through AscendCL, with `/dev/davinci*` as a
fallback, and publishes them as the logical `NPU` resource.

```shell #test id="ray-detects-npus"
python - <<'PY'
import ray
from ray._private.accelerators import NPUAcceleratorManager

ray.init(include_dashboard=False, log_to_driver=False)
resources = ray.cluster_resources()
count = int(resources.get("NPU", 0))
assert count == 2, resources
print("Ray NPU resources:", count)
print("Ascend type:", NPUAcceleratorManager.get_current_node_accelerator_type())
ray.shutdown()
PY
```

```shell #test-result id="ray-detects-npus" fuzzy="xxx"
Ray NPU resources: 2
Ascend type: xxx
```

## Verify Task and Actor isolation

This follows the upstream Task/Actor example. The actor keeps one NPU
reserved while the task receives the other. Ray writes the allocated physical
ID into `ASCEND_RT_VISIBLE_DEVICES`; `torch_npu` then performs a real operation
through that isolated view.

```shell #test id="ray-isolates-npus"
python - <<'PY'
import os
import ray

ray.init(resources={"NPU": 2}, include_dashboard=False, log_to_driver=False)

def assigned_npu():
    import torch
    import torch_npu

    ids = [str(value) for value in ray.get_runtime_context().get_accelerator_ids()["NPU"]]
    assert len(ids) == 1, ids
    visible = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    assert visible == ids[0], (visible, ids)
    value = torch.ones(4, device="npu:0").sum().cpu().item()
    return ids[0], value

@ray.remote(resources={"NPU": 1})
class NPUActor:
    def ping(self):
        return assigned_npu()

@ray.remote(resources={"NPU": 1})
def npu_task():
    return assigned_npu()

actor = NPUActor.remote()
actor_result = ray.get(actor.ping.remote())
task_result = ray.get(npu_task.remote())
ids = sorted([actor_result[0], task_result[0]])
values = sorted([actor_result[1], task_result[1]])
assert ids == ["0", "1"], ids
assert values == [4.0, 4.0], values
print("assigned NPU IDs:", ",".join(ids))
print("tensor sums:", ",".join(str(value) for value in values))
ray.kill(actor)
ray.shutdown()
PY
```

```shell #test-result id="ray-isolates-npus"
assigned NPU IDs: 0,1
tensor sums: 4.0,4.0
```

Ray's NPU resource is logical. Fractional requests such as
`resources={"NPU": 0.25}` allow sharing but do not provide memory or compute
isolation; applications remain responsible for safe sharing.
