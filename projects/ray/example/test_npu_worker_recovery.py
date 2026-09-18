"""An NPU task failure must not leave the device permanently reserved."""

import os

import pytest
import ray


def test_npu_is_reusable_after_worker_error() -> None:
    expected_id = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert expected_id and "," not in expected_id

    ray.init(include_dashboard=False, log_to_driver=False)

    @ray.remote(resources={"NPU": 1}, max_retries=0)
    def failing_task() -> None:
        import torch
        import torch_npu  # noqa: F401

        assert torch.ones(1, device="npu:0").cpu().item() == 1.0
        raise RuntimeError("intentional NPU task failure")

    @ray.remote(resources={"NPU": 1})
    def replacement_task() -> tuple[str, float]:
        import torch
        import torch_npu  # noqa: F401

        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == str(ids[0])
        return str(ids[0]), float(torch.ones(2, device="npu:0").sum().cpu().item())

    try:
        with pytest.raises(ray.exceptions.RayTaskError, match="intentional NPU"):
            ray.get(failing_task.remote(), timeout=45)
        assert ray.get(replacement_task.remote(), timeout=60) == (expected_id, 2.0)
    finally:
        ray.shutdown()
