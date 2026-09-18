"""Fractional NPU resources share scheduling capacity, not memory isolation."""

import os

import ray


def test_fractional_npu_allocation() -> None:
    expected_id = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert expected_id and "," not in expected_id

    ray.init(include_dashboard=False, log_to_driver=False)

    @ray.remote(resources={"NPU": 0.25})
    def fractional_task() -> str:
        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1, ids
        visible = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
        assert visible == str(ids[0])
        return visible

    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 1
        assigned = ray.get([fractional_task.remote() for _ in range(4)], timeout=45)
        assert assigned == [expected_id] * 4
    finally:
        ray.shutdown()
