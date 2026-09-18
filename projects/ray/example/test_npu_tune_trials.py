"""Ray Tune trials must reserve NPUs and execute real NPU work."""

import asyncio
import os
import tempfile

import ray


def test_tune_trials_reserve_npus() -> None:
    from ray import tune

    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    expected = set(visible.split(",")) if visible else set()
    assert len(expected) == 2, expected

    @ray.remote(num_cpus=0, max_concurrency=2)
    class TrialBarrier:
        def __init__(self):
            self.arrivals = set()
            self.both_ready = asyncio.Event()

        async def arrive(self, device_id: str) -> list[str]:
            self.arrivals.add(device_id)
            if len(self.arrivals) == 2:
                self.both_ready.set()
            await asyncio.wait_for(self.both_ready.wait(), timeout=60)
            return sorted(self.arrivals)

    def objective(config: dict, barrier) -> None:
        import torch
        import torch_npu  # noqa: F401

        ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
        assert len(ids) == 1, ids
        physical_id = str(ids[0])
        assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == physical_id
        value = torch.ones(4, device="npu:0").sum().cpu().item()
        assert set(ray.get(barrier.arrive.remote(physical_id), timeout=70)) == expected
        tune.report({"npu_id": physical_id, "value": float(value)})

    ray.init(include_dashboard=False, log_to_driver=False)
    try:
        assert int(ray.cluster_resources().get("NPU", 0)) == 2
        barrier = TrialBarrier.remote()
        with tempfile.TemporaryDirectory() as results_dir:
            analysis = tune.run(
                tune.with_parameters(objective, barrier=barrier),
                resources_per_trial={"NPU": 1},
                num_samples=2,
                max_concurrent_trials=2,
                storage_path=results_dir,
                verbose=0,
                raise_on_failed_trial=True,
            )
            assert len(analysis.trials) == 2
            assert {trial.last_result["npu_id"] for trial in analysis.trials} == expected
            for trial in analysis.trials:
                assert trial.last_result["value"] == 4.0
    finally:
        ray.shutdown()
