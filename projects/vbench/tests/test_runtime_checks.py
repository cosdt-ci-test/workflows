"""Real CPU tensors exercise test-side probes; NPU acceptance stays in CI."""

import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import torch


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/runtime_checks.py"


class TestRuntimeChecks(unittest.TestCase):
    def _module(self):
        self.assertTrue(_SCRIPT.is_file(), "runtime validation should live in the guard")
        spec = importlib.util.spec_from_file_location("vbench_runtime_checks", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_checkpoint_allows_generated_position_and_rejects_missing_weights(self):
        checks = self._module()
        model = torch.nn.Linear(4, 2)
        model.register_parameter("pos_embed", torch.nn.Parameter(torch.zeros(1, 4)))
        state = {key: value for key, value in model.state_dict().items() if key != "pos_embed"}
        checks.check_checkpoint(model, state)
        del state["weight"]
        with self.assertRaisesRegex(RuntimeError, "weight"):
            checks.check_checkpoint(model, state)

    def test_hook_observes_real_forwards_and_restores_factory(self):
        checks = self._module()
        model = torch.nn.Linear(4, 2)
        factory = lambda: model
        module = types.SimpleNamespace(create_model=factory)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save(model.state_dict(), checkpoint)
            with checks.watch_human_action(module, checkpoint, "cpu") as observed:
                evaluated = module.create_model()
                for _ in range(5):
                    evaluated(torch.ones(1, 4))
            self.assertEqual(observed, ["cpu"] * 5)
        self.assertIs(module.create_model, factory)
        self.assertFalse(model._forward_hooks)

    def test_probe_rejects_wrong_device(self):
        checks = self._module()
        with self.assertRaisesRegex(AssertionError, "device"):
            checks.check_forward(torch.nn.Linear(4, 2), (torch.ones(1, 4),),
                                 torch.ones(1, 2), torch.device("meta"))

    def test_probe_rejects_nonfinite_logits(self):
        checks = self._module()
        with self.assertRaisesRegex(AssertionError, "finite"):
            checks.check_forward(torch.nn.Linear(4, 2), (torch.ones(1, 4),),
                                 torch.full((1, 2), float("nan")), torch.device("cpu"))

    def test_factory_and_hooks_are_restored_when_evaluation_fails(self):
        checks = self._module()
        model = torch.nn.Linear(4, 2)
        factory = lambda: model
        module = types.SimpleNamespace(create_model=factory)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save(model.state_dict(), checkpoint)
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                with checks.watch_human_action(module, checkpoint, "cpu"):
                    module.create_model()
                    raise RuntimeError("evaluation failed")
        self.assertIs(module.create_model, factory)
        self.assertFalse(model._forward_hooks)

    def _api_fixture(self, root):
        model = torch.nn.Linear(4, 2)
        module = types.SimpleNamespace(create_model=lambda: model)
        checkpoint = root / "model.pth"
        torch.save(model.state_dict(), checkpoint)
        output = root / "output"
        videos = [{"video_path": f"video-{i}.mp4", "video_results": False,
                   "cor_num_per_video": 0} for i in range(5)]
        for entry in ("cli", "api"):
            directory = output / entry
            directory.mkdir(parents=True)
            (directory / "example_eval_results.json").write_text(
                json.dumps({"human_action": [0.0, videos]}))
        return model, module, checkpoint, output

    def test_api_guard_checks_real_forwards_and_both_result_files(self):
        checks = self._module()
        self.assertTrue(hasattr(checks, "check_api"))
        with tempfile.TemporaryDirectory() as directory:
            model, module, checkpoint, output = self._api_fixture(Path(directory))
            with checks.check_api(module, checkpoint, device="cpu", output_root=output):
                evaluated = module.create_model()
                for _ in range(5):
                    evaluated(torch.ones(1, 4))
            self.assertFalse(model._forward_hooks)

    def test_api_guard_rejects_an_example_that_does_not_run_the_model(self):
        checks = self._module()
        self.assertTrue(hasattr(checks, "check_api"))
        with tempfile.TemporaryDirectory() as directory:
            _, module, checkpoint, output = self._api_fixture(Path(directory))
            with self.assertRaisesRegex(AssertionError, "five forwards"):
                with checks.check_api(module, checkpoint, device="cpu", output_root=output):
                    pass

    def test_api_guard_rejects_invalid_scores_and_removes_hooks(self):
        checks = self._module()
        self.assertTrue(hasattr(checks, "check_api"))
        with tempfile.TemporaryDirectory() as directory:
            model, module, checkpoint, output = self._api_fixture(Path(directory))
            result = output / "api/example_eval_results.json"
            data = json.loads(result.read_text())
            data["human_action"][0] = float("nan")
            result.write_text(json.dumps(data))
            with self.assertRaisesRegex(AssertionError, "score"):
                with checks.check_api(module, checkpoint, device="cpu", output_root=output):
                    evaluated = module.create_model()
                    for _ in range(5):
                        evaluated(torch.ones(1, 4))
            self.assertFalse(model._forward_hooks)


if __name__ == "__main__":
    unittest.main()
