"""Real CPU tensors exercise test-side probes; NPU acceptance stays in CI."""

import importlib.util
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


if __name__ == "__main__":
    unittest.main()
