"""Exercise patched upstream code, without pretending CPU tests prove NPU support.

Set VBENCH_UPSTREAM_ROOT to a pristine upstream clone with v0.1.4 and master
available. The NPU document test remains the hardware acceptance test.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


_PROJECT = Path(__file__).resolve().parents[1]
_UPSTREAM = os.environ.get("VBENCH_UPSTREAM_ROOT")
_FILES = (
    "setup.py", "MANIFEST.in", "evaluate.py", "vbench/__init__.py", "vbench/distributed.py",
    "vbench/launch/evaluate.py", "vbench/cli/evaluate.py", "vbench/human_action.py",
    "vbench/third_party/umt/models/modeling_finetune.py",
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(_UPSTREAM, "set VBENCH_UPSTREAM_ROOT to test the actual upstream patch")
class TestAscendPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.roots = {}
        for ref in ("v0.1.4", "master"):
            root = Path(cls.temp.name) / ref
            root.mkdir()
            archive = subprocess.run(
                ["git", "-C", _UPSTREAM, "archive", ref, *_FILES],
                capture_output=True, check=True,
            ).stdout
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(root, filter="data")
            cls.roots[ref] = root

    def _patched(self, ref="v0.1.4"):
        root = self.roots[ref]
        if not (root / "vbench/device.py").exists():
            script = _PROJECT / "scripts/apply_patches.py"
            self.assertTrue(script.is_file(), "Ascend patch application is not implemented")
            subprocess.run([sys.executable, str(script), str(root)], check=True)
        return root

    def test_patch_applies_to_release_and_master_and_all_python_compiles(self):
        for ref in self.roots:
            with self.subTest(ref=ref):
                root = self._patched(ref)
                for path in root.rglob("*.py"):
                    compile(path.read_text(), str(path), "exec")

    def test_cpu_device_is_explicit_and_unavailable_npu_fails(self):
        import torch

        device = _load(self._patched() / "vbench/device.py", "vbench_device_test")
        self.assertEqual(device.get_device("cpu"), torch.device("cpu"))
        with mock.patch.dict(sys.modules, {"torch_npu": None}):
            with self.assertRaisesRegex(RuntimeError, "torch_npu"):
                device.get_device("npu:0")

    def test_single_process_does_not_start_nccl(self):
        import torch

        distributed = _load(self._patched() / "vbench/distributed.py", "dist_test")
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with mock.patch.object(torch.distributed, "init_process_group") as init:
                distributed.dist_init(torch.device("cpu"))
        init.assert_not_called()
        self.assertEqual(distributed.get_world_size(), 1)

    def test_cli_preserves_paths_and_propagates_child_failure(self):
        root = self._patched()
        cli = _load(root / "vbench/cli/evaluate.py", "cli_test")
        args = argparse.Namespace(
            ngpus=1, videos_path="videos with spaces", dimension="human_action",
            device="npu:0", output_path="results with spaces", func=None,
        )
        error = subprocess.CalledProcessError(7, ["python"])
        with mock.patch.object(cli.subprocess, "run", side_effect=error) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                cli.evaluate(args)
        argv = run.call_args.args[0]
        self.assertIsInstance(argv, list)
        self.assertEqual(argv[argv.index("--videos_path") + 1], args.videos_path)
        self.assertEqual(argv[argv.index("--device") + 1], "npu:0")
        self.assertTrue(run.call_args.kwargs["check"])

    def test_human_action_install_profile_does_not_resolve_other_evaluators(self):
        root = self._patched()
        tree = ast.parse((root / "setup.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "fetch_requirements")
        namespace = {"os": os, "__file__": str(root / "setup.py")}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "setup.py", "exec"), namespace)
        with mock.patch.dict(os.environ, {"VBENCH_INSTALL_PROFILE": "human_action"}):
            dependencies = namespace["fetch_requirements"]()
        self.assertIn("timm==0.9.12", dependencies)
        self.assertTrue(any(line.startswith("decord") for line in dependencies))
        self.assertFalse(any("pyiqa" in line or "detectron2" in line for line in dependencies))

    def test_checkpoint_allows_generated_position_but_rejects_missing_weights(self):
        import torch

        root = self._patched()
        tree = ast.parse((root / "vbench/human_action.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "_load_weights"]
        self.assertEqual(len(functions), 1, "checkpoint validation must preserve generated position embeddings")
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "human_action.py", "exec"), namespace)
        model = torch.nn.Linear(4, 2)
        model.register_parameter("pos_embed", torch.nn.Parameter(torch.zeros(1, 4)))
        weights = {key: value for key, value in model.state_dict().items() if key != "pos_embed"}
        namespace["_load_weights"](model, weights)
        del weights["weight"]
        with self.assertRaisesRegex(RuntimeError, "weight"):
            namespace["_load_weights"](model, weights)

    def test_original_umt_forward_uses_16_frame_tensor(self):
        import torch
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Overwriting vit_large_patch16_224")
            umt = _load(self._patched() / "vbench/third_party/umt/models/modeling_finetune.py", "vbench_umt_test")
        # Small test configuration of the actual upstream implementation;
        # the Quick Start itself always loads the original ViT-L checkpoint.
        model = umt.VisionTransformer(img_size=32, patch_size=16, embed_dim=32,
                                      depth=1, num_heads=4, num_classes=400,
                                      all_frames=16, tubelet_size=1)
        model.eval()
        with torch.inference_mode():
            logits = model(torch.randn(1, 3, 16, 32, 32))
        self.assertEqual(tuple(logits.shape), (1, 400))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
