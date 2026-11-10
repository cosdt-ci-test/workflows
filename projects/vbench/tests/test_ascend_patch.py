"""Exercise the release patch against actual upstream sources.

VBENCH_UPSTREAM_ROOT supplies a clone; VBENCH_UPSTREAM_REF defaults to the
validated release v0.1.4. The workflow still resolves the latest release.
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
_REF = os.environ.get("VBENCH_UPSTREAM_REF", "v0.1.4")
_FILES = (
    "setup.py", "MANIFEST.in", "requirements.txt", "evaluate.py",
    "vbench/__init__.py", "vbench/distributed.py", "vbench/launch/evaluate.py",
    "vbench/cli/evaluate.py", "vbench/human_action.py",
    "vbench/third_party/umt/models/modeling_finetune.py",
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(_UPSTREAM, "set VBENCH_UPSTREAM_ROOT to test the actual release patch")
class TestAscendPatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        archive = subprocess.run(
            ["git", "-C", _UPSTREAM, "archive", _REF, *_FILES],
            capture_output=True, check=True,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(cls.root, filter="data")
        cls.original = {name: (cls.root / name).read_bytes() for name in _FILES}
        subprocess.run(
            [sys.executable, str(_PROJECT / "scripts/apply_patches.py"), str(cls.root)],
            check=True,
        )

    def _initialize(self, **globals):
        tree = ast.parse((self.root / "vbench/launch/evaluate.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "initialize_device"]
        self.assertEqual(len(functions), 1, "device initialization belongs in the CLI launcher")
        exec(compile(ast.Module(body=functions, type_ignores=[]), "launcher.py", "exec"), globals)
        return globals["initialize_device"]

    def test_api_root_entry_and_shared_distributed_code_are_unchanged(self):
        for name in ("evaluate.py", "vbench/__init__.py", "vbench/distributed.py"):
            with self.subTest(name=name):
                self.assertEqual((self.root / name).read_bytes(), self.original[name])
        self.assertFalse((self.root / "vbench/device.py").exists())

    def test_human_action_scoring_and_result_schema_are_unchanged(self):
        def function(source):
            tree = ast.parse(source)
            result = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name == "human_action")
            # A single device log is the only instrumentation in upstream code.
            result.body = [node for node in result.body if not (
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "print"
            )]
            return ast.dump(result)
        self.assertEqual(function((self.root / "vbench/human_action.py").read_text()),
                         function(self.original["vbench/human_action.py"].decode()))

    def test_single_npu_does_not_initialize_collectives(self):
        npu = types.SimpleNamespace(is_available=lambda: True, set_device=mock.Mock())
        torch = types.SimpleNamespace(npu=npu, device=lambda value: value)
        dist_init = mock.Mock()
        initialize = self._initialize(torch=torch, os=os, dist_init=dist_init)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with mock.patch.dict(sys.modules, {"torch_npu": types.ModuleType("torch_npu")}):
                self.assertEqual(initialize("npu:0"), "npu:0")
        dist_init.assert_not_called()
        npu.set_device.assert_called_once_with("npu:0")

    def test_cuda_keeps_upstream_distributed_initialization(self):
        import torch

        dist_init = mock.Mock()
        initialize = self._initialize(torch=torch, os=os, dist_init=dist_init)
        self.assertEqual(initialize("cuda"), torch.device("cuda"))
        dist_init.assert_called_once_with()

    def test_missing_npu_package_fails_without_cpu_fallback(self):
        import torch

        initialize = self._initialize(torch=torch, os=os, dist_init=mock.Mock())
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with mock.patch.dict(sys.modules, {"torch_npu": None}):
                with self.assertRaises(ImportError):
                    initialize("npu:0")

    def test_unavailable_npu_fails_without_initializing_cuda(self):
        npu = types.SimpleNamespace(is_available=lambda: False)
        dist_init = mock.Mock()
        initialize = self._initialize(torch=types.SimpleNamespace(npu=npu), os=os,
                                      dist_init=dist_init)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with mock.patch.dict(sys.modules, {"torch_npu": types.ModuleType("torch_npu")}):
                with self.assertRaisesRegex(RuntimeError, "not available"):
                    initialize("npu:0")
        dist_init.assert_not_called()

    def test_multi_npu_is_rejected_before_collectives(self):
        dist_init = mock.Mock()
        initialize = self._initialize(torch=None, os=os, dist_init=dist_init)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaisesRegex(RuntimeError, "single NPU"):
                initialize("npu:0")
        dist_init.assert_not_called()

    def test_cli_still_passes_device_and_propagates_child_failure(self):
        cli = _load(self.root / "vbench/cli/evaluate.py", "vbench_cli_test")
        args = argparse.Namespace(ngpus=1, videos_path="videos", dimension="human_action",
                                  device="npu:0", output_path="results", func=None)
        error = subprocess.CalledProcessError(7, "evaluation")
        with mock.patch.object(cli.subprocess, "run", side_effect=error) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                cli.evaluate(args)
        self.assertIn("--device npu:0", run.call_args.args[0])
        self.assertTrue(run.call_args.kwargs["check"])

    def test_install_profile_leaves_version_pins_in_project_constraints(self):
        tree = ast.parse((self.root / "setup.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "fetch_requirements")
        namespace = {"os": os, "__file__": str(self.root / "setup.py")}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "setup.py", "exec"), namespace)
        with mock.patch.dict(os.environ, {"VBENCH_INSTALL_PROFILE": "human_action"}):
            dependencies = namespace["fetch_requirements"]()
        self.assertIn("timm>=0.9", dependencies)
        self.assertIn("decord", dependencies)
        self.assertFalse(any("pyiqa" in line or "detectron2" in line for line in dependencies))
        with mock.patch.dict(os.environ, {"VBENCH_INSTALL_PROFILE": "full"}):
            self.assertEqual(namespace["fetch_requirements"](),
                             [line for line in self.original["requirements.txt"].decode().splitlines()
                              if "@" not in line])

    def test_patch_is_single_and_all_patched_python_compiles(self):
        self.assertEqual(len(list((_PROJECT / "patches").glob("*.patch"))), 1)
        for path in self.root.rglob("*.py"):
            compile(path.read_text(), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
