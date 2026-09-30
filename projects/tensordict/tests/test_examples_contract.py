"""Static contract for the first TensorDict NPU tutorial guard."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "projects" / "tensordict"
MANIFEST_PATH = PROJECT / "examples_manifest.yaml"
TUTORIALS = {
    "data_fashion.py",
    "export.py",
    "functional.py",
    "serialization_speed.py",
    "streamed_tensordict.py",
    "tensorclass_fashion.py",
    "tensorclass_imagenet.py",
    "tensordict_keys.py",
    "tensordict_memory.py",
    "tensordict_module.py",
    "tensordict_preallocation.py",
    "tensordict_shapes.py",
    "tensordict_slicing.py",
    "zarr_storage.py",
}
SUPPORTED = {
    "tensordict_keys.py",
    "tensordict_shapes.py",
    "tensordict_preallocation.py",
}


class TensorDictExamplesContract(unittest.TestCase):
    def test_manifest_accounts_for_all_upstream_tutorials(self) -> None:
        text = MANIFEST_PATH.read_text(encoding="utf-8")
        manifest = yaml.safe_load(text)
        self.assertEqual(manifest["scan"]["root"], "tutorials/sphinx_tuto")
        self.assertEqual(manifest["scan"]["include_extensions"], [".py"])

        supported = manifest["supported"]
        unsupported = manifest["unsupported"]
        supported_paths = {item["path"] for item in supported}
        unsupported_paths = set(unsupported)
        prefix = "tutorials/sphinx_tuto/"
        self.assertEqual({path.removeprefix(prefix) for path in supported_paths}, SUPPORTED)
        self.assertFalse(supported_paths & unsupported_paths)
        self.assertEqual(
            {path.removeprefix(prefix) for path in supported_paths | unsupported_paths},
            TUTORIALS,
        )
        self.assertEqual(len(supported), 3)
        self.assertEqual(len(unsupported), 11)

        lines = text.splitlines()
        for item in supported:
            path = item["path"]
            self.assertEqual(item["profile"], "tensordict_npu")
            self.assertEqual(item["runner"], "linux-aarch64-a2-1")
            self.assertIn("cann:9.1.0-910b", item["image"])
            self.assertGreater(item["timeout_minutes"], 0)
            path_line = next(i for i, line in enumerate(lines) if line.strip() == f"- path: {path}")
            self.assertTrue(lines[path_line - 1].lstrip().startswith("# "), path)
        for path in unsupported:
            line = next(line for line in lines if line.lstrip().startswith(f"- {path}  # "))
            self.assertTrue(any("\u4e00" <= char <= "\u9fff" for char in line), path)

    def test_launcher_requires_real_npu_results(self) -> None:
        run = (PROJECT / "scripts" / "run_example.sh").read_text(encoding="utf-8")
        setup = (PROJECT / "scripts" / "setup_example.sh").read_text(encoding="utf-8")
        self.assertIn('torch.set_default_device("npu:0")', run)
        self.assertIn('runpy.run_path(sys.argv[1], run_name="__main__")', run)
        self.assertIn('leaf.device.type != "npu"', run)
        self.assertIn('"NPU is unavailable; refusing CPU fallback"', run)
        self.assertIn('uv pip install --system --no-deps -e "$TARGET_ROOT"', setup)
        self.assertNotIn("pip install torch", setup)

    def test_thin_trigger_and_registry(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "tensordict-examples.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("uses: ./.github/workflows/examples-template.yml", workflow)
        self.assertIn("upstream_repo: pytorch/tensordict", workflow)
        self.assertIn("# schedule:", workflow)
        self.assertNotIn("  schedule:\n", workflow)
        projects = (ROOT / "projects.yaml").read_text(encoding="utf-8")
        self.assertIn("examples: .github/workflows/tensordict-examples.yml", projects)


if __name__ == "__main__":
    unittest.main()
