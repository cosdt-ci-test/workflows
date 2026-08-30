"""Static contract for the TensorFlow 1.15 + TF Adapter 9.1.0 guard."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
for _path in (_SRC, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from workflows.markdown_doc_test_base import MarkdownDocTestBase  # noqa: E402


class _Parser(MarkdownDocTestBase):
    pass


class TestTensorFlowProjectContract(unittest.TestCase):
    def test_project_registry_points_to_the_quick_start_workflow(self) -> None:
        registry = yaml.safe_load(
            (_REPO_ROOT / "projects.yaml").read_text(encoding="utf-8")
        )
        projects = [
            item for item in registry["projects"] if item["name"] == "tensorflow"
        ]
        self.assertEqual(len(projects), 1)
        project = projects[0]
        self.assertEqual(project["upstream_repo"], "tensorflow/tensorflow")
        self.assertEqual(project["runner"], "linux-aarch64-a2-1")
        self.assertEqual(
            project["workflows"],
            {"quick_start": ".github/workflows/tensorflow-quick-start.yml"},
        )

    def test_workflow_uses_fixed_tf115_ref_on_one_a2_npu(self) -> None:
        workflow = (
            _REPO_ROOT / ".github" / "workflows" / "tensorflow-quick-start.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", workflow)
        self.assertIn("project: tensorflow", workflow)
        self.assertIn("test_runner: '[\"linux-aarch64-a2-1\"]'", workflow)
        self.assertIn("fixed_ref: v1.15.0", workflow)
        self.assertIn("upstream_repo: tensorflow/tensorflow", workflow)
        self.assertIn(
            "cann:9.1.0-910b-ubuntu22.04-py3.12-devel",
            workflow,
        )
        self.assertIn("container_options: ''", workflow)
        self.assertNotIn("--device=/dev/davinci", workflow)
        self.assertNotIn("/usr/local/Ascend/driver", workflow)

    def test_shared_engine_supports_an_optional_fixed_ref(self) -> None:
        template = (
            _REPO_ROOT / ".github" / "workflows" / "quick-start-template.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("fixed_ref:", template)
        self.assertIn("FIXED_REF: ${{ inputs.fixed_ref }}", template)
        self.assertIn('if [ -n "$FIXED_REF" ]; then', template)
        self.assertIn('ref="$FIXED_REF"', template)
        self.assertIn("/commits/$FIXED_REF", template)

    def test_doc_pins_the_official_compatible_stack_and_npu_flow(self) -> None:
        doc_path = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "docs"
            / "Quick-start-Ascend.md"
        )
        text = doc_path.read_text(encoding="utf-8")
        commands, results = _Parser().parse(text)
        test_ids = {command.id for command in commands if hasattr(command, "id")}
        self.assertEqual(
            test_ids,
            {
                "check-python",
                "install-tensorflow",
                "install-tf-adapter",
                "run-npu-optimizer",
            },
        )
        self.assertEqual(set(results), test_ids)
        test_commands = {
            command.id: command.cmd
            for command in commands
            if hasattr(command, "id")
        }
        self.assertNotIn("pip install", test_commands["install-tensorflow"])
        self.assertNotIn("curl ", test_commands["install-tensorflow"])
        self.assertNotIn("pip install", test_commands["install-tf-adapter"])
        self.assertNotIn("curl ", test_commands["install-tf-adapter"])

        required_fragments = (
            "| Python | 3.7.10 |",
            "| TensorFlow | 1.15.0 (`v1.15.0`) |",
            "| CANN | 9.1.0 |",
            "| TF Adapter branch | 9.1.0 |",
            "| TF Adapter wheel release | `tfa_v0.0.49_9.1.0` |",
            "| npu_bridge | 1.15.0 |",
            "| HDF5 | 1.10.5 |",
            "| h5py | 2.8.0 |",
            "tensorflow-1.15.0-cp37-cp37m-manylinux2014_aarch64.whl",
            "npu_bridge-1.15.0-py3-none-manylinux2014_aarch64.whl",
            "from npu_bridge.npu_init import *",
            'custom_op.name = "NpuOptimizer"',
            "RewriterConfig.OFF",
            "ASCEND_DEVICE_ID=0",
            "TensorFlow Ascend Quick Start PASSED",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)

        official_paths = (
            "docs/zh/tfadapter_1/installation/tensorflow-1-15_install.md",
            "docs/zh/tfadapter_1/installation/tfadapter_install.md",
            "docs/zh/tfadapter_1/quick_start.md",
        )
        for path in official_paths:
            with self.subTest(path=path):
                self.assertIn(path, text)

    def test_e2e_test_sources_cann_and_is_npu_gated(self) -> None:
        test_file = (
            _REPO_ROOT
            / "projects"
            / "tensorflow"
            / "tests"
            / "test_quick_start_ascend.py"
        ).read_text(encoding="utf-8")
        self.assertIn("MarkdownDocTestBase", test_file)
        self.assertIn("sys.path.insert", test_file)
        self.assertIn("NPU_READY", test_file)
        self.assertIn("/usr/local/Ascend/cann/set_env.sh", test_file)
        self.assertIn("TFPLUGIN_INSTALL_PATH", test_file)
        self.assertIn("ASCEND_DEVICE_ID", test_file)
        self.assertIn("ERR99999", test_file)


if __name__ == "__main__":
    unittest.main()
