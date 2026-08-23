"""Static contract for the Ray Ascend guard project."""

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


class TestRayProjectContract(unittest.TestCase):
    def test_project_registry_points_to_both_workflows(self) -> None:
        registry = yaml.safe_load(
            (_REPO_ROOT / "projects.yaml").read_text(encoding="utf-8")
        )
        ray_projects = [
            item for item in registry["projects"] if item["name"] == "ray"
        ]
        self.assertEqual(len(ray_projects), 1)
        ray_project = ray_projects[0]
        self.assertEqual(ray_project["upstream_repo"], "ray-project/ray")
        self.assertEqual(
            ray_project["workflows"],
            {
                "examples": ".github/workflows/ray-examples.yml",
                "quick_start": ".github/workflows/ray-quick-start.yml",
            },
        )

    def test_manifest_guards_only_the_two_selected_upstream_tests(self) -> None:
        manifest_path = _REPO_ROOT / "projects" / "ray" / "examples_manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        expected_paths = {
            "python/ray/tests/accelerators/test_npu.py",
            "python/ray/train/tests/test_torch_device_manager.py",
        }
        self.assertEqual(set(manifest["scan"]["paths"]), expected_paths)
        self.assertEqual(
            {entry["path"] for entry in manifest["supported"]}, expected_paths
        )
        self.assertEqual(
            {entry["profile"] for entry in manifest["supported"]},
            {"core", "train"},
        )
        self.assertEqual(manifest["unsupported"], [])

    def test_quick_start_covers_environment_detection_and_isolation(self) -> None:
        doc_path = _REPO_ROOT / "projects" / "ray" / "docs" / "Quick-start-Ascend.md"
        commands, results = _Parser().parse(doc_path.read_text(encoding="utf-8"))
        test_ids = {command.id for command in commands if hasattr(command, "id")}
        self.assertEqual(
            test_ids,
            {"check-environment", "ray-detects-npus", "ray-isolates-npus"},
        )
        self.assertEqual(set(results), test_ids)
        text = doc_path.read_text(encoding="utf-8")
        self.assertIn("ray-core/scheduling/accelerators", text)
        self.assertIn('resources={"NPU": 1}', text)
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES", text)

    def test_project_scripts_and_workflows_exist(self) -> None:
        expected = [
            "projects/ray/scripts/check_manifest.py",
            "projects/ray/scripts/setup_example.sh",
            "projects/ray/scripts/run_example.sh",
            "projects/ray/tests/test_quick_start_ascend.py",
            ".github/workflows/ray-examples.yml",
            ".github/workflows/ray-quick-start.yml",
        ]
        missing = [path for path in expected if not (_REPO_ROOT / path).is_file()]
        self.assertEqual(missing, [])

        quick_workflow = (
            _REPO_ROOT / ".github" / "workflows" / "ray-quick-start.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("uses: ./.github/workflows/quick-start-template.yml", quick_workflow)

    def test_example_setup_installs_the_target_commit_aarch64_wheel(self) -> None:
        setup_path = _REPO_ROOT / "projects" / "ray" / "scripts" / "setup_example.sh"
        text = setup_path.read_text(encoding="utf-8")
        self.assertIn("ray-wheels/master", text)
        self.assertIn("python/ray/_version.py", text)
        self.assertIn("manylinux2014_aarch64.whl", text)
        self.assertIn("requirement=\"ray[${extras}] @ ${wheel_url}\"", text)
        self.assertIn("install_target_ray train", text)

    def test_example_workflow_uses_the_project_local_checker(self) -> None:
        workflow_path = _REPO_ROOT / ".github" / "workflows" / "ray-examples.yml"
        text = workflow_path.read_text(encoding="utf-8")
        self.assertIn("python/ray/tests/accelerators/test_npu.py", text)
        self.assertIn("python/ray/train/tests/test_torch_device_manager.py", text)
        self.assertIn("projects/ray/scripts/check_manifest.py", text)
        self.assertIn("sha=master", text)


if __name__ == "__main__":
    unittest.main()
