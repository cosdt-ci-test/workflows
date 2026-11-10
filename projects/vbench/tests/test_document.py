"""Validate runnable document blocks and the reusable-workflow interface."""

import ast
import json
import re
import subprocess
import unittest
from pathlib import Path

import yaml
from workflows.markdown_doc_test_base import MarkdownDocTestBase, TestCommand


_PROJECT = Path(__file__).resolve().parents[1]
_ROOT = _PROJECT.parents[1]


class TestDocument(unittest.TestCase):
    def test_guide_uses_dynamic_vbench_release_with_explicit_dependencies(self):
        text = (_PROJECT / "docs/Quick-start-Ascend.md").read_text()
        visible = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        self.assertIn("| VBench | xxx |", visible)
        self.assertIn("xxx 表示最新的版本号。", visible)
        self.assertIn("`<ref>` 替换为 VBench 当前最新 release 的标签", visible)
        self.assertNotIn("v0.1.4", visible)
        self.assertIn("2.9.0.post2", visible)
        self.assertIn("0.24.0", visible)

    def test_python_checks_are_standalone_and_explained(self):
        text = (_PROJECT / "docs/Quick-start-Ascend.md").read_text()
        visible = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        self.assertNotIn("python - <<'PY'", visible)
        blocks = list(re.finditer(r'(?m)^```python #test id="([^"]+)"', visible))
        self.assertEqual(
            {block.group(1) for block in blocks},
            {"install-torch", "install-vbench", "human-action-api"},
        )
        for block in blocks:
            introduction = visible[:block.start()].rstrip().splitlines()[-1]
            self.assertIn("以下代码用 Python 执行", introduction)

    def test_each_expected_output_is_introduced_for_readers(self):
        text = (_PROJECT / "docs/Quick-start-Ascend.md").read_text()
        visible = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        outputs = list(re.finditer(r'(?m)^```text(?: #test-result id="[^"]+")?', visible))
        self.assertEqual(len(outputs), 7)
        for block in outputs:
            introduction = visible[:block.start()].rstrip().splitlines()[-1]
            self.assertEqual(introduction, "输出结果如下：")

    def test_guide_names_the_tested_ascend_runtime(self):
        text = (_PROJECT / "docs/Quick-start-Ascend.md").read_text()
        visible = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        self.assertTrue(visible.startswith("# VBench\n"))
        self.assertIn(
            "swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-ubuntu22.04-py3.12",
            visible,
        )
        for component, version in (
            ("Python", "3.12.13"),
            ("CANN", "9.1.0"),
            ("torch", "2.9.0+cpu"),
            ("torch_npu", "2.9.0.post2"),
        ):
            self.assertIn(f"| {component} | {version} |", visible)
        self.assertIn("Python 3.12.13", visible)
        self.assertNotIn("Python 3.12...", visible)

    def test_rendered_guide_is_for_vbench_users_not_runner_maintainers(self):
        text = (_PROJECT / "docs/Quick-start-Ascend.md").read_text()
        visible = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        for internal in ("cache-seed", "manifest", "/data/ci-cache", "Runner", "CI",
                         "工作流", "看护", "runtime_checks", "watch_human_action"):
            with self.subTest(internal=internal):
                self.assertFalse(internal in visible, f"Maintainer-only content: {internal}")
        self.assertIn("python -m pip install", visible)
        self.assertIn("uv", visible)
        self.assertIn("constraints-npu.txt", visible)
        self.assertIn("<ref>", visible)

    def test_user_api_example_only_needs_the_public_vbench_api(self):
        runner = type("Parser", (MarkdownDocTestBase,), {})()
        commands, _ = runner.parse((_PROJECT / "docs/Quick-start-Ascend.md").read_text())
        command = next(c for c in commands if isinstance(c, TestCommand)
                       and c.id == "human-action-api")
        tree = ast.parse(command.cmd)
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertEqual(imports, {"vbench"})
        self.assertFalse(any(isinstance(node, (ast.Assert, ast.With)) for node in ast.walk(tree)))
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and node.func.attr == "evaluate" for node in ast.walk(tree)))

    def test_runner_wraps_the_exact_user_api_example_with_test_side_checks(self):
        from projects.vbench.tests import test_quick_start_ascend as guard

        body = 'print("original user example")\n'
        self.assertTrue(hasattr(guard.TestQuickStartAscend, "_checked_api"),
                        "NPU and result validation must move out of the user example")
        checked = guard.TestQuickStartAscend._checked_api(body)
        tree = ast.parse(checked)
        context = next(node for node in tree.body if isinstance(node, ast.With))
        self.assertEqual(context.items[0].context_expr.func.id, "check_api")
        prefix = tree.body[:tree.body.index(context)]
        self.assertTrue(any(isinstance(node, ast.Import) and
                            any(alias.name == "torch_npu" for alias in node.names)
                            for node in prefix), "Register the NPU backend before creating the device probe")
        self.assertEqual([ast.dump(node) for node in context.body],
                         [ast.dump(node) for node in ast.parse(body).body])

    def test_every_document_step_has_valid_syntax_and_expected_output(self):
        runner = type("Parser", (MarkdownDocTestBase,), {})()
        commands, results = runner.parse((_PROJECT / "docs/Quick-start-Ascend.md").read_text())
        self.assertEqual(len(results), 7)
        for command in commands:
            body = command.cmd.replace("<ref>", "v0.1.4")
            if isinstance(command, TestCommand):
                self.assertIn(command.id, results)
                if command.language == "python":
                    compile(body, command.id, "exec")
                    continue
            checked = subprocess.run(["bash", "-n"], input=body, text=True, capture_output=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_workflow_inputs_are_accepted_by_current_shared_template(self):
        # BaseLoader preserves the literal GitHub Actions key "on".
        workflow = yaml.load((_ROOT / ".github/workflows/vbench-quick-start.yml").read_text(), Loader=yaml.BaseLoader)
        template = yaml.load((_ROOT / ".github/workflows/quick-start-template.yml").read_text(), Loader=yaml.BaseLoader)
        job = workflow["jobs"]["vbench-quick-start"]
        contract = template["on"]["workflow_call"]["inputs"]
        self.assertFalse(job["with"].keys() - contract.keys())
        for name, definition in contract.items():
            if definition.get("required") == "true":
                self.assertIn(name, job["with"])
        self.assertEqual(json.loads(job["with"]["test_runner"]), ["linux-aarch64-a2-1"])
        registry = yaml.safe_load((_ROOT / "projects.yaml").read_text())
        project = next(p for p in registry["projects"] if p["name"] == "vbench")
        self.assertEqual((_ROOT / project["dir"]).resolve(), _PROJECT)
        self.assertEqual(job["with"]["upstream_repo"], project["upstream_repo"])


if __name__ == "__main__":
    unittest.main()
