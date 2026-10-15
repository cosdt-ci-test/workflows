"""Validate runnable document blocks and the reusable-workflow interface."""

import json
import subprocess
import unittest
from pathlib import Path

import yaml
from workflows.markdown_doc_test_base import MarkdownDocTestBase, TestCommand


_PROJECT = Path(__file__).resolve().parents[1]
_ROOT = _PROJECT.parents[1]


class TestDocument(unittest.TestCase):
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
