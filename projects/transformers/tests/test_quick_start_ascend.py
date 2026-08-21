import os
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC = PROJECT_ROOT / "docs" / "Quick-start-Ascend.md"
TARGET_ROOT = Path(
    os.environ.get("TARGET_ROOT", PROJECT_ROOT.parents[2] / "transformers")
).resolve()


def extract_doctest_code(text: str) -> str:
    """Extract the first Python console block using upstream-style prompts.

    The guard document has one executable pycon block. This deliberately
    mirrors the prompt/continuation shape used by upstream doctests without
    pretending to run every Markdown block in the document.
    """
    marker = "```pycon"
    start = text.find(marker)
    if start < 0:
        raise ValueError("Quick Start document has no pycon block")
    body_start = text.find("\n", start) + 1
    body_end = text.find("```", body_start)
    if body_end < 0:
        raise ValueError("Quick Start pycon block is not closed")

    commands: list[str] = []
    for line in text[body_start:body_end].splitlines():
        if line.startswith(">>> "):
            commands.append(line[4:])
        elif line.startswith("... "):
            if not commands:
                raise ValueError("continuation appears before a command")
            commands[-1] += "\n" + line[4:]
        elif line.strip() in {">>>", "..."}:
            continue
        elif line.strip():
            raise ValueError(f"unexpected line in pycon block: {line!r}")
    if not commands:
        raise ValueError("Quick Start pycon block has no commands")
    return "\n".join(commands)


class QuickStartAscendTest(unittest.TestCase):
    def test_documented_pipeline_runs(self):
        self.assertTrue(TARGET_ROOT.is_dir(), f"target checkout not found: {TARGET_ROOT}")
        # Run in the monitored checkout so the document is tested against the
        # exact transformers source selected by the workflow, not an implicit
        # installation from the workflow repository's parent directory.
        code = extract_doctest_code(DOC.read_text(encoding="utf-8"))
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=TARGET_ROOT,
            check=False,
            text=True,
            capture_output=True,
            timeout=900,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        self.assertEqual(completed.returncode, 0)
        # The fixed prefix checks that generation produced output, rather than
        # merely importing transformers and exiting successfully.
        self.assertIn("The secret to baking a good cake is", completed.stdout)


if __name__ == "__main__":
    unittest.main()
