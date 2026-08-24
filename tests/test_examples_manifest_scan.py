#!/usr/bin/env python3
"""Tests for shared example scan rules and manifest check diffs."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from examples_manifest_scan import (  # noqa: E402
    load_scan,
    scan_examples,
)

CHECKER = SCRIPTS / 'check_examples_manifest.py'


def write_tree(root: Path, files: list[str]) -> None:
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('x\n', encoding='utf-8')


class LoadScanTests(unittest.TestCase):
    def test_unknown_unit_fails(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            load_scan({'unit': 'trees'})
        self.assertIn('scan.unit', str(caught.exception))

    def test_max_depth_must_be_positive_int(self) -> None:
        with self.assertRaises(SystemExit):
            load_scan({'unit': 'mixed', 'max_depth': 0})
        with self.assertRaises(SystemExit):
            load_scan({'unit': 'mixed', 'max_depth': True})

    def test_directories_marker_cannot_be_blank(self) -> None:
        with self.assertRaises(SystemExit):
            load_scan({'unit': 'directories', 'marker': '   '})

    def test_mixed_whitespace_marker_fails(self) -> None:
        with self.assertRaises(SystemExit):
            load_scan({'unit': 'mixed', 'marker': '   '})

    def test_mixed_empty_marker_means_all_dirs(self) -> None:
        scan = load_scan({'unit': 'mixed', 'marker': ''})
        self.assertEqual(scan['marker'], '')
        self.assertEqual(scan['include_extensions'], ('.sh', '.py'))

    def test_empty_extension_fails(self) -> None:
        with self.assertRaises(SystemExit):
            load_scan({
                'unit': 'files',
                'include_extensions': ['.sh', ''],
            })


class ScanBehaviorTests(unittest.TestCase):
    def test_files_unit_is_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/a.sh',
                'examples/nested/b.py',
                'examples/nested/c.yaml',
                'examples/skip.txt',
            ])
            found = scan_examples(root, load_scan({'unit': 'files'}))
            self.assertEqual(found, [
                'examples/a.sh',
                'examples/nested/b.py',
                'examples/nested/c.yaml',
            ])

    def test_directories_unit_requires_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/cli/CMakeLists.txt',
                'examples/python/whisper.py',
                'examples/wchess/libwchess/CMakeLists.txt',
            ])
            found = scan_examples(root, load_scan({
                'unit': 'directories',
                'marker': 'CMakeLists.txt',
                'max_depth': 1,
            }))
            self.assertEqual(found, ['examples/cli'])

    def test_mixed_finds_top_level_dirs_and_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/cli/CMakeLists.txt',
                'examples/python/whisper_processor.py',
                'examples/server.py',
                'examples/generate-karaoke.sh',
                'examples/helpers.js',
                'examples/common.cpp',
            ])
            found = scan_examples(root, load_scan({
                'unit': 'mixed',
                'max_depth': 1,
                'include_extensions': ['.sh', '.py'],
            }))
            self.assertEqual(found, [
                'examples/cli',
                'examples/generate-karaoke.sh',
                'examples/python',
                'examples/server.py',
            ])

    def test_mixed_does_not_split_nested_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/python/whisper_processor.py',
                'examples/python/test_whisper_processor.py',
            ])
            found = scan_examples(root, load_scan({
                'unit': 'mixed',
                'max_depth': 1,
            }))
            self.assertEqual(found, ['examples/python'])

    def test_mixed_dedupes_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/z-dir/keep.txt',
                'examples/a.sh',
                'examples/m-dir/keep.txt',
            ])
            found = scan_examples(root, load_scan({
                'unit': 'mixed',
                'max_depth': 1,
                'include_extensions': ['.sh'],
            }))
            self.assertEqual(found, [
                'examples/a.sh',
                'examples/m-dir',
                'examples/z-dir',
            ])


class ManifestCheckTests(unittest.TestCase):
    def _run(
        self,
        target_root: Path,
        manifest_text: str,
    ) -> subprocess.CompletedProcess[str]:
        manifest = target_root / 'examples_manifest.yaml'
        manifest.write_text(manifest_text, encoding='utf-8')
        result_json = target_root / 'result.json'
        return subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                '--target-root', str(target_root),
                '--manifest', str(manifest),
                '--result-json', str(result_json),
                '--target-repo', 'org/proj',
                '--target-ref', 'master',
                '--trigger', 'review',
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS),
        )

    def test_manual_existing_path_is_not_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/cli/CMakeLists.txt',
                'examples/server.py',
            ])
            proc = self._run(root, """\
version: 1
scan:
  root: examples
  unit: directories
  marker: CMakeLists.txt
  max_depth: 1
supported:
  - path: examples/cli
    profile: cann
    runner: linux-aarch64-a2-1
    npu_devices: '0'
    image: swr.example/cann:tag
    timeout_minutes: 60
unsupported:
  - examples/server.py
""")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads((root / 'result.json').read_text())
            self.assertEqual(result['new_paths'], [])
            self.assertEqual(result['stale_paths'], [])

    def test_missing_listed_paths_are_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, ['examples/cli/CMakeLists.txt'])
            proc = self._run(root, """\
version: 1
scan:
  root: examples
  unit: directories
  marker: CMakeLists.txt
  max_depth: 1
supported:
  - path: examples/cli
    profile: cann
    runner: linux-aarch64-a2-1
    npu_devices: '0'
    image: swr.example/cann:tag
    timeout_minutes: 60
unsupported:
  - examples/gone.sh
""")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads((root / 'result.json').read_text())
            self.assertEqual(result['stale_paths'], ['examples/gone.sh'])

    def test_missing_supported_path_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, ['examples/keep/CMakeLists.txt'])
            proc = self._run(root, """\
version: 1
scan:
  root: examples
  unit: directories
  marker: CMakeLists.txt
  max_depth: 1
supported:
  - path: examples/missing
    profile: cann
    runner: linux-aarch64-a2-1
    npu_devices: '0'
    image: swr.example/cann:tag
    timeout_minutes: 60
unsupported: []
""")
            self.assertEqual(proc.returncode, 1)
            result = json.loads((root / 'result.json').read_text())
            self.assertEqual(result['stale_paths'], ['examples/missing'])
            self.assertIn('supported example missing', proc.stderr)

    def test_new_scanned_path_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tree(root, [
                'examples/cli/CMakeLists.txt',
                'examples/server.py',
            ])
            proc = self._run(root, """\
version: 1
scan:
  root: examples
  unit: mixed
  max_depth: 1
  include_extensions: ['.sh', '.py']
supported:
  - path: examples/cli
    profile: cann
    runner: linux-aarch64-a2-1
    npu_devices: '0'
    image: swr.example/cann:tag
    timeout_minutes: 60
unsupported: []
""")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads((root / 'result.json').read_text())
            self.assertEqual(result['new_paths'], ['examples/server.py'])


if __name__ == '__main__':
    unittest.main()
