"""Exercise VBench's adapter with the real shared seeder and temporary fixtures."""

import hashlib
import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


_PROJECT = Path(__file__).resolve().parents[1]
_ROOT = _PROJECT.parents[1]


class TestCacheSeed(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        script = self.root / "projects/vbench/scripts/prepare_assets.py"
        script.parent.mkdir(parents=True)
        shutil.copy2(_PROJECT / "scripts/prepare_assets.py", script)
        (self.root / "scripts").mkdir()
        shutil.copy2(_ROOT / "scripts/cache_seed.py", self.root / "scripts/cache_seed.py")
        spec = importlib.util.spec_from_file_location("vbench_seed_assets", script)
        self.assets = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.assets)
        self.bundle = self.root / "cache-seed/vbench"
        self.cache = self.root / "cache"
        self.relative_paths = [Path("lavie/human_action") / f"{self.assets.PROMPT}-{i}.mp4"
                               for i in range(5)]

    def _write_bundle(self):
        # These bytes are unit-test fixtures, never staged as benchmark inputs.
        for i, relative in enumerate(self.relative_paths):
            source = self.bundle / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(f"unit-test fixture {i}".encode())
        self._write_manifest()

    def _write_manifest(self):
        entries = [{"path": str(relative),
                    "sha256": hashlib.sha256((self.bundle / relative).read_bytes()).hexdigest()}
                   for relative in self.relative_paths]
        (self.bundle / "manifest.yaml").write_text(yaml.safe_dump({"files": entries}))

    def _seed(self):
        self.assertTrue(hasattr(self.assets, "seed_samples"),
                        "VBench must restore its videos through the shared cache seeder")
        return self.assets.seed_samples(self.cache)

    def test_bundle_is_delivered_to_vbench_cache_not_huggingface_cache(self):
        self._write_bundle()
        self.assertTrue(self._seed())
        for relative in self.relative_paths:
            self.assertEqual((self.cache / relative).read_bytes(), (self.bundle / relative).read_bytes())
        self.assertFalse((self.cache / "hub").exists())

    def test_valid_cached_files_are_not_rewritten(self):
        self._write_bundle()
        self._seed()
        before = [(self.cache / relative).stat().st_mtime_ns for relative in self.relative_paths]
        self._seed()
        self.assertEqual(before, [(self.cache / relative).stat().st_mtime_ns
                                  for relative in self.relative_paths])

    def test_corrupt_cached_video_is_repaired_from_bundle(self):
        self._write_bundle()
        self._seed()
        relative = self.relative_paths[0]
        (self.cache / relative).write_bytes(b"truncated cached video")
        self._seed()
        self.assertEqual((self.cache / relative).read_bytes(), (self.bundle / relative).read_bytes())

    def test_absent_bundle_preserves_manual_cache_and_download_mode(self):
        self.assertFalse(self._seed())
        self.assertFalse(self.cache.exists())

    def test_incomplete_manifest_is_not_reported_as_success(self):
        self._write_bundle()
        manifest = self.bundle / "manifest.yaml"
        data = yaml.safe_load(manifest.read_text())
        data["files"].pop()
        manifest.write_text(yaml.safe_dump(data))
        self.assertTrue(hasattr(self.assets, "seed_samples"))
        with self.assertRaisesRegex(ValueError, "five original LaVie videos"):
            self.assets.seed_samples(self.cache)
        self.assertFalse(self.cache.exists())

    def test_corrupt_bundle_fails_instead_of_downloading_a_replacement(self):
        self._write_bundle()
        relative = self.relative_paths[0]
        (self.bundle / relative).write_bytes(b"corrupt source")
        self.assertTrue(hasattr(self.assets, "seed_samples"))
        with self.assertRaises(subprocess.CalledProcessError):
            self.assets.seed_samples(self.cache)
        self.assertFalse((self.cache / relative).exists())

    def test_prepare_samples_restores_and_decodes_bundle_without_google_drive(self):
        import cv2
        import numpy as np

        self._write_bundle()
        for relative in self.relative_paths:
            writer = cv2.VideoWriter(str(self.bundle / relative),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 8, (32, 32))
            self.assertTrue(writer.isOpened())
            for _ in range(16):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
        self._write_manifest()
        self.assertTrue(hasattr(self.assets, "seed_samples"))
        paths = self.assets.prepare_samples(self.cache, self.root / "target")
        self.assertEqual(len(paths), 5)
        for relative, output in zip(self.relative_paths, paths):
            self.assertEqual(output.read_bytes(), (self.bundle / relative).read_bytes())
        self.assertFalse((self.cache / "lavie.zip").exists())


if __name__ == "__main__":
    unittest.main()
