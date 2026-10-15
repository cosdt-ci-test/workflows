"""Asset integrity and original-Prompt selection, without network access."""

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_assets.py"


class TestAssets(unittest.TestCase):
    def _module(self):
        self.assertTrue(_SCRIPT.is_file(), "VBench asset preparation is not implemented")
        spec = importlib.util.spec_from_file_location("vbench_assets", _SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_extracts_original_five_videos_and_ignores_other_dimensions(self):
        assets = self._module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "lavie.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for i in range(5):
                    z.writestr(f"lavie/human_action/{assets.PROMPT}-{i}.mp4", b"video")
                z.writestr("lavie/scene/other.mp4", b"other")
            paths = assets.extract_samples(archive, root / "videos")
            self.assertEqual(len(paths), 5)
            self.assertEqual({p.name for p in paths},
                             {f"{assets.PROMPT}-{i}.mp4" for i in range(5)})
            self.assertFalse((root / "videos/other.mp4").exists())

    def test_missing_video_fails_before_reporting_assets_ready(self):
        assets = self._module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "partial.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(f"lavie/human_action/{assets.PROMPT}-0.mp4", b"video")
            with self.assertRaisesRegex(ValueError, "Missing"):
                assets.extract_samples(archive, root / "videos")

    def test_subset_metadata_preserves_upstream_prompt(self):
        assets = self._module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "upstream.json"
            selected = {"prompt_en": assets.PROMPT, "dimension": ["human_action"]}
            source.write_text(json.dumps([selected, {"prompt_en": "other", "dimension": ["scene"]}]))
            assets.write_subset_info(source, root / "subset.json")
            self.assertEqual(json.loads((root / "subset.json").read_text()), [selected])

    def test_checksum_rejects_truncated_checkpoint(self):
        assets = self._module()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pth"
            checkpoint.write_bytes(b"partial download")
            self.assertFalse(assets.check_sha256(checkpoint, assets.UMT_SHA256))

    def test_preloaded_videos_work_without_google_drive(self):
        import cv2
        import numpy as np

        assets = self._module()
        self.assertTrue(hasattr(assets, "prepare_samples"), "preloaded sample cache must be reusable")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            videos = cache / "lavie/human_action"
            videos.mkdir(parents=True)
            for i in range(5):
                writer = cv2.VideoWriter(str(videos / f"{assets.PROMPT}-{i}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 8, (32, 32))
                for _ in range(16):
                    writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
                writer.release()
            output = Path(directory) / "target"
            paths = assets.prepare_samples(cache, output)
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertFalse((cache / "lavie.zip").exists())


if __name__ == "__main__":
    unittest.main()
