"""Regression tests for workflows.modelscope_cache.purge_corrupt_models.

The masked-dir + symlink layout is what modelscope 1.37.0 actually
produces for dotted model ids (verified against a real
``snapshot_download('Qwen/Qwen2.5-3B-Instruct')``):

    <cache>/hub/models/Qwen/Qwen2___5-3B-Instruct/   # masked dir, files here
    <cache>/hub/models/Qwen/Qwen2.5-3B-Instruct      # symlink -> masked dir

These tests pin the purge behavior on that layout: the symlink entry
must be skipped, the masked dir purged once, and the purge must not
crash on ``shutil.rmtree(symlink)``.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from workflows.modelscope_cache import purge_corrupt_models  # noqa: E402

# A genuinely valid minimal safetensors payload (one tiny tensor),
# byte-for-byte what safetensors.torch.save({"w": torch.zeros(1)})
# produces — note the two trailing spaces padding the JSON header to
# its declared length of 56. safe_open accepts it.
_VALID_SAFETENSORS = (
    b'8\x00\x00\x00\x00\x00\x00\x00'
    b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}  '
    b'\x00\x00\x00\x00'
)

# Truncated garbage: header-length prefix claims 1 KB, body missing.
_CORRUPT_SAFETENSORS = b'\x00\x04\x00\x00\x00\x00\x00\x00{"a":'


def _make_model_cache(root: Path) -> tuple[Path, Path]:
    """Create one dotted-id model in modelscope's masked + symlink layout."""
    org = root / 'hub' / 'models' / 'Qwen'
    org.mkdir(parents=True)
    masked = org / 'Qwen2___5-3B-Instruct'
    masked.mkdir()
    (masked / 'model-00001.safetensors').write_bytes(_VALID_SAFETENSORS)
    link = org / 'Qwen2.5-3B-Instruct'
    link.symlink_to(masked)
    return masked, link


class TestPurgeCorruptModels(unittest.TestCase):

    def test_symlink_entry_skipped_on_healthy_cache(self) -> None:
        """Dotted-id model: masked dir + symlink both exist, shards healthy.

        The purge must treat them as ONE model (validate shards once),
        not two — and must not crash.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masked, _link = _make_model_cache(root)
            # Purge must not crash and must not delete anything.
            purge_corrupt_models(root)
            self.assertTrue(masked.exists())

    def test_purge_survives_symlink_and_removes_masked_dir(self) -> None:
        """Corrupt shard in a dotted-id model: the purge must remove the
        masked dir (the real directory) without tripping over the
        same-named symlink entry.

        Regression: with plain ``is_dir()`` the symlink also matched
        and, depending on glob order, ``shutil.rmtree(symlink)`` raised
        ``OSError`` and crashed the whole purge.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            masked, link = _make_model_cache(root)
            # Corrupt the model: replace the healthy shard with garbage.
            (masked / 'model-00001.safetensors').write_bytes(
                _CORRUPT_SAFETENSORS
            )
            purge_corrupt_models(root)
            # Masked dir purged -> modelscope re-downloads next run.
            self.assertFalse(masked.exists())
            # Symlink now dangles; modelscope recreates it on re-download.
            self.assertTrue(link.is_symlink())

    def test_no_op_when_cache_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            purge_corrupt_models(Path(tmp) / 'nope')  # must not raise

    def test_undotted_model_without_symlink_still_purged(self) -> None:
        """Ids without '.' (no masking, no symlink) keep working."""
        with tempfile.TemporaryDirectory() as tmp:
            org = Path(tmp) / 'hub' / 'models' / 'AI-ModelScope'
            org.mkdir(parents=True)
            model = org / 'stable-diffusion-v1-5'
            model.mkdir()
            (model / 'shard.safetensors').write_bytes(_CORRUPT_SAFETENSORS)
            purge_corrupt_models(Path(tmp))
            self.assertFalse(model.exists())


if __name__ == '__main__':
    unittest.main()
