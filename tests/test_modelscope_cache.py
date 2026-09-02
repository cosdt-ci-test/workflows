"""Regression tests for workflows.modelscope_cache.purge_corrupt_models
plus diagnostic pinning for the cache mount / disk-usage layer.

The masked-dir + symlink layout is what modelscope 1.37.0 actually
produces for dotted model ids (verified against a real
``snapshot_download('Qwen/Qwen2.5-3B-Instruct')``):

    <cache>/hub/models/Qwen/Qwen2___5-3B-Instruct/   # masked dir, files here
    <cache>/hub/models/Qwen/Qwen2.5-3B-Instruct      # symlink -> masked dir

Two test groups:

- ``TestPurgeCorruptModels`` pins the purge behavior on that layout:
  the symlink entry must be skipped, the masked dir purged once,
  and the purge must not crash on ``shutil.rmtree(symlink)``.
- ``TestMountDiagnostics`` pins ``_log_mount_info`` so each test's
  captured stdout carries a ``[mount-diag]`` line showing what
  disk the cache_root was on — distinguishes a tmpfs/overlay
  tmpdir from a real bind mount like
  ``/data/ci-cache/modelscope/<project>/`` when "stale shards" /
  "disk full" / "purge didn't help" need root-causing from a CI
  log alone.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from workflows.modelscope_cache import purge_corrupt_models  # noqa: E402


def _log_mount_info(label: str, path: Path) -> None:
    """Diagnostic helper: print disk usage for ``path`` so a failing
    test's captured stdout shows the mount state of the cache_root
    the purge ran on. Distinguishes a tmpfs / overlay tmpdir from a
    real bind mount like ``/data/ci-cache/modelscope/<project>/`` —
    exactly the distinction that matters when "disk full" /
    "stale shards" / "purge didn't help" need root-causing from a
    CI log alone.

    No-op for inaccessible paths: prints the ``OSError`` name +
    message instead of raising, matching the no-op spirit of
    ``test_no_op_when_cache_absent``.
    """
    try:
        u = shutil.disk_usage(str(path))
    except OSError as e:
        print(f'[mount-diag] {label}: {path} -> {type(e).__name__}: {e}')
        return
    gb = 1 << 30
    print(
        f'[mount-diag] {label}: path={path} '
        f'total={u.total/gb:.1f}GB used={u.used/gb:.1f}GB '
        f'free={u.free/gb:.1f}GB'
    )

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
            _log_mount_info('test-tmp', root)
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
            _log_mount_info('test-tmp', root)
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
            _log_mount_info('test-tmp', Path(tmp))
            purge_corrupt_models(Path(tmp) / 'nope')  # must not raise

    def test_undotted_model_without_symlink_still_purged(self) -> None:
        """Ids without '.' (no masking, no symlink) keep working."""
        with tempfile.TemporaryDirectory() as tmp:
            _log_mount_info('test-tmp', Path(tmp))
            org = Path(tmp) / 'hub' / 'models' / 'AI-ModelScope'
            org.mkdir(parents=True)
            model = org / 'stable-diffusion-v1-5'
            model.mkdir()
            (model / 'shard.safetensors').write_bytes(_CORRUPT_SAFETENSORS)
            purge_corrupt_models(Path(tmp))
            self.assertFalse(model.exists())


class TestMountDiagnostics(unittest.TestCase):
    """Pin ``_log_mount_info`` so future refactors don't silently
    drop the disk-usage line from test output.

    The helper itself is a thin ``shutil.disk_usage`` wrapper; the
    tests below pin two invariants: (a) a real path emits the
    expected ``[mount-diag]`` line with ``label`` / ``path`` /
    ``total`` / ``used`` / ``free`` fields, (b) a missing path
    doesn't raise — prints the ``OSError`` name instead. Same
    no-op spirit as ``test_no_op_when_cache_absent``.
    """

    def test_logs_disk_usage_for_existing_path(self) -> None:
        """Happy path: real tmpdir prints ``label`` + ``path`` +
        ``total=/used=/free=`` on a single ``[mount-diag]`` line.
        Capture stdout so we assert line shape, not incidental
        ordering or extra fields.
        """
        import io
        from contextlib import redirect_stdout

        with (
            tempfile.TemporaryDirectory() as tmp,
            redirect_stdout(io.StringIO()) as buf,
        ):
            _log_mount_info('label-x', Path(tmp))
        out = buf.getvalue()
        self.assertIn('[mount-diag] label-x:', out)
        self.assertIn('total=', out)
        self.assertIn('used=', out)
        self.assertIn('free=', out)
        self.assertIn(tmp, out)

    def test_logs_gracefully_for_nonexistent_path(self) -> None:
        """Missing path must not raise — the ``OSError`` name
        (typically ``FileNotFoundError`` on POSIX) is emitted so the
        diagnostic line is still informative in a failure log.
        Mirrors ``test_no_op_when_cache_absent``.
        """
        import io
        from contextlib import redirect_stdout

        ghost = (
            Path(tempfile.gettempdir())
            / f'definitely-not-here-{time.time_ns()}'
        )
        with redirect_stdout(io.StringIO()) as buf:
            _log_mount_info('ghost', ghost)  # must not raise
        out = buf.getvalue()
        self.assertIn('[mount-diag] ghost:', out)
        self.assertIn(str(ghost), out)
        # Either FileNotFoundError or its OSError parent.
        self.assertRegex(out, r'(FileNotFoundError|OSError)')


if __name__ == '__main__':
    unittest.main()
