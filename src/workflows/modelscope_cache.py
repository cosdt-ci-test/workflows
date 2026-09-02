"""Modelscope hub cache validation: purge stale safetensors shards.

Shared utility used by the ``prepare_environment`` step of every
project whose quick-start test downloads model weights via
``modelscope`` (ms-swift, peft, diffusers, torchtune, torchtitan
as of writing). Lives in the ``workflows`` namespace package so
each project's test can ``from workflows.modelscope_cache
import ...`` instead of copy-pasting the same logic.

Why this exists
---------------

The runner uses a host-side bind mount
(``/data/ci-cache/modelscope/<project>/...``) so the modelscope
cache survives across CI runs. Useful for cache hits, but it also
lets stale partial ``.safetensors`` shards from interrupted runs
(kill -9, OOM, network drop) leak into the next run. ``transformers``
then trips on ``safetensors.safe_open`` with::

    safetensors._safetensors_rust.SafetensorError:
        Error while deserializing header: incomplete metadata,
        file not fully covered

This module walks the cache, uses the native ``safetensors`` loader
to validate each shard's header, and ``rmtree``-s the parent model
dir on any failure. ``modelscope`` will re-download the whole model
on next access.

Layout
------

ModelScope stores models under::

    <cache_root>/hub/models/<org>/<model>/<revision>/*.safetensors

Unlike HuggingFace Hub's ``blobs/`` + symlink pattern — we walk the
full model dir to cover both layouts, and rely on
``safetensors.safe_open`` to resolve symlinks transparently.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def ensure_safetensors() -> None:
    """Defensively install ``safetensors`` if not already importable.

    The CANN base image used by the runner may not ship
    ``safetensors``; torch < 4.20 doesn't hard-depend on it. The
    validation helpers below need it, so callers should invoke
    this once in their ``prepare_environment`` before
    ``purge_corrupt_models``. No-op when already installed.

    Inherits the parent env, so any ``PIP_INDEX_URL`` /
    ``PIP_CONSTRAINT`` / ``UV_*`` configured by the workflow
    carries through to the install.
    """
    try:
        import safetensors  # noqa: F401
    except ImportError:
        subprocess.run(
            ['python', '-m', 'pip', 'install', 'safetensors'],
            check=True,
        )


def resolve_modelscope_cache() -> Path:
    """Return the modelscope cache root the same way modelscope does:
    prefer ``$MODELSCOPE_CACHE`` if set, otherwise
    ``~/.cache/modelscope``. Computed at call time (not import time)
    so tests / subprocesses that mutate the env get the right value."""
    return Path(
        os.environ.get('MODELSCOPE_CACHE')
        or str(Path.home() / '.cache' / 'modelscope')
    )


def safetensors_header_ok(path: Path) -> bool:
    """Use safetensors' native loader to validate the file header.
    Returns True iff ``safe_open`` accepts the file (header parses,
    tensor offsets fit within the file). ``SafetensorError`` or
    ``OSError`` means the shard is unusable.

    Lazy import: the module itself loads fine on machines that
    don't have ``safetensors`` installed yet — the import only
    fires when a caller actually invokes this function. Callers
    should run ``ensure_safetensors()`` once first if they intend
    to call this in an environment where ``safetensors`` may be
    missing.
    """
    from safetensors import safe_open, SafetensorError  # noqa: I001
    try:
        # framework='numpy' instead of 'pt': only ``.keys()`` is read so
        # the framework is a no-op for our use case, but safetensors
        # 0.8.0 makes ``framework`` required and 'pt' would force
        # ``import torch`` — bare CANN 9.1.0 image doesn't ship torch.
        with safe_open(str(path), framework='numpy') as f:
            list(f.keys())  # force header read
    except (SafetensorError, OSError):
        return False
    return True


def diagnose_mount_environment(cache_root: Path | None = None) -> None:
    """Probe the container's view of ``cache_root`` to disambiguate
    "the bind mount isn't visible" cases.

    Companion to ``purge_corrupt_models`` — called at the entry of
    ``purge_corrupt_models`` so every ``prepare_environment``
    invocation captures the cache_root's environment context
    regardless of cache-state outcome (HIT / PARTIAL / CORRUPT /
    miss).

    Each probe answers a specific question:

    - ``findmnt -T <path>`` — canonical "is this a mountpoint".
      Empty output means it isn't. Reveals the underlying
      ``SOURCE`` (``nfs ...:ascend-ci-share`` vs ``/dev/sdXN`` vs
      ``overlay``) and ``FSTYPE`` so the CI log itself can answer
      "what disk is this on" without ops-side debugging.
    - ``stat`` on the path *and* its parent (dev/inode) —
      mountpoint detection without ``findmnt``: a separate
      mountpoint lives on a different ``st_dev`` than its parent.
      Bind-mount targets nested deeper than the actual
      mount-point typically show ``same`` device — confirms with
      (3) below.
    - ``/proc/self/mountinfo`` — kernel-level mount table, more
      authoritative than ``mount`` (which reads ``/etc/mtab`` /
      ``/proc/mounts`` and can be filtered in some runtimes).
    - ``hostname`` + ``/proc/1/cgroup`` — confirm we're inside a
      Kubernetes pod (cgroup paths contain ``kubepods/``);
      bind-mount behavior and capability drops differ there.
    - ``CapBnd`` from ``/proc/self/status`` — ``CAP_SYS_ADMIN``
      (bit 21) is required for bind mounts on most runtimes;
      absence explains silent bind-mount failures and missing
      ``findmnt`` detail.
    - ``df -h`` — backing filesystem / size for when the path
      *is* a mountpoint (sanity check that it's actually
      persistent, and matches ``shutil.disk_usage`` results
      against the fs backing store).

    All probes are read-only and have 5 s timeouts; missing tools
    print ``(not available)`` rather than raising, matching the
    no-op spirit of the unit tests' ``test_no_op_when_cache_absent``.
    """
    cache_root = cache_root or resolve_modelscope_cache()
    print('mount-diag: probing container mount environment')

    # findmnt — canonical "is this a mountpoint"
    print(f'  findmnt -T {cache_root}:')
    try:
        out = subprocess.run(
            ['findmnt', '-T', str(cache_root)],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.splitlines():
                print(f'    {line}')
        elif out.returncode == 0:
            print('    (empty — path is NOT a mountpoint)')
        else:
            print(
                f'    findmnt failed (rc={out.returncode}): '
                f'{out.stderr.strip()[:200]}'
            )
    except FileNotFoundError:
        print('    (findmnt not available)')
    except subprocess.TimeoutExpired:
        print('    (findmnt timed out)')

    # stat (dev, inode) — corroborating mountpoint check
    print(f'  stat -c "%d:%i" {cache_root} vs parent:')
    try:
        path_stat = cache_root.stat()
        parent_stat = cache_root.parent.stat()
        print(
            f'    {cache_root}: '
            f'dev={path_stat.st_dev} inode={path_stat.st_ino}'
        )
        print(
            f'    {cache_root.parent}: '
            f'dev={parent_stat.st_dev} inode={parent_stat.st_ino}'
        )
        if path_stat.st_dev != parent_stat.st_dev:
            print(
                '    → different device than parent → mountpoint '
                '(separate filesystem)'
            )
        else:
            print(
                '    → same device as parent → NOT a mountpoint '
                '(regular directory on the parent fs)'
            )
    except OSError as e:
        print(f'    stat failed: {e}')

    # /proc/self/mountinfo — kernel view
    print(f'  /proc/self/mountinfo entries for {cache_root}:')
    try:
        with open('/proc/self/mountinfo', encoding='utf-8') as fh:
            matches = [
                line.rstrip() for line in fh if str(cache_root) in line
            ]
        if matches:
            for line in matches:
                print(f'    {line}')
        else:
            print(f'    (no entry for {cache_root})')
    except OSError as e:
        print(f'    read /proc/self/mountinfo failed: {e}')

    # host identity — K8s detection
    print('  host identity:')
    try:
        out = subprocess.run(
            ['hostname'],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            print(f'    hostname: {out.stdout.strip()}')
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        with open('/proc/1/cgroup', encoding='utf-8') as fh:
            cgroup_lines = fh.read().splitlines()
        is_k8s = any('kubepods' in line for line in cgroup_lines)
        print(
            f'    /proc/1/cgroup: {"K8s pod" if is_k8s else "non-K8s"}'
        )
        for line in cgroup_lines[:5]:
            print(f'      {line}')
        if len(cgroup_lines) > 5:
            print(f'      ... and {len(cgroup_lines) - 5} more')
    except OSError as e:
        print(f'    read /proc/1/cgroup failed: {e}')

    # capabilities — CAP_SYS_ADMIN (bit 21) required for bind mounts
    print('  capabilities (from /proc/self/status):')
    try:
        cap_lines: list[str] = []
        cap_bnd: int | None = None
        with open('/proc/self/status', encoding='utf-8') as fh:
            for line in fh:
                if line.startswith('Cap'):
                    cap_lines.append(line.rstrip())
                if line.startswith('CapBnd:'):
                    cap_bnd = int(line.split()[1], 16)
        if cap_lines:
            for line in cap_lines:
                print(f'    {line}')
        else:
            print('    (no Cap lines in /proc/self/status)')
        if cap_bnd is not None:
            has_sys_admin = bool(cap_bnd & (1 << 21))
            print(
                f'    CAP_SYS_ADMIN (bit 21): '
                f'{"YES" if has_sys_admin else "NO"} '
                f'(CapBnd=0x{cap_bnd:x})'
            )
    except OSError as e:
        print(f'    read /proc/self/status failed: {e}')

    # df — backing filesystem / size
    print(f'  df -h {cache_root}:')
    try:
        out = subprocess.run(
            ['df', '-h', str(cache_root)],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.splitlines():
                print(f'    {line}')
        elif out.returncode == 0:
            print('    (empty)')
        else:
            print(f'    df failed: {out.stderr.strip()[:200]}')
    except FileNotFoundError:
        print('    (df not available)')
    except subprocess.TimeoutExpired:
        print('    (df timed out)')


def purge_corrupt_models(cache_root: Path) -> None:
    """Scan every ``*.safetensors`` file under each model dir and purge
    the model dir if any shard is corrupt. ``modelscope`` will
    re-download the whole model on next access. No-op when the
    cache root is absent (fresh container, or first-time setup).

    Walks the full model dir (not just ``blobs/``) because
    ModelScope's layout is
    ``<model_dir>/<revision>/*.safetensors`` — unlike HuggingFace
    Hub which uses ``blobs/`` + symlinks. ``safe_open`` resolves
    symlinks transparently, so this also catches a future
    modelscope release that switches to the HF-style layout.

    For dotted model ids (``Qwen2.5-...``), modelscope stores the
    files in a masked dir (``.`` → ``___``) plus a *symlink* named
    after the original id for readability. ``is_dir()`` follows the
    symlink, so without the ``not is_symlink()`` filter both entries
    match: each shard gets validated twice, and when a corrupt shard
    triggers the purge, ``shutil.rmtree`` on the symlink entry raises
    ``OSError`` (rmtree refuses symlinks by design) and crashes
    ``prepare_environment`` — exactly in the scenario the purge
    exists for. Skipping the symlink entries keeps the purge on the
    masked dir only.

    Parameters
    ----------
    cache_root : Path
        The modelscope cache root (i.e. the value of
        ``$MODELSCOPE_CACHE`` or its default ``~/.cache/modelscope``).
        Passed in as a parameter so the function is reusable from
        tests with a tmp dir and doesn't carry an implicit dependency
        on a module-level constant.
    """
    diagnose_mount_environment(cache_root)
    hub_models = cache_root / 'hub' / 'models'
    if not hub_models.exists():
        print(f'cache: miss ({hub_models} not present yet); nothing to validate')
        return
    model_dirs = [
        d for d in hub_models.glob('*/*')
        if d.is_dir() and not d.is_symlink()
    ]
    purged = 0
    for model_dir in model_dirs:
        corrupt = [
            p for p in model_dir.rglob('*.safetensors')
            if not safetensors_header_ok(p)
        ]
        if not corrupt:
            continue
        print(
            f'cache: purging {model_dir.parent.name}/{model_dir.name} '
            f'({len(corrupt)} corrupt shard(s)); modelscope will re-download'
        )
        shutil.rmtree(model_dir)
        purged += 1
    if purged:
        print(
            f'cache: partial — validated {len(model_dirs)} model dir(s), '
            f'purged {purged}'
        )
    else:
        print(
            f'cache: hit {len(model_dirs)} model dir(s), all healthy'
        )