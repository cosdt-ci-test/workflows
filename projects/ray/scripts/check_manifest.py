#!/usr/bin/env python3
"""Check Ray's two explicit upstream NPU test paths and emit the common matrix."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import yaml


_WORKFLOWS_ROOT = Path(__file__).resolve().parents[3]
_COMMON_PATH = _WORKFLOWS_ROOT / "scripts" / "check_examples_manifest.py"
_SPEC = importlib.util.spec_from_file_location(
    "common_examples_manifest", _COMMON_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_COMMON = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMMON)


def load_manifest(path: Path) -> tuple[list[str], list[dict], list[str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_paths = (data.get("scan") or {}).get("paths")
    if not isinstance(raw_paths, list) or not raw_paths:
        raise SystemExit("Ray manifest scan.paths must be a non-empty list")

    paths: list[str] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise SystemExit("Ray manifest paths must be non-empty strings")
        candidate = Path(raw_path.strip())
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SystemExit(f"Ray manifest path must be relative: {raw_path!r}")
        normalized = candidate.as_posix()
        if normalized in paths:
            raise SystemExit(f"duplicate Ray manifest path: {normalized}")
        paths.append(normalized)

    supported = data.get("supported") or []
    unsupported = data.get("unsupported") or []
    listed = {item["path"] for item in supported} | set(unsupported)
    if set(paths) != listed:
        raise SystemExit(
            "Ray manifest scan.paths must match supported + unsupported paths"
        )
    return paths, supported, unsupported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--trigger", default="")
    parser.add_argument("--target-repo", default="")
    parser.add_argument("--target-ref", default="")
    args = parser.parse_args()

    target_root = Path(args.target_root).resolve()
    paths, supported, _unsupported = load_manifest(Path(args.manifest))
    existing = {path for path in paths if (target_root / path).exists()}
    supported_paths = {item["path"] for item in supported}
    listed = set(paths)
    new_paths = sorted(existing - listed)
    stale_paths = sorted(listed - existing)

    _COMMON.write_result(
        args.result_json,
        args.trigger,
        args.target_repo,
        args.target_ref,
        new_paths,
        stale_paths,
        supported,
    )
    _COMMON.write_github_output(_COMMON.matrix_entries(supported))

    missing_supported = sorted(supported_paths - existing)
    if missing_supported:
        for path in missing_supported:
            print(f"supported upstream Ray path missing: {path}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
