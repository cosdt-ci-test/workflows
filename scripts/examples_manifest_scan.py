"""Shared example-scan config and discovery for manifest tools.

bootstrap_manifest.py and check_examples_manifest.py must replay the
same rules. This module only validates scan config and walks a tree;
it does not classify supported vs unsupported.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_SCAN_ROOT = 'examples'
DEFAULT_INCLUDE_EXTENSIONS = ('.sh', '.py', '.yaml')
DEFAULT_MIXED_INCLUDE_EXTENSIONS = ('.sh', '.py')
DEFAULT_SCAN_UNIT = 'files'
DEFAULT_DIR_MARKER = 'CMakeLists.txt'
DEFAULT_DIR_MAX_DEPTH = 1
SCAN_UNITS = ('files', 'directories', 'mixed')


def normalize_extension(ext: str) -> str:
    if not isinstance(ext, str):
        raise SystemExit(
            f'scan.include_extensions items must be strings, got {type(ext).__name__}')
    ext = ext.strip()
    if not ext:
        raise SystemExit('empty value in scan.include_extensions')
    return ext if ext.startswith('.') else f'.{ext}'


def normalize_extensions(value, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or not value:
        raise SystemExit('scan.include_extensions must be a non-empty list')
    return tuple(normalize_extension(item) for item in value)


def _normalize_marker(unit: str, marker):
    if unit == 'directories':
        if marker is None:
            return DEFAULT_DIR_MARKER
        if not isinstance(marker, str) or not marker.strip():
            raise SystemExit('scan.marker must be a non-empty string')
        return marker.strip()
    if unit == 'mixed':
        if marker is None or marker == '':
            return ''
        if not isinstance(marker, str):
            raise SystemExit(
                f'scan.marker must be a string or empty, got {type(marker).__name__}')
        if not marker.strip():
            raise SystemExit(
                'scan.marker must be a non-empty string or explicitly empty')
        return marker.strip()
    return None


def load_scan(scan: dict) -> dict:
    unit = scan.get('unit') or DEFAULT_SCAN_UNIT
    if unit not in SCAN_UNITS:
        raise SystemExit(
            f'scan.unit must be one of {SCAN_UNITS}, got {unit!r}')
    max_depth = scan.get('max_depth', DEFAULT_DIR_MAX_DEPTH)
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise SystemExit(
            f'scan.max_depth must be an integer >= 1, got {max_depth!r}')
    if unit == 'mixed':
        default_extensions = DEFAULT_MIXED_INCLUDE_EXTENSIONS
    else:
        default_extensions = DEFAULT_INCLUDE_EXTENSIONS
    return {
        'scan_root': scan.get('root') or DEFAULT_SCAN_ROOT,
        'unit': unit,
        'marker': _normalize_marker(unit, scan.get('marker')),
        'max_depth': max_depth,
        'include_extensions': normalize_extensions(
            scan.get('include_extensions'), default_extensions),
    }


def scan_file_units(
        examples_root: Path, target_root: Path,
        include_extensions: tuple[str, ...],
        max_depth: int | None = None) -> list[str]:
    found: list[str] = []
    for path in sorted(examples_root.rglob('*')):
        if not path.is_file() or path.suffix not in include_extensions:
            continue
        if (max_depth is not None
                and len(path.relative_to(examples_root).parts) > max_depth):
            continue
        found.append(path.relative_to(target_root).as_posix())
    return found


def scan_directory_units(examples_root: Path, target_root: Path,
                         marker: str | None, max_depth: int) -> list[str]:
    found: list[str] = []

    def visit(directory: Path, depth: int) -> None:
        if depth >= max_depth:
            return
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            if not marker or (child / marker).is_file():
                found.append(child.relative_to(target_root).as_posix())
            visit(child, depth + 1)

    visit(examples_root, 0)
    return found


def scan_examples(target_root: Path, scan: dict) -> list[str]:
    examples_root = target_root / scan['scan_root']
    if not examples_root.is_dir():
        raise SystemExit(f"{scan['scan_root']}/ not found under {target_root}")
    unit = scan['unit']
    if unit == 'directories':
        found = scan_directory_units(
            examples_root, target_root, scan['marker'], scan['max_depth'])
    elif unit == 'mixed':
        found = scan_directory_units(
            examples_root, target_root, scan['marker'], scan['max_depth'])
        found.extend(scan_file_units(
            examples_root, target_root, scan['include_extensions'],
            scan['max_depth']))
    else:
        found = scan_file_units(
            examples_root, target_root, scan['include_extensions'])
    return sorted(set(found))
