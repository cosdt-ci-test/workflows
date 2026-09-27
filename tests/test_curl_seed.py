"""Contract tests for HF and generic URL cache-seed downloads."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from scripts import curl_seed


REPO_ROOT = Path(__file__).resolve().parents[1]


def asset_spec(payload: bytes, path: str = "third_party/slime/kernel.zip") -> dict:
    return {
        "url": "https://example.invalid/kernel.zip",
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_existing_hf_seed_format_remains_supported(tmp_path, monkeypatch) -> None:
    torchtune = REPO_ROOT / "cache-seed" / "torchtune"
    assert curl_seed.load_spec(torchtune)
    assert curl_seed.load_assets(torchtune) == []

    payload = b"model weights"
    monkeypatch.setattr(curl_seed, "fetch_sha", lambda _hf_id: "commit123")
    def download_hf(_url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)

    monkeypatch.setattr(curl_seed, "curl_download", download_hf)
    curl_seed.plant(
        "org/model",
        [{"name": "model.safetensors", "sha256": hashlib.sha256(payload).hexdigest()}],
        tmp_path,
    )
    repo = tmp_path / "hub" / "models--org--model"
    assert (repo / "refs" / "main").read_text() == "commit123"
    assert (repo / "snapshots" / "commit123" / "model.safetensors").read_bytes() == payload


def test_generic_asset_is_atomic_and_skips_verified_hit(tmp_path, monkeypatch) -> None:
    payload = b"verified archive"
    spec = asset_spec(payload)
    target = tmp_path / spec["path"]
    calls = []

    def download(_url: str, partial: Path) -> None:
        calls.append(partial)
        assert not target.exists()
        partial.write_bytes(payload)

    monkeypatch.setattr(curl_seed, "curl_download", download)
    curl_seed.plant_asset(spec, tmp_path)
    curl_seed.plant_asset(spec, tmp_path)
    assert target.read_bytes() == payload
    assert len(calls) == 1
    assert not target.with_name(target.name + ".part").exists()


def test_generic_asset_resumes_partial_file(tmp_path, monkeypatch) -> None:
    spec = asset_spec(b"abcdef")
    target = tmp_path / spec["path"]
    target.parent.mkdir(parents=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(b"abc")

    def resume(_url: str, dest: Path) -> None:
        assert dest == partial
        assert dest.read_bytes() == b"abc"
        with dest.open("ab") as output:
            output.write(b"def")

    monkeypatch.setattr(curl_seed, "curl_download", resume)
    curl_seed.plant_asset(spec, tmp_path)
    assert target.read_bytes() == b"abcdef"


def test_generic_asset_publishes_already_complete_partial(tmp_path, monkeypatch) -> None:
    spec = asset_spec(b"complete")
    target = tmp_path / spec["path"]
    target.parent.mkdir(parents=True)
    target.with_name(target.name + ".part").write_bytes(b"complete")
    monkeypatch.setattr(
        curl_seed, "curl_download",
        lambda *_args: pytest.fail("verified partial should not be downloaded"),
    )
    curl_seed.plant_asset(spec, tmp_path)
    assert target.read_bytes() == b"complete"


def test_generic_asset_restarts_full_corrupt_partial(tmp_path, monkeypatch) -> None:
    payload = b"correct"
    spec = asset_spec(payload)
    spec["size"] = len(payload)
    target = tmp_path / spec["path"]
    target.parent.mkdir(parents=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(b"corrupt")

    def download(_url: str, dest: Path) -> None:
        assert not dest.exists()
        dest.write_bytes(payload)

    monkeypatch.setattr(curl_seed, "curl_download", download)
    curl_seed.plant_asset(spec, tmp_path)
    assert target.read_bytes() == payload


def test_generic_asset_rejects_bad_checksum_without_publishing(tmp_path, monkeypatch) -> None:
    spec = asset_spec(b"expected")
    monkeypatch.setattr(
        curl_seed, "curl_download",
        lambda _url, dest: dest.write_bytes(b"unexpected"),
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        curl_seed.plant_asset(spec, tmp_path)
    target = tmp_path / spec["path"]
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a\\b", "a/../b", "a//b"])
def test_generic_asset_rejects_unsafe_path(tmp_path, monkeypatch, path) -> None:
    monkeypatch.setattr(
        curl_seed, "curl_download",
        lambda *_args: pytest.fail("unsafe asset path reached download"),
    )
    with pytest.raises(ValueError, match="safe relative"):
        curl_seed.plant_asset(asset_spec(b"x", path), tmp_path)


def test_slime_seed_contains_only_yaml_and_matches_setup() -> None:
    import yaml

    seed_dir = REPO_ROOT / "cache-seed" / "slime"
    assert {path.name for path in seed_dir.iterdir()} == {"curl_seeds.yaml"}
    data = yaml.safe_load((seed_dir / "curl_seeds.yaml").read_text())
    assert data["version"] == 1
    assert len(data["assets"]) == 1
    asset = data["assets"][0]
    assert asset["path"].startswith("third_party/slime/")
    assert asset["path"].endswith(asset["url"].rsplit("/", 1)[1])
    assert asset["size"] == 12675026

    setup = (REPO_ROOT / "projects" / "slime" / "scripts" / "setup_example.sh").read_text()
    assert asset["sha256"] in setup
    assert "SHARED_CACHE_ROOT" in setup
    assert "sgl-kernel-npu shared cache hit" in setup
    assert "downloading directly" in setup


def test_seeder_dispatches_assets_without_hf_seeds(tmp_path, monkeypatch) -> None:
    seed_dir = tmp_path / "seed" / "slime"
    seed_dir.mkdir(parents=True)
    (seed_dir / "curl_seeds.yaml").write_text(
        "assets:\n  - url: https://example.invalid/a.zip\n"
        "    path: third_party/slime/a.zip\n"
        f"    sha256: {hashlib.sha256(b'a').hexdigest()}\n",
        encoding="utf-8",
    )
    seen = []
    monkeypatch.setattr(curl_seed, "SEED_ROOT", seed_dir.parent)
    monkeypatch.setattr(curl_seed, "plant_asset", lambda spec, root: seen.append((spec, root)))
    monkeypatch.setattr(
        sys, "argv", ["curl_seed.py", "--projects", "slime", "--root", str(tmp_path / "cache")]
    )
    assert curl_seed.main() == 0
    assert len(seen) == 1
    assert seen[0][0]["path"] == "third_party/slime/a.zip"
