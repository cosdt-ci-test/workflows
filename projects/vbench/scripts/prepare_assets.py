"""Cache the upstream UMT checkpoint and five original LaVie benchmark videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


PROMPT = "A person is strumming guitar"
LAVIE_DRIVE_ID = "1hviZzsInIgJA96ppVj4B2DHhTZWeM4nc"
UMT_REVISION = "83290ee59b4e2c0a4d198857edbc79dd186b8d44"
UMT_FILENAME = "l16_ptk710_ftk710_ftk400_f16_res224.pth"
UMT_SHA256 = "bfee78a03bf806fbc0c216309e8f92792011b7db6556fa860e9d869c1e69fecd"


def check_sha256(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == expected


def extract_samples(archive: Path, destination: Path) -> list[Path]:
    expected = [f"{PROMPT}-{i}.mp4" for i in range(5)]
    with zipfile.ZipFile(archive) as zipped:
        members = {
            Path(name).name: name for name in zipped.namelist()
            if "/human_action/" in f"/{name}" and Path(name).name in expected
        }
        missing = set(expected) - members.keys()
        if missing:
            raise ValueError(f"Missing original LaVie videos: {sorted(missing)}")
        destination.mkdir(parents=True, exist_ok=True)
        for name in expected:
            with zipped.open(members[name]) as source:
                with (destination / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
    return [destination / name for name in expected]


def write_subset_info(source: Path, destination: Path) -> None:
    info = json.loads(source.read_text(encoding="utf-8"))
    selected = [item for item in info if item["prompt_en"] == PROMPT
                and "human_action" in item["dimension"]]
    if len(selected) != 1:
        raise ValueError("The original human_action Prompt changed; review this guard")
    destination.write_text(json.dumps(selected, indent=2), encoding="utf-8")


def prepare_samples(cache: Path, destination: Path) -> list[Path]:
    import cv2

    cached = cache / "lavie/human_action"
    paths = [cached / f"{PROMPT}-{i}.mp4" for i in range(5)]
    if not all(path.is_file() for path in paths):
        archive = cache / "lavie.zip"
        if not zipfile.is_zipfile(archive):
            import gdown

            cache.mkdir(parents=True, exist_ok=True)
            print(f"Downloading official LaVie archive: Google Drive {LAVIE_DRIVE_ID}", flush=True)
            try:
                gdown.download(id=LAVIE_DRIVE_ID, output=str(archive), quiet=False,
                               resume=True, timeout=(30, 120), retries=3)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot download official LaVie samples. Supply lavie.zip in {cache}, "
                    f"or place the five original '{PROMPT}-[0-4].mp4' files in {cached}. "
                    "The HF dataset mirror requires approved access."
                ) from exc
        paths = extract_samples(archive, cached)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Cannot decode cached upstream video: {path}; replace it from the official archive")
        finally:
            capture.release()
        output = destination / path.name
        shutil.copy2(path, output)
        output_paths.append(output)
    return output_paths


def prepare_assets(target: Path, cache: Path) -> None:
    checkpoint = cache / "umt_model" / UMT_FILENAME
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not check_sha256(checkpoint, UMT_SHA256):
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = f"{endpoint}/OpenGVLab/VBench_Used_Models/resolve/{UMT_REVISION}/{UMT_FILENAME}"
        temporary = checkpoint.with_suffix(".pth.part")
        print(f"Downloading official UMT checkpoint: {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as output:
                downloaded = 0
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    print(f"UMT downloaded: {downloaded / 1024**2:.0f} MiB", flush=True)
        if not check_sha256(temporary, UMT_SHA256):
            raise ValueError("UMT checkpoint SHA256 mismatch; cached file was not replaced")
        temporary.replace(checkpoint)
    print(f"UMT checkpoint verified: {checkpoint}", flush=True)

    videos = target / "sampled_videos" / "lavie" / "human_action"
    paths = prepare_samples(cache, videos)
    write_subset_info(target / "vbench/VBench_full_info.json", target / "human_action_full_info.json")
    print(f"Original LaVie human_action videos: {len(paths)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("--cache", type=Path,
                        default=Path(os.environ.get("VBENCH_CACHE_DIR", Path.home() / ".cache/vbench")))
    args = parser.parse_args()
    prepare_assets(args.target.resolve(), args.cache.resolve())
