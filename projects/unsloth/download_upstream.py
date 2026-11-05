"""Download the pinned upstream source archives without fetching Git objects."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile


def download_source(repository, revision, parent):
    if repository not in {"unsloth", "unsloth-zoo"} or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected a known upstream repository and a full commit SHA")
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / repository
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    url = f"https://codeload.github.com/unslothai/{repository}/tar.gz/{revision}"
    print(f"Downloading {repository} at {revision}", flush=True)
    with tempfile.TemporaryDirectory(prefix=f".{repository}-", dir=parent) as directory:
        staging = Path(directory).resolve()
        assert staging.is_relative_to(parent)
        archive = staging / "source.tar.gz"
        subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "20", "--max-time", "180",
            "--retry", "3", "--retry-all-errors", "--retry-delay", "5",
            "--output", str(archive), url,
        ], check=True)
        root_name = f"{repository}-{revision}"
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            if not members or any(
                member.name.split("/")[0] != root_name for member in members
            ):
                raise ValueError("Unexpected source archive root")
            source.extractall(staging, filter="data")
        extracted = staging / root_name
        if not (extracted / "pyproject.toml").is_file():
            raise ValueError("Source archive has no pyproject.toml")
        if not extracted.resolve().is_relative_to(staging):
            raise ValueError("Source archive resolved outside staging")
        with archive.open("rb") as stream:
            archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        print(f"Verified archive: {repository} commit={revision} sha256={archive_hash}", flush=True)
        extracted.rename(destination)


if __name__ == "__main__":
    download_source("unsloth", os.environ["UNSLOTH_REF"], "upstream")
    download_source("unsloth-zoo", os.environ["ZOO_REF"], "upstream")
