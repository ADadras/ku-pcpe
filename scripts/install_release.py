"""Verify and install the three archives supplied in this repository."""

import argparse
import hashlib
import json
from pathlib import Path
from ku_pcpe.validation import install_bundle

ROOT = Path(__file__).resolve().parents[1]


def install(output):
    records = json.loads((ROOT / "configs/release_assets.json").read_text())
    for record in records:
        archive = ROOT / "data/bundles" / record["file"]
        if archive.stat().st_size != record["bytes"]:
            raise ValueError("Archive size mismatch: " + archive.name)
        install_bundle(archive, record["sha256"], output)
    metadata = output / "kupcp_completed_public_20261023_v1"
    expected = set()
    for line in (metadata / "checksums.sha256").read_text().splitlines():
        checksum, relative = line.split(None, 1)
        relative = relative.strip()
        path = metadata / relative
        if not path.resolve().is_relative_to(metadata.resolve()):
            raise ValueError("Metadata checksum path escapes its root")
        if relative in expected:
            raise ValueError("Duplicate metadata checksum entry")
        expected.add(relative)
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        if value != checksum:
            raise ValueError("Metadata file checksum mismatch: " + relative)
    actual = {
        p.relative_to(metadata).as_posix() for p in metadata.rglob("*") if p.is_file()
    }
    if actual != expected | {"checksums.sha256"}:
        raise ValueError("Metadata inventory differs from checksum manifest")
    print(
        json.dumps(
            {
                "status": "PASS",
                "archives": len(records),
                "complete_metadata_files": len(expected),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/data")
    install(parser.parse_args().output)
