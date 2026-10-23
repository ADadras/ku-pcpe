"""Verify bundled archives, evaluator integrity and the distribution boundary."""

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def verify_website_assets(root):
    """Allow only the exact website illustrations reviewed with the paper."""
    site = root / "website/site"
    records = json.loads((root / "website/ASSET_MANIFEST.json").read_text())["sha256"]
    actual = {
        path.relative_to(site).as_posix()
        for path in (site / "assets").rglob("*")
        if path.is_file()
    }
    if actual != set(records):
        raise ValueError("Unlisted or missing website asset")
    for relative, expected in records.items():
        path = site / relative
        if (
            not path.resolve().is_relative_to((site / "assets").resolve())
            or path.is_symlink()
            or digest(path) != expected
        ):
            raise ValueError("Unapproved website asset: " + relative)
    return {site / relative for relative in records if Path(relative).suffix == ".webp"}


def verify(root=ROOT):
    root = Path(root)
    records = json.loads((root / "configs/release_assets.json").read_text())
    if len(records) != 3:
        raise ValueError("The release requires exactly three archives")
    checked = []
    for record in records:
        path = root / "data/bundles" / record["file"]
        if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise ValueError("Archive size or checksum mismatch: " + path.name)
        if path.stat().st_size >= 100 * 1024**2:
            raise ValueError("Archive exceeds GitHub's per-file limit")
        with tarfile.open(path) as archive:
            members = archive.getmembers()
            for member in members:
                parts = Path(member.name).parts
                if (
                    not parts
                    or Path(member.name).is_absolute()
                    or ".." in parts
                    or parts[0] != record["root"]
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("Unsafe archive member: " + member.name)
                if Path(member.name).suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".webp",
                    ".sqlite",
                    ".db",
                    ".pt",
                    ".pth",
                    ".safetensors",
                }:
                    raise ValueError("Excluded source image, database or weight file")
                if member.uid or member.gid or member.uname or member.gname:
                    raise ValueError("Archive contains owner metadata")
        checked.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": record["sha256"],
            }
        )
    frozen = json.loads((root / "configs/evaluator_sha256.json").read_text())
    for relative, expected in frozen.items():
        if digest(root / relative) != expected:
            raise ValueError("Frozen evaluator changed: " + relative)
    approved_illustrations = verify_website_assets(root)
    excluded = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
    }
    scanned = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or relative.parts[0] in {"assets", "outputs"}
            or any(
                part in excluded or part.endswith(".egg-info")
                for part in relative.parts
            )
        ):
            continue
        if path.stat().st_size >= 100 * 1024**2:
            raise ValueError("File exceeds GitHub limit: " + path.name)
        if (
            path.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".webp",
                ".sqlite",
                ".db",
                ".pt",
                ".pth",
                ".safetensors",
            }
            and path not in approved_illustrations
        ) or path.name.startswith(".env"):
            raise ValueError("Excluded private or upstream asset: " + path.name)
        if path.suffix in {
            ".py",
            ".md",
            ".txt",
            ".json",
            ".csv",
            ".yml",
            ".cff",
            ".html",
            ".css",
            ".js",
            ".bib",
        }:
            text = path.read_text()
            if re.search(
                r"gh[pousr]_[A-Za-z0-9]{30,}|^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----$",
                text,
                re.M,
            ):
                raise ValueError("Credential pattern: " + path.name)
            # Tests intentionally contain rejected path/field examples.
            if "tests" not in path.parts and re.search(
                r"/(?:home|share|Users)/[A-Za-z0-9][^\s\"']+", text
            ):
                raise ValueError("Personal infrastructure path: " + path.name)
        scanned += 1
    result = {
        "status": "PASS",
        "archives": checked,
        "frozen_evaluator_files": len(frozen),
        "approved_website_illustrations": len(approved_illustrations),
        "source_files_scanned": scanned,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    verify(parser.parse_args().root)
