"""Portable file, schema, and privacy checks for release artifacts."""

from pathlib import Path
import hashlib, json, tarfile

BLACKLIST = [
    "/home/",
    "/share/",
    "ssh://",
    "annotator_id",
    "action_types",
    '"actions"',
    "browser_token",
    "api_key",
    "password",
    "annotations.sqlite",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_text(text):
    for key in BLACKLIST:
        if key in text:
            raise ValueError("Private field or infrastructure reference: " + key)


def privacy_scan(root):
    scanned = 0
    for p in Path(root).rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        if p.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".webp",
            ".sqlite",
            ".db",
            ".pt",
            ".pth",
            ".safetensors",
        ]:
            raise ValueError("Excluded binary asset: " + p.name)
        if p.suffix.lower() in [
            ".json",
            ".jsonl",
            ".csv",
            ".md",
            ".txt",
            ".cff",
            ".yml",
            ".yaml",
        ]:
            # The blacklist policy documentation names rejected fields intentionally.
            if p.name not in ["SECURITY_AND_PRIVACY.md"]:
                inspect_text(p.read_text())
        scanned += 1
    return scanned


def verify_replay(root):
    root = Path(root)
    rows = json.loads((root / "checksums.json").read_text())
    paths = set()
    for r in rows:
        p = root / r["path"]
        if not p.resolve().is_relative_to(root.resolve()):
            raise ValueError("Path escape")
        if r["path"] in paths:
            raise ValueError("Duplicate manifest entry")
        paths.add(r["path"])
        assert p.stat().st_size == r["bytes"] and digest(p) == r["sha256"]
    actual = {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    } - {"checksums.json"}
    assert actual == paths
    return dict(status="PASS", files=len(rows), privacy_files=privacy_scan(root))


def install_bundle(archive, sha256, output):
    archive = Path(archive)
    output = Path(output)
    if digest(archive) != sha256:
        raise ValueError("Bundle checksum mismatch")
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar.getmembers():
            if (
                not (output / m.name).resolve().is_relative_to(output.resolve())
                or m.issym()
                or m.islnk()
            ):
                raise ValueError("Unsafe archive entry")
        tar.extractall(output, filter="data")
