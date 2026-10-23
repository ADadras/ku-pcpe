"""Download an annotation archive from a supplied HTTPS URL."""

import argparse, urllib.request
from pathlib import Path
from ku_pcpe.validation import digest

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--sha256", required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if not a.url.startswith("https://"):
    raise ValueError("Expected an author-published HTTPS URL")
if a.output.exists():
    raise FileExistsError(a.output)
with urllib.request.urlopen(a.url) as source, a.output.open("wb") as target:
    while chunk := source.read(1 << 20):
        target.write(chunk)
if digest(a.output) != a.sha256:
    raise ValueError("Downloaded checksum mismatch")
