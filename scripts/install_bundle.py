import argparse, json
from pathlib import Path
from ku_pcpe.validation import install_bundle

p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--bundle-dir", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
for r in json.loads(a.manifest.read_text()):
    install_bundle(a.bundle_dir / r["file"], r["sha256"], a.output)
print("Verified and installed both versioned bundles")
