"""Write split-aware local source mapping; never copy source photographs."""

import argparse, csv, json
from pathlib import Path
from PIL import Image
from ku_pcpe.data import jsonl, image_key

p = argparse.ArgumentParser()
p.add_argument("--annotations", type=Path, required=True)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
rows = []
seen = {}
for split, bundle in [
    ("train", a.annotations / "annotations/single_annotator_train"),
    ("test", a.annotations / "annotations/expert_test/expert_01"),
]:
    for r in jsonl(bundle / "annotations.jsonl"):
        key = image_key(split, r["image_file"])
        path = a.source / (split + "_img") / r["image_file"]
        with Image.open(path) as image:
            assert image.size == (r["width"], r["height"])
        seen[key] = path
        rows.append(dict(task_id=r["task_id"], split=split, image_file=r["image_file"]))
with a.output.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(json.dumps(dict(status="PASS", tasks=len(rows), split_local_images=len(seen))))
