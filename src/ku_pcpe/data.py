from pathlib import Path
import csv, json
import numpy as np
from PIL import Image
from .metrics import dice_score

CLASSES = [
    "horizontal",
    "vertical",
    "diagonal",
    "curved",
    "triangle",
    "rule_of_thirds",
    "center",
    "symmetric",
    "pattern",
]


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def read_record_mask(record, root):
    path = Path(root) / record["mask_path"]
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Mask path escapes bundle")
    with Image.open(path) as image:
        a = np.asarray(image.convert("L"))
    if a.shape != (int(record["height"]), int(record["width"])):
        raise ValueError("Mask dimension mismatch")
    if not np.isin(a, [0, 255]).all():
        raise ValueError("Nonbinary annotation")
    if int((a > 0).sum()) != int(record["area_px"]):
        raise ValueError("Mask area mismatch")
    return a > 0


def keyed(records):
    result = {r["task_id"]: r for r in records}
    if len(result) != len(records):
        raise ValueError("Duplicate task ID")
    return result


def image_key(split, image_file):
    if split not in ["train", "test"]:
        raise ValueError("Unknown split")
    if Path(image_file).name != image_file:
        raise ValueError("Expected a split-local filename")
    return split, image_file


def align(masks):
    if not masks:
        raise ValueError("Missing expert masks")
    return [masks[0]] + [
        ~m if dice_score(~m, masks[0]) > dice_score(m, masks[0]) else m
        for m in masks[1:]
    ]


def consensus(masks, aligned=True):
    if len(masks) != 3:
        raise ValueError("Expected exactly three experts")
    if len({m.shape for m in masks}) != 1:
        raise ValueError("Expert shape mismatch")
    return np.stack(align(masks) if aligned else masks).mean(0) >= 0.5


def load_release(root):
    root = Path(root)
    train = root / "annotations/single_annotator_train"
    records = jsonl(train / "annotations.jsonl")
    keyed(records)
    bundles = [root / f"annotations/expert_test/expert_0{i}" for i in range(1, 4)]
    tables = [keyed(jsonl(b / "annotations.jsonl")) for b in bundles]
    assert len(records) == 4093 and all(len(t) == 1327 for t in tables)
    assert all(set(t) == set(tables[0]) for t in tables)
    assert not set(keyed(records)) & set(tables[0])
    assert len({r["image_id"] for r in records}) == 3169
    assert len({r["image_id"] for r in tables[0].values()}) == 1073
    assert set(r["class_id"] for r in records) == set(CLASSES)
    return train, records, bundles, tables
