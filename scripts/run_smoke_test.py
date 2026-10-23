"""CPU-only end-to-end synthetic example; no source photos, credentials, or network."""

import json, socket, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from ku_pcpe.data import consensus, image_key
from ku_pcpe.baselines import score_prediction
from ku_pcpe.proposals import parse_ids, union_masks, exact_oracles


def blocked(*args, **kwargs):
    raise RuntimeError("Network is forbidden in the synthetic fixture")


socket.socket.connect = blocked
socket.create_connection = blocked
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    a = np.zeros((20, 30), bool)
    a[2:10, 3:13] = True
    b = np.zeros_like(a)
    b[12:17, 20:26] = True
    ids, _ = parse_ids('selected: [1,1,"2",99]', 2)
    assert ids == [1, 2]
    scores = []
    for label, target in [("center", a), ("pattern", b)]:
        for split in ["train", "test"]:
            root = base / split
            root.mkdir(exist_ok=True)
            Image.fromarray(target.astype("uint8") * 255).save(root / (label + ".png"))
        truth = consensus([target, target, ~target], True)
        assert np.array_equal(truth, target)
        pred = exact_oracles([a, b], truth)["exact_up_to_3"][0]
        metrics = score_prediction(pred, truth)
        assert metrics["dice"] == 1.0
        scores.append(dict(label=label, dice=metrics["dice"]))
    assert image_key("train", "0001.jpg") != image_key("test", "0001.jpg")
    print(
        json.dumps(
            dict(status="PASS", tasks=2, split_collision_avoided=True, scores=scores),
            sort_keys=True,
        )
    )
