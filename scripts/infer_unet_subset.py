"""Optional CPU inference on a predeclared one-task-per-class test subset."""

import argparse, hashlib, json
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from ku_pcpe.data import CLASSES, jsonl
from ku_pcpe.metrics import dice_score
from ku_pcpe.unet import SmallUNet

p = argparse.ArgumentParser()
p.add_argument("--annotations", type=Path, required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--images", type=Path, required=True)
p.add_argument("--replay", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
# Fixed before inference/comparison, independent of target quality or observed score.
records = jsonl(a.annotations / "annotations/expert_test/expert_01/annotations.jsonl")
subset = [
    min((r for r in records if r["class_id"] == c), key=lambda r: r["task_id"])
    for c in CLASSES
]
torch.set_num_threads(2)
checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
assert checkpoint["classes"] == CLASSES
args = checkpoint["args"]
assert args["size"] == 384
model = SmallUNet(in_ch=12, base=args["base"])
model.load_state_dict(checkpoint["model"])
model.eval()
rows = []
for rec in subset:
    with Image.open(a.images / rec["image_file"]) as image:
        rgb = (
            np.asarray(
                image.convert("RGB").resize((384, 384), Image.Resampling.BILINEAR)
            ).astype(np.float32)
            / 255
        )
    label = np.zeros((384, 384, 9), np.float32)
    label[..., CLASSES.index(rec["class_id"])] = 1
    x = np.concatenate([rgb, label], 2).transpose(2, 0, 1)
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(x[None])))[0, 0].numpy()
    pred = (
        cv2.resize(prob, (rec["width"], rec["height"]), interpolation=cv2.INTER_LINEAR)
        >= 0.5
    )
    with Image.open(
        a.replay
        / "supervised/unet_label_seed13"
        / (rec["task_id"].replace(":", "__") + ".png")
    ) as image:
        saved = np.asarray(image.convert("L")) > 0
    fraction = float((pred != saved).mean())
    dice = dice_score(pred, saved)
    rows.append(
        dict(
            task_id=rec["task_id"],
            class_id=rec["class_id"],
            dice_to_saved=dice,
            differing_pixel_fraction=fraction,
            passed=dice >= 0.999 and fraction <= 0.001,
        )
    )
result = dict(
    status="PASS" if all(r["passed"] for r in rows) else "FAIL",
    scope="CPU label-conditioned U-Net, one lexicographically first task per class; no full inference or training claim",
    torch=torch.__version__,
    checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
    predeclared_tolerance=dict(min_dice=0.999, max_pixel_difference_fraction=0.001),
    tasks=rows,
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
if result["status"] != "PASS":
    raise RuntimeError("Subset predictions exceed predeclared tolerance")
