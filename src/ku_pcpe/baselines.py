from __future__ import annotations
import hashlib
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .data import CLASSES, read_record_mask
from .metrics import boundary_f1, dice_score, iou_score, overcover


def resize_bool(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((w, h), Image.Resampling.BILINEAR)
    return np.asarray(img) >= 128


def top_fraction_mask(score: np.ndarray, frac: float) -> np.ndarray:
    frac = float(np.clip(frac, 0.001, 0.95))
    threshold = np.quantile(score.reshape(-1), 1.0 - frac)
    return score >= threshold


def random_area_mask(h: int, w: int, frac: float, key: str) -> np.ndarray:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    n = int(round(float(np.clip(frac, 0.001, 0.95)) * h * w))
    flat = np.zeros(h * w, dtype=bool)
    flat[rng.choice(h * w, size=n, replace=False)] = True
    return flat.reshape(h, w)


def template_mask(class_id: str, h: int, w: int) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    if class_id == "horizontal":
        band = max(8, int(h * 0.16))
        cy = h // 2
        draw.rectangle([0, cy - band // 2, w, cy + band // 2], fill=255)
    elif class_id == "vertical":
        band = max(8, int(w * 0.16))
        cx = w // 2
        draw.rectangle([cx - band // 2, 0, cx + band // 2, h], fill=255)
    elif class_id == "diagonal":
        width = max(12, int(min(h, w) * 0.08))
        draw.line([0, h, w, 0], fill=255, width=width)
        draw.line([0, 0, w, h], fill=160, width=max(4, width // 2))
    elif class_id == "curved":
        width = max(12, int(min(h, w) * 0.08))
        draw.arc(
            [int(w * 0.12), int(h * 0.08), int(w * 0.88), int(h * 1.08)],
            190,
            350,
            fill=255,
            width=width,
        )
    elif class_id == "triangle":
        pts = [
            (w // 2, int(h * 0.18)),
            (int(w * 0.18), int(h * 0.82)),
            (int(w * 0.82), int(h * 0.82)),
        ]
        draw.polygon(pts, fill=255)
    elif class_id == "rule_of_thirds":
        band_w = max(8, int(w * 0.10))
        band_h = max(8, int(h * 0.10))
        for x in (w // 3, 2 * w // 3):
            draw.rectangle([x - band_w // 2, 0, x + band_w // 2, h], fill=180)
        for y in (h // 3, 2 * h // 3):
            draw.rectangle([0, y - band_h // 2, w, y + band_h // 2], fill=180)
    elif class_id == "center":
        draw.ellipse(
            [int(w * 0.32), int(h * 0.25), int(w * 0.68), int(h * 0.75)], fill=255
        )
    elif class_id == "symmetric":
        band = max(8, int(w * 0.08))
        cx = w // 2
        draw.rectangle([cx - band // 2, 0, cx + band // 2, h], fill=255)
        draw.rectangle(
            [int(w * 0.12), int(h * 0.25), int(w * 0.40), int(h * 0.75)], fill=160
        )
        draw.rectangle(
            [int(w * 0.60), int(h * 0.25), int(w * 0.88), int(h * 0.75)], fill=160
        )
    elif class_id == "pattern":
        step_x = max(24, w // 5)
        step_y = max(24, h // 4)
        r = max(8, min(step_x, step_y) // 5)
        for y in range(step_y // 2, h, step_y):
            for x in range(step_x // 2, w, step_x):
                draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
    else:
        draw.rectangle([0, 0, w, h], fill=255)
    return np.asarray(img) > 0


def learn_class_templates(
    train_records: list[dict], train_export: Path, size: int = 192
) -> dict[str, dict]:
    accum = {c: np.zeros((size, size), dtype=np.float32) for c in CLASSES}
    counts = {c: 0 for c in CLASSES}
    areas: dict[str, list[float]] = defaultdict(list)
    for rec in train_records:
        c = rec["class_id"]
        mask = read_record_mask(rec, train_export)
        small = cv2.resize(
            mask.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA
        )
        accum[c] += small
        counts[c] += 1
        areas[c].append(float(mask.mean()))
    templates = {}
    for c in CLASSES:
        avg = accum[c] / max(1, counts[c])
        area = float(np.median(areas[c])) if areas[c] else 0.2
        templates[c] = {"score": avg, "area": area, "count": counts[c]}
    return templates


def predict_learned_template(template: dict, h: int, w: int) -> np.ndarray:
    score = cv2.resize(template["score"], (w, h), interpolation=cv2.INTER_LINEAR)
    return top_fraction_mask(score, float(template["area"]))


def score_prediction(pred: np.ndarray, target: np.ndarray) -> dict:
    scale = 256.0 / max(target.shape)
    if scale < 1.0:
        new_h = max(1, int(round(target.shape[0] * scale)))
        new_w = max(1, int(round(target.shape[1] * scale)))
        pred_b = resize_bool(pred, new_h, new_w)
        target_b = resize_bool(target, new_h, new_w)
    else:
        pred_b = pred
        target_b = target
    return {
        "iou": iou_score(pred, target),
        "dice": dice_score(pred, target),
        "boundary_f1": boundary_f1(pred_b, target_b, tol=2),
        "overcover": overcover(pred, target),
        "area_frac": float(pred.mean()),
    }
