from __future__ import annotations
import itertools, re
import cv2
from functools import lru_cache
import numpy as np
from pycocotools import mask as mask_utils
from .metrics import dice_score, iou_score
from .baselines import predict_learned_template, template_mask


def decode_rle(rle: dict) -> np.ndarray:
    rr = dict(rle)
    if isinstance(rr["counts"], str):
        rr["counts"] = rr["counts"].encode("ascii")
    return mask_utils.decode(rr).astype(bool)


def normalize_masks(
    masks: list[np.ndarray], target_shape: tuple[int, int]
) -> list[np.ndarray]:
    h, w = target_shape
    out = []
    for mask in masks:
        if mask.shape != target_shape:
            mask = (
                cv2.resize(
                    mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
                > 0
            )
        out.append(mask.astype(bool))
    return out


def union_masks(masks: list[np.ndarray], target_shape: tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.zeros(target_shape, dtype=bool)
    out = np.zeros(target_shape, dtype=bool)
    for mask in masks:
        out |= mask
    return out


def ranked_union(
    masks: list[np.ndarray], scores: list[float], k: int, target_shape: tuple[int, int]
) -> np.ndarray:
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)[:k]
    return union_masks([masks[i] for i in order], target_shape)


def template_scores(
    masks: list[np.ndarray],
    target_shape: tuple[int, int],
    class_id: str,
    learned_templates: dict,
) -> list[float]:
    h, w = target_shape
    learned = predict_learned_template(learned_templates[class_id], h, w)
    geom = template_mask(class_id, h, w)
    return [
        0.7 * iou_score(mask, learned) + 0.3 * iou_score(mask, geom) for mask in masks
    ]


def parse_ids(text: str, num_candidates: int) -> tuple[list[int], str]:
    match = re.search(r"\[[^\]]*\]", text)
    source = match.group(0) if match else text
    ids = []
    for tok in re.findall(r"\d+", source):
        val = int(tok)
        if 1 <= val <= num_candidates and val not in ids:
            ids.append(val)
        if len(ids) == 3:
            break
    invalid = ""
    if not ids:
        invalid = "no_valid_ids"
    return ids, invalid


@lru_cache(None)
def subsets(n, exact_three=False):
    sizes = [min(3, n)] if exact_three else range(min(3, n) + 1)
    return [
        (sum(1 << i for i in ids), tuple(i + 1 for i in ids))
        for k in sizes
        for ids in itertools.combinations(range(n), k)
    ]


def exact_oracles(masks, target):
    """Enumerate all allowed subsets via exact integer pixel counts (subset sums)."""
    n = len(masks)
    if n > 12:
        raise ValueError("Exact control restricted to frozen visible bank")
    membership = np.zeros(target.shape, np.uint16)
    for i, m in enumerate(masks):
        membership |= m.astype(np.uint16) << i
    hist = np.bincount(membership.ravel(), minlength=1 << n).astype(np.int64)
    inter = np.bincount(membership[target], minlength=1 << n).astype(np.int64)
    # After the transform, h[s] counts pixels whose membership is a subset of s.
    for bit in range(n):
        step = 1 << bit
        for a in (hist, inter):
            view = a.reshape(-1, 2 * step)
            view[:, step:] += view[:, :step]
    area_target = int(target.sum())
    output = {}
    for name, exactly in [("exact_up_to_3", False), ("exact_3", True)]:
        candidates = subsets(n, exactly)
        scores = []
        for bits, ids in candidates:
            complement = ((1 << n) - 1) ^ bits
            area = target.size - int(hist[complement])
            tp = area_target - int(inter[complement])
            scores.append(2 * tp / (area + area_target) if area + area_target else 1.0)
        best = int(np.argmax(scores))  # fewest then lexicographic IDs on ties
        ids = candidates[best][1]
        pred = np.zeros_like(target)
        for i in ids:
            pred |= masks[i - 1]
        assert abs(dice_score(pred, target) - scores[best]) < 1e-14
        output[name] = (pred, ids)
    return output


def greedy_oracle_union(masks, target, k=3):
    """Exact original greedy rule with integer bitsets; force min(k,n) additions."""
    target_bits = int.from_bytes(np.packbits(target.reshape(-1)).tobytes(), "little")
    bits = [
        int.from_bytes(np.packbits(m.reshape(-1)).tobytes(), "little") for m in masks
    ]
    target_area = int(target.sum())
    current = 0
    chosen = []
    for _ in range(min(k, len(masks))):
        best_i = None
        best = -1.0
        for i, m in enumerate(bits):
            if i in chosen:
                continue
            union = current | m
            denom = union.bit_count() + target_area
            score = 2 * (union & target_bits).bit_count() / denom if denom else 1.0
            if score > best:
                best_i = i
                best = score
        chosen.append(best_i)
        current |= bits[best_i]
    return union_masks([masks[i] for i in chosen], target.shape)
