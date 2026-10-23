from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = as_bool(pred)
    target = as_bool(target)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(pred, target).sum()
    return float(inter / union)


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = as_bool(pred)
    target = as_bool(target)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    inter = np.logical_and(pred, target).sum()
    return float(2.0 * inter / denom)


def overcover(pred: np.ndarray, target: np.ndarray) -> float:
    pred = as_bool(pred)
    target = as_bool(target)
    extra = np.logical_and(pred, ~target).sum()
    pred_area = pred.sum()
    if pred_area == 0:
        return 0.0
    return float(extra / pred_area)


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = as_bool(mask)
    if mask.sum() == 0:
        return mask
    eroded = binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def boundary_f1(pred: np.ndarray, target: np.ndarray, tol: int = 2) -> float:
    pb = boundary(pred)
    tb = boundary(target)
    if pb.sum() == 0 and tb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or tb.sum() == 0:
        return 0.0
    dist_to_t = distance_transform_edt(~tb)
    dist_to_p = distance_transform_edt(~pb)
    p_match = (dist_to_t[pb] <= tol).sum()
    t_match = (dist_to_p[tb] <= tol).sum()
    precision = p_match / max(1, pb.sum())
    recall = t_match / max(1, tb.sum())
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
