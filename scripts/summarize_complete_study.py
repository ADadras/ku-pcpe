"""Recompute process tables and disjoint task strata using every completed rater."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import multiprocessing as mp
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

GEOMETRY_EVENTS = frozenset(
    {
        "add_dynamic_segment",
        "accept_region",
        "add_positive_point",
        "add_negative_point",
        "apply_manual_refinement",
        "split_segment",
        "remove_region",
        "clear_current_region",
        "undo_refinement",
    }
)
MANUAL_EVENTS = frozenset(
    {
        "enter_manual_refinement",
        "apply_manual_refinement",
        "exit_manual_refinement",
        "undo_refinement",
    }
)
GROUPS = ("no_geometry_action", "any_geometry_action")
METRICS = ("dice", "iou", "boundary_f1", "overcover", "hit_at_0_5")
METHODS = (
    "Random matched area",
    "Train class template",
    "SAM2.1 Oracle@3",
    "SAM2.1 Template@3",
    "U-Net no label",
    "U-Net + label",
    "SegFormer-B0",
    "Qwen3-VL-32B proposals",
)
PRIMARY = (
    "SAM2.1 Oracle@3",
    "SAM2.1 Template@3",
    "SegFormer-B0",
    "Qwen3-VL-32B proposals",
)
AGREEMENT = ("raw_dice", "orientation_invariant_dice", "boundary_f1", "raw_pair_kappa")


def csv_read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def csv_write(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def task_groups(test_records, flags):
    """Each task contributes once, after checking all three distinct raters."""
    grouped = defaultdict(list)
    for row in test_records:
        grouped[row["task_id"]].append(row)
    result = {}
    raters = {r["annotator_id"] for r in test_records}
    if len(raters) != 3:
        raise ValueError("Expected exactly three test raters")
    for task, rows in grouped.items():
        if len(rows) != 3 or {r["annotator_id"] for r in rows} != raters:
            raise ValueError(f"Missing or duplicated expert for {task}")
        result[task] = GROUPS[int(any(flags[r["annotation_id"]] for r in rows))]
    return result, grouped


def transition_boundary(mask):
    # Historical descriptive agreement: transitions on BOTH sides, no image frame.
    b = np.zeros_like(mask)
    vertical = mask[1:] != mask[:-1]
    horizontal = mask[:, 1:] != mask[:, :-1]
    b[1:] |= vertical
    b[:-1] |= vertical
    b[:, 1:] |= horizontal
    b[:, :-1] |= horizontal
    return b


def descriptive_boundary_f1(a, b):
    h, w = a.shape
    scale = min(1.0, 256 / max(h, w))
    size = (round(w * scale), round(h * scale))
    a = np.asarray(Image.fromarray(a).resize(size, Image.Resampling.NEAREST))
    b = np.asarray(Image.fromarray(b).resize(size, Image.Resampling.NEAREST))
    a, b = transition_boundary(a), transition_boundary(b)
    if not a.any() or not b.any():
        return float(not a.any() and not b.any())
    # The original descriptive tolerance is a 5x5 square (Chebyshev <= 2).
    precision = maximum_filter(b, size=5, mode="constant")[a].mean()
    recall = maximum_filter(a, size=5, mode="constant")[b].mean()
    return (
        float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )


def pair_statistics(item):
    task, root, rows = item
    # Keep the published reference slot fixed. Public IDs are release-specific;
    # this mapping was derived by exact mask identity, not participant identity.
    order = {"rater_04": 0, "rater_01": 1, "rater_03": 2}
    rows = sorted(rows, key=lambda r: order[r["annotator_id"]])
    masks = []
    for r in rows:
        with Image.open(Path(root) / r["mask_path"]) as im:
            masks.append(np.asarray(im) > 0)
    n = masks[0].size
    areas = [int(x.sum()) for x in masks]
    pairs = {}
    raw, kappas, boundaries, oriented = [], [], [], []
    for i, j in itertools.combinations(range(3), 2):
        inter = int(np.logical_and(masks[i], masks[j]).sum())
        pairs[i, j] = inter
        raw.append(2 * inter / (areas[i] + areas[j]) if areas[i] + areas[j] else 1.0)
        complement_denom = areas[i] + n - areas[j]
        complement_dice = (
            2 * (areas[i] - inter) / complement_denom if complement_denom else 1.0
        )
        oriented.append(max(raw[-1], complement_dice))
        chance = (areas[i] * areas[j] + (n - areas[i]) * (n - areas[j])) / n**2
        observed = (n - areas[i] - areas[j] + 2 * inter) / n
        kappas.append((observed - chance) / (1 - chance) if chance < 1 else 1.0)
        boundaries.append(descriptive_boundary_f1(masks[i], masks[j]))
    return {
        "task_id": task,
        "class_id": rows[0]["class_id"],
        "raw_dice": float(np.mean(raw)),
        "orientation_invariant_dice": float(np.mean(oriented)),
        "boundary_f1": float(np.mean(boundaries)),
        "raw_pair_kappa": float(np.mean(kappas)),
    }


def interval(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None, None, None
    rng = np.random.default_rng(seed)
    boot = np.empty(samples)
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        boot[start:stop] = values[
            rng.integers(0, len(values), (stop - start, len(values)))
        ].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def add_interval(row, name, values, samples, seed):
    point, low, high = interval(values, samples, seed)
    row.update({name: point, name + "_ci_low": low, name + "_ci_high": high})


def compare_expected(out, expected):
    maximum = 0.0
    expected_files = sorted(Path(expected).glob("*.csv"))
    if not expected_files:
        raise ValueError("Expected directory contains no CSVs")
    for p in expected_files:
        before, after = csv_read(p), csv_read(Path(out) / p.name)
        if len(before) != len(after):
            raise ValueError(f"Expected row count mismatch: {p.name}")
        for x, y in zip(before, after, strict=True):
            if x.keys() != y.keys():
                raise ValueError(f"Expected columns mismatch: {p.name}")
            for key, value in x.items():
                try:
                    delta = abs(float(value) - float(y[key]))
                except ValueError:
                    if value != y[key]:
                        raise ValueError(
                            f"Expected text mismatch: {p.name}/{key}"
                        ) from None
                else:
                    maximum = max(maximum, delta)
                    if not np.isfinite(delta) or delta > 1e-12:
                        raise ValueError(f"Expected number mismatch: {p.name}/{key}")
    return maximum


def modality_statistics(values):
    return {
        "ratings": len(values),
        "mean_rating": float(values.mean()),
        "segment_percent": 100 * float((values < 1 / 3).mean()),
        "mixed_percent": 100 * float(((values >= 1 / 3) & (values <= 2 / 3)).mean()),
        "line_percent": 100 * float((values > 2 / 3).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument(
        "--benchmark-results",
        type=Path,
        required=True,
        help="Tier 1 canonical_per_task.csv, before any stratification",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-dir", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=920)
    args = parser.parse_args()
    if args.workers < 1 or args.samples < 1 or args.seed < 0:
        parser.error("workers and samples must be positive; seed must be nonnegative")
    started = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    root = args.metadata_root
    records = csv_read(root / "annotations.csv")
    annotations = {r["annotation_id"]: r for r in records}
    if len(records) != 8074 or len(annotations) != 8074:
        raise ValueError("Annotation inventory differs from this versioned release")
    event_types, totals = defaultdict(set), Counter()
    for r in csv_read(root / "tool_timeline.csv"):
        ident = r["annotation_id"]
        if ident not in annotations:
            raise ValueError("Timeline references an unknown annotation")
        event_types[ident].add(r["event_type"])
        totals[ident] += int(r["event_count"])
    flags = {}
    for r in records:
        ident = r["annotation_id"]
        if not re.fullmatch(r"rater_\d{2}", r["annotator_id"]):
            raise ValueError("Expected public rater pseudonym")
        if totals[ident] != int(r["recorded_action_count"]) or totals[ident] <= 0:
            raise ValueError("Missing/inconsistent retained action records")
        flags[ident] = bool(event_types[ident] & GEOMETRY_EVENTS)
    test = [r for r in records if r["split"] == "val"]
    groups, test_by_task = task_groups(test, flags)
    if len(test) != 3981 or len(groups) != 1327:
        raise ValueError("Test annotation/task coverage mismatch")
    csv_write(
        out / "annotation_workflows.csv",
        [
            {
                **{
                    k: r[k]
                    for k in (
                        "annotation_id",
                        "annotator_id",
                        "split",
                        "task_id",
                        "class_id",
                    )
                },
                "geometry_action": int(flags[r["annotation_id"]]),
                "recorded_actions": totals[r["annotation_id"]],
            }
            for r in sorted(records, key=lambda x: int(x["annotation_id"]))
        ],
    )
    csv_write(
        out / "task_workflows.csv",
        [
            {
                "task_id": t,
                "class_id": test_by_task[t][0]["class_id"],
                "workflow": groups[t],
                "experts_with_geometry_action": sum(
                    flags[r["annotation_id"]] for r in test_by_task[t]
                ),
            }
            for t in sorted(groups)
        ],
    )
    process = []
    for name, subset in [
        ("train", [r for r in records if r["split"] == "train"]),
        ("test_pooled", test),
    ] + [
        (rater, [r for r in records if r["annotator_id"] == rater])
        for rater in sorted({r["annotator_id"] for r in records})
    ]:
        n = len(subset)
        row = {
            "set": name,
            "annotations": n,
            "unique_tasks": len({r["task_id"] for r in subset}),
            "raters": len({r["annotator_id"] for r in subset}),
            "no_geometry_actions": sum(not flags[r["annotation_id"]] for r in subset),
            "geometry_actions": sum(flags[r["annotation_id"]] for r in subset),
            "manual_interaction": sum(
                bool(event_types[r["annotation_id"]] & MANUAL_EVENTS) for r in subset
            ),
            "manual_applied": sum(
                "apply_manual_refinement" in event_types[r["annotation_id"]]
                for r in subset
            ),
            "split_operation": sum(
                "split_segment" in event_types[r["annotation_id"]] for r in subset
            ),
            "dynamic_addition": sum(
                "add_dynamic_segment" in event_types[r["annotation_id"]] for r in subset
            ),
            "median_recorded_actions": float(
                np.median([totals[r["annotation_id"]] for r in subset])
            ),
        }
        for key in [
            "no_geometry_actions",
            "geometry_actions",
            "manual_interaction",
            "manual_applied",
            "split_operation",
            "dynamic_addition",
        ]:
            row[key + "_percent"] = 100 * row[key] / n
        assert row["no_geometry_actions"] + row["geometry_actions"] == n
        process.append(row)
    csv_write(out / "process_summary.csv", process)
    modality = []
    modality_classes = []
    subsets = [
        (
            row["set"],
            [
                r
                for r in records
                if (
                    r["split"] == "train"
                    if row["set"] == "train"
                    else r["split"] == "val"
                    if row["set"] == "test_pooled"
                    else r["annotator_id"] == row["set"]
                )
            ],
        )
        for row in process
    ]
    for name, subset in subsets:
        values = np.array([float(r["preference_score"]) for r in subset])
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError("Missing or invalid modality score")
        modality.append({"set": name, **modality_statistics(values)})
        if name in ("train", "test_pooled"):
            for cls in sorted({r["class_id"] for r in subset}):
                values = np.array(
                    [
                        float(r["preference_score"])
                        for r in subset
                        if r["class_id"] == cls
                    ]
                )
                modality_classes.append(
                    {"set": name, "class_id": cls, **modality_statistics(values)}
                )
    csv_write(out / "modality_summary.csv", modality)
    csv_write(out / "modality_per_class.csv", modality_classes)

    print("Process summaries calculated for all four raters", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        agreement = list(
            pool.imap(
                pair_statistics,
                [(t, str(root), test_by_task[t]) for t in sorted(test_by_task)],
                chunksize=4,
            )
        )
    csv_write(out / "expert_agreement_per_task.csv", agreement)
    agreement = {r["task_id"]: r for r in agreement}
    predictions = {}
    for r in csv_read(args.benchmark_results):
        if r["target"] != "Ambiguity-aware consensus" or r["method"] not in METHODS:
            continue
        key = (r["task_id"], r["method"])
        if key in predictions or r["task_id"] not in groups:
            raise ValueError("Duplicate/unknown benchmark task")
        predictions[key] = {m: float(r[m]) for m in METRICS}
    if set(predictions) != {(t, m) for t in groups for m in METHODS}:
        raise ValueError("Missing benchmark predictions")
    summary, agree_full, models, per_class = [], [], [], []
    classes = sorted({r["class_id"] for r in test})
    for gi, group in enumerate(GROUPS):
        tasks = sorted(t for t in groups if groups[t] == group)
        row = {"workflow": group, "tasks": len(tasks)}
        add_interval(
            row,
            "expert_pair_dice",
            [agreement[t]["raw_dice"] for t in tasks],
            args.samples,
            args.seed + gi * 100,
        )
        for mi, method in enumerate(PRIMARY):
            add_interval(
                row,
                method,
                [predictions[t, method]["dice"] for t in tasks],
                args.samples,
                args.seed + gi * 100 + mi + 1,
            )
        summary.append(row)
        row = {"workflow": group, "tasks": len(tasks)}
        for mi, metric in enumerate(AGREEMENT):
            add_interval(
                row,
                metric,
                [agreement[t][metric] for t in tasks],
                args.samples,
                args.seed + 1000 + gi * 100 + mi,
            )
        agree_full.append(row)
        for mi, method in enumerate(METHODS):
            row = {"workflow": group, "tasks": len(tasks), "method": method}
            for ki, metric in enumerate(METRICS):
                add_interval(
                    row,
                    metric,
                    [predictions[t, method][metric] for t in tasks],
                    args.samples,
                    args.seed + 2000 + gi * 1000 + mi * 100 + ki,
                )
            models.append(row)
        for cls in classes:
            selected = [t for t in tasks if agreement[t]["class_id"] == cls]
            row = {
                "workflow": group,
                "class_id": cls,
                "tasks": len(selected),
                "expert_pair_dice": float(
                    np.mean([agreement[t]["raw_dice"] for t in selected])
                )
                if selected
                else None,
            }
            for method in METHODS:
                row[method] = (
                    float(np.mean([predictions[t, method]["dice"] for t in selected]))
                    if selected
                    else None
                )
            per_class.append(row)
    csv_write(out / "workflow_summary.csv", summary)
    csv_write(out / "workflow_agreement_full.csv", agree_full)
    csv_write(out / "workflow_model_metrics.csv", models)
    csv_write(out / "workflow_per_class.csv", per_class)
    print("All-expert task strata and 10,000-draw intervals calculated", flush=True)
    reweighted = []
    by_class = {(r["workflow"], r["class_id"]): r for r in per_class}
    common = [c for c in classes if all(by_class[g, c]["tasks"] for g in GROUPS)]
    for metric in ["expert_pair_dice"] + list(METHODS):
        first = float(np.mean([by_class[GROUPS[0], c][metric] for c in common]))
        second = float(np.mean([by_class[GROUPS[1], c][metric] for c in common]))
        reweighted.append(
            {
                "metric": metric,
                "common_classes": len(common),
                "no_geometry_action": first,
                "any_geometry_action": second,
                "difference_no_minus_any": first - second,
                "classes_no_lower_than_any": sum(
                    by_class[GROUPS[0], c][metric] >= by_class[GROUPS[1], c][metric]
                    for c in common
                ),
            }
        )
    csv_write(out / "workflow_class_balanced.csv", reweighted)
    sensitivity = []
    for ri, rater in enumerate(sorted({r["annotator_id"] for r in test})):
        subsets = {
            g: sorted(
                r["task_id"]
                for r in test
                if r["annotator_id"] == rater
                and GROUPS[int(flags[r["annotation_id"]])] == g
            )
            for g in GROUPS
        }
        assert sum(map(len, subsets.values())) == 1327
        for gi, group in enumerate(GROUPS):
            tasks = subsets[group]
            row = {"rater": rater, "workflow": group, "tasks": len(tasks)}
            add_interval(
                row,
                "expert_pair_dice",
                [agreement[t]["raw_dice"] for t in tasks],
                args.samples,
                args.seed + 5000 + ri * 1000 + gi * 100,
            )
            for mi, method in enumerate(PRIMARY):
                add_interval(
                    row,
                    method,
                    [predictions[t, method]["dice"] for t in tasks],
                    args.samples,
                    args.seed + 5000 + ri * 1000 + gi * 100 + mi + 1,
                )
            sensitivity.append(row)
    csv_write(out / "per_rater_sensitivity.csv", sensitivity)
    reconstruction = []
    for method in METHODS:
        for metric in METRICS:
            whole = float(
                np.mean([predictions[t, method][metric] for t in sorted(groups)])
            )
            recombined = (
                sum(r["tasks"] * r[metric] for r in models if r["method"] == method)
                / 1327
            )
            delta = abs(whole - recombined)
            if delta > 1e-12:
                raise ValueError(
                    "Task groups do not reconstruct the frozen overall benchmark"
                )
            reconstruction.append(delta)
    maximum = compare_expected(out, args.expected_dir) if args.expected_dir else None
    status = {
        "status": "PASS",
        "annotations": 8074,
        "test_annotations": 3981,
        "test_tasks": 1327,
        "task_groups": {r["workflow"]: r["tasks"] for r in summary},
        "raters": 4,
        "bootstrap_samples": args.samples,
        "seed": args.seed,
        "resampling_unit": "image-label task",
        "image_clustering": "Not performed; multiple labels on an image may be correlated",
        "grouping": "No geometry action in all three retained records versus at least one action in any record",
        "geometry_event_types": sorted(GEOMETRY_EVENTS),
        "frozen_benchmark_reconstruction_max_difference": max(reconstruction),
        "expected_max_difference": maximum,
        "elapsed_seconds": time.monotonic() - started,
        "metadata_manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "canonical_per_task_sha256": hashlib.sha256(
            args.benchmark_results.read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
