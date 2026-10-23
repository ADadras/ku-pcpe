"""Evaluate independently exported annotations and saved predictions on CPU."""

from pathlib import Path
from collections import defaultdict
import csv, json, multiprocessing as mp, time
import cv2
import numpy as np
from PIL import Image
from .data import jsonl, load_release, read_record_mask, consensus, align
from .metrics import dice_score, boundary_f1, iou_score, overcover
from .baselines import (
    learn_class_templates,
    random_area_mask,
    predict_learned_template,
    score_prediction,
)
from .proposals import (
    decode_rle,
    normalize_masks,
    union_masks,
    ranked_union,
    template_scores,
    greedy_oracle_union,
    exact_oracles,
    parse_ids,
)

METRICS = ["dice", "iou", "boundary_f1", "overcover", "hit_at_0_5"]
G = {}
CANONICAL = [
    "Random matched area",
    "Train class template",
    "SAM2.1 Oracle@3",
    "SAM2.1 Template@3",
    "U-Net no label",
    "U-Net + label",
    "SegFormer-B0",
    "Qwen3-VL-32B proposals",
]


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def norm(mask, shape):
    if mask.shape == shape:
        return mask.astype(bool)
    return (
        cv2.resize(
            mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
        > 0
    )


def worker(task):
    bundles, tables = G["bundles"], G["tables"]
    rec = tables[0][task]
    masks = [read_record_mask(t[task], b) for b, t in zip(bundles, tables)]
    targets = {
        **{f"Expert {chr(65 + i)}": m for i, m in enumerate(masks)},
        "Strict consensus": consensus(masks, False),
        "Ambiguity-aware consensus": consensus(masks, True),
    }
    shape = masks[0].shape
    h, w = shape
    proposals = normalize_masks(
        [
            decode_rle(m["segmentation"])
            for m in G["proposals"][rec["image_id"]]["masks"]
        ],
        shape,
    )
    visible = proposals[:12]
    template = G["templates"][rec["class_id"]]
    scores = template_scores(proposals, shape, rec["class_id"], G["templates"])
    fixed = {
        "Random matched area": random_area_mask(h, w, template["area"], task),
        "Train class template": predict_learned_template(template, h, w),
        "SAM2.1 Template@3": ranked_union(proposals, scores, 3, shape),
    }
    for family, method in [
        ("unet_nolabel_seed13", "U-Net no label"),
        ("unet_label_seed13", "U-Net + label"),
        ("segformer_b0_label_seed13", "SegFormer-B0"),
    ]:
        path = G["replay"] / "supervised" / family / (task.replace(":", "__") + ".png")
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved prediction: {family}, {task}")
        with Image.open(path) as image:
            fixed[method] = norm(np.asarray(image.convert("L")) > 0, shape)
    vlms = {}
    for model, table in G["vlms"].items():
        row = table[task]
        assert row["image_id"] == rec["image_id"] and row["class_id"] == rec["class_id"]
        assert row["num_candidates"] == len(visible)
        ids, _ = parse_ids(row["raw_answer"], len(visible))
        assert ids == row["selected_ids"]
        vlms[model] = union_masks([visible[i - 1] for i in ids], shape)
    fixed["Qwen3-VL-32B proposals"] = vlms["Qwen/Qwen3-VL-32B-Instruct"]
    canonical = []
    matched = []
    for target_name, target in targets.items():
        predictions = {
            **fixed,
            "SAM2.1 Oracle@3": greedy_oracle_union(proposals, target, 3),
        }
        for method in CANONICAL:
            metrics = score_prediction(predictions[method], target)
            canonical.append(
                dict(
                    method=method,
                    task_id=task,
                    image_id=rec["image_id"],
                    class_id=rec["class_id"],
                    target=target_name,
                    **metrics,
                    hit_at_0_5=int(metrics["dice"] >= 0.5),
                )
            )
        if target_name not in ["Strict consensus", "Ambiguity-aware consensus"]:
            continue
        exact = exact_oracles(visible, target)
        predictions = {
            **vlms,
            "Template@3 (12)": ranked_union(visible, scores[:12], 3, shape),
            "Greedy Oracle@3 (12)": greedy_oracle_union(visible, target, 3),
            "Exact Oracle@3 (12, exactly 3)": exact["exact_3"][0],
            "Exact Oracle@3 (12, up to 3)": exact["exact_up_to_3"][0],
        }
        for method, pred in predictions.items():
            metrics = score_prediction(pred, target)
            matched.append(
                dict(
                    method=method,
                    task_id=task,
                    image_id=rec["image_id"],
                    class_id=rec["class_id"],
                    target=target_name,
                    **metrics,
                    hit_at_0_5=int(metrics["dice"] >= 0.5),
                    invalid=int(
                        method in vlms and not G["vlms"][method][task]["selected_ids"]
                    ),
                )
            )
    human = []
    for held, m in enumerate(masks):
        peer = align([masks[i] for i in range(3) if i != held])
        target = np.stack(peer).mean(0) >= 0.5
        held_eval = ~m if dice_score(~m, target) > dice_score(m, target) else m
        scale = min(1.0, 256 / max(shape))
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        a = (
            cv2.resize(target.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            > 0
        )
        b = (
            cv2.resize(
                held_eval.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
            )
            > 0
        )
        d = dice_score(target, held_eval)
        human.append(
            dict(
                dice=d,
                iou=iou_score(target, held_eval),
                boundary_f1=boundary_f1(a, b, 2),
                overcover=overcover(target, held_eval),
                hit_at_0_5=int(d >= 0.5),
            )
        )
    return (
        canonical,
        matched,
        dict(
            task_id=task, **{m: float(np.mean([r[m] for r in human])) for m in METRICS}
        ),
    )


def bootstrap(values, seed, samples=10000):
    data = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.concatenate(
        [
            data[rng.integers(0, len(data), (min(200, samples - i), len(data)))].mean(1)
            for i in range(0, samples, 200)
        ]
    )
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def summarize(rows, kind, by_class=False):
    keys = (["method", "target"] if kind == "canonical" else ["target", "method"]) + (
        ["class_id"] if by_class else []
    )
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for gi, (key, group) in enumerate(sorted(groups.items())):
        r = dict(zip(keys, key))
        r["tasks"] = len(group)
        for mi, m in enumerate(METRICS + (["invalid"] if kind == "matched" else [])):
            values = [float(x[m]) for x in group]
            r[m] = float(np.mean(values))
            if not by_class:
                r[m + "_ci_low"], r[m + "_ci_high"] = bootstrap(
                    values, 920 + gi * 100 + mi if kind == "canonical" else 920 + gi
                )
        out.append(r)
    return out


def compare(rows, path, keys, fields):
    expected = {tuple(r[k] for k in keys): r for r in csv.DictReader(Path(path).open())}
    actual = {tuple(r[k] for k in keys): r for r in rows}
    if set(expected) != set(actual):
        raise ValueError(f"Summary key mismatch for {Path(path).name}")
    errors = []
    maxdiff = 0.0
    for key, r in actual.items():
        for m in fields:
            if m not in r or m not in expected[key]:
                continue
            d = abs(float(r[m]) - float(expected[key][m]))
            maxdiff = max(d, maxdiff)
            if d > 1e-12:
                errors.append(dict(key=key, metric=m, difference=d))
    return dict(
        status="FAIL" if errors else "PASS",
        rows=len(rows),
        max_difference=maxdiff,
        errors=errors[:20],
    )


def paired(canonical, matched):
    lookup = {(r["target"], r["method"], r["task_id"]): r for r in matched}
    for r in canonical:
        method = {
            "SAM2.1 Oracle@3": "Greedy Oracle@3 (full)",
            "SAM2.1 Template@3": "Template@3 (full)",
        }.get(r["method"])
        if method:
            lookup[(r["target"], method, r["task_id"])] = r
    pairs = [
        ("Qwen/Qwen3-VL-32B-Instruct", "Template@3 (12)"),
        ("Greedy Oracle@3 (full)", "Greedy Oracle@3 (12)"),
        ("Template@3 (full)", "Template@3 (12)"),
        ("Exact Oracle@3 (12, exactly 3)", "Greedy Oracle@3 (12)"),
        ("Exact Oracle@3 (12, up to 3)", "Exact Oracle@3 (12, exactly 3)"),
    ]
    out = []
    for target in ["Ambiguity-aware consensus", "Strict consensus"]:
        for a, b in pairs:
            for metric in METRICS:
                values = [
                    float(lookup[(target, a, t)][metric])
                    - float(lookup[(target, b, t)][metric])
                    for t in G["tasks"]
                ]
                lo, hi = bootstrap(values, 920)
                out.append(
                    dict(
                        target=target,
                        method_a=a,
                        method_b=b,
                        metric=metric,
                        tasks=len(values),
                        mean_a_minus_b=float(np.mean(values)),
                        ci_low=lo,
                        ci_high=hi,
                    )
                )
    return out


def evaluate(annotations, replay, expected, output, workers=2, resume=False):
    start = time.time()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=resume)
    replay = Path(replay)
    expected = Path(expected)
    train, records, bundles, tables = load_release(annotations)
    byid = {r["task_id"]: r for r in records}
    order = json.loads((replay / "train_order.json").read_text())
    assert set(order) == set(byid)
    templates = learn_class_templates([byid[t] for t in order], train)
    proposals = {r["image_id"]: r for r in jsonl(replay / "proposals.jsonl")}
    assert len(proposals) == 1073
    vlms = {}
    for p in sorted((replay / "vlm").glob("*.jsonl")):
        rows = jsonl(p)
        model = rows[0]["model"]
        table = {r["task_id"]: r for r in rows}
        assert len(table) == len(rows) == 1327 and set(table) == set(tables[0])
        vlms[model] = table
    assert len(vlms) == 7
    G.update(
        bundles=bundles,
        tables=tables,
        replay=replay,
        templates=templates,
        proposals=proposals,
        vlms=vlms,
        tasks=sorted(tables[0]),
    )
    from .cache import TaskCache, digest

    signature = dict(
        annotations=digest(Path(annotations) / "checksums.csv"),
        replay=digest(replay / "checksums.json"),
        code={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
    )
    cache = TaskCache(out / "task_cache", signature, resume)
    completed = {
        task: cached for task in G["tasks"] if (cached := cache.load(task)) is not None
    }
    pending = [task for task in G["tasks"] if task not in completed]
    print(
        f"Reusing {len(completed)} completed task checkpoints; {len(pending)} remaining",
        flush=True,
    )
    with mp.get_context("fork").Pool(workers) as pool:
        for task, result in zip(pending, pool.imap(worker, pending, chunksize=2)):
            cache.save(task, result)
            completed[task] = result
            if len(completed) % 100 == 0:
                print(f"Evaluated {len(completed)}/1327", flush=True)
    canonical = []
    matched = []
    human = []
    # Strict original task ordering preserves every bootstrap draw and rounding.
    for task in G["tasks"]:
        a, b, c = completed[task]
        assert len(a) == 40 and len(b) == 22 and c["task_id"] == task
        canonical.extend(a)
        matched.extend(b)
        human.append(c)
    write(out / "canonical_per_task.csv", canonical)
    write(out / "matched12_per_task.csv", matched)
    write(out / "human_per_task.csv", human)
    print("Scoring complete; computing fixed-seed confidence intervals", flush=True)
    cs = summarize(canonical, "canonical")
    ms = summarize(matched, "matched")
    pc = summarize(matched, "matched", True)
    pairs = paired(canonical, matched)
    for name, rows in [
        ("canonical_summary.csv", cs),
        ("matched12_summary.csv", ms),
        ("matched12_per_class.csv", pc),
        ("paired_intervals.csv", pairs),
    ]:
        write(out / name, rows)
    checks = {}
    fields = METRICS + [m + s for m in METRICS for s in ["_ci_low", "_ci_high"]]
    checks["canonical"] = compare(
        cs, expected / "canonical.csv", ["method", "target"], fields
    )
    checks["matched12"] = compare(
        ms,
        expected / "matched12.csv",
        ["target", "method"],
        fields + ["invalid", "invalid_ci_low", "invalid_ci_high"],
    )
    checks["matched12_per_class"] = compare(
        pc,
        expected / "matched12_per_class.csv",
        ["target", "method", "class_id"],
        METRICS + ["invalid"],
    )
    checks["paired"] = compare(
        pairs,
        expected / "paired_intervals.csv",
        ["target", "method_a", "method_b", "metric"],
        ["mean_a_minus_b", "ci_low", "ci_high"],
    )
    hs = {m: float(np.mean([r[m] for r in human])) for m in METRICS}
    write(out / "human_summary.csv", [dict(method="Human LOO", tasks=1327, **hs)])
    previous = json.loads((expected / "human.json").read_text())
    diff = {m: hs[m] - previous[m] for m in METRICS}
    checks["human"] = dict(
        status="PASS" if all(abs(d) < 1e-12 for d in diff.values()) else "FAIL",
        differences=diff,
    )
    result = dict(
        status="PASS"
        if all(c["status"] == "PASS" for c in checks.values())
        else "FAIL",
        tasks=1327,
        canonical_rows=len(canonical),
        matched_rows=len(matched),
        elapsed_seconds=time.time() - start,
        checks=checks,
    )
    (out / "status.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "PASS":
        raise RuntimeError("Saved-prediction reproduction differs; see status.json")
    return result
