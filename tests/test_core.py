import io, itertools
import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco
from ku_pcpe.data import align, consensus, keyed, image_key, read_record_mask
from ku_pcpe.metrics import dice_score, iou_score, overcover, boundary_f1
from ku_pcpe.baselines import score_prediction, resize_bool
from ku_pcpe.proposals import (
    parse_ids,
    greedy_oracle_union,
    exact_oracles,
    ranked_union,
    union_masks,
    normalize_masks,
)
from ku_pcpe.validation import inspect_text, install_bundle


def test_split_label_and_cardinality():
    assert image_key("train", "0001.jpg") != image_key("test", "0001.jpg")
    with pytest.raises(ValueError):
        image_key("test", "../x.jpg")
    with pytest.raises(ValueError):
        keyed([dict(task_id="x"), dict(task_id="x")])
    with pytest.raises(ValueError):
        consensus([np.zeros((3, 3), bool)])
    a = np.eye(5, dtype=bool)
    b = ~a
    assert not np.array_equal(consensus([a, a, a]), consensus([b, b, b]))


def test_alignment_and_ties():
    a = np.eye(6, dtype=bool)
    b = ~a
    assert np.array_equal(consensus([a, b, a]), a)
    assert np.array_equal(consensus([a, b, a], False), a)
    x = np.array([[1, 1, 0, 0]], bool)
    ref = np.array([[1, 0, 1, 0]], bool)
    assert dice_score(x, ref) == dice_score(~x, ref)
    assert np.array_equal(align([ref, x])[1], x)


@pytest.mark.parametrize("shape", [(1, 1), (23, 71), (701, 101), (101, 701)])
def test_metrics_empty_resize_boundary(shape):
    a = np.zeros(shape, bool)
    b = np.ones(shape, bool)
    assert dice_score(a, a) == iou_score(a, a) == boundary_f1(a, a) == 1
    assert dice_score(a, b) == 0 and overcover(a, b) == 0
    assert overcover(b, a) == 1 and boundary_f1(a, b) == 0
    assert all(np.isfinite(x) for x in score_prediction(a, b).values())
    assert resize_bool(a, 7, 11).shape == (7, 11)


def test_hit_equality_and_overcoverage():
    a = np.array([[1, 1, 1]], bool)
    b = np.array([[1, 0, 0]], bool)
    assert dice_score(a, b) == 0.5 and int(dice_score(a, b) >= 0.5) == 1
    assert overcover(a, b) == pytest.approx(2 / 3)


def test_two_pixel_boundary_tolerance():
    a = np.zeros((35, 65), bool)
    a[10:15, 10:20] = 1
    b = np.roll(a, 2, axis=1)
    c = np.roll(a, 7, axis=1)
    assert boundary_f1(a, b, 2) > boundary_f1(a, c, 2)


@pytest.mark.parametrize(
    "text,n,wanted",
    [
        ("[1,2,3,4]", 4, [1, 2, 3]),
        ('text [2,2,"1",99]', 3, [2, 1]),
        ("malformed ID=2", 3, [2]),
        ("[]", 3, []),
        ("blah", 3, []),
        ("[0,9]", 3, []),
        ("[1]", 0, []),
        ("[2] then [1]", 3, [2]),
    ],
)
def test_frozen_parser(text, n, wanted):
    assert parse_ids(text, n)[0] == wanted


@pytest.mark.parametrize("n", range(13))
def test_greedy_exact_and_fewer_candidates(n):
    rng = np.random.default_rng(920 + n)
    target = rng.random((9, 13)) < 0.4
    masks = [rng.random(target.shape) < 0.35 for _ in range(n)]
    exact = exact_oracles(masks, target)
    greedy = greedy_oracle_union(masks, target, 3)
    current = np.zeros_like(target)
    remaining = list(range(n))
    for _ in range(min(3, n)):
        i = max(remaining, key=lambda x: dice_score(current | masks[x], target))
        current |= masks[i]
        remaining.remove(i)
    assert np.array_equal(greedy, current)
    assert np.array_equal(greedy, greedy_oracle_union(masks, target, 3))
    assert dice_score(greedy, target) <= dice_score(exact["exact_3"][0], target) + 1e-14
    values = []
    for k in range(min(3, n) + 1):
        for ids in itertools.combinations(range(n), k):
            values.append(
                dice_score(union_masks([masks[i] for i in ids], target.shape), target)
            )
    assert dice_score(exact["exact_up_to_3"][0], target) == max(values)
    selected = ranked_union(masks[:12], list(range(len(masks[:12]))), 3, target.shape)
    assert selected.shape == target.shape


@pytest.mark.parametrize("shape", [(19, 27), (55, 15)])
def test_holes_components_roundtrip(shape):
    a = np.zeros(shape, bool)
    a[1:8, 1:8] = 1
    a[3:5, 3:5] = 0
    a[-3:, -3:] = 1
    rle = coco.encode(np.asfortranarray(a.astype("uint8")))
    b = coco.decode(rle).astype(bool)
    assert np.array_equal(a, b)
    buf = io.BytesIO()
    Image.fromarray(a.astype("uint8") * 255).save(buf, format="PNG")
    buf.seek(0)
    assert np.array_equal(np.asarray(Image.open(buf)) > 0, a)
    assert int(coco.area(rle)) == int(a.sum())
    box = coco.toBbox(rle)
    assert box[2] > 0 and box[3] > 0


@pytest.mark.parametrize(
    "bad",
    [
        "/home/user/a",
        "/share/x",
        '"annotator_id": "private"',
        '"actions": []',
        "api_key=abc",
        "annotations.sqlite3",
    ],
)
def test_privacy_rejects(bad):
    with pytest.raises(ValueError):
        inspect_text(bad)


def test_bad_masks_and_dimension(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(np.full((3, 4), 128, "uint8")).save(p)
    rec = dict(mask_path="x.png", height=3, width=4, area_px=12)
    with pytest.raises(ValueError):
        read_record_mask(rec, tmp_path)
    rec["width"] = 5
    with pytest.raises(ValueError):
        read_record_mask(rec, tmp_path)


def test_macro_and_absent_class():
    from ku_pcpe.evaluate import summarize

    rows = [
        dict(
            method="x",
            target="t",
            class_id=c,
            **{
                k: v
                for k in [
                    "dice",
                    "iou",
                    "boundary_f1",
                    "overcover",
                    "hit_at_0_5",
                    "invalid",
                ]
            },
        )
        for c, v in [("center", 1.0), ("center", 0.0), ("pattern", 0.25)]
    ]
    result = summarize(rows, "matched", True)
    assert (
        len(result) == 2
        and result[0]["tasks"] == 2
        and result[0]["dice"] == 0.5
        and result[1]["dice"] == 0.25
    )


def test_missing_mask_and_path_escape(tmp_path):
    rec = dict(mask_path="missing.png", width=5, height=5, area_px=1)
    with pytest.raises(FileNotFoundError):
        read_record_mask(rec, tmp_path)
    rec["mask_path"] = "../outside.png"
    with pytest.raises(ValueError):
        read_record_mask(rec, tmp_path)


def test_exact_rejects_more_than_visible_bank():
    target = np.zeros((2, 3), bool)
    with pytest.raises(ValueError):
        exact_oracles([target] * 13, target)


def test_boundary_tolerance_after_downsampling():
    target = np.zeros((64, 512), bool)
    target[:, 200:220] = True
    within = np.roll(target, 4, axis=1)
    beyond = np.roll(target, 6, axis=1)
    assert score_prediction(within, target)["boundary_f1"] == 1
    assert score_prediction(beyond, target)["boundary_f1"] < 0.6


def test_bundle_hash_and_archive_escape(tmp_path):
    import hashlib, tarfile
    from ku_pcpe.validation import install_bundle

    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        install_bundle(archive, "0" * 64, tmp_path / "out")
    with pytest.raises(ValueError):
        install_bundle(
            archive, hashlib.sha256(archive.read_bytes()).hexdigest(), tmp_path / "out"
        )


def test_atomic_resume_cache(tmp_path):
    from ku_pcpe.cache import TaskCache

    cache = TaskCache(tmp_path / "cache", {"inputs": "fixed"})
    assert cache.load("task") is None
    value = [[{"dice": 0.5}], [], {"task_id": "task"}]
    cache.save("task", value)
    restored = TaskCache(tmp_path / "cache", {"inputs": "fixed"}, resume=True)
    assert restored.load("task") == value
    with pytest.raises(ValueError):
        TaskCache(tmp_path / "cache", {"inputs": "changed"}, resume=True)
    path = restored.path("task")
    import json

    row = json.loads(path.read_text())
    row["result"][0][0]["dice"] = 0.9
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        restored.load("task")
