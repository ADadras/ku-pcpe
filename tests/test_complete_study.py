"""Guard against tripling tasks or accepting incomplete expert coverage."""

import importlib.util
from pathlib import Path

import pytest
import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location(
    "complete_study", Path(__file__).parents[1] / "scripts/summarize_complete_study.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def records():
    return [
        dict(task_id="t", annotation_id=str(i), annotator_id=r)
        for i, r in enumerate(("rater_01", "rater_03", "rater_04"))
    ]


def test_three_records_form_one_task():
    rows = records()
    flags = {r["annotation_id"]: False for r in rows}
    groups, joined = study.task_groups(rows, flags)
    assert groups == {"t": "no_geometry_action"}
    assert len(joined["t"]) == 3
    flags["1"] = True
    assert study.task_groups(rows, flags)[0] == {"t": "any_geometry_action"}


@pytest.mark.parametrize("bad", ["missing", "duplicate", "extra"])
def test_reject_incomplete_or_duplicate_experts(bad):
    rows = records()
    rows += [
        {**r, "task_id": "other", "annotation_id": r["annotation_id"] + "x"}
        for r in rows
    ]
    if bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows[-1]["annotator_id"] = rows[-2]["annotator_id"]
    else:
        rows.append(dict(rows[-1]))
    with pytest.raises(ValueError, match="Missing or duplicated"):
        study.task_groups(rows, {r["annotation_id"]: False for r in rows})


def test_pair_metrics_on_complementary_masks(tmp_path):
    a = np.zeros((10, 20), dtype=bool)
    a[:, :10] = True
    rows = []
    for i, (rater, mask) in enumerate(
        zip(("rater_04", "rater_01", "rater_03"), (a, ~a, a))
    ):
        name = f"{i}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(tmp_path / name)
        rows.append(dict(mask_path=name, annotator_id=rater, class_id="horizontal"))
    scores = study.pair_statistics(("t", str(tmp_path), rows))
    assert scores["raw_dice"] == pytest.approx(1 / 3)
    assert scores["orientation_invariant_dice"] == 1
    assert scores["boundary_f1"] == 1
    assert scores["raw_pair_kappa"] == pytest.approx(-1 / 3)


def test_expected_directory_must_contain_results(tmp_path):
    with pytest.raises(ValueError, match="no CSVs"):
        study.compare_expected(tmp_path, tmp_path)
