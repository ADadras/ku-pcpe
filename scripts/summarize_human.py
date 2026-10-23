"""Finish the human-reference table from independently computed per-task LOO metrics."""

import argparse, csv, json
from pathlib import Path
import numpy as np
from ku_pcpe.evaluate import METRICS, bootstrap, write

p = argparse.ArgumentParser()
p.add_argument("--results", type=Path, required=True)
p.add_argument("--expected", type=Path, required=True)
a = p.parse_args()
rows = list(csv.DictReader((a.results / "human_per_task.csv").open()))
assert len(rows) == 1327 and len({r["task_id"] for r in rows}) == 1327
summary = dict(tasks=len(rows))
for i, m in enumerate(METRICS):
    values = [float(r[m]) for r in rows]
    summary[m] = float(np.mean(values))
    summary[m + "_ci_low"], summary[m + "_ci_high"] = bootstrap(values, 920 + i)
expected = next(csv.DictReader(a.expected.open()))
differences = {k: float(v) - float(expected[k]) for k, v in summary.items()}
assert all(abs(v) <= 1e-12 for v in differences.values()), differences
write(a.results / "human_ci.csv", [summary])
(a.results / "human_ci_status.json").write_text(
    json.dumps(dict(status="PASS", differences=differences), indent=2)
)
print("Human mean and all five confidence intervals PASS at 1e-12")
