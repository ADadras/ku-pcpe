# Dataset archive: annotations and study metadata

The main KU-PCPE dataset archive, `kupcp_completed_public_20261023_v1.tar.gz`, contains the complete annotation sets from three test experts (1,327 records each) and one training annotator (4,093 records). It includes a segment-versus-line score, recorded sensitivity/settings, component masks, tool counts and relative event timelines for every annotation. Its participant index lists release-specific pseudonyms and completed-set counts; it does not publish demographic or experience profiles. Partial annotation sets are intentionally excluded.

This is the primary dataset download. The two archives listed separately in the main README support reproduction: one preserves the annotations in the evaluator’s required layout, and the other supplies saved model outputs and proposals. Their final annotation masks are identical to those in this dataset archive. The evaluator uses its preserved layout; the complete-study analysis uses the main dataset archive directly. The archive is included in `data/bundles/`. Its exact bytes and SHA-256 are listed in `configs/complete_metadata_manifest.json` and `configs/release_assets.json`.

Files to inspect:

- `annotations.csv` / `annotations.jsonl.gz`: all 8,074 annotations and their safe metadata; JSONL also includes final and component region RLEs.
- `participants.csv`: the four completed annotator/split sets.
- `tool_usage_by_task.csv`, `tool_usage_by_annotator.csv`, `tool_usage_by_class.csv`: observed tool use at the stated aggregation levels.
- `tool_timeline.csv` / `tool_event_counts.csv`: relative event bins and controlled event totals.
- `data_dictionary.json`, `manifest.json`, `validation.json`, `checksums.sha256`: schema, scope, exporter checks and file digests.
- `masks/` and `coco_union_masks.json`: PNG and COCO encodings of the final masks. Application RLEs are row-major; COCO RLEs are column-major.

An independent local check on 23 October 2026 verified every archive checksum, all 8,074 PNG/union/COCO masks and 35,604 component-region masks. All final masks are pixel-identical to the frozen benchmark release; all previously released expert ratings also agree. All 66,648 timeline rows reconcile to 176,128 recorded actions, including every annotation, and per-task/class/annotator totals were recomputed. No mask, prediction, target, scoring code or benchmark score changed. This is verification of the supplied archive, not an independent audit of the live source database.

The updated paper uses all four completed annotators for process and modality summaries. Its main workflow analysis assigns a task to no geometry action only if no expert recorded an event from the original geometry-action set, and to any geometry action otherwise. This yields 615/712 tasks, each counted once. Per-rater sensitivity also partitions all 1,327 tasks separately for each test expert. Model predictions and targets remain fixed; there is no training or inference in this recalculation. The new export does not reconstruct discarded drafts or unretained revision histories. It supplies per-annotation UI sensitivity, not the historical automatic proposal-generator configuration.

Timing uses five-second bins within observed clock segments and caps values at 30 minutes. It is not exact active working time, model latency or inference-call counts. The exporter retains at most 4,000 source events per task; this archive has no task at that cap. It reports 305 tasks with clock decreases and 134 capped task durations. Original participant identifiers, calendar times, free text, raw coordinates/payloads and original photos are excluded. The small pseudonymous cohort, including its unique training rater, is not guaranteed irreversibly anonymous.

## Reproduce the updated tables

The one-command reproduction in the main README installs this archive and regenerates these tables automatically. To run this analysis separately after Tier 1 has produced `canonical_per_task.csv`, install the included archive with `scripts/install_release.py` and run:

```bash
cd /path/to/kupcp_completed_public_20261023_v1
sha256sum --check --quiet checksums.sha256
cd /path/to/ku-pcpe-release
python -I scripts/summarize_complete_study.py \
  --metadata-root /path/to/kupcp_completed_public_20261023_v1 \
  --benchmark-results /path/to/tier1/results/canonical_per_task.csv \
  --expected-dir expected/workflow_all_raters \
  --output /path/to/new-workflow-results
```

Dependencies are already pinned in `requirements.lock`; no research repository or private database is imported. Two CPU workers decode final masks. The script validates rater/task coverage and action totals, recomputes human agreement from PNGs, joins fixed benchmark scores, and estimates 95% intervals with 10,000 task-bootstrap draws (base seed 920, deterministic per-table offsets). It also checks that weighted group scores reconstruct all 40 overall method/metric means within 1e-12. Expected CSVs are comparison references, never substituted as results. Output `status.json` records PASS only after all requested checks pass, input/script digests, counts, and elapsed seconds. Repeated labels on an image can be correlated; bootstrap resamples tasks, not independent images.

Outputs include pooled/per-rater process summaries; all-expert workflow means, intervals, per-class and class-balanced results; six per-expert sensitivity rows; and modality distributions by split, class, and rater. The 3,981 test annotations must not be described as 3,981 unique tasks. Training has 4,093 tasks and one rater; test has 1,327 tasks and three raters. A single symmetric task belongs to the no-action group, so the class-wise comparison there is descriptive and unstable.

Historical descriptive human B-F1 uses Pillow nearest-neighbor resizing (maximum side 256), both sides of transitions without image-frame edges, and a 5x5 square tolerance (Chebyshev distance <= 2). Complement-aware pair Dice takes max(Dice(A,B), Dice(A,~B)) in the published pair order, then averages three pairs. Public pseudonyms map to that order as rater_04, rater_01, rater_03, established by exact final-mask identity. This is separate from joint majority alignment and the benchmark model B-F1 convention. Recomputing the earlier one-expert strata reproduces all four published point estimates exactly; those earlier strata remain available as sensitivity rows, rather than defining the updated main analysis.
