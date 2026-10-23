# KU-PCP Elements (KU-PCPE)

[![CPU checks](https://github.com/ADadras/ku-pcpe/actions/workflows/tests.yml/badge.svg)](https://github.com/ADadras/ku-pcpe/actions/workflows/tests.yml)

**Grounding Image Composition with Label-Conditioned Visual Evidence** · WACV 2027, Evaluations & Datasets<br>
Armin Dadras, Tingyu Lin, Robert Sablatnig, Franziska Proksa, Markus Seidl<br>
Computer Vision Lab, TU Wien · UAS St. Pölten

KU-PCPE asks which visible regions support a composition label that applies to a photograph. It extends KU-PCP with masks selected by people for nine composition classes. This repository includes the annotations, sanitized study metadata, saved predictions and proposals, and an independent evaluator that reproduces the benchmark tables.

| | Training | Test |
|---|---:|---:|
| Images | 3,169 | 1,073 |
| Tasks (one image and one label) | 4,093 | 1,327 |
| Annotators per task | 1 | 3 |
| Individual masks | 4,093 | 3,981 |

**The original photograph dataset and model weights are not included.** The project page contains only 18 resized paper illustrations and their thumbnails, with upstream rights retained. You can reproduce the reported metrics from the supplied masks, proposals and saved selections without photos or a GPU. This is distinct from running model inference or retraining models. [Protocol and limits](docs/PROTOCOLS.md) · [Dataset card](DATASET_CARD.md) · [Data license](data/LICENSE.md)

## Project page

The interactive [project page is included under `website/`](website/README.md), with the paper, supplement, examples from all nine classes and full/matched-12 benchmark charts. The page supports GitHub Pages and includes a manual workflow for validation and deployment. [Preview and publication instructions](website/README.md#preview-and-check)

## Download the dataset

**[Download KU-PCPE: annotations and metadata — 69.5 MiB](data/bundles/kupcp_completed_public_20261023_v1.tar.gz?raw=true)**

KU-PCPE is **one dataset**. This archive contains all 8,074 annotations from the four completed annotators, including component masks, ratings, tool counts and reduced relative timelines. Use this download to work with the annotations and study metadata. Original photographs must be obtained separately.

[Dataset guide](docs/COMPLETE_STUDY_METADATA.md) · [License](data/LICENSE.md) · [Checksums](data/bundles/SHA256SUMS)

## Additional files for reproducing the paper

These are supporting files for the same dataset. They are not additional datasets.

| Reproduction asset | Size | Why it is provided |
|---|---:|---|
| [Annotations for the evaluator](data/bundles/ku_pcpe_annotations_1.0.0-rc.2.tar.gz?raw=true) | 37.0 MiB | Copies of the same final masks in the layout used by the tested evaluator, plus the consensus masks and validator |
| [Saved predictions and proposals](data/bundles/ku_pcpe_replay_1.0.0-rc.2.tar.gz?raw=true) | 30.6 MiB | Saved prediction masks, the original proposals and their order, and responses/selections from all seven VLMs |

To reproduce the benchmark and study tables, use the dataset archive and both supporting archives with the command below. A normal clone includes all three files without Git LFS or a separate hosting account. The repeated masks preserve the evaluator's validated inputs; the complete dataset supplies the metadata for the updated study tables. [File manifest](configs/release_assets.json)

## Install and check

Use **Python 3.12 on Linux**. Dependencies are pinned. From a fresh clone:

```sh
git clone https://github.com/ADadras/ku-pcpe.git
cd ku-pcpe
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-build-isolation --no-deps .
.venv/bin/python scripts/verify_release.py
.venv/bin/python -m pytest -q
.venv/bin/python -I scripts/run_smoke_test.py
```

The tests and synthetic example take seconds and need no original images or GPU. The release check verifies all three archive digests and the frozen evaluator files. The fixture ends with `"status": "PASS"`.

## Reproduce the tables

After installation, run:

```sh
.venv/bin/python -I scripts/reproduce.py --workers 2 --output outputs/reproduction
```

This verifies and installs the included archives, validates the masks and replay assets, runs all 1,327 test tasks, recomputes uncertainty, and regenerates the complete-study tables. It ends with `outputs/reproduction/reproduction_status.json` containing `"status": "PASS"` only if every step succeeds. Expected tables are checked **after** calculation, with a tolerance of **1e-12**. They are never substituted for predictions or computed scores.

Use a new output directory for each full run. Each step has its own log. Eight workers were used for the recorded cluster check, with a 10 GiB memory allocation and no GPU; two workers reduce memory use. Allow about 2 GiB of disk space for the checkout, environment, unpacked data and results. Runtime depends on the CPU; measured results are in [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md).

| Paper results | Recomputed output under the chosen output directory |
|---|---|
| Main benchmark and rescoring against individual experts or consensus | `benchmark/canonical_summary.csv` |
| Comparison using the same 12 proposals | `benchmark/matched12_summary.csv`, `benchmark/matched12_per_class.csv` |
| Paired differences and confidence intervals | `benchmark/paired_intervals.csv` |
| Human reference and its uncertainty | `benchmark/human_summary.csv`, `benchmark/human_ci.csv` |
| Process and modality summaries for all four annotators | `complete_study/process_summary.csv`, `complete_study/modality_summary.csv`, `complete_study/modality_per_class.csv` |
| Updated workflow analysis (615/712 disjoint task groups) | `complete_study/workflow_summary.csv`, `complete_study/workflow_model_metrics.csv`, `complete_study/workflow_per_class.csv` |
| Final checks | `benchmark/status.json`, `benchmark/human_ci_status.json`, `complete_study/status.json`, `reproduction_status.json` |

To inspect data without evaluating models:

```sh
.venv/bin/python -I scripts/install_release.py --output assets/data
.venv/bin/python -I assets/data/release_candidate_v1.0.0-rc.2/evaluate_smoke_test.py
```

Detailed individual commands and the definitions behind each table are in [the protocol](docs/PROTOCOLS.md) and [the metadata guide](docs/COMPLETE_STUDY_METADATA.md). The evaluator checkpoints completed tasks; its lower-level CLI supports `--resume` with unchanged code and inputs. The full wrapper intentionally starts a new run.

## Scope and interpretation

The benchmark uses positive labels from KU-PCP. It does not test arbitrary labels that do not apply to an image. Training uses one annotator; three different experts annotate the test set. Images may have several labels, so tasks from the same image can be correlated.

A strict consensus mask uses a pixel majority. The second consensus first compares experts 2 and 3 with expert 1, reverses a selection only when its complement gives strictly better Dice, and then takes the majority. Ties retain the original selection. Model predictions are never complemented. The historical human-reference calculation follows a separate convention documented in the protocol.

Oracles use the target mask and are references for measurement, not deployable methods. The full-bank oracle is greedy, not a guaranteed optimum. The exact oracle checks every combination of up to three of the same 12 proposals shown to VLMs. Parser behavior, invalid selections, bootstrap seeds, resizing and metric conventions are frozen and tested.

The annotation interface offered proposed regions. This can introduce shared bias, and there has been no independent audit without proposals. Workflow comparisons are observational. Differences between training and test masks cannot isolate annotator effects because the images differ too. The complete metadata uses pseudonyms and reduced relative time bins, which do not guarantee irreversible anonymity in a small cohort.

The main command reproduces metrics from saved predictions. Auxiliary classifier, attribution/deletion, modality-prediction and original geometric-detector experiments are documented, but their inference is not reproduced by this command. A separate CPU check reproduced U-Net predictions on nine predeclared examples: [inference instructions and evidence](docs/CPU_INFERENCE.md). Full GPU inference and retraining have not been rerun in this independent package.

## Source images, licenses and citation

To run inference, obtain original KU-PCP photographs separately under the distributor’s terms. Keep train/test paths distinct: the same filename can refer to different images in the two splits. `scripts/reconstruct_images.py` checks split-aware filenames and dimensions. The original source is credited in [NOTICE.md](NOTICE.md).

- Project code: [MIT](LICENSE).
- KU-PCPE annotations, sanitized metadata and derived replay assets: [CC BY 4.0](data/LICENSE.md).
- Source photographs, upstream model code and checkpoints retain their original terms. The full photograph dataset and checkpoints are excluded; the website's limited paper illustrations are described in [its attribution notice](website/NOTICE.md).

Permission covers distribution of the annotations and sanitized metadata. See the [privacy boundary](SECURITY_AND_PRIVACY.md).

```bibtex
@inproceedings{dadras2027kupcpe,
  title = {Grounding Image Composition with Label-Conditioned Visual Evidence},
  author = {Dadras, Armin and Lin, Tingyu and Sablatnig, Robert and Proksa, Franziska and Seidl, Markus},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year = {2027}
}
```

Please also cite Jun-Tae Lee, Han-Ul Kim, Chul Lee and Chang-Su Kim, *Photographic composition classification and dominant geometric element detection for outdoor scenes*, JVCIR 55:91–105, 2018. Maintenance: Armin Dadras, adadras@cvl.tuwien.ac.at.
