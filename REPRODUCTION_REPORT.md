# Reproduction report — PASS

Verified on 23 October 2026 (Europe/Vienna) from a clean copy of this package, with a newly created Python 3.12.12 environment on Linux x86_64. The complete command in the README ran successfully. Original research source files and the earlier evaluator checkout were denied by Linux Landlock; TCP connections were denied after installation. No original photographs, private database, model weights or GPU were available to the calculation.

[Machine-readable evidence](verification/release_check_2026-10-23.json) records the source-file hashes, archive digests, individual checks and numerical differences. The files needed to repeat the calculation are included here.

## Commands executed

The test used a new virtual environment and no system-site packages or inherited research `PYTHONPATH`:

```sh
python3.12 -m venv /path/to/fresh-environment
/path/to/fresh-environment/bin/pip install -r requirements.lock
/path/to/fresh-environment/bin/pip install --no-build-isolation --no-deps .
/path/to/fresh-environment/bin/ruff check --select E9,F63,F7,F82 src scripts tests
/path/to/fresh-environment/bin/ruff format --check src scripts tests
/path/to/fresh-environment/bin/python -I scripts/reproduce.py --workers 8 --output /path/to/new-results
```

The last three commands ran under `scripts/sandbox_run.py`, with explicit read/write roots and probes confirming that the old source files and TCP were inaccessible. The example commands use generic paths. Dependency installation was allowed network access before isolation.

## Results

| Check | Result |
|---|---|
| Pinned installation, fatal-error lint and formatting | PASS |
| CPU tests | 50 passed |
| Synthetic example | PASS; two labels on one image; exact Dice 1.0 |
| Three complete archive digests and all installed metadata checksums | PASS |
| Annotation masks, targets, dimensions, counts and replay files | PASS |
| Canonical benchmark | 53,080 rows; 40 summaries; maximum expected-table difference 0 |
| Comparison using the same 12 proposals | 29,194 rows; 22 summaries; 198 class rows; maximum difference 0 |
| Paired uncertainty | 50 rows; maximum difference 0 |
| Human reference | 1,327 task values and all five confidence intervals pass at 1e-12 |
| Complete-study analysis | All 8,074 annotations and four raters; nine expected CSVs; maximum difference 0 |
| Workflow groups | 615 with no recorded geometry action; 712 with an action; each task counted once |
| Recombination of workflow groups | Maximum difference 1.11e-16, below 1e-12 |

The evaluator computes predictions/regions, metrics and uncertainty before comparing them with expected tables. The seven saved VLM responses are parsed again, the original ordered proposals are decoded, and template and greedy/exact controls are reselected. No test targets tune a predictor; only the explicitly designated oracles use targets for selection.

The historical parser behavior and metric conventions are retained deliberately. Pinned dependencies emit deprecation warnings; these are recorded rather than fixed by changing the numerical protocol.

## Resources

The Slurm check requested 8 CPU workers, 10 GiB memory and no GPU. Scheduler elapsed time was **8 minutes 42 seconds**, including setup. The full Python harness measured 520.58 seconds; the reproduction command measured 505.23 seconds. Maximum child-process RSS was 6,285,696 KiB. Slurm batch MaxRSS was 10,388,168 KiB under its process-accounting scope; shared memory can be counted differently from the child-process statistic. These are measured resource records, not model-inference timings.

A separate newly created laptop environment using Python 3.12.14 also passed all 50 CPU tests and the synthetic example.

## Data and license scope

The complete export is supplied byte-for-byte as received from the author. Its final masks match the frozen benchmark; the current archive digests are listed in `data/bundles/SHA256SUMS`. All mask, prediction, proposal and response payloads match those used in the numerical check. [Current package integrity](verification/package_integrity_2026-10-29.json) records the archive checksums and validation results. The original complete-export audit also checked the component masks and COCO encodings. The current installer validates every complete-export file checksum, and the analysis recomputes the published process, workflow and modality tables.

Permission covers distribution of the annotation layer and sanitized metadata. Original photographs, model weights, private reviews, credentials and participant identities are excluded. Released pseudonyms and reduced relative timings are not a guarantee of irreversible anonymity.

## Limits

This check reproduces numerical results from saved predictions, proposals and annotations. It is not a new inference run for the seven VLMs or a retraining experiment. Auxiliary classifier, attribution/deletion, modality-prediction and geometric-detector inference are documented but are not rerun by the main command. The separate nine-task U-Net CPU inference check is described in [docs/CPU_INFERENCE.md](docs/CPU_INFERENCE.md); full GPU inference and training have not been rerun in this independent package.
