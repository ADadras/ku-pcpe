# One dataset and separate reproduction assets

## Dataset download

**[Download KU-PCPE annotations and metadata — 69.5 MiB](kupcp_completed_public_20261023_v1.tar.gz?raw=true)**

This is the complete dataset: all 8,074 annotations and study metadata for the four completed annotators. Original photographs are obtained separately.

## Additional files for reproducing the paper

- [Annotations for the evaluator — 37.0 MiB](ku_pcpe_annotations_1.0.0-rc.2.tar.gz?raw=true): the same final masks arranged for the tested evaluator, with consensus masks and validation tools.
- [Saved predictions and proposals — 30.6 MiB](ku_pcpe_replay_1.0.0-rc.2.tar.gz?raw=true): model outputs, the ordered proposals, and responses/selections from all seven VLMs.

These two files support reproduction of the paper; they are not additional datasets. The full reproduction command uses all three archives. See the [main README](../../README.md#reproduce-the-tables) and [SHA-256 checksums](SHA256SUMS).
