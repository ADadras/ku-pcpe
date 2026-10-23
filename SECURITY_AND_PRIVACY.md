# Release boundary

This repository contains evaluation code, expected summary tables, three versioned data archives and the project page. The full KU-PCP photograph dataset, raw databases, original participant identities, calendar timestamps, free text, cursor coordinates, credentials, private reviews and model weights are excluded. The website includes only 18 resized photographs selected for the paper and nine thumbnails; their exact hashes are listed in `website/ASSET_MANIFEST.json` and their upstream rights are retained.

The frozen benchmark archives use pseudonymous expert slots and retain an earlier workflow categorization from one expert. The complete study archive adds released pseudonyms, scalar settings, tool counts and relative event bins for all four completed annotators. It excludes incomplete sets and demographic profiles. Relative times are rounded to five-second bins and capped at 30 minutes. Small-cohort pseudonyms do not guarantee irreversible anonymity.

`scripts/verify_release.py` checks archive bytes and SHA-256 digests, safe paths, file types, stripped archive-owner metadata, frozen evaluator hashes and source files for credential patterns and personal infrastructure paths. Its only photo exception is the exact hash-listed website illustrations; new or modified website assets fail validation until deliberately reviewed and recorded. The installer checks every complete-metadata file against its checksum manifest. The annotation validator and complete-study analysis verify mask and task coverage. These checks supplement the recorded privacy review; they cannot establish permissions by themselves.

Permission covers distribution of the annotations and sanitized metadata. Original images and upstream checkpoints retain their own terms.

The optional Linux clean-room harness uses Landlock to limit filesystem access to explicit roots and deny TCP connections. It checks isolation; it does not claim to secure arbitrary untrusted code.
