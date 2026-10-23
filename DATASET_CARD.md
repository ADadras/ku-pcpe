# Dataset card

KU-PCPE contains positive image-label grounding tasks for Horizontal, Vertical, Diagonal, Curved, Triangle, Rule of thirds, Center, Symmetric, and Pattern. A binary mask marks visible regions selected as support for a label already judged applicable in KU-PCP. Masks may contain objects, object parts, disconnected regions, background, or negative space. Abstract lines and axes are not rasterized as region targets.

The training split contains 4,093 tasks from 3,169 images and one annotator. The test split contains 1,327 tasks from 1,073 images, three expert masks per task, and strict and ambiguity-aware consensus targets. There are 8,074 individual annotations in total.

## Distribution

The primary dataset download is `kupcp_completed_public_20261023_v1.tar.gz`. It contains annotations and sanitized study metadata for all four completed annotators: ratings, component masks, scalar settings, tool counts, and relative event timelines. It is one dataset. Two separate reproduction archives arrange the same final masks for the evaluator and supply saved predictions, ordered proposals, and VLM responses.

Original photographs must be obtained independently under the KU-PCP distributor's terms. The data archives contain no photographs. The project page includes 18 resized research illustrations and nine thumbnails; their original rights remain in force. See [website attribution](website/NOTICE.md).

## Collection and intended use

The dataset supports research on queried-label region evidence and human/model agreement in photographic composition. It does not establish universal aesthetic quality or one correct interpretation for every photograph. Three test experts worked independently, had formal arts/design education and more than five years of professional visual practice, and could select, add, split, remove or refine proposed regions. They received the nine-class guide reproduced in the supplement. A separate annotator produced the training masks.

Known limitations include proposal-assisted collection, one-annotator training supervision, positive-only queries, and exploratory modality ratings with modest agreement. An independent annotation audit without proposals has not been collected. Training and test use different images and annotators, so their differences do not isolate an annotator effect. Tasks sharing one image may be correlated.

## Metadata and privacy

The masks describe photographic structure rather than identity or sensitive attributes. Original participant identities, calendar timestamps, coordinates, free text, raw geometry payloads, and private runtime metadata are excluded. Relative timelines use five-second bins and are capped at 30 minutes. Pseudonyms in a small cohort do not guarantee irreversible anonymity.

Process summaries use all 4,093 training and 3,981 test annotations. The paper's workflow comparison groups each of the 1,327 test tasks once: 615 have no geometry action recorded by any expert, and 712 have an action recorded by at least one expert. The evaluator's supporting annotation archive retains earlier categories from one expert (846/481 tasks) for protocol compatibility; those are not the paper's primary workflow groups. Discarded drafts and all later mask revisions cannot be reconstructed from retained events, including 71 symmetric-mask replacements. These records are not additional independent annotations. See [the metadata guide](docs/COMPLETE_STUDY_METADATA.md).

## Validation

Checks cover task/expert cardinality, mask dimensions and binary values, foreground areas, lossless PNG round trips, both consensus constructions, checksums, split-local joins and complete task coverage. All 8,074 final masks match between the primary dataset and the evaluator's annotation assets. Exact-file overlap between source-image splits was zero in the source-image check; near-duplicate scenes were not assessed.

The repository pins `Pillow==11.3.0` for the annotation validator. Run the installation and reproduction commands in the [README](README.md) to check the archives and recompute the benchmark, uncertainty, process summaries and workflow tables.

## Class distribution (image-label tasks)

| Class | Train | Test |
|---|---:|---:|
| horizontal | 758 | 211 |
| vertical | 195 | 66 |
| diagonal | 471 | 189 |
| curved | 235 | 92 |
| triangle | 374 | 171 |
| rule_of_thirds | 923 | 125 |
| center | 750 | 311 |
| symmetric | 232 | 99 |
| pattern | 155 | 63 |

## License and maintenance

The annotation layer, sanitized metadata and derived replay assets are licensed under CC BY 4.0; project code is MIT licensed. Permission covers distribution of the annotations and sanitized metadata. KU-PCP photographs and upstream checkpoints retain their original terms. See [NOTICE](NOTICE.md), [data license](data/LICENSE.md), and [citation](CITATION.cff).

Maintenance contact: Armin Dadras, adadras@cvl.tuwien.ac.at.
