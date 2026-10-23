# Optional representative CPU inference

A separate nine-class inference check uses the label-conditioned U-Net checkpoint. It uses the lexicographically first task in each of nine classes, fixed before inference/comparison. This is a representative subset, not full GPU/VLM inference or training reproduction. No target masks tune the model. The tolerance was declared before execution: prediction Dice to the saved mask >=.999 and differing-pixel fraction <=.001.

Install base requirements and the project as in README, then install `torch==2.9.1+cpu` from `https://download.pytorch.org/whl/cpu`. `requirements-inference.freeze.txt` records the exact environment actually executed. Independently obtain the permitted KU-PCP test images and the author's label-conditioned U-Net checkpoint; weights/photos are not in the public bundles.

```sh
.venv/bin/python -I scripts/infer_unet_subset.py --annotations assets/data/release_candidate_v1.0.0-rc.2 --checkpoint assets/private/unet_label.pt --images assets/private/test_img --replay assets/data/replay_bundle_v1.0.0-rc.2 --output outputs/cpu_subset.json
```

Executed check: PASS for all nine predeclared tasks. Minimum Dice to saved predictions: 0.9997801846448983; maximum differing-pixel fraction: 8.409288194444444e-05. Floating CPU/CUDA differences are reported rather than claimed pixel-identical. Checkpoint SHA-256: `4a1c42d483580025154330936ac8b71cfa675579da0353a631c3d67970335c46`. The command writes per-task results to the chosen output file. Original photographs and the checkpoint must be supplied separately.
