"""Reproduce the benchmark and complete-study tables from the bundled assets."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--data", type=Path, default=ROOT / "assets/data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or platform.system() != "Linux":
        parser.error("The frozen evaluator requires Python 3.12 on Linux")
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = args.data.resolve()
    started = time.monotonic()
    report = {
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "workers": args.workers,
        "steps": [],
    }
    environment = dict(os.environ)
    for name in ["PYTHONPATH", "KUPCP_ROOT"]:
        environment.pop(name, None)
    environment.update(
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENCV_FOR_THREADS_NUM="1",
    )

    def run(name, command):
        print("Starting " + name, flush=True)
        with (output / (name + ".log")).open("w") as log:
            log.write("COMMAND " + json.dumps(command) + "\n")
            log.flush()
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        report["steps"].append({"step": name, "status": "PASS"})
        print(name + " PASS", flush=True)

    python = sys.executable
    try:
        run("release_integrity", [python, "-I", "scripts/verify_release.py"])
        run("pytest", [python, "-I", "-m", "pytest", "-q"])
        run("synthetic_fixture", [python, "-I", "scripts/run_smoke_test.py"])
        run(
            "install_assets",
            [python, "-I", "scripts/install_release.py", "--output", str(data)],
        )
        annotation = data / "release_candidate_v1.0.0-rc.2"
        replay = data / "replay_bundle_v1.0.0-rc.2"
        run(
            "annotation_validation",
            [python, "-I", str(annotation / "evaluate_smoke_test.py")],
        )
        run(
            "replay_validation",
            [
                python,
                "-I",
                "-c",
                "from ku_pcpe.validation import verify_replay; print(verify_replay("
                + repr(str(replay))
                + "))",
            ],
        )
        run(
            "benchmark",
            [
                python,
                "-I",
                "-m",
                "ku_pcpe.cli",
                "--annotations",
                str(annotation),
                "--replay",
                str(replay),
                "--expected",
                "expected",
                "--output",
                str(output / "benchmark"),
                "--workers",
                str(args.workers),
            ],
        )
        run(
            "human_intervals",
            [
                python,
                "-I",
                "scripts/summarize_human.py",
                "--results",
                str(output / "benchmark"),
                "--expected",
                "expected/human_ci.csv",
            ],
        )
        run(
            "complete_study",
            [
                python,
                "-I",
                "scripts/summarize_complete_study.py",
                "--metadata-root",
                str(data / "kupcp_completed_public_20261023_v1"),
                "--benchmark-results",
                str(output / "benchmark/canonical_per_task.csv"),
                "--expected-dir",
                "expected/workflow_all_raters",
                "--output",
                str(output / "complete_study"),
                "--workers",
                str(min(2, args.workers)),
            ],
        )
        report.update(
            status="PASS",
            benchmark=json.loads((output / "benchmark/status.json").read_text()),
            complete_study=json.loads(
                (output / "complete_study/status.json").read_text()
            ),
        )
    except Exception as error:
        report.update(status="FAIL", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "reproduction_status.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
